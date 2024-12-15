# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Define DenseBlock and Generator for CycleGAN
class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=256):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(growth_rate)
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_dense_blocks=5):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        channels = 256
        dense_blocks = []
        for _ in range(num_dense_blocks):
            dense_blocks.append(DenseBlock(channels, growth_rate=256))
            channels += 256
        self.transfer = nn.Sequential(*dense_blocks)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.transfer(x)
        x = self.upsampling(x)
        return self.final(x)

# Load the trained CycleGAN model
G_XtoY = Generator().to(device)
checkpoint_path = "/gpfsnyu/scratch/yl10337/cycleGAN_checkpoints/checkpoint_epoch_52.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
G_XtoY.load_state_dict(checkpoint['G_XtoY'])
G_XtoY.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Inference function
def infer_and_save_grid(input_string, source_folder, model, transform, device, output_path):
    """
    Performs inference on the input string, generating a vertical grid of calligraphy works arranged
    from right to left for each separated part of the string.

    Args:
        input_string (str): Chinese character string, separated by ',' and '.'.
        source_folder (str): Path to source images.
        model (torch.nn.Module): Trained Generator model (G_XtoY).
        transform (torchvision.transforms.Compose): Preprocessing transformations.
        device (torch.device): Device for inference.
        output_path (str): Path to save the final grid image.
    """
    # Split the input string by ',' and '.'
    entries = [entry.strip() for entry in input_string.replace('.', ',').split(',') if entry.strip()]
    
    # Determine grid size
    max_length = max(len(entry) for entry in entries)
    num_entries = len(entries)

    # Initialize the grid array
    grid = np.ones((num_entries * 128, max_length * 128)) * 255  # White background

    for row_idx, entry in enumerate(entries):
        for col_idx, char in enumerate(reversed(entry)):  # Right to left
            img_path = os.path.join(source_folder, f"{char}.png")
            if os.path.exists(img_path):
                # Load and preprocess the image
                source_img = Image.open(img_path).convert("L")
                processed_img = transform(source_img).unsqueeze(0).to(device)

                # Generate the calligraphy image
                with torch.no_grad():
                    generated_img = model(processed_img).squeeze().cpu().numpy()
                    generated_img = (generated_img * 0.5 + 0.5) * 255  # Unnormalize to [0, 255]
                    generated_img = generated_img.astype(np.uint8)

                # Place the image in the grid
                grid[row_idx * 128:(row_idx + 1) * 128, 
                     (max_length - col_idx - 1) * 128:(max_length - col_idx) * 128] = generated_img

    # Save the grid image
    Image.fromarray(grid).convert("L").save(output_path)


# Example usage with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate calligraphy grid from input string using CycleGAN")
    parser.add_argument("--input_string", type=str, required=True, help="Chinese character string (e.g., 枯藤老树昏鸦,小桥流水人家,古道西风瘦马.夕阳西下,断肠人在天涯)")
    parser.add_argument("--source_folder", default="/gpfsnyu/scratch/yl10337/normal_pingfang",type=str, required=False, help="Path to source images")
    parser.add_argument("--output_file", default="/gpfsnyu/scratch/yl10337/cycleGAN_calligraphy_grid.png", type=str, required=False, help="Path to save the output grid image")
    args = parser.parse_args()

    infer_and_save_grid(args.input_string, args.source_folder, G_XtoY, transform, device, args.output_file)
    print(f"Calligraphy grid saved to {args.output_file}")