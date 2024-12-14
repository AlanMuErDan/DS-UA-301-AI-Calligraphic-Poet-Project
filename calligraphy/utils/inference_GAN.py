# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Load Generator model and checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Define DenseBlock and Generator model
class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=256):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(growth_rate)
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_dense_blocks=5):
        super(Generator, self).__init__()
        # Encoder
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
        # Transfer with Dense Blocks
        channels = 256
        dense_blocks = []
        for _ in range(num_dense_blocks):
            dense_blocks.append(DenseBlock(channels, growth_rate=256))
            channels += 256
        self.transfer = nn.Sequential(*dense_blocks)
        
        # Decoder
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

generator = Generator(1, 1).to(device)
checkpoint_path = "/gpfsnyu/scratch/yl10337/GAN_checkpoints/checkpoint_epoch_99.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

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
        model (torch.nn.Module): Trained Generator model.
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

# Example usage
if __name__ == "__main__":
    input_string = "暗吹散残红,长随风.暗云低垂."
    source_image_folder = "/gpfsnyu/scratch/yl10337/normal_pingfang"
    output_file_path = "/gpfsnyu/scratch/yl10337/calligraphy_grid.png"
    infer_and_save_grid(input_string, source_image_folder, generator, transform, device, output_file_path)
    print(f"Calligraphy grid saved to {output_file_path}")