# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
checkpoint_path = "/gpfsnyu/scratch/yl10337/cycleGAN_checkpoints/checkpoint_epoch_3.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
G_XtoY.load_state_dict(checkpoint["G_XtoY"])
G_XtoY.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the processing function
def process_image_folder(input_folder, output_folder, model, transform, device):
    """
    Processes all images in the input folder using the trained CycleGAN model and saves them
    to the output folder with the same filenames.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        model (torch.nn.Module): Trained Generator model.
        transform (torchvision.transforms.Compose): Transformations for input images.
        device (torch.device): Device for computation (CPU or GPU).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Load and preprocess the image
        image = Image.open(input_path).convert("L")
        processed_image = transform(image).unsqueeze(0).to(device)

        # Generate the output using the model
        with torch.no_grad():
            output = model(processed_image).squeeze().cpu().numpy()
            output = (output * 0.5 + 0.5) * 255  # Denormalize to [0, 255]
            output = output.astype(np.uint8)

        # Save the processed image
        Image.fromarray(output).save(output_path)

# Example usage
if __name__ == "__main__":
    input_folder = "/gpfsnyu/scratch/yl10337/bdsr_source"
    output_folder = "/gpfsnyu/scratch/yl10337/cycleGAN_processed_bdsr"
    process_image_folder(input_folder, output_folder, G_XtoY, transform, device)
    print(f"Processed images saved to {output_folder}")