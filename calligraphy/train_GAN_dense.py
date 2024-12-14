import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
import random
import numpy as np

# Define paths for paired data (replace with your own paths)
input_source = "/gpfsnyu/scratch/yl10337/bdsr_source"
input_target = "/gpfsnyu/scratch/yl10337/bdsr/1000"
checkpoint_dir = "/gpfsnyu/scratch/yl10337/GAN_checkpoints"
output_images_dir = "/gpfsnyu/scratch/yl10337/GAN_outputimages"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# Define a simple dataset for paired data
class PairedDataset(Dataset):
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_files = sorted(os.listdir(source_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return min(len(self.source_files), len(self.target_files))

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_dir, self.source_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        source_image = Image.open(source_path).convert("L")  # Grayscale
        target_image = Image.open(target_path).convert("L")  # Grayscale

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return source_image, target_image, self.source_files[idx]

# Define transformations and DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = PairedDataset(input_source, input_target, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# DenseBlock for Generator
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
        return torch.cat([x, self.block(x)], 1)  # Concatenate along the channel dimension

# Generator with Dense Blocks
class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_dense_blocks=5):
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

        # Dense Blocks
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

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.L1Loss()  # Pixel-wise loss for paired data

# Pre-select 5 fixed characters
fixed_indices = random.sample(range(len(dataset)), 5)
fixed_samples = [dataset[idx] for idx in fixed_indices]

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    epoch_loss_G = 0
    epoch_loss_D = 0

    for i, (source, target, _) in enumerate(tqdm(data_loader)):
        source, target = source.to(device), target.to(device)
        
        # Get the output size from the discriminator for the current input shape
        valid = torch.ones_like(discriminator(target))
        fake = torch.zeros_like(discriminator(target))

        # Train Generator
        optimizer_G.zero_grad()
        fake_target = generator(source)
        pred_fake = discriminator(fake_target)
        
        # Adversarial loss and pixelwise loss
        loss_G_adv = adversarial_loss(pred_fake, valid)
        loss_G_pixel = pixelwise_loss(fake_target, target)
        loss_G = loss_G_adv + 100 * loss_G_pixel  # Weighted sum of losses
        
        loss_G.backward()
        optimizer_G.step()
        epoch_loss_G += loss_G.item()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(target)
        loss_D_real = adversarial_loss(pred_real, valid)

        pred_fake = discriminator(fake_target.detach())
        loss_D_fake = adversarial_loss(pred_fake, fake)
        
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        epoch_loss_D += loss_D.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss G: {epoch_loss_G / len(data_loader):.4f}, Loss D: {epoch_loss_D / len(data_loader):.4f}")

    # Save checkpoint every epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

    # Save images for fixed characters
    generator.eval()
    for idx, (source, target, filename) in enumerate(fixed_samples):
        source = source.unsqueeze(0).to(device)
        with torch.no_grad():
            generated = generator(source).squeeze().cpu().numpy()
            generated = (generated * 0.5 + 0.5) * 255  # Unnormalize and scale to [0, 255]
            generated = generated.astype(np.uint8)

            # Save images
            output_dir = os.path.join(output_images_dir, f"epoch_{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            generated_image_path = os.path.join(output_dir, f"generated_{filename}")
            Image.fromarray(generated).save(generated_image_path)
    print(f"Generated images for epoch {epoch+1} saved.")