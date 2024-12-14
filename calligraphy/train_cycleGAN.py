import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
from tqdm import tqdm
import numpy as np

# --------------------------------------
# Define your file paths
# --------------------------------------
train_source_data_path = "/gpfsnyu/scratch/yl10337/bdsr/bdsr_source"
train_target_data_path = "/gpfsnyu/scratch/yl10337/bdsr/1000"
checkpoint_dir = "/gpfsnyu/scratch/yl10337/cycleGAN_checkpoints"
output_images_dir = "/gpfsnyu/scratch/yl10337/cycleGAN_outputimages"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(os.listdir(root))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("L")  # Single-channel grayscale
        if self.transform:
            img = self.transform(img)
        return img, self.files[index]

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Single-channel grayscale normalization
])

# Load datasets
train_source_dataset = CustomDataset(train_source_data_path, transform=transform)
train_target_dataset = CustomDataset(train_target_data_path, transform=transform)

train_source_loader = DataLoader(train_source_dataset, batch_size=1, shuffle=True)
train_target_loader = DataLoader(train_target_dataset, batch_size=1, shuffle=True)

# Define the Generator and Discriminator
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
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize models, optimizers, and loss functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
G_XtoY = Generator().to(device)
G_YtoX = Generator().to(device)
D_X = Discriminator().to(device)
D_Y = Discriminator().to(device)

optimizer_G = optim.Adam(list(G_XtoY.parameters()) + list(G_YtoX.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_X = optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()

# Pre-select 5 fixed images
fixed_images = random.sample(os.listdir(train_source_data_path), 5)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    G_XtoY.train()
    G_YtoX.train()
    for source, target in tqdm(zip(train_source_loader, train_target_loader), total=min(len(train_source_loader), len(train_target_loader))):
        source, target = source[0].to(device), target[0].to(device)

        # Train Generators
        optimizer_G.zero_grad()
        fake_target = G_XtoY(source)
        reconstructed_source = G_YtoX(fake_target)
        loss_cycle_X = cycle_loss(reconstructed_source, source)
        fake_source = G_YtoX(target)
        reconstructed_target = G_XtoY(fake_source)
        loss_cycle_Y = cycle_loss(reconstructed_target, target)
        loss_G = adversarial_loss(D_Y(fake_target), torch.ones_like(D_Y(fake_target))) + \
                 adversarial_loss(D_X(fake_source), torch.ones_like(D_X(fake_source))) + \
                 10 * (loss_cycle_X + loss_cycle_Y)
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminators
        optimizer_D_X.zero_grad()
        loss_D_X = adversarial_loss(D_X(source), torch.ones_like(D_X(source))) + \
                   adversarial_loss(D_X(fake_source.detach()), torch.zeros_like(D_X(fake_source)))
        loss_D_X.backward()
        optimizer_D_X.step()

        optimizer_D_Y.zero_grad()
        loss_D_Y = adversarial_loss(D_Y(target), torch.ones_like(D_Y(target))) + \
                   adversarial_loss(D_Y(fake_target.detach()), torch.zeros_like(D_Y(fake_target)))
        loss_D_Y.backward()
        optimizer_D_Y.step()

    # Save checkpoints
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'G_XtoY': G_XtoY.state_dict(),
        'G_YtoX': G_YtoX.state_dict(),
        'D_X': D_X.state_dict(),
        'D_Y': D_Y.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_X': optimizer_D_X.state_dict(),
        'optimizer_D_Y': optimizer_D_Y.state_dict()
    }, checkpoint_path)

    # Generate and save fixed character outputs
    G_XtoY.eval()
    output_dir = os.path.join(output_images_dir, f"epoch_{epoch+1}")
    os.makedirs(output_dir, exist_ok=True)
    for img_name in fixed_images:
        img_path = os.path.join(train_source_data_path, img_name)
        img = Image.open(img_path).convert("L")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_img = G_XtoY(img).squeeze().cpu().numpy()
            fake_img = (fake_img * 0.5 + 0.5) * 255  # Denormalize
            fake_img = fake_img.astype(np.uint8)
            Image.fromarray(fake_img).save(os.path.join(output_dir, f"generated_{img_name}"))

    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Checkpoint and outputs saved.")