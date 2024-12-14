import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.models import densenet121

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

#----------------------------------
# change this to your own directory
#----------------------------------
train_dir = "./train"
test_dir = "./test"

# Directory to save models
model_save_dir = "./densenet121_binary_checkpoints"
os.makedirs(model_save_dir, exist_ok=True)

# Transformations
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images for DenseNet
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
}

#---------------------------------------------
# choose the classes you want to classify
#---------------------------------------------
selected_classes = ['class1', 'class2']  

# filter dataset
def filter_dataset(dataset, class_names):
    assert all(cls in dataset.classes for cls in class_names), \
        f"Selected classes {class_names} not found in dataset classes {dataset.classes}"
    
    selected_indices = [dataset.class_to_idx[cls] for cls in class_names]
    filtered_samples = [
        (path, selected_indices.index(label))  
        for path, label in dataset.samples if label in selected_indices
    ]
    
    dataset.samples = filtered_samples
    dataset.classes = class_names
    dataset.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    return dataset

# prepare dataloader
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform["train"])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform["test"])
train_dataset = filter_dataset(train_dataset, selected_classes)
test_dataset = filter_dataset(test_dataset, selected_classes)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model
model = densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)  # 修改分类器
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train function
def train_model_with_checkpoints(model, dataloaders, criterion, optimizer, num_epochs=10, save_dir="./densenet121_binary_checkpoints"):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  
                dataloader = dataloaders["train"]
            else:
                model.eval()  
                dataloader = dataloaders["test"]

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloader, desc=f"{phase.capitalize()} Phase", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # save model 
        checkpoint_path = os.path.join(save_dir, f"densenet121_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    return model

dataloaders = {
    "train": train_loader,
    "test": test_loader,
}

trained_model = train_model_with_checkpoints(model, dataloaders, criterion, optimizer, num_epochs=10, save_dir=model_save_dir)