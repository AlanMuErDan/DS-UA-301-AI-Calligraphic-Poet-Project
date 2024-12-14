import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Define a function to calculate style consistency score
def calculate_style_consistency_score(model, image1, image2, device):
    """
    Calculate the style consistency score based on cosine similarity of features extracted by the model.

    Args:
        model (torch.nn.Module): Pretrained DenseNet model as a feature extractor.
        image1 (PIL.Image or torch.Tensor): First image.
        image2 (PIL.Image or torch.Tensor): Second image.
        device (torch.device): Device to perform computation.

    Returns:
        float: Cosine similarity score between extracted features of image1 and image2.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(image1, Image.Image):
            image1 = transform(image1).unsqueeze(0).to(device)
        if isinstance(image2, Image.Image):
            image2 = transform(image2).unsqueeze(0).to(device)
        features1 = model(image1).squeeze().cpu().numpy()
        features2 = model(image2).squeeze().cpu().numpy()
        similarity = cosine_similarity([features1], [features2])[0, 0]
    return similarity


# Main testing block
if __name__ == "__main__":
    # ------------------------------------------
    # load checkpoint your checkpoint here 
    # ------------------------------------------
    model_checkpoint_path = "checkpoint.pth"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 2) 
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.classifier = nn.Identity() 
    model = model.to(device)

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # generate random images
    random_image1 = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
    random_image2 = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
    score = calculate_style_consistency_score(model, random_image1, random_image2, device)
    print(f"Style Consistency Score between random images: {score:.4f}")