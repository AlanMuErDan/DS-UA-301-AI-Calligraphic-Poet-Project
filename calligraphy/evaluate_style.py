# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import the style consistency score function
from utils.calculate_style_consistency_score import calculate_style_consistency_score

def evaluate_style_consistency(generated_images_path, real_images_path, model, transform, device):
    """
    Evaluate style consistency by calculating the average cosine similarity score
    between features of generated images and real images.

    Args:
        generated_images_path (str): Path to the directory containing generated images.
        real_images_path (str): Path to the directory containing real images.
        model (torch.nn.Module): Pretrained DenseNet model used as a feature extractor.
        transform (torchvision.transforms.Compose): Transformations to apply to the images.
        device (torch.device): Device for computation (CPU or GPU).

    Returns:
        float: Average style consistency score.
    """
    generated_images = sorted(os.listdir(generated_images_path))
    real_images = sorted(os.listdir(real_images_path))

    # Ensure both directories have the same number of images
    assert len(generated_images) == len(real_images), "Mismatch in the number of images between directories."

    scores = []
    for gen_img_name, real_img_name in tqdm(zip(generated_images, real_images), total=len(generated_images), desc="Evaluating"):
        gen_img_path = os.path.join(generated_images_path, gen_img_name)
        real_img_path = os.path.join(real_images_path, real_img_name)

        # Load images
        gen_img = Image.open(gen_img_path).convert("RGB")
        real_img = Image.open(real_img_path).convert("RGB")

        # Calculate the style consistency score for the pair
        score = calculate_style_consistency_score(model, gen_img, real_img, device)
        scores.append(score)

    # Compute the average score
    average_score = np.mean(scores)
    return average_score

if __name__ == "__main__":
    # ------------------------------------------
    # Load the checkpoint
    # ------------------------------------------
    model_checkpoint_path = "/gpfsnyu/scratch/yl10337/densenet121_binary_checkpoints/densenet121_csl.pth"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # Replace classifier for feature extraction
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.classifier = nn.Identity()  # Use DenseNet as a feature extractor
    model = model.to(device)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Define paths for generated and real images
    generated_images_path = "/gpfsnyu/scratch/yl10337/cycleGAN_processed_bdsr"
    real_images_path = "/gpfsnyu/scratch/yl10337/bdsr/1000"

    # Calculate the average style consistency score
    avg_score = evaluate_style_consistency(generated_images_path, real_images_path, model, transform, device)
    print(f"Average Style Consistency Score: {avg_score:.4f}")