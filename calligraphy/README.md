# Calligraphy Generation

## Data

Data needed is varied for **GAN** and **CycleGAN** model. 

## GAN

## CycleGAN

## Comparison

We use two evaluation metrics to evaluate different models:

- **Consistent Score:** We trained a CNN-based classification model to differentiate between various calligraphy styles. After comparing several architectures, we selected DenseNet as the base model. The well-trained DenseNet model serves as a style-consistent feature extractor. To rigorously evaluate the model’s ability to learn styles, we calculate the cosine similarity between the generated images and the real images. You can train your own feature extractor in the *train_classifier* folder. The calculation function is stored in *utils* folder name ``calculate_style_consistency_score.py``. 