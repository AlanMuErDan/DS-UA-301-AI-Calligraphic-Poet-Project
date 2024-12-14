# Transfer Learning and Fine tuning
# Sources/Tutorials That I Referenced or used code from: 
# for fine tuning using transformers (Transformers Tutorial 1): https://huggingface.co/docs/transformers/main/en/tasks/image_captioning 
# for fine tuning using transformers (Transformers Tutorial 2): https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb 
# for how to freeze specific layers of a model: https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6

# Documentation For Models/Libraries referenced: 
# GIT model: https://huggingface.co/docs/transformers/main/en/model_doc/git 
# Dataset Loader: https://huggingface.co/docs/datasets/en/loading 
# ImageFolder: https://huggingface.co/docs/datasets/main/en/image_dataset#imagefolder 
# MBart-50: https://huggingface.co/docs/transformers/en/model_doc/mbart 
# specific pretrained model: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
# AutoTokenizer and AutoImageProcessor: https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer 


# References/Documentation used for Model Evaluation 
# https://www.nltk.org/api/nltk.translate.bleu_score.html#module-nltk.translate.bleu_score
# https://www.nltk.org/api/nltk.translate.meteor_score.html#module-nltk.translate.meteor_score

import warnings
import os
from datasets import load_dataset
from transformers import AutoImageProcessor
from transformers import MBart50TokenizerFast
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import random

import nltk
nltk.download('omw')
nltk.download('wordnet')

warnings.filterwarnings("ignore")






# Model Training 



# loading the training dataset and correponding captions 
train_dir = os.path.join(os.getcwd(), 'data/output_train')
image_train_dataset = load_dataset("imagefolder", data_dir = train_dir, split="train") # doing train just so the data don't split


# preprocess the image and text data so they can be used in the model

# For image use the default image processor
processor = AutoImageProcessor.from_pretrained("microsoft/git-base")
processed_train_image = processor(images=image_train_dataset['image'], return_tensors = 'pt')

# for the captions, we cannot use Bert with the GIT model because the captions are in Chinese
# need to use other libaries: use MBart50 instead
# this is also the tokenizer used to generate the captions 
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
processed_train_captions = tokenizer(image_train_dataset['text'], return_tensors = 'pt', padding = True)


# load the data 
caption_dataloader = DataLoader(processed_train_captions['input_ids'], batch_size=32)
caption_batch = next(iter(caption_dataloader))

image_dataloader = DataLoader(processed_train_image['pixel_values'], batch_size=32)
image_batch = next(iter(image_dataloader))

attention_mask_dataloader = DataLoader(processed_train_captions['attention_mask'], batch_size=32)
attention_mask_batch = next(iter(attention_mask_dataloader))

# this is the number of batches that will be processed in each epoch
# this is to make sure that the GPU does not run out of memory
# feel free to change the batch size above for the DataLoader depending on your specific GPU 
count = 0
for idx, batch in enumerate(caption_dataloader):
  count+=1






# Making Predictions 

# validation data
valid_dir = os.path.join(os.getcwd(), 'data/output_valid')
image_valid_dataset = load_dataset("imagefolder", data_dir = valid_dir, split="train") # doing train just so the data don't split


# get list of captions
# will need this later to calculate the evaluation metrics 
valid_caps = []
with open('data/valid.txt', 'r') as f:
    captions = f.readlines()
    for cap in captions:
        valid_caps.append(cap.replace(",", "。").replace("\n","")) 


# This will loop through potential configurations to find the best hyperparameters
# Use random search 
# referenced code from Transformers Tutorial 2 for the model training/fine-tuning part
# see beginning for link/details

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rates = [5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]
epochs = [10,12,14,16,18,20]

def train_git(learning_rate, num):
    print("____________________")
    print("Number of Epochs: ", num)
    print("Learning Rate: ", learning_rate)
    
    # getting model and optimizer
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.to(device)

    for name, param in model.named_parameters(): 
        if "image_encoder" in name:
            param.requires_grad = False # freeze these layers
        else:
            param.requires_grad = True 

    # I am assuming that the image_encoder layers extracts the features from the image
    # and that the encoder layers are what turns the image features to text
    # since we want keep the image weights, the image_encoder layers will be frozen 
    # everything else will be changed 
    
    not_freeze = [param for param in model.parameters() if param.requires_grad]
 
    optimizer = torch.optim.AdamW(not_freeze, lr=learning_rate)
    model.train()

    
    # model training/fine-tuning
    for epoch in range(num):
      print("Epoch: ", epoch + 1)
    
      # change it to iterable object so we can train in batches
      caption_iter = iter(caption_dataloader)
      image_iter = iter(image_dataloader)
      attention_mask_iter = iter(attention_mask_dataloader)
    
      for i in range(count): #this depends on number of batches
        input_ids = next(caption_iter).to(device)
        pixel_values = next(image_iter).to(device)
        attention_mask = next(attention_mask_iter).to(device)
    
        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask = attention_mask,
                        labels=input_ids)
    
        loss = outputs.loss
        if i == count-1:
            print("Loss:", loss)
    
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()




    # process images
    pred_valid_captions = []
    batch_size = 32
    num_iters = int(image_valid_dataset.shape[0] / batch_size) + 1 # add one because it rounds down 
    
    # predict captions
    for i in range(1, num_iters +1):    
        start = batch_size * (i-1)
        end = batch_size * i
    
        processed_valid_image = processor(images=image_valid_dataset['image'][start:end], return_tensors = 'pt')
        generated_ids = model.generate(pixel_values=processed_valid_image['pixel_values'].to(device), max_length = 10)
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        pred_valid_captions = [*pred_valid_captions, *generated_caption]
    
    # remove the "_ " in front of each caption 
    for i in range(len(pred_valid_captions)):
        pred_valid_captions[i] = pred_valid_captions[i].replace("_ ", "")


    # evaluation  
    sum = 0
    for i in range(len(valid_caps)):
        bleu = nltk.translate.bleu_score.sentence_bleu([tokenizer.tokenize(valid_caps[i])[1:]], tokenizer.tokenize(pred_valid_captions[i])[1:], weights=[1])
        sum+= bleu
    bleu_one = sum / len(valid_caps)
    print("bleu1 score: ",bleu_one)        

    
    sum = 0
    for i in range(len(valid_caps)):
        bleu = nltk.translate.bleu_score.sentence_bleu([tokenizer.tokenize(valid_caps[i])[1:]], tokenizer.tokenize(pred_valid_captions[i])[1:],weights=(0.5, 0.5))
        sum+= bleu
    bleu_two = sum / len(valid_caps)
    print("bleu2 score: ",bleu_two)

        
    sum = 0
    for i in range(len(valid_caps)):
        meteor = nltk.translate.meteor_score.single_meteor_score(tokenizer.tokenize(valid_caps[i])[1:], tokenizer.tokenize(pred_valid_captions[i])[1:], gamma = 0)
        sum+= meteor
    meteor = sum / len(valid_caps)
    print('meteor score: ', meteor)
    

    # delete current model and optimizer to clear GPU memory
    del model, optimizer
    torch.cuda.empty_cache()




# random search 
num_trials = 2

for i in range(num_trials):
    temp_lr_index = int(round(random.uniform(0, len(learning_rates) - 1),0))
    temp_lr = learning_rates[temp_lr_index]

    temp_epoch_index = int(round(random.uniform(0, len(epochs) - 1),0))
    temp_epoch = epochs[temp_epoch_index]

    train_git(temp_lr, temp_epoch)





# from running the previous code, the best hyperparameters are: num_epoch = 18 and learning_rate = 0.0001
# run the model with the optimized hyperparameters

# getting model and optimizer
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
model.resize_token_embeddings(tokenizer.vocab_size)
model.to(device)

for name, param in model.named_parameters(): 
    if "image_encoder" in name:
        param.requires_grad = False # freeze these layers
    else:
        param.requires_grad = True 

not_freeze = [param for param in model.parameters() if param.requires_grad]

optimizer = torch.optim.AdamW(not_freeze, lr=0.0001)
model.train()


# model training/fine-tuning
for epoch in range(18):
  print("Epoch: ", epoch + 1)

  # change it to iterable object so we can train in batches
  caption_iter = iter(caption_dataloader)
  image_iter = iter(image_dataloader)
  attention_mask_iter = iter(attention_mask_dataloader)

  for i in range(count): #this depends on number of batches
    input_ids = next(caption_iter).to(device)
    pixel_values = next(image_iter).to(device)
    attention_mask = next(attention_mask_iter).to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask = attention_mask,
                    labels=input_ids)

    loss = outputs.loss
    if i == count-1:
        print("Loss:", loss)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()


# process images
pred_valid_captions = []
batch_size = 32
num_iters = int(image_valid_dataset.shape[0] / batch_size) + 1 # add one because it rounds down 

# predict captions
for i in range(1, num_iters +1):    
    start = batch_size * (i-1)
    end = batch_size * i

    processed_valid_image = processor(images=image_valid_dataset['image'][start:end], return_tensors = 'pt')
    generated_ids = model.generate(pixel_values=processed_valid_image['pixel_values'].to(device), max_length = 10)
    generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    pred_valid_captions = [*pred_valid_captions, *generated_caption]

# remove the "_ " in front of each caption 
for i in range(len(pred_valid_captions)):
    pred_valid_captions[i] = pred_valid_captions[i].replace("_ ", "")



# evaluation  
sum = 0
for i in range(len(valid_caps)):
    bleu = nltk.translate.bleu_score.sentence_bleu([tokenizer.tokenize(valid_caps[i])[1:]], tokenizer.tokenize(pred_valid_captions[i])[1:], weights=[1])
    sum+= bleu
bleu_one = sum / len(valid_caps)
print("bleu1 score: ",bleu_one)        


sum = 0
for i in range(len(valid_caps)):
    bleu = nltk.translate.bleu_score.sentence_bleu([tokenizer.tokenize(valid_caps[i])[1:]], tokenizer.tokenize(pred_valid_captions[i])[1:],weights=(0.5, 0.5))
    sum+= bleu
bleu_two = sum / len(valid_caps)
print("bleu2 score: ",bleu_two)

    
sum = 0
for i in range(len(valid_caps)):
    meteor = nltk.translate.meteor_score.single_meteor_score(tokenizer.tokenize(valid_caps[i])[1:], tokenizer.tokenize(pred_valid_captions[i])[1:], gamma = 0)
    sum+= meteor
meteor = sum / len(valid_caps)
print('meteor score: ', meteor)


# save the model weights
torch.save(model, os.path.join(os.getcwd(), 'final_model'))

