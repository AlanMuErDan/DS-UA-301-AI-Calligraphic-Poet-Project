# Sources
# Documentation that I used/referenced code from: 
# GIT: https://huggingface.co/docs/transformers/en/model_doc/git#transformers.GitForCausalLM.forward.example
# MBart-50: https://huggingface.co/docs/transformers/en/model_doc/mbart
# specific pretrained model: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

import warnings
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoProcessor
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re
import os

warnings.filterwarnings("ignore")


# I will be using the GIT model to generate English captions

model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Get the pretrained image processor
processor = AutoProcessor.from_pretrained("microsoft/git-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# using mbart-large-50-many-to-many-mmt model to translate captions from Chinese to English 
# importing the model and tokenizer 
translate_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translate_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
translate_tokenizer.src_lang = "en_XX" # set source language to English 





# creating the captions for the validation set 
image_valid_dataset = load_dataset("imagefolder", data_dir = 'data/output_valid', split="train") # doing train just so the data don't split

valid_captions = []
batch_size = 50 # update in batches so GPU does not run out of memory 
num_iters = int(image_valid_dataset.shape[0] / batch_size) 

for i in range(1, num_iters +1):    
    start = batch_size * (i-1)
    end = batch_size * i

    processed_valid_image = processor(images=image_valid_dataset['image'][start:end], padding="max_length", return_tensors = 'pt')
    generated_ids = model.generate(pixel_values=processed_valid_image['pixel_values'].to(device), max_length = 10)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    valid_captions = [*valid_captions, *generated_caption]

# when manually checking the captions, I realized that the model used for translation was not able to recognize "against"
# replace 'against' with 'with' in captions
for i in range(len(valid_captions)):
     valid_captions[i] = valid_captions[i].replace("against", "with")


# generate the translated validation captions (English to Chinese)
translated_valid_caps = []
for i in range(1, num_iters +1):
    start = batch_size * (i-1)
    end = batch_size * i

    encoded_caps = translate_tokenizer(valid_captions[start:end], return_tensors="pt", padding=True)
    generated_tokens = translate_model.generate(**encoded_caps, forced_bos_token_id=translate_tokenizer.lang_code_to_id["zh_CN"])
    translated_caps = translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    translated_valid_caps = [*translated_valid_caps, *translated_caps]


# after manually checking the captions, the rest of the untranslated English words are probably because it is not recognized 
# in most cases, removing those words do not affect the overall meaning
# so I will remove the English words
for i in range(len(translated_valid_caps)):
    translated_valid_caps[i] = re.sub('[a-zA-Z]', "", translated_valid_caps[i])


# write the validation captions to a file 
f = open("data/valid.txt", "w")
for cap in translated_valid_caps:
    f.write(cap)
    f.write("\n")
f.close()




# repeat the same process for the captions for the training set 
image_train_dataset = load_dataset("imagefolder", data_dir = 'data/output_train', split="train") # doing train just so the data don't split

# caption generation 
train_captions = []
batch_size = 50
num_iters = int(image_train_dataset.shape[0] / batch_size) + 1 # add 1 because it rounds down

for i in range(1, num_iters +1):    
    start = batch_size * (i-1)
    end = batch_size * i

    processed_train_image = processor(images=image_train_dataset['image'][start:end], padding="max_length", return_tensors = 'pt')
    generated_ids = model.generate(pixel_values=processed_train_image['pixel_values'].to(device), max_length = 10)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    train_captions = [*train_captions, *generated_caption]

# replace 'against' with 'with' in captions
for i in range(len(train_captions)):
    train_captions[i] = train_captions[i].replace("against", "with")
    

# translate captions 
translated_train_caps = []
for i in range(1, num_iters +1):
    start = batch_size * (i-1)
    end = batch_size * i

    encoded_caps = translate_tokenizer(train_captions[start:end], return_tensors="pt", padding=True)
    generated_tokens = translate_model.generate(**encoded_caps, forced_bos_token_id=translate_tokenizer.lang_code_to_id["zh_CN"])
    translated_caps = translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    translated_train_caps = [*translated_train_caps, *translated_caps]

for i in range(len(translated_train_caps)):
    translated_train_caps[i] = re.sub('[a-zA-Z][#-]', "", translated_train_caps[i])
    translated_train_caps[i] = re.sub('[#-]', "", translated_train_caps[i])

# write to file 
f = open("data/train.txt", "w")
for cap in translated_train_caps:
    f.write(cap)
    f.write("\n")
f.close()





# write csv file to match images with captions 
# this is used for loading the data and captions during model creation phase

# training set 
train_dir = os.path.join(os.getcwd(), 'data/output_train')

# get name of files (excluding the extensions like '.jpeg')
file_names = []
for filename in os.listdir(train_dir):
    file_names.append(filename)
file_names.sort()
file_names = file_names[1:] # first item is not image 

# get list of captions from text file of captions 
train_caps = []
with open('data/train.txt', 'r') as f:
    captions = f.readlines()
    for cap in captions:
        train_caps.append(cap.replace(",", "。").replace("\n", "")) # replace , with 。because comma will interfere with csv file

# write csv file to match up image with caption
with open(os.path.join(train_dir, "metadata.csv"), 'w') as f:
    f.write("file_name,text\n") # specifies format
    for i in range(len(file_names)):
        f.write(file_names[i] + "," + train_caps[i] + "\n")


# validation set 
valid_dir = os.path.join(os.getcwd(), 'data/output_valid')

# get name of files (without the extensions like '.jpeg')
file_names = []
for filename in os.listdir(valid_dir):
    file_names.append(filename)
file_names.sort()
file_names = file_names[1:] # first item is not image 

# get list of captions
valid_caps = []
with open('data/valid.txt', 'r') as f:
    captions = f.readlines()
    for cap in captions:
        valid_caps.append(cap.replace(",", "。").replace("\n","")) # replace , with 。because comma will interfere with csv file

# write csv file to match up image with caption
with open(os.path.join(valid_dir, "metadata.csv"), 'w') as f:
    f.write("file_name,text\n") # specifies format
    for i in range(len(file_names)):
        f.write(file_names[i] + "," + valid_caps[i]+ "\n")


