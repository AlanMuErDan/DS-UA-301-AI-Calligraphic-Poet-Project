# Captioning Images

## Data
The image dataset is compiled from 8 different datasets on Kaggle, cited below:

1. W. Ahmed, <em>Boat vs. Sea Images Dataset</em>, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/waqasahmedbasharat/boat-vs-sea-images-dataset?select=sea. [Accessed: Dec. 14, 2024].
2. B. Cretu and A. Iftene, <em>Flowers-299 </em>, Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/bogdancretu/flower299. [Accessed: Dec. 14, 2024].
3. Hariharan, <em>River vs Lake </em>, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/hariharanalm/river-vs-lake. [Accessed: Dec. 14, 2024].
4. ProtectorYao, <em>Visual China</em>, Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/protectoryao/visual-china. [Accessed: Dec. 14, 2024].
5. DeepNets,<em> Landscape Recognition | Image Dataset | 12K Images</em>, Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images. [Accessed: Dec. 14, 2024].
6. G. Ajayi, <em>Multi-class Weather Dataset for Image Classification</em>, Mendeley Data, 2018. [Online]. Available: http://dx.doi.org/10.17632/4drtyfjtfy.1. [Accessed: Dec. 14, 2024].
7. J. Bhathena,<em> Weather Image Recognition</em>, Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset. [Accessed: Dec. 14, 2024].
8. A. Olariu, <em>Leaf Detection</em>, Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/alexo98/leaf-detection?select=train. [Accessed: Dec. 14, 2024].

The resulting data is under the [data folder](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/tree/main/caption/data) in the [output_test](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/tree/main/caption/data/output_test), [output_train](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/tree/main/caption/data/output_train), and [output_valid](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/tree/main/caption/data/output_valid) folders.
For more information regarding how the dataset is compiled, check out [this notebook](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/blob/main/caption/DatasetCompilation.ipynb). 


Since the images in the dataset are compiled from different datasets that are not initially intended for image captioning, we have to create our own reference captions for the images in the train and valid sets. We used the pretrained model GIT to generate captions first in English and then used the pretrained model MBart50 to translate the captions into Chinese. Each image should have one reference caption. For more information regarding the dataset labeling process, check out the [code](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/blob/main/caption/DatasetLabeling.py). 


## Model 
For the image captioning section, our goal is to finetune the GIT transformer model to output Chinese image captions (it is initially intended to output English image captions). We froze the layers associated with identifying image features and fine tuned the layers related to the word embedding and caption generation. We also ran a random search to determine the most optimal hyperparameters for the model. The model is evaluated using BLEU-1, BLEU-2, and METEOR scores. Compared to the original model, we were able to achieve lower BLEU scores but higher METEOR scores. The best model weights have been saved into a file named 'final_model' and is linked [here](https://drive.google.com/file/d/17q0t5qMDtpMwTMP_EzT2C8gsa2vFxoi_/view?usp=sharing). 

For more information regarding the Model Creation Process, check out the [code](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/blob/main/caption/ModelCreation.py).


## Using the Model 
If you want to try the model for yourself, you will have to download the 'final_model' weights from the Google Drive [here](https://drive.google.com/file/d/17q0t5qMDtpMwTMP_EzT2C8gsa2vFxoi_/view?usp=sharing).

For an example of how to run the model using the test images provided, check out [this notebook](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/blob/main/caption/Prediction.ipynb). The notebook is set to output a random image from the test set and show its generated caption. 

Additionally, you may wish to try the model using your own images. In this case, you will have to upload your image into the [demo folder](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/tree/main/caption/data/demo) under data. You can then use [this notebook](https://github.com/AlanMuErDan/DS-UA-301-AI-Calligraphic-Poet-Project/blob/main/caption/Demo.ipynb) as a reference.  

