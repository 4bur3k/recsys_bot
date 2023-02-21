import numpy as np
import pandas as pd
import os, shutil 
import tensorflow as tf
import tensorflow.keras as keras
from keras import Model
from keras.applications import DenseNet121
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from keras.utils import plot_model
import cv2
import pathlib
from sklearn.metrics.pairwise import linear_kernel
import glob 


PATH = './data/kaggleinputfashion-product-images-dataset/fashion-dataset/'


def make_dirs():
    pass

def init_data():

    df = pd.read_csv(PATH + "styles.csv", nrows=6000, error_bad_lines=False)
    df['image'] = df.apply(lambda x: str(x['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)
    
    return df
    

def init_model():
    #image dim
    img_width, img_height, chnl = 200, 200, 3

    # DenseNet121
    densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnl))
    densenet.trainable = False

    # Add Layer Embedding
    model = keras.Sequential([
        densenet,
        GlobalMaxPooling2D()
    ])

    return model

def get_embedings():
    return pd.read_csv('embeddings.csv')

def get_recomendations_list(article, indices,  df, cosine_sim):
    idx = indices[article]

    # Get the pairwsie similarity scores of all clothes with that one
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the clothes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar clothes
    sim_scores = sim_scores[1:6]

    # Get the clothes indices
    cloth_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['image'].iloc[cloth_indices], cloth_indices

def clear_dir():
    folder = './tmp/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def recomendation(article, df, model, embeddings):
    clear_dir()

    df = init_data()
    model = init_model()
    embeddings = get_embedings()

    cosine_sim = linear_kernel(embeddings, embeddings)
    indices = pd.Series(range(len(df)), index=df.index)

    recommendation_list, _ = get_recomendations_list(article, indices, df, cosine_sim)

    chosen_img =  cv2.imread(PATH + 'images/' + df.iloc[article].image)
    cv2.imwrite('original.jpg', chosen_img)

    for i in recommendation_list:
        img =  cv2.imread(PATH + 'images/'+ i)
        cv2.imwrite(f'./tmp/{i}', img)
    
    return True

