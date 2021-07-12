import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Functions

def preprocess(data):
    data = data.fillna(0)
    return data

def predict_personality(data, model):
    #numtowords = {'0':'one', '1':'two', '2':'three', '3': 'four', '4':'five', '5':'six', '6':'seven',
                  #'7':'eight', '8':'nine', '9':'ten'}
    preprocessed_data = preprocess(data)
    predicted_personality = model.predict(data)
    #visualization = visualize_score(num2words(predicted_personality[0]))
    return int(predicted_personality)