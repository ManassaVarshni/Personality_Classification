import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

normalized_all_types_scores = {'one': {'extroversion_score': 0.9817906905951671,
  'neuroticism_score': 0.397636004315784,
  'agreeableness_score': 0.7383559228997083,
  'conscientiousness_score': 0.19979774568518632,
  'openness_score': 1.0},
 'two': {'extroversion_score': 0.0,
  'neuroticism_score': 1.0,
  'agreeableness_score': 0.22711787450823331,
  'conscientiousness_score': 0.28796843134158007,
  'openness_score': 0.6141907042104143},
 'three': {'extroversion_score': 0.5881381349450099,
  'neuroticism_score': 0.0,
  'agreeableness_score': 0.18638050878272705,
  'conscientiousness_score': 0.0,
  'openness_score': 0.0},
 'four': {'extroversion_score': 1.0,
  'neuroticism_score': 0.3737653288204355,
  'agreeableness_score': 1.0,
  'conscientiousness_score': 1.0,
  'openness_score': 0.9673975203125824},
 'five': {'extroversion_score': 0.2730247134154843,
  'neuroticism_score': 0.9702670617166861,
  'agreeableness_score': 0.7235604948596414,
  'conscientiousness_score': 0.08067322449021798,
  'openness_score': 0.9985402788691458},
 'six': {'extroversion_score': 0.6336903904467563,
  'neuroticism_score': 0.6893632419423631,
  'agreeableness_score': 0.4740562283208338,
  'conscientiousness_score': 0.46118097782862033,
  'openness_score': 0.5440521819930965},
 'seven': {'extroversion_score': 0.3036158736902516,
  'neuroticism_score': 0.9302051156659882,
  'agreeableness_score': 0.8394488506341384,
  'conscientiousness_score': 0.927826040165919,
  'openness_score': 0.9114877035251373},
 'eight': {'extroversion_score': 0.25465439866994205,
  'neuroticism_score': 0.3845958233023361,
  'agreeableness_score': 0.8557004997301368,
  'conscientiousness_score': 0.7259010075543252,
  'openness_score': 0.8607322315446722},
 'nine': {'extroversion_score': 0.2480448904184913,
  'neuroticism_score': 0.5079644385122419,
  'agreeableness_score': 0.0,
  'conscientiousness_score': 0.5917745168140207,
  'openness_score': 0.994424010909459},
 'ten': {'extroversion_score': 0.9962164746276573,
  'neuroticism_score': 0.9600278298847502,
  'agreeableness_score': 0.7868346559745573,
  'conscientiousness_score': 0.3924490659860161,
  'openness_score': 0.9670591654520466}}

# Functions

def preprocess(data):
    data = data.fillna(0)
    return data

def predict_personality(data, model):
    preprocessed_data = preprocess(data)
    predicted_personality = model.predict(data)
    #visualization = visualize_score(num2words(predicted_personality[0]))
    return int(predicted_personality)

def calculateScores(personality_type, normalized_all_types_scores):
    temp_dict = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'ten'}
    result =  temp_dict[personality_type]
    return normalized_all_types_scores[result]