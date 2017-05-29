#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer

# Reads the json object as a panda object
def read_data(train_filepath, test_filepath):
    return pd.read_json(train_filepath), pd.read_json(test_filepath)
    

# Removes unwanted features from the pandas object
def remove_unused_features(train_data, test_data, features_used):
    # List of all features
    features = list(train_data)
    
    # Iterate through all features, removing the uneeded ones
    for feature in features:
        if feature not in features_used and feature != 'interest_level':
            train_data.drop(feature, axis=1, inplace=True)
            test_data.drop(feature, axis=1, inplace=True)

def add_features(train_data, test_data):
    features = ['price', 'bedrooms', 'bathrooms', 'num_photos', 'manager_id', 
                'building_id']
    
    # The number of photos
    train_data['num_photos'] = train_data['photos'].apply(len)
    #price per sqft equation based on Darnell's breakdown: https://www.kaggle.com/arnaldcat/a-proxy-for-sqft-and-the-interest-on-1-2-baths
    train_data['price_per_sqft'] = (train_data['price']/(1 + train_data['bedrooms'].clip(1,4) + 0.5*train_data['bathrooms'].clip(0,2)))
    test_data['num_photos'] = test_data['photos'].apply(len)
    test_data['price_per_sqft'] = (test_data['price']/(1 + test_data['bedrooms'].clip(1,4) + 0.5*test_data['bathrooms'].clip(0,2)))
    
    remove_unused_features(train_data, test_data, features)

# Converts the interest level into integers
def interest_level_to_int(interest_level):
    if interest_level == 'high':
        return 0

    elif interest_level == 'medium':
        return 1

    else: # low interest
        return 2

# Creates a Y for evaluation
def create_label(train_data):
    return np.array(train_data['interest_level'].apply(interest_level_to_int))

def prepare_data(train_data):
    train_data = train_data.as_matrix()
    train_data = np.delete(train_data, [2], 1)

    for f in range(len(train_data[0])): 
        if train_data[f].dtype=='object': 
            lbl = preprocessing.LabelEncoder() 
            lbl.fit(train_data[f]) 
            train_data[f] = lbl.transform(train_data[f])
            

train_data, test_data = read_data('train.json', 'train.json')

add_features(train_data, test_data)

train_y = create_label(train_data)

# train = xgb.DMatrix(train_data, label=train_y)
# train.save_binary('train.buffer')
