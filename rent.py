#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing

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
    
    # List of all features
    features = list(test_data)
    
    # Iterate through all features, removing the uneeded ones
    for feature in features:
        if feature not in features_used and feature != 'interest_level':
            test_data.drop(feature, axis=1, inplace=True)

def add_features(train_data, test_data):
    features = ['price', 'bedrooms', 'bathrooms', 'num_photos',
                'price_per_sqft', 'manager_id', 'building_id', 
                'display_address', 'street_address']#'school_district_id']
    
    # The number of photos
    train_data['num_photos'] = train_data['photos'].apply(len)
    test_data['num_photos'] = test_data['photos'].apply(len)

    #price per sqft equation based on Darnell's breakdown: 
    # www.kaggle.com/arnaldcat/a-proxy-for-sqft-and-the-interest-on-1-2-baths
    train_data['price_per_sqft'] = train_data['price'] / (1 \
              + train_data['bedrooms'].clip(1,4) \
              + 0.5 * train_data['bathrooms'].clip(0,2))
    test_data['price_per_sqft'] = test_data['price'] / (1 \
             + test_data['bedrooms'].clip(1,4) \
             + 0.5 * test_data['bathrooms'].clip(0,2))
    
    # School district id
    #train_data['school_district_id'] = train_data.apply(lambda x: district(
           # x['latitude'], x['longitude']), axis=1)
    #test_data['school_district_id'] = test_data.apply(lambda x: district(
           # x['latitude'], x['longitude']), axis=1)
    
    remove_unused_features(train_data, test_data, features)
    return features

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

def remove_interest_col(train_data):
    interest_levels = ['high', 'medium', 'low']
    
    for col in range(len(train_data[0])):
        if train_data[0][col] in interest_levels:
            return np.delete(train_data, col, 1)

def prepare_data(train_data, test_data, features):
    print(features)

    for feature in features:
        if train_data[feature].dtype=='object':
            print(feature)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_data[feature].values) + list(test_data[feature].values))
            train_data[feature] = lbl.transform(list(train_data[feature].values))

    for feature in features:
        if test_data[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_data[feature].values) + list(test_data[feature].values))
            test_data[feature] = lbl.transform(list(test_data[feature].values))
  
    train_data = train_data.as_matrix()
    test_data = test_data.as_matrix()

    train_data = remove_interest_col(train_data)
    return train_data, test_data
            

train_data, test_data = read_data('train.json', 'test.json')

features = add_features(train_data, test_data)

train_y = create_label(train_data)

train_data, test_data = prepare_data(train_data, test_data, features)

train = xgb.DMatrix(train_data, label=train_y)

params = {}
params['objective'] = 'multi:softprob'
params['num_class'] = 3
params['eval_metric'] = 'mlogloss'

boost = xgb.train(params, train)

boost.dump_model('dump.raw.txt')

print("Training Complete")

test = xgb.DMatrix(test_data)

ypred = boost.predict(test)

train_data, test_data = read_data('train.json', 'test.json')

out_df = pd.DataFrame(ypred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_data.listing_id.values
out_df.to_csv("coolbros.csv", index=False)

