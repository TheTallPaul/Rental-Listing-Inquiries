#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Using SRK's introduction to XGBoost was used as a general guideline for 
# this program, specifically for reading the data in as a Pandas dataframe
# and adding additional feautres to that dataframe.
# https://www.kaggle.com/sudalairajkumar/xgb-starter-in-python

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing, metrics

# Using a csv containing the school ids and building ids, builds a dictionary 
# keying the former to the latter
def dist(file):
	f = open(file, "r")
	district = {}
	i = 1
	for line in f:
		control = 0
		buildID = ""
		schoolID = ""
		for i in range (len(line)):
			if control == 1 and line[i] != ",":
				buildID += line[i]
			if control == 2 and line[i] != "\n":
				schoolID += line[i]
			if line[i] == ',':
				control += 1
		district[buildID] = schoolID
	return district
	
def parseCount(file):
	f = open(file,"r")
	dict = {}
	for line in f:
		id = ""
		count = ""
		control = False
		for i in range (len(line)):
			if control == True:
				count += line[i]
			if line[i] != "," and control == False:
				id += line[i]
			else:
				control = True
		dict[id] = int(count)
	return dict

trainWordCount = parseCount("wordCountTraining.csv")
testWordCount = parseCount("wordCountTesting.csv")

trainDist = dist("schoolDistricts.csv")
testDist = dist("schoolDistrictsTest.csv")

def keyToWordCountTest(key):
	return testWordCount[key]

def keyToWordCountTrain(key):
	return trainWordCount[key]
	
def keyToDictTrain(key):
	return str(trainDist[key])

def keyToDictTest(key):
	return str(testDist[key])
	
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
                'price_per_sqft', 'manager_id', 'building_id', 'num_features',
                'latitude', 'longitude', 'school_district_id', 
                'description_word_count']
    
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
    
    # Number of Features
    train_data['num_features'] = train_data['features'].apply(len)
    test_data['num_features'] = test_data['features'].apply(len)
	
    # School district id
    train_data['school_district_id'] = train_data['building_id']\
        .apply(keyToDictTrain)
    test_data['school_district_id'] = test_data['building_id']\
        .apply(keyToDictTest)
	
	#description word counts
    train_data['description_word_count'] = train_data['building_id']\
        .apply(keyToWordCountTrain)
    test_data['description_word_count'] = test_data['building_id']\
        .apply(keyToWordCountTest)
	
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

    
    #encode categorical features in training
    for feature in features:
        if train_data[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            # fit to unique labels in training + testing categorical feature
            lbl.fit(list(train_data[feature].values) \
                    + list(test_data[feature].values))
            train_data[feature] = lbl.transform(
                    list(train_data[feature].values))
            
    #encode categorical features in testing
    for feature in features:
        if test_data[feature].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            # fit to unique labels in training + testing categorical feature
            lbl.fit(list(train_data[feature].values) \
                    + list(test_data[feature].values))
            test_data[feature] = lbl.transform(
                    list(test_data[feature].values))
  
    train_data = train_data.as_matrix()
    test_data = test_data.as_matrix()

    train_data = remove_interest_col(train_data)
    return train_data, test_data
            
# Get the features
train_data, test_data = read_data('train.json', 'test.json')
features = add_features(train_data, test_data)

# Create the matrix that will be evaluated with the training
train_y = create_label(train_data)

# Classify categorical features
train_data, test_data = prepare_data(train_data, test_data, features)

# Save as an DMatrix
train = xgb.DMatrix(train_data, label=train_y)

# Set paramaters for learning
params = {}
params['objective']   = 'multi:softprob'
params['num_class']   = 3
params['eval_metric'] = 'mlogloss'
params['eta']         = 0.5

# Perform training
boost = xgb.train(params, train)
boost.dump_model('dump.raw.txt')

# Print the log loss for the training
print("Training Complete")
ypred_train = boost.predict(train)
print("Training Log Loss:", metrics.log_loss(
        train_y, ypred_train, labels=[0, 1, 2]))

# Plot the importance of features
xgb.plot_importance(boost)

# Make preidctions on the test data
test = xgb.DMatrix(test_data)
ypred = boost.predict(test)

# Move the predictions into a csv file
# Using SRK's method for reading the prediction into a csv
# https://www.kaggle.com/sudalairajkumar/xgb-starter-in-python (In[ 10])
train_data, test_data = read_data('train.json', 'test.json')
out_df = pd.DataFrame(ypred)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_data.listing_id.values
out_df.to_csv("rent_predictions.csv", index=False)