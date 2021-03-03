# -*- coding: utf-8 -*-
"""
KNN
"""
import pandas as pd
import numpy as np
from scipy import stats

def train_test_split(x, y, test_size, random_state=1):
    """
    Take in as parameters:
    x = x datafarme of features and their values
    y = true labels
    test_size = ratio of the test size
    random_state = random_state at which to cycle through the data
    Return x_train, x_test, y_train, y_test
    """
    x_train = x.sample(frac=1-test_size, replace=False, random_state=random_state)
    x_test = x[~x.index.isin(x_train.index)]

    y_train = y[y.index.isin(x_train.index)]
    y_test = y[y.index.isin(x_test.index)]

    return x_train, x_test, y_train, y_test

def knn_classifier(x_train, x_test, y_train, k=3, train_sample_ratio=0.1):
    """
    KNN Classifier
    """
    euclid_predictions = list()
    manhattan_predictions = list()

    for row in range(len(x_test.values)):

        if train_sample_ratio==1.0:
            #when k==1 to get 100% accuracy avoid sub sampling the data
            x_train_sample = x_train
        else:
            #change random state on each oteration below to select a new sub sample each time
            x_train_sample = x_train.sample(frac=train_sample_ratio, replace=False, random_state=row) #create a random sample of 10% of the training data to calculate distances from

        euclid_dist = ((((x_train_sample - x_test.iloc[row])**2).sum(axis=1))**0.5) #euclidean distance
        manhat_dist = np.abs(x_train_sample - x_test.iloc[row]).sum(axis=1) #manhattan distance

        idx = pd.IndexSlice
        euclid_dist_sorted_index = euclid_dist.sort_values().index #get sorted indices
        euclid_neighbors = y_train.loc[idx[euclid_dist_sorted_index]][:k] #select top k neighbors
        euclid_majority_class = stats.mode(euclid_neighbors)[0][0] #select the majority class
        euclid_predictions.append(euclid_majority_class) #append prediction to the euclid prediction list

        manhat_dist_sorted_index = manhat_dist.sort_values().index #sorted indices for manhatttan distance
        manhat_neighbors = y_train.loc[idx[manhat_dist_sorted_index]][:k] #select top k nieghbors
        manhat_majority_class = stats.mode(manhat_neighbors)[0][0] #select the majority class
        manhattan_predictions.append(manhat_majority_class) #append prediction to the manhattan prediction list

    return euclid_predictions, manhattan_predictions

def measure_accuracy(predictions, test):
    """
    Measure classification accuracy and misclassification rate
    """
    match = [1 if predictions[y] == test.iloc[y] else 0 for y in range(len(test))]
    accuracy = sum(match)/len(match)
    misclassification_rate = 1-accuracy

    return accuracy, misclassification_rate
