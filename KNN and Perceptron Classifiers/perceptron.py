import numpy as np
import pandas as pd

def perceptron(x, y, decay_rate=0.1):
    """
    The function takes as input:
    x: data
    y: class labels for the data
    Return:
    w: perceptron weight vector
    """
    w = np.ones(x.shape[1]+1)
    instances = x.shape[0]
    for j in range(100):    #outer loop for multiple itrations over the entire data
        for i in range(instances): #iterations over samples
            b_i = 1/((i+1)**(decay_rate)) #initialize the stepsize as 1/t^2 where t is each iteration(add 1 to avoid division by 0)
            x_i = np.append(x.iloc[i],1) #add a constant to each sample
            if np.dot(w, x_i) > 0:
                y_hat = 1
            if np.dot(w, x_i) <= 0:
                y_hat = -1
            if y.iloc[i] == y_hat:
                continue
            else:
                w = w + b_i*y.iloc[i]*x_i  #update the weight vector
    return w

def predict(x, w):
    """
    The function takes in:
    x: data to be made predictions on
    w: weight vector
    Returns:
    y_hat: predicted classes for each sample
    """
    instances = x.shape[0]
    y_hat = list()
    for i in range(instances):
        scalar = np.dot(w, np.append(x.iloc[i],1)) #dot product for the decision rule
        if scalar>0:
            y_hat.append(1)
        if scalar<=0:
            y_hat.append(-1)
    return y_hat

def multi_class_perceptron(x_train, y_train, x_test, labels, decay_rate):
    """
    Multi-class classification
    Returns:
    w_dict: weight dictionary for each model using one vs all
    preditions: predictions from each model
    """
    w_dict = dict()
    predictions = dict()
    for label in labels:
        y = y_train
        y = pd.Series([1 if y_train.iloc[i]==label else -1 for i in range(len(y_train))], index=y_train.index) #build y for multi class using one vs. all
        w = perceptron(x_train, y, decay_rate=decay_rate) #fit the perceptron

        y_pred = predict(x_test, w) #make predictions on x_test and append to the predictions list

        predictions[label] = y_pred

        w_dict[label] = w #add weights for each class to the weight dictionary

    return w_dict, predictions

def multi_class_perceptron_accuracy(y_test, predictions, labels):
    """
    Measure the accuracy of the multi-class perceptron.
    Take predictions from each model and combine them.
    Returns accuracy and misclassification rate.
    """
    accurate_predictions = list()
    for label in labels:
        prediction_list = predictions.get(label)
        for i, p in enumerate(prediction_list):
            if ((p == 1) & (y_test.iloc[i]==label))|((p == -1) & (y_test.iloc[i]!=label)):
                accurate_predictions.append(1)
    total_test_set = len(y_test)*len(labels)
    accuracy = sum(accurate_predictions)/total_test_set
    misclassification_rate = 1-accuracy

    return accuracy, misclassification_rate
