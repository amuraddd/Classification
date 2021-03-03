"""
Model Evaluation
"""
import pandas as pd
import numpy as np
from load_data import process_a4a, process_iris
from knn import train_test_split, knn_classifier, measure_accuracy

a4a = pd.read_csv('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a', header=None)
a4a = process_a4a(a4a, 123)
a4a.rename(columns={0:'Class'}, inplace=True)
a4a = a4a.copy().sample(frac=1, random_state=32)

x = a4a.iloc[:,1:]
x.fillna(0, inplace=True)
y = np.squeeze(a4a.iloc[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   test_size=0.25,
                                                   random_state=1)

euclid_pred, manhat_pred = knn_classifier(x_train, x_test, y_train, k=5) #make predictions
euclid_accuracy, euclid_misclassification_rate = measure_accuracy(euclid_pred, y_test) #euclidean accuracy
manhat_accuracy, manhat_misclassification_rate = measure_accuracy(manhat_pred, y_test) #manhattan accuracy

print('CLassification results for the a4a dataset')
print(f'Euclidean Distance Accuracy, {round(euclid_accuracy*100,2)}% - Euclidean Distance Misclassification Rate, {round(euclid_misclassification_rate*100,2)}%')
print(f'Manhattan Distance Accuracy, {round(manhat_accuracy*100,2)}% - Manhattan Distance Misclassification Rate, {round(manhat_misclassification_rate*100,2)}%')

###############################################################################################################################

iris  = pd.read_csv('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale', header=None)
iris = process_iris(iris,4)
iris.rename(columns={0:'Class'}, inplace=True)
iris = iris.copy().sample(frac=1, random_state=45)

x = iris.iloc[:,1:]
x.fillna(x.mean(), inplace=True)
y = np.squeeze(iris.iloc[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   test_size=0.25,
                                                   random_state=1)

euclid_pred, manhat_pred = knn_classifier(x_train, x_test, y_train, k=5) #make predictions
euclid_accuracy, euclid_misclassification_rate = measure_accuracy(euclid_pred, y_test) #euclidean accuracy
manhat_accuracy, manhat_misclassification_rate = measure_accuracy(manhat_pred, y_test) #manhattan accuracy

print('Classification results for the IRIS dataset')
print(f'Euclidean Distance Accuracy, {round(euclid_accuracy*100,2)}% - Euclidean Distance Misclassification Rate, {round(euclid_misclassification_rate*100,2)}%')
print(f'Manhattan Distance Accuracy, {round(manhat_accuracy*100,2)}% - Manhattan Distance Misclassification Rate, {round(manhat_misclassification_rate*100,2)}%')
