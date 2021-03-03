import numpy as np
import pandas as pd
from load_data import process_a4a, process_iris
from knn import train_test_split, measure_accuracy
from perceptron import perceptron, predict, multi_class_perceptron, multi_class_perceptron_accuracy
import matplotlib.pyplot as plt

#a4a evaluation
a4a = pd.read_csv('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a', header=None)
a4a = process_a4a(a4a, 123)
a4a.rename(columns={0:'Class'}, inplace=True)
a4a = a4a.copy().sample(frac=1, random_state=123)

x = a4a.iloc[:,1:]
x.fillna(0, inplace=True)
y = np.squeeze(a4a.iloc[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                   test_size=0.25,
                                                   random_state=0)

w = perceptron(x_train, y_train, decay_rate=0.1)
y_pred = predict(x_test, w)
accuracy, misclassification_rate = measure_accuracy(y_pred, y_test)
print('Perceptron Accuracy and Misclassification Rate on the A4A dataset')
print(f'Accuracy: {round(accuracy*100,2)}%, Misclassification Rate: {round(misclassification_rate*100,2)}%')

features = list(x_train.columns)
features.append('124/b')

important_features = pd.DataFrame([np.abs(w)], index=['Abs Value of Feature Weight'], columns=features)
important_features.sort_values(by=['Abs Value of Feature Weight'], axis=1, ascending=False, inplace=True)

top_20_features = important_features.iloc[0,0:20]

fig, ax = plt.subplots(figsize=(15,6))
x = np.arange(len(top_20_features.keys()))
width = 0.45

rects = ax.bar(x - width/15, top_20_features, width, label='Abs Value Feature Weight', color='#28abb9')

ax.set_ylabel('Absolute Value of feature weight')
ax.set_xlabel('Feature')
ax.set_title('A4A feature importance by absolute value of feature weight')
ax.set_xticks(x)
ax.set_xticklabels(list(top_20_features.keys()))
ax.legend()

def vals(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,2)),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
vals(rects)
plt.show()

#IRIS evaluation
iris  = pd.read_csv('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale', header=None)
iris = process_iris(iris,4)
iris.rename(columns={0:'Class'}, inplace=True)
iris = iris.copy().sample(frac=1, random_state=123)

x = iris.iloc[:,1:]
x.fillna(x.mean(), inplace=True)
y = np.squeeze(iris.iloc[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                   test_size=0.25,
                                                   random_state=123)

decay_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
weight_matrix = dict()
iris_evaluation = dict()
for i, decay_rate in enumerate(decay_grid):
    w_dict, predictions = multi_class_perceptron(x_train, y_train, x_test, labels=[1,2,3], decay_rate=decay_rate)
    weight_matrix[decay_rate] = w_dict

    accuracy, misclassification_rate = multi_class_perceptron_accuracy(y_test, predictions, labels=[1,2,3])

    iris_evaluation[decay_rate] = [round(accuracy,2), round(misclassification_rate,2)]

iris_evaluation_df = pd.DataFrame(iris_evaluation, index=['Accuracy', 'Misclassification Rate'])
feature_importance = weight_matrix.get(0.8)
features = list(x_train.columns)
features.append('5/b')
feature_importance_df = pd.DataFrame(feature_importance, index=features)

fig, ax = plt.subplots(figsize=(15,6))
x = np.arange(len(feature_importance_df.index))
width = 0.20

rects_1 = ax.bar(x - width/1.25, np.abs(feature_importance[1]), width, label='Abs Value Feature Weights for Class 1', color='#213e3b')
rects_2 = ax.bar(x + width/2, np.abs(feature_importance[2]), width, label='Abs Value Feature Weights for Class 2', color='#41aea9')
rects_3 = ax.bar(x + width*1.7, np.abs(feature_importance[3]), width, label='Abs Value Feature Weights for Class 3', color='#8ac4d0')

ax.set_ylabel('Absolute Value of feature weight')
ax.set_xlabel('Feature')
ax.set_title('IRIS feature importance by absolute value of feature weight')
ax.set_xticks(x)
ax.set_xticklabels(list(feature_importance_df.index))
ax.legend()

def vals(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,2)),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
vals(rects_1)
vals(rects_2)
vals(rects_3)
plt.show()

iris_w_dict, predictions = multi_class_perceptron(x_train, y_train, x_test, labels=[1,2,3], decay_rate=0.8)
accuracy, misclassification_rate = multi_class_perceptron_accuracy(y_test, predictions, labels=[1,2,3])
print(f'Accuracy: {round(accuracy*100,2)}%, Misclassification Rate: {round(misclassification_rate*100,2)}%')
