"""
Authors: Ashwani Kashyap, Anshul Pardhi
"""

from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import yaml
from pathlib import Path
import dataset.BelgiumTSC.BTSC_adapted as btsc_adapted
import numpy as np
from sklearn.metrics import accuracy_score

scriptDir = Path(__file__).parent
configPath = scriptDir.joinpath("config.yml")
treeTextPath = scriptDir.joinpath("treeText.txt")

N_TRAIN = 300
N_VALIDATION = 100
N_TEST = -1

with open(configPath, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# # default data set
# df = pd.read_csv('dataset/Social_Network_Ads.csv')
# header = list(df.columns)

# # overwrite your data set here
# # header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
# # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
# # data-set link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/
# # df = pd.read_csv('data_set/breast-cancer.csv')


# lst = df.values.tolist()

# splitting the data set into train and test
# trainDF, testDF = model_selection.train_test_split(lst, test_size=0.2)

dataset = btsc_adapted.load_rand_data(N_TRAIN, N_VALIDATION, N_TEST)
(
    train_inputs,
    train_classes,
    validation_inputs,
    validation_classes,
    test_inputs,
    test_classes,
) = (
    dataset["train_inputs"],
    dataset["train_classes"],
    dataset["validation_inputs"],
    dataset["validation_classes"],
    dataset["test_inputs"],
    dataset["test_classes"],
)


# Reshape labels_array to have the same number of dimensions as data_array
train_classes_reshaped = train_classes[:, np.newaxis]
# Concatenate data_array and labels_array along the second axis
trainDF = np.concatenate((train_inputs, train_classes_reshaped), axis=1)

# Reshape labels_array to have the same number of dimensions as data_array
validation_classes_reshaped = validation_classes[:, np.newaxis]
# Concatenate data_array and labels_array along the second axis
validationDF = np.concatenate((validation_inputs, validation_classes_reshaped), axis=1)

# # Reshape labels_array to have the same number of dimensions as data_array
# test_classes_reshaped = test_classes[:, np.newaxis]
# # Concatenate data_array and labels_array along the second axis
# testDF = np.concatenate((test_inputs, test_classes_reshaped), axis=1)

header = [f"feature_{i}" for i in range(len(trainDF[0]) - 1)]
header.append("label")

# load the tree
with open('./treeText.txt', mode='r+') as fin:
    treeText = fin.read()

t = parseTreeStructure(treeText)

pred = []
for row in validation_inputs:
    pred.append(classify(row, t))

print(pred)
print(validation_classes)
accu = accuracy_score(validation_classes, pred)
print(f'Reached accuracy: {accu}')