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

scriptDir = Path(__file__).parent
configPath = scriptDir.joinpath("config.yml")
treeTextPath = scriptDir.joinpath('treeText.txt')

N_TRAIN = 500
N_TEST = 100

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

train_inputs, test_inputs, train_classes, test_classes = btsc_adapted.load_data()

# Reshape labels_array to have the same number of dimensions as data_array
train_classes_reshaped = train_classes[:, np.newaxis]
# Concatenate data_array and labels_array along the second axis
trainDF = np.concatenate((train_inputs, train_classes_reshaped), axis=1)
trainDF = trainDF.tolist()

# Reshape labels_array to have the same number of dimensions as data_array
test_classes_reshaped = test_classes[:, np.newaxis]
# Concatenate data_array and labels_array along the second axis
testDF = np.concatenate((test_inputs, test_classes_reshaped), axis=1)
testDF = testDF.tolist()

trainDF, testDF = np.array(trainDF[0:N_TRAIN]), np.array(testDF[0:N_TEST])

header = [f"feature{i}" for i in range(len(trainDF[0]) - 1)]
header.append("label")

# building the tree
t = build_tree(trainDF, header, config)

# get leaf and inner nodes
print("\nLeaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes ****************")
innerNodes = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# print tree
maxAccuracy = computeAccuracy(testDF, t)
print("\nTree before pruning with accuracy: " + str(maxAccuracy * 100) + "\n")
print_tree(t)
with open(treeTextPath, mode='w+') as fout:
    exportTreeText(t, fout)

# TODO: You have to decide on a pruning strategy
# Pruning strategy
nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(testDF, t)
        print(
            "Pruned node_id: "
            + str(node.id)
            + " to achieve accuracy: "
            + str(currentAccuracy * 100)
            + "%"
        )
        # print("Pruned Tree")
        # print_tree(t)
        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            nodeIdToPrune = node.id
        t = build_tree(trainDF, header, config)
        if maxAccuracy == 1:
            break

if nodeIdToPrune != -1:
    t = build_tree(trainDF, header, config)
    prune_tree(t, [nodeIdToPrune])
    print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
else:
    t = build_tree(trainDF, header, config)
    print("\nPruning strategy did'nt increased accuracy")

print("\n********************************************************************")
print(
    "*********** Final Tree with accuracy: "
    + str(maxAccuracy * 100)
    + "%  ************"
)
print("********************************************************************\n")
print_tree(t)
