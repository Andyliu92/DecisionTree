"""
Authors: Ashwani Kashyap, Anshul Pardhi
"""

from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import yaml
from pathlib import Path
import dataset.BelgiumTSC.BTSC_adapted as btsc_adapted
import dataset.iris.iris as iris
import numpy as np
import argparse

scriptDir = Path(__file__).parent
outputDie = scriptDir.joinpath("outputs")

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--n_train", type=int, default=300, help="Use n_train samples for training"
)
parser.add_argument(
    "--n_validation",
    type=int,
    default=100,
    help="Use n_validation samples for validation",
)
parser.add_argument(
    "--n_test", type=int, default=-1, help="Use n_test samples for testing"
)
parser.add_argument(
    "--output_path", type=str, default="treeText.txt", help="Tree text output path"
)
parser.add_argument(
    "--config", type=str, default="config.yml", help="The config file path"
)

args = parser.parse_args()

N_TRAIN = args.n_train
N_VALIDATION = args.n_validation
N_TEST = args.n_test
treeTextPath = scriptDir.joinpath(args.output_path)
configPath = scriptDir.joinpath(args.config)

if not treeTextPath.parent.exists():
    treeTextPath.parent.mkdir(parents=True)

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

# dataset = btsc_adapted.load_rand_data(N_TRAIN, N_VALIDATION, N_TEST)
(
    train_inputs,
    train_classes,
    validation_inputs,
    validation_classes,
    test_inputs,
    test_classes,
) = iris.load_data_2feature()

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

# building the tree
T_ORIGINAL = build_tree(trainDF, header, config)
t = copyTree(T_ORIGINAL)

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
maxAccuracy = computeAccuracy(validationDF, t)
print("\nTree before pruning with accuracy: " + str(maxAccuracy * 100) + "\n")
print_tree(t)
with open(treeTextPath, mode="w+") as fout:
    exportTreeText(t, fout)

# TODO: You have to decide on a pruning strategy
# Pruning strategy
nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(validationDF, t)
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
        t = copyTree(T_ORIGINAL)
        if maxAccuracy == 1:
            break

if nodeIdToPrune != -1:
    t = copyTree(T_ORIGINAL)
    prune_tree(t, [nodeIdToPrune])
    print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
else:
    t = copyTree(T_ORIGINAL)
    print("\nPruning strategy did'nt increased accuracy")

print_tree(t)
with open(treeTextPath, mode="w+") as fout:
    exportTreeText(t, fout)

print("\n********************************************************************")
print(
    "*********** Final Tree with accuracy: "
    + str(maxAccuracy * 100)
    + "%  ************"
)
print("********************************************************************\n")
