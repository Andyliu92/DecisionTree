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
import argparse
from DecisionTree import copyTree, exportTreeText

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--tree_text",
    type=str,
    default="treeText.txt",
    help="The path of the trained tree text",
)
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
    "--config", type=str, default="config.yml", help="The config file path"
)

args = parser.parse_args()

scriptDir = Path(__file__).parent
treeTextPath = scriptDir.joinpath(args.tree_text)
configPath = scriptDir.joinpath(args.config)

N_TRAIN = args.n_train
N_VALIDATION = args.n_validation
N_TEST = args.n_test

with open(configPath, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def main():
    dataset = btsc_adapted.load_rand_data(N_TRAIN, N_VALIDATION, N_TEST)
    (
        validation_inputs,
        validation_classes,
    ) = (
        dataset["validation_inputs"],
        dataset["validation_classes"],
    )

    T_ORIGINAL = loadTree(treeTextPath)
    pred(T_ORIGINAL, validation_inputs, validation_classes)

    sampleTimes = config["weightVar"]["sampleTimes"]
    accu = 0
    for i in range(sampleTimes):
        t = copyTree(T_ORIGINAL)
        
        addWeightVariation(t)
        accu += pred(t, validation_inputs, validation_classes, printResult=False)
    accu /= sampleTimes
    print(f'Average inference accuracy: {accu}')


def pred(
    rootNode: Union[Decision_Node, Leaf],
    inputs: np.ndarray,
    classes: np.ndarray,
    printResult=True,
) -> float:
    pred = []
    for row in inputs:
        pred.append(classify(row, rootNode))

    accu = accuracy_score(classes, pred)
    if printResult:
        print(f"Inference accuracy: {accu}")

    return float(accu)


def loadTree(treeTextPath: Path) -> Union[Leaf, Decision_Node]:
    # load the tree
    with open(treeTextPath, mode="r") as fin:
        treeText = fin.read()

    return parseTreeStructure(treeText)


def addWeightVariation(node: Union[Decision_Node, Leaf]):
    assert (
        config["hasWeightVar"] == True
    ), "config file indicates no variation but addWeightVariation() is called!"

    if isinstance(node, Leaf):
        return
    else:
        q = node.question
        q.setValue(q.getValue() + np.random.normal(0, config["weightVar"]["stdDev"]))

        addWeightVariation(node.true_branch)
        addWeightVariation(node.false_branch)


if __name__ == "__main__":
    main()
