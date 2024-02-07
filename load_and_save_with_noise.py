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
from DecisionTree import *
import multiprocessing

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--tree_text",
    type=str,
    default="treeText.txt",
    help="The path of the trained tree text",
)
parser.add_argument(
    "--config", type=str, default="config.yml", help="The config file path"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="treeText_var.txt",
    help="The path of the decision tree with variation to be saved",
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

args = parser.parse_args()

N_TRAIN = args.n_train
N_VALIDATION = args.n_validation
N_TEST = args.n_test

scriptDir = Path(__file__).parent
treeTextPath = scriptDir.joinpath(args.tree_text)
configPath = scriptDir.joinpath(args.config)
outputTreeTextPath = scriptDir.joinpath(args.output_path)

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
    t = copyTree(T_ORIGINAL)
    addWeightVariation(t, config)
    # print(f'software accu: {pred(t, validation_inputs, validation_classes, printResult=False)}')
    with open(outputTreeTextPath, mode="w+") as fout:
        exportTreeText(t, fout)


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

if __name__ == "__main__":
    main()
