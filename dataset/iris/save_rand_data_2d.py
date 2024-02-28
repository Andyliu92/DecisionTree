import numpy as np
from pathlib import Path
import os
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import iris

scriptDir = Path(__file__).parent
datasetDir = scriptDir

def load_data_2feature(feature1ID=0, feature2ID=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable (species)

    X = X[:, :2]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

outputPath = datasetDir.joinpath(
    f"iris_2feature.npz"
)

(train_inputs, validation_inputs, train_classes, validation_classes) = (
    load_data_2feature()
)

np.savez(
    outputPath,
    train_inputs=train_inputs,
    train_classes=train_classes,
    validation_inputs=validation_inputs,
    validation_classes=validation_classes,
    test_inputs=[],
    test_classes=[]
)

print(f'saved file to {outputPath}')