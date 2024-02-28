from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np
from pathlib import Path

scriptDir = Path(__file__).parent

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable (species)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

def load_data_2feature():
    dataset = np.load(scriptDir.joinpath(f"iris_2feature.npz"))
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

    return (
        train_inputs,
        train_classes,
        validation_inputs,
        validation_classes,
        test_inputs,
        test_classes,
    )
