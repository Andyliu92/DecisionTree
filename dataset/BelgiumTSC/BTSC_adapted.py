from scipy.io import loadmat
from typing import Tuple
import numpy as np
from pathlib import Path

scriptDir = Path(__file__).parent

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat_load = loadmat(scriptDir.joinpath('dataset_adapted.mat'))

    image_size = 16
    train_inputs = mat_load["train_inputs"]
    test_inputs = mat_load["test_inputs"]

    train_inputs = train_inputs.reshape(
        (train_inputs.shape[0], image_size * image_size)
    )
    test_inputs = test_inputs.reshape((test_inputs.shape[0], image_size * image_size))

    train_classes = mat_load["train_classes"]
    test_classes = mat_load["test_classes"]

    train_inputs, test_inputs, train_classes, test_classes = (
        train_inputs,
        test_inputs,
        train_classes.flatten(),
        test_classes.flatten(),
    )

    return train_inputs, test_inputs, train_classes, test_classes
