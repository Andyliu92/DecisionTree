import numpy as np
from pathlib import Path
import os
import sys
from copy import deepcopy

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import BTSC_adapted

scriptDir = Path(__file__).parent
datasetDir = scriptDir

N_TRAIN = 300
N_VALIDATION = 100
N_TEST = -1  # use all test set

outputPath = datasetDir.joinpath(
    f"{N_TRAIN}train_{N_VALIDATION}validation_{N_TEST}test.npz"
)

train_inputs, test_inputs, train_classes, test_classes = BTSC_adapted.load_data()

assert train_inputs.shape[0] == train_classes.shape[0]

numTrainRows = train_inputs.shape[0]
randIndex = np.random.permutation(numTrainRows)

train_inputs, train_classes = train_inputs[randIndex], train_classes[randIndex]

validation_inputs = deepcopy(train_inputs)
validation_classes = deepcopy(train_classes)

assert N_TRAIN <= train_inputs.shape[0]

train_inputs, train_classes = train_inputs[0:N_TRAIN, :], train_classes[0:N_TRAIN]

assert N_TRAIN + N_VALIDATION <= validation_inputs.shape[0]

validation_inputs, validation_classes = (
    validation_inputs[N_TRAIN : N_TRAIN + N_VALIDATION, :],
    validation_classes[N_TRAIN : N_TRAIN + N_VALIDATION],
)

assert test_inputs.shape[0] == test_classes.shape[0]
assert N_TEST <= test_inputs.shape[0]

numTestRows = test_inputs.shape[0]
randIndex = np.random.permutation(numTestRows)

test_inputs, test_classes = test_inputs[randIndex], test_classes[randIndex]
test_inputs, test_classes = test_inputs[0:N_TEST, :], test_classes[0:N_TEST]

np.savez(
    outputPath,
    train_inputs=train_inputs,
    train_classes=train_classes,
    validation_inputs=validation_inputs,
    validation_classes=validation_classes,
    test_inputs=test_inputs,
    test_classes=test_classes
)

print(f'saved file to {outputPath}')