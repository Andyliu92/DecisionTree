# some part of the code comes from https://github.com/HewlettPackard/X-TIME/blob/main/training/xtime/datasets/_gas_concentrations.py
import numpy as np
import pandas as pd
from typing import Union, Tuple
from openml.datasets import get_dataset as get_openml_dataset
from openml.datasets.dataset import OpenMLDataset
from sklearn.model_selection import train_test_split


def load_dataset(testSize=0.3):
    # Fetch dataset and its description from OpenML. Will be cached in ${HOME}/.openml
    data: OpenMLDataset = get_openml_dataset(
        dataset_id="gas-drift-different-concentrations",
        version=1,
        error_if_multiple=True,
        download_data=True,
    )

    # Load from local cache
    x, y, _, _ = data.get_data(
        target=data.default_target_attribute, dataset_format="dataframe"
    )

    randomState: int = 0
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=testSize, random_state=randomState, shuffle=True
    )
    print("Finished loading 'gas-drift-different-concentrations' dataset.")

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy() 

    x_train, x_test, y_train, y_test = x_train.astype(np.int32), x_test.astype(np.int32), y_train.astype(np.int32), y_test.astype(np.int32)

    y_train -= 1
    y_test -= 1

    # x_test = x_test[0:99]
    # y_test = y_test[0:99]

    return x_train, x_test, y_train, y_test