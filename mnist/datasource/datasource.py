from torch.utils.data import TensorDataset
import pickle
import gzip
import torch
from typing import Tuple
from os.path import join


class CustomDataource:
    """
    Datasource class with method to produce the required Dataset instances with training and validation data.
    """

    FILENAME = "mnist.pkl.gz"

    def __init__(self, path: str):
        self.path = path

    def transform(self) -> Tuple[TensorDataset, TensorDataset]:
        """
        Returns dataset instances fror training and validation
        """
        with gzip.open((join(self.path, self.FILENAME)), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
                f, encoding="latin-1"
            )

        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid)
        )

        train_dataset = TensorDataset(x_train, y_train)
        validation_dataset = TensorDataset(x_valid, y_valid)

        return train_dataset, validation_dataset
