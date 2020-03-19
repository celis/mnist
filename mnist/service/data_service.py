import requests
from os.path import join
from pathlib import Path


class DataService:
    """
    """

    URL = "http://deeplearning.net/data/mnist"
    FILENAME = "mnist.pkl.gz"

    def __init__(self):
        self._data = None

    def load(self):
        """
        :return:
        """
        self._data = requests.get(join(self.URL, self.FILENAME)).content

    def save(self, path: str = "input_data"):
        """
        :return:
        """
        mnist_path = Path(path)
        (mnist_path / self.FILENAME).open("wb").write(self._data)
