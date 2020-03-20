import requests
from os.path import join
from pathlib import Path


class DataService:
    """
    """

    URL = "http://deeplearning.net/data/mnist"
    FILENAME = "mnist.pkl.gz"

    def __init__(self):
        pass

    def save(self, path: str = "input_data"):
        """
        :return:
        """
        mnist_path = Path(path)
        if not (mnist_path / self.FILENAME).exists():
            data = requests.get(join(self.URL, self.FILENAME)).content
            (mnist_path / self.FILENAME).open("wb").write(data)
