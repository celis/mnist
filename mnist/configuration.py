from typing import Dict
import json


class Configuration:
    """
    Configures the model training pipeline
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.parameters = self._read_parameters()

    def _read_parameters(self) -> Dict:
        """
        reads parameters from configs
        """
        parameters = json.load(open(self.config_path, "r"))
        return parameters
