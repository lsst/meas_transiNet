from lsst.daf.butler import Formatter
import torch
from io import BytesIO
from . import utils

__all__ = ["PytorchCheckpointFormatter", "PytorchCheckpointFormatter"]


class PytorchCheckpointFormatter(Formatter):
    """Formatter for Pytorch Checkpoint files.
    """
    extension = ".tar"

    def read(self, component=None):
        return torch.load(self.fileDescriptor.location.path)

    def write(self, inMemoryDataset):
        torch.save(inMemoryDataset, self.fileDescriptor.location.path)

class PythonFileFormatter(Formatter):
    """Formatter for Python files.
    """
    extension = ".py"

    def readFile(self, path):
        with open(path, 'r') as file:
            return file.read()

    def writeFile(self, path, content):
        with open(path, 'w') as file:
            file.write(content)

    def read(self, component=None):
        path = self.fileDescriptor.location.path

        # Dynamically load as module
        module = utils.load_module_from_file(path)
        return module

    def write(self, inMemoryDataset):
        path = self.fileDescriptor.location.path
        self.writeFile(path, inMemoryDataset)

class BinaryFormatter(Formatter):
    """Formatter for binary files.
    """
    extension = ".bin"

    def read(self, component=None):
        with open(self.fileDescriptor.location.path, "rb") as f:
            return BytesIO(f.read())

    def write(self, inMemoryDataset):
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset.getvalue())
