from lsst.daf.butler import Formatter
import torch
from io import BytesIO
#from . import utils
import dataclasses

__all__ = ["PytorchCheckpointFormatter", "PytorchCheckpointFormatter",
           "NNModelPackageFormatter", "NNModelPackagePayload"]


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

class NNModelPackagePayload():
    """ A thin wrapper around the payload of a NNModelPackageFormatter,
    which simply carries an in-memory file between the formatter and the
    storage adapter of model pacakges.
    """
    def __init__(self):
        self.bytes = BytesIO()

class NNModelPackageFormatter(Formatter):
    """Formatter for NN model packages.
    """
    extension = ".zip"

    def read(self, component=None):
        payload = NNModelPackagePayload()
        with open(self.fileDescriptor.location.path, "rb") as f:
            payload.bytes = BytesIO(f.read())
        return payload

    def write(self, inMemoryDataset):
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset.bytes.getvalue())
            print("Wrote model package to", self.fileDescriptor.location.path)


# payload = NNModelPackagePayload
# payload.bytes = BytesIO(b"hello")
