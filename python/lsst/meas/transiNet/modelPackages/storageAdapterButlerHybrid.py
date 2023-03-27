import os
import glob

from .storageAdapterBase import StorageAdapterBase
from lsst.daf.butler import Formatter

import torch

__all__ = ["StorageAdapterButlerHybrid", "BinaryFormatter", "PytorchCheckpointFormatter"]

class BinaryFormatter(Formatter):
    extension = ".tar"

    def read(self, component=None):
        with open(self.fileDescriptor.location.path, "rb") as f:
            return f.read()
        #return read_data(self.fileDescriptor.location.path)

    def write(self, inMemoryDataset):
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset)


class PytorchCheckpointFormatter(Formatter):
    extension = ".tar"

    def read(self, component=None):
        return torch.load(self.fileDescriptor.location.path)

    def write(self, inMemoryDataset):
        torch.save(inMemoryDataset, self.fileDescriptor.location.path)


class StorageAdapterButlerHybrid(StorageAdapterBase):
    """ An adapter for interfacing with ModelPackages stored in the
    'neighbor' mode.

    ButlerHybrid mode means the pretrained weights are stored in the
    butler repository.
    """
    def __init__(self, model_package_name):

        raise NotImplementedError("ButlerHybrid mode is not yet implemented")
