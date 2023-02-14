# StorageAdapter: a class factory for creating two types
# of storage adapters: StorageAdapterLocal and SotrageAdapterNeighbor.
# These represent possible storage types/modes for ModelPackages.

from .nnModelPackageAdapterLocal import NNModelPackageAdapterLocal
from .nnModelPackageAdapterNeighbor import NNModelPackageAdapterNeighbor
from . import utils

import torch

class StorageAdapter:

    @classmethod
    def create(cls, modelPackageName, storageMode):
        '''
        Factory method for creating a storage adapter,
        based on the storageMode parameter.

        Returns:
            A storage adapter object.
        '''

        if storageMode == 'local':
            return NNModelPackageAdapterLocal(modelPackageName)
        elif storageMode == 'neighbor':
            return NNModelPackageAdapterNeighbor(modelPackageName)
        else:
            raise Exception("Unknown storage type: " + storageMode)

