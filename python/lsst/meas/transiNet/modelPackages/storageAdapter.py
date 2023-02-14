# StorageAdapter: a class factory for creating two types
# of storage adapters: StorageAdapterLocal and SotrageAdapterNeighbor.
# These represent possible storage types/modes for ModelPackages.

from .storageAdapterLocal import StorageAdapterLocal
from .storageAdapterNeighbor import StorageAdapterNeighbor


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
            return StorageAdapterLocal(modelPackageName)
        elif storageMode == 'neighbor':
            return StorageAdapterNeighbor(modelPackageName)
        else:
            raise Exception("Unknown storage mode: " + storageMode)
