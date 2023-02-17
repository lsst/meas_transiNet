# StorageAdapter: a class factory for creating two types
# of storage adapters: StorageAdapterLocal and SotrageAdapterNeighbor.
# These represent possible storage types/modes for ModelPackages.

from .storageAdapterLocal import StorageAdapterLocal
from .storageAdapterNeighbor import StorageAdapterNeighbor


class StorageAdapter:

    # A dict mapping storage modes to storage adapter class.
    #
    # This is necessary to guarantee that the user does not
    # specify a too customized storage mode -- to try to prevent
    # source injection attacks.
    storageAdapterClasses = {
        'local': StorageAdapterLocal,
        'neighbor': StorageAdapterNeighbor,
    }

    @classmethod
    def create(cls, modelPackageName, storageMode, **kwargs):
        '''
        Factory method for creating a storage adapter,
        based on the storageMode parameter.

        Returns:
            A storage adapter object.
        '''

        # Check that the storage mode is valid.
        if storageMode not in cls.storageAdapterClasses:
            raise ValueError('Invalid storage mode: ' + storageMode)

        # Create the storage adapter in one line.
        return cls.storageAdapterClasses[storageMode](modelPackageName, **kwargs)
