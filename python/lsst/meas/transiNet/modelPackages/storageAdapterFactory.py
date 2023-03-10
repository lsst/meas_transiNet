from .storageAdapterLocal import StorageAdapterLocal
from .storageAdapterNeighbor import StorageAdapterNeighbor


class StorageAdapterFactory:
    """A class factory for creating two types of storage adapters:
    StorageAdapterLocal and SotrageAdapterNeighbor. These represent
    possible storage types/modes for ModelPackages.
    """

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
    def create(cls, modelPackageName, storageMode):
        """ Factory method to create a storage adapter
        based on the storageMode parameter.

        Parameters
        ----------
        modelPackageName : `str`
            The name of the model package.
        storageMode : `str`
            The storage mode for the model package.

        Returns
        -------
        storageAdapter : {StorageAdapterLocal, StorageAdapterNeighbor}
            A storage adapter object, based on the storageMode parameter.
        """
        # Check that the storage mode is valid.
        if storageMode not in cls.storageAdapterClasses:
            raise ValueError('Invalid storage mode: ' + storageMode)

        # Create and return the storage adapter.
        storageAdapter = cls.storageAdapterClasses[storageMode](modelPackageName)
        return storageAdapter
