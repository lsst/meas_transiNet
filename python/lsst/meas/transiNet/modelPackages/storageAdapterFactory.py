from .storageAdapterLocal import StorageAdapterLocal
from .storageAdapterNeighbor import StorageAdapterNeighbor
from .storageAdapterButler import StorageAdapterButler


class StorageAdapterFactory:
    """A factory for storage adapters, which represent possible storage
    types/modes for ModelPackages.

    As of March 2023, the following modes are supported:
    neighbor
        A ModelPackage stored inside the ``rbClassifier_data`` Git repository.
    local
        A ModelPackage stored inside the ``meas_transiNet`` Git repository.
    """

    # A dict mapping storage modes to storage adapter class.
    #
    # This is necessary to guarantee that the user does not
    # specify a too customized storage mode -- to try to prevent
    # source injection attacks.
    storageAdapterClasses = {
        'local': StorageAdapterLocal,
        'neighbor': StorageAdapterNeighbor,
        'butler': StorageAdapterButler,
    }

    @classmethod
    def create(cls, modelPackageName, storageMode, **kwargs):
        """ Factory method to create a storage adapter
        based on the storageMode parameter.

        Parameters
        ----------
        modelPackageName : `str`
            The name of the model package.
        storageMode : `str`
            The storage mode for the model package.
        **kwargs
            Additional keyword arguments to pass to the storage adapter.

        Returns
        -------
        storageAdapter : `StorageAdapterBase`
            A storage adapter object, based on the storageMode parameter. It is
            an instance of one of the classes in the
            storageAdapterFactory.storageAdapterClasses dict.
        """
        # Check that the storage mode is valid.
        # Convert to lower case to make it case insensitive.
        storageMode = storageMode.lower()
        if storageMode not in cls.storageAdapterClasses:
            raise ValueError('Invalid storage mode: ' + storageMode)

        # Drop any None-valued kwargs.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Create and return the storage adapter.
        storageAdapter = cls.storageAdapterClasses[storageMode](modelPackageName, **kwargs)
        return storageAdapter
