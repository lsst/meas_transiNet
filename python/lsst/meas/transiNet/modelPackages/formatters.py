from lsst.daf.butler import Formatter
from io import BytesIO

__all__ = ["NNModelPackageFormatter", "NNModelPackagePayload"]


class NNModelPackagePayload():
    """A thin wrapper around the payload of a NNModelPackageFormatter,
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
        """Read a dataset.

        Parameters
        ----------
        component : `str`, optional
            Component to read from the file.

        Returns
        -------
        payload : `NNModelPackagePayload`
            The requested data as a Python object.
        """
        payload = NNModelPackagePayload()
        with open(self.fileDescriptor.location.path, "rb") as f:
            payload.bytes = BytesIO(f.read())
        return payload

    def write(self, inMemoryDataset):
        """Write a Dataset.

        Parameters
        ----------
        inMemoryDataset : `object`
            The Dataset to store.

        """
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset.bytes.getvalue())

    def fromBytes(self, serializedDataset: bytes, component=None):
        """Read serialized data into a Dataset or its component.

        Parameters
        ----------
        serializedDataset : `bytes`
            Bytes object to unserialize.
        component : `str`, optional
            Component to read from the Dataset. Only used if the `StorageClass`
            for reading differed from the `StorageClass` used to write the
            file.

        Returns
        -------
        payload : `NNModelPackagePayload`
            The requested data as a Python object.
        """
        payload = NNModelPackagePayload()
        payload.bytes = BytesIO(serializedDataset)
        return payload

    def toBytes(self, inMemoryDataset):
        """Serialize the Dataset to bytes based on formatter.

        Parameters
        ----------
        inMemoryDataset : `object`
            The Python object to serialize.

        Returns
        -------
        serializedDataset : `bytes`
            Bytes representing the serialized dataset.
        """
        return inMemoryDataset.bytes.getvalue()
