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
        payload = NNModelPackagePayload()
        with open(self.fileDescriptor.location.path, "rb") as f:
            payload.bytes = BytesIO(f.read())
        return payload

    def write(self, inMemoryDataset):
        with open(self.fileDescriptor.location.path, "wb") as f:
            f.write(inMemoryDataset.bytes.getvalue())
