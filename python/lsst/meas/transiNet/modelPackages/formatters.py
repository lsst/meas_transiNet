from lsst.daf.butler import FormatterV2
from lsst.resources import ResourcePath
from io import BytesIO
from typing import Any

__all__ = ["NNModelPackageFormatter", "NNModelPackagePayload"]


class NNModelPackagePayload():
    """A thin wrapper around the payload of a NNModelPackageFormatter,
    which simply carries an in-memory file between the formatter and the
    storage adapter of model pacakges.
    """
    def __init__(self):
        self.bytes = BytesIO()


class NNModelPackageFormatter(FormatterV2):
    """Formatter for NN model packages.
    """
    default_extension = ".zip"
    can_read_from_uri = True

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
        """Read a dataset.

        Parameters
        ----------
        uri : `lsst.ResourcePath`
            Location of the file to read.
        component : `str` or `None`, optional
            Component to read from the file.
        expected_size : `int`, optional
            Expected size of the file.

        Returns
        -------
        payload : `NNModelPackagePayload`
            The requested data as a Python object.
        """
        payload = NNModelPackagePayload()
        payload.bytes = BytesIO(uri.read())
        return payload

    def to_bytes(self, in_memory_dataset: Any) -> bytes:
        return in_memory_dataset.bytes.getvalue()
