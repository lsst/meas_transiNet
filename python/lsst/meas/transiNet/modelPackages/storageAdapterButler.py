from .storageAdapterBase import StorageAdapterBase
from lsst.meas.transiNet.modelPackages.formatters import NNModelPackagePayload
from lsst.daf.butler import DatasetType
from . import utils

import torch
import zipfile
import io
import yaml

__all__ = ["StorageAdapterButler"]


class StorageAdapterButler(StorageAdapterBase):
    """ An adapter for interfacing with butler model packages.

    In this mode, all components of a model package are stored in the
    a Butler repository.

    Parameters
    ----------
    model_package_name : `str`
        The name of the ModelPackage to be loaded.
    butler : `lsst.daf.butler.Butler`
        The butler instance used for loading the model package.
        This is used in the "offline" mode, where the model package
        is not preloaded, but is fetched from the butler repository
        manually.
    butler_loaded_package : `io.BytesIO`
        The package pre-loaded by the graph builder.
        This is a data blob representing a `pretrainedModelPackage` dataset
        directly loaded from the butler repository.
        It is only set when we are in the "online" mode of functionality.
    """

    dataset_type_name = 'pretrainedModelPackage'
    packages_parent_collection = 'pretrained_models'

    def __init__(self, model_package_name, butler=None, butler_loaded_package=None):
        super().__init__(model_package_name)

        self.model_package_name = model_package_name
        self.butler = butler

        self.model_file = self.checkpoint_file = self.metadata_file = None

        # butler and butler_loaded_package are mutually exclusive.
        if butler is not None and butler_loaded_package is not None:
            raise ValueError('butler and butler_loaded_package are mutually exclusive')

        # Use the butler_loaded_package if it is provided.
        if butler_loaded_package is not None:
            self.from_payload(butler_loaded_package)

        # If the butler is provided, we are in the "offline" mode. Let's go
        # and fetch the model package from the butler repository.
        if butler is not None:
            self.fetch()

    @classmethod
    def from_other(cls, other, use_name=None):
        """
        Create a new instance of this class from another instance, which
        can be of a different mode.

        Parameters
        ----------
        other : `StorageAdapterBase`
            The instance to create a new instance from.
        """

        instance = cls(model_package_name=use_name or other.model_package_name)

        if hasattr(other, 'model_file'):
            instance.model_file = other.model_file
            instance.checkpoint_file = other.checkpoint_file
            instance.metadata_file = other.metadata_file
        else:
            with open(other.model_filename, mode="rb") as f:
                instance.model_file = io.BytesIO(f.read())
            with open(other.checkpoint_filename, mode="rb") as f:
                instance.checkpoint_file = io.BytesIO(f.read())
            with open(other.metadata_filename, mode="rb") as f:
                instance.metadata_file = io.BytesIO(f.read())

        return instance

    def from_payload(self, payload):
        """
        Decompress the payload into the memory and save each component
        as an in-memory file.

        Parameters
        ----------
        payload : `NNModelPackagePayload`
            The payload to create the instance from.

        """
        with zipfile.ZipFile(payload.bytes, mode="r") as zf:
            with zf.open('checkpoint') as f:
                self.checkpoint_file = io.BytesIO(f.read())
            with zf.open('architecture') as f:
                self.model_file = io.BytesIO(f.read())
            with zf.open('metadata') as f:
                self.metadata_file = io.BytesIO(f.read())

    def to_payload(self):
        """
        Compress the model package into a payload.

        Returns
        -------
        payload : `NNModelPackagePayload`
            The payload containing the compressed model package.
        """

        payload = NNModelPackagePayload()

        with zipfile.ZipFile(payload.bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('checkpoint', self.checkpoint_file.read())
            zf.writestr('architecture', self.model_file.read())
            zf.writestr('metadata', self.metadata_file.read())

        return payload

    def fetch(self):
        """Fetch the model package from the butler repository, decompress it
        into the memory and save each component as an in-memory file.

        In case self.preloaded_package is not None, the fetching from the
        butler repository is already done, which is the "normal" case.
        """

        # If we have already loaded the package, there's nothing left to do.
        if self.model_file is not None:
            return

        # Fetching needs a butler object.
        if self.butler is None:
            raise ValueError('The `butler` object is required for fetching the model package')

        # Fetch the model package from the butler repository.
        results = self.butler.registry.queryDatasets(StorageAdapterButler.dataset_type_name,
                                                     collections=f'{StorageAdapterButler.packages_parent_collection}/{self.model_package_name}')  # noqa: E501
        payload = self.butler.get(list(results)[0])
        self.from_payload(payload)

    def load_arch(self, device):
        """
        Load and return the model architecture
        (no loading of pre-trained weights).

        Parameters
        ----------
        device : `torch.device`
            Device to load the model on.

        Returns
        -------
        model : `torch.nn.Module`
            The model architecture. The exact type of this object
            is model-specific.

        See Also
        --------
        load_weights
        """

        module = utils.load_module_from_memory(self.model_file)
        model = utils.import_model_from_module(module).to(device)
        return model

    def load_weights(self, device):
        """
        Load and return a checkpoint of a neural network model.

        Parameters
        ----------
        device : `torch.device`
            Device to load the pretrained weights on.
            Only loading on CPU can be used in this case (Butler mode).


        Returns
        -------
        network_data : `dict`
            Dictionary containing a saved network state in PyTorch format,
            composed of the trained weights, optimizer state, and other
            useful metadata.

        See Also
        --------
        load_arch
        """
        if device != 'cpu':
            raise RuntimeError('storageAdapterButler only supports loading on CPU')
        network_data = torch.load(self.checkpoint_file, map_location=device)
        return network_data

    def load_metadata(self):
        """
        Load and return the metadata associated with the model package.

        Returns
        -------
        metadata : `dict`
            Dictionary containing the metadata associated with the model.
        """

        metadata = yaml.safe_load(self.metadata_file)
        return metadata

    @staticmethod
    def ingest(model_package, butler, model_package_name=None):
        """
        Ingest a model package to the butler repository.

        Parameters
        ----------
        model_package : nnModelPackage
            The model package to be ingested.
        butler : `lsst.daf.butler.Butler`
            The butler instance to use for ingesting.
        model_package_name : `str`, optional
            The name of the model package to be ingested.
        """

        # Check if the input model package is of a proper type.
        if model_package.adapter is StorageAdapterButler:
            raise ValueError('The input model package cannot be of the butler type')

        # Choose the name of the model package to be ingested.
        if model_package_name is None:
            the_name = model_package.model_package_name
        else:
            the_name = model_package_name

        # Create the destination run collection.
        run_collection = f"{StorageAdapterButler.packages_parent_collection}/{the_name}"
        butler.registry.registerRun(run_collection)

        # Create the dataset type (and register it, just in case).
        data_id = {}
        dataset_type = DatasetType(StorageAdapterButler.dataset_type_name,
                                   dimensions=[],
                                   storageClass="NNModelPackagePayload",
                                   universe=butler.registry.dimensions)

        # Register the dataset type.
        def register_dataset_type(butler, dataset_type_name, dataset_type):
            try:  # Do nothing if the dataset type is already registered
                butler.registry.getDatasetType(dataset_type_name)
            except KeyError:
                butler.registry.registerDatasetType(dataset_type)
        register_dataset_type(butler, StorageAdapterButler.dataset_type_name, dataset_type)

        # Create an instance of StorageAdapterButler, and ingest its payload.
        payload = StorageAdapterButler.from_other(model_package.adapter).to_payload()
        butler.put(payload,
                   dataset_type,
                   data_id,
                   run=run_collection)
