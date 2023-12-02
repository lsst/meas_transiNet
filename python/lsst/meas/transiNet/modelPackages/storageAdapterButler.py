from .storageAdapterBase import StorageAdapterBase
from lsst.meas.transiNet.modelPackages.formatters import BinaryFormatter
from lsst.daf.butler import DatasetType, FileDataset, DatasetRef
from . import utils

import torch
import zipfile
import tempfile
import io
import os
import yaml

__all__ = ["StorageAdapterButler"]


class StorageAdapterButler(StorageAdapterBase):
    """ An adapter for interfacing with butler model packages.

    In this mode, all components of a model package  are stored in the
    a Butler repository.

    Parameters
    ----------
    model_package_name : `str`
        The name of the ModelPackage to be loaded.
    butler : `butler.core.Butler`
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
        self.butler_loaded_package = butler_loaded_package
        self.model_file = self.checkpoint_file = self.metadata_file = None

        self.fetch()

    @staticmethod
    def lookupFunction(config, dataSetType, registry, dataId, collections):
        """Lookup function that locates the pretrained weights of a model
        package in the Butler repository.

        All parameters are automatically set by the graph builder, except
        for the `config` parameter, which is manually set in the init()
        method of the client task's `Connections` class.

        Parameters
        ----------
        config : `lsst.pipe.base.PipelineTaskConfig`
            The configuration of the client pipeline task.
        dataSetType : `lsst.daf.butler.DatasetType`
            The `DatasetType` being queried.
        registry : `lsst.daf.butler.Registry`
            The `Registry` to use to find datasets.
        dataId : `dict`
            The `DataId` to use to find datasets -- ignored.
        collections : `str` or `list` of `str`
            The collection or collections to search for datasets -- ignored.

        Returns
        -------
        ref : `list` of `lsst.daf.butler.DatasetRef`
            List of DatasetRefs for the requested dataset.
            Ideally, there should be only one.
        """

        results = registry.queryDatasets(dataSetType,
                                         collections=f'{StorageAdapterButler.packages_parent_collection}/{config.modelPackageName}')
        return list(results)

    def fetch(self):
        """Fetch the model package from the butler repository, decompress it
        into the memory and save each component as an in-memory file.

        In case self.preloaded_package is not None, the fetching from the
        butler repository is already done, which is the "normal" case.
        """

        # Check if at least one of the properties is non-empty.
        if self.butler is None and self.butler_loaded_package is None:
            raise ValueError('Either butler or butler_loaded_package must be non-empty')

        # If fetch() has already been called, do nothing.
        if self.model_file is not None:
            return

        # Do the fetching from butler, if needed.
        if self.butler_loaded_package is None: # We are not using a preloaded package. Use butler.
            results = self.butler.registry.queryDatasets(StorageAdapterButler.dataset_type_name,
                                                         collections=f'{StorageAdapterButler.packages_parent_collection}/{self.model_package_name}')
            # fetch the object using butler
            self.butler_loaded_package = self.butler.get(list(results)[0])

        # The object in the memory is a zip file. We need to decompress it.
        with zipfile.ZipFile(self.butler_loaded_package, 'r') as zip_ref:
            with zip_ref.open('checkpoint') as f:
                self.checkpoint_file = io.BytesIO(f.read())
            with zip_ref.open('architecture') as f:
                self.model_file = io.BytesIO(f.read())
            with zip_ref.open('metadata') as f:
                self.metadata_file = io.BytesIO(f.read())

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
        run_collection = StorageAdapterButler.packages_parent_collection+"/"+the_name
        butler.registry.registerRun(run_collection)

        # Create the dataset type (and register it, just in case).
        data_id = {}
        dataset_type = DatasetType(StorageAdapterButler.dataset_type_name,
                               dimensions=[],
                               storageClass="ModelPackage",
                               universe=butler.registry.dimensions)

        def register_dataset_type(butler, dataset_type_name, dataset_type):
            try: # Do nothing if the dataset type is already registered
                butler.registry.getDatasetType(dataset_type_name)
            except KeyError:
                butler.registry.registerDatasetType(dataset_type)

        register_dataset_type(butler, StorageAdapterButler.dataset_type_name, dataset_type)

        # Create a temporary file and Zip all the three components into it.
        temp_fd, temp_path = tempfile.mkstemp(suffix='.bin')
        with os.fdopen(temp_fd, 'wb') as tmp:
            with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(model_package.adapter.checkpoint_filename, arcname="checkpoint")
                zf.write(model_package.adapter.model_filename, arcname="architecture")
                zf.write(model_package.adapter.metadata_filename, arcname="metadata")

        file_dataset = FileDataset(path=temp_path,
                                   refs=DatasetRef(dataset_type, data_id, run=run_collection),
                                   formatter=BinaryFormatter)

        # Ingest the file into the butler repo
        butler.ingest(file_dataset, transfer='copy')

        # Remove the temporary file
        os.remove(temp_path)
