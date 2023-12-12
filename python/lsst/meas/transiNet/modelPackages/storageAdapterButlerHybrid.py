import os
import glob

from .storageAdapterBase import StorageAdapterBase

__all__ = ["StorageAdapterButlerHybrid"]


class StorageAdapterButlerHybrid(StorageAdapterBase):
    """An adapter for interfacing with butler-hybrid model packages.

    In this mode, pretrained weights are stored in the Butler repository.
    The model architecture and other components are stored on the disk,
    similar to the 'local' mode.

    Parameters
    ----------
    model_package_name : `str`
        The name of the ModelPackage to be loaded.
    """

    def __init__(self, model_package_name, preloaded_weights):
        super().__init__(model_package_name)

        self.preloaded_weights = preloaded_weights
        self.model_filename = self.get_filenames()

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
                                         collections=f'pretrained_models/{config.modelPackageName}')
        return list(results)

    @staticmethod
    def get_base_path():
        """Return the base model packages storage path for this mode.

        Note that this is all similar to the 'local' mode, but only the
        weights are stored in the Butler repository. The ModelPackage
        adapter takes care of finding the right butler object and loading
        the weights from there.

        Returns
        -------
        `str`
            The base path to the model packages storage.

        """
        try:
            base_path = os.environ['MEAS_TRANSINET_DIR']
        except KeyError:
            raise RuntimeError("The environment variable MEAS_TRANSINET_DIR is not set.") from None

        return os.path.join(base_path, 'model_packages')

    def get_filenames(self):
        """Find and return absolute paths to on-disk components.

        These is every component, except the weights, which are
        already loaded from the Butler repository.

        Returns
        -------
        model_filename : `str`
            The full path to the .py file containing the model architecture.

        Raises
        ------
        FileNotFoundError
            If the model package is not found.
        """
        dir_name = os.path.join(self.get_base_path(),
                                self.model_package_name)

        # We do not assume default file names in case of the 'local' mode.
        # For now we rely on a hacky pattern matching approach:
        # There should be one and only one file named arch*.py under the dir.
        try:
            model_filename = glob.glob(f'{dir_name}/arch*.py')[0]
        except IndexError:
            raise FileNotFoundError("Cannot find model architecture.")

        return model_filename

    def load_weights(self, device):
        """Load the pretrained weights.

        In this case, they should
        already be loaded by the Butler and passed to client task as
        an in-memory object.
        """

        # TODO: device!
        return self.preloaded_weights
