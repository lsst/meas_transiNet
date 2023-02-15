from . import utils
import torch

import lsst.pipe.base.Task


class StorageAdapterBase(lsst.pipe.base.Task):
    """
    Base class for StorageAdapter* adapters
    """
    def __init__(self, model_package_name):
        self.model_package_name = model_package_name

    def fetch(self):
        """
        Derived classes must implement any potentially
        needed fetching operation in this method.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    def load_model(self):
        """
        Load and return the model architecture
        (no loading of pre-trained weights yet)

        Parameters
        ----------

        Returns
        -------
        model : unknown subclass of nn.Module
        """

        model = utils.import_model(self.model_filename)
        return model

    def load_weights(self, device):
        """
        Load and return a network checkpoint

        Parameters
        ----------

        Returns
        -------
        network_data : dict
        """

        network_data = torch.load(self.checkpoint_filename, map_location=device)
        return network_data
