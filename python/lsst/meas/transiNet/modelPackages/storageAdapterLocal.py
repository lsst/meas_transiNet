import os
import torch

from .storageAdapterBase import StorageAdapterBase
from . import utils

__all__ = ["StorageAdapterLocal"]


class StorageAdapterLocal(StorageAdapterBase):
    """
    An adapter class for interfacing with ModelPackages stored in
    'local' mode: those of which both the code and pretrained weights
    reside in the same Git repository as that of rbTransiNetInterface
    """
    def __init__(self, model_package_name):
        super().__init__(model_package_name)

        self.fetch()
        self.model_filename, self.checkpoint_filename = self.get_filenames()

    def fetch(self):
        pass

    def get_filenames(self):
        """

        Parameters
        ----------

        Returns
        -------
        model_filename : string
        checkpoint_filename : string

        Raises:
        -------
        FileNotFoundError
        """
        dir_name = os.path.join(os.getenv('MEAS_TRANSINET_DIR'),
                                "model_packages",
                                self.model_package_name)

        try:
            model_filename = os.path.join(dir_name, 'model.py')  # For now assume fixed filenames
            checkpoint_filename = os.path.join(dir_name, 'checkpoint.pth.tar')  # For now assume fixed filenames
        except FileNotFoundError:
            raise FileNotFoundError("Cannot find model architecture or checkpoint file")

        return model_filename, checkpoint_filename

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
