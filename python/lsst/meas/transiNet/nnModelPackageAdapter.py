import os
import torch

import lsst.utils

from .nnModelPackageAdapterBase import NNModelPackageAdapterBase
from . import utils

__all__ = ["NNModelPackageAdapter"]


class NNModelPackageAdapter(NNModelPackageAdapterBase):
    """
    An adapter class for interfacing with ModelPackages stored in
    'neighbor' mode: those of which both the code and pretrained weights
    reside in the neighbor Git repository, namely rbClassifier_data
    Each model package is assumed to be a directory under the
    "model_packages" folder, in root of that repository.
    """
    def __init__(self, model_package_name):
        super().__init__(model_package_name)

        self.fetch()
        self.model_filename, self.checkpoint_filename = self.get_filenames()

    def fetch(self):
        pass

    def get_filenames(self):
        """
        Find and return absolute paths to the architecture and checkpoint files

        Parameters
        ----------

        Returns
        -------
        model_filename : string
        checkpoint_filename : string

        """
        model_directory = os.path.join(lsst.utils.getPackageDir("rbClassifier_data"),
                                       self.model_package_name)
        model_filename = os.path.join(model_directory, self.model_package_name + ".py")
        checkpoint_filename = os.path.join(model_directory, self.model_package_name + ".pth.tar")

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
