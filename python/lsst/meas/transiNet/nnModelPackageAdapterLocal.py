import os

import lsst.utils

from .nnModelPackageAdapter import NNModelPackageAdapter

__all__ = ["NNModelPackageAdapterLocal"]


class NNModelPackageAdapterLocal(NNModelPackageAdapter):
    """
    An adapter class for interfacing with ModelPackages stored in
    'local' mode: those of which both the code and pretrained weights
    reside in the same Git repository as that of rbTransiNetInterface
    """

    def get_filenames(self):
        """

        Parameters
        ----------

        Returns
        -------
        model_filename : string
        checkpoint_filename : string

        """
        dir_name = os.path.join(lsst.utils.getPackageDir('meas_transiNet'),
                                "model_packages",
                                self.model_package_name)
        model_filename = os.path.join(dir_name, 'model.py')  # For now assume fixed filenames
        checkpoint_filename = os.path.join(dir_name, 'checkpoint.pth.tar')  # For now assume fixed filenames

        return model_filename, checkpoint_filename
