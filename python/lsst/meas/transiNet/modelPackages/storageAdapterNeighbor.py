import os
import glob

from .storageAdapterBase import StorageAdapterBase

__all__ = ["StorageAdapterNeighbor"]


class StorageAdapterNeighbor(StorageAdapterBase):
    """ An adapter for interfacing with ModelPackages stored in the
    'neighbor' mode.

    Neighbor mode means both the code and pretrained weights
    reside in the neighbor Git repository, namely rbClassifier_data.
    Each model package is assumed to be a directory under the
    "model_packages" folder, in the root of that repository.
    """
    def __init__(self, model_package_name):
        super().__init__(model_package_name)

        self.fetch()
        self.model_filename, self.checkpoint_filename = self.get_filenames()

    @staticmethod
    def get_base_path():
        """
        Returns the base model packages storage path for this mode.

        Returns
        -------
        str

        """
        try:
            base_path = os.environ['RBCLASSIFIER_DATA_DIR']
        except KeyError:
            raise KeyError("The environment variable RBCLASSIFIER_DATA_DIR is not set.")

        return os.path.join(base_path, 'model_packages')

    def get_filenames(self):
        """
        Find and return absolute paths to the architecture and checkpoint files

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
        dir_name = os.path.join(self.get_base_path(), self.model_package_name)

        # We do not assume default file names in case of the 'neighbor' mode.
        # For now we rely on a hacky pattern matching approach:
        # There should be one and only one file named arch*.py under the dir.
        # There should be one and only one file named *.pth.tar under the dir.
        try:
            model_filename = glob.glob(f'{dir_name}/arch*.py')[0]
            checkpoint_filename = glob.glob(f'{dir_name}/*.pth.tar')[0]
        except IndexError:
            raise FileNotFoundError("Cannot find model architecture or checkpoint file")

        return model_filename, checkpoint_filename