import os
import glob

from .storageAdapterBase import StorageAdapterBase

__all__ = ["StorageAdapterNeighbor"]


class StorageAdapterNeighbor(StorageAdapterBase):
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
        dir_name = os.path.join(os.getenv('RBCLASSIFIER_DATA_DIR'),
                                "model_packages",
                                self.model_package_name)

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