import os
import glob

from .storageAdapterBase import StorageAdapterBase

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

    @staticmethod
    def get_base_path():
        """
        Returns the base model packages storage path for this mode.

        Returns
        -------
        str

        """
        return os.path.join(os.getenv('MEAS_TRANSINET_DIR'),
                            'model_packages')

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
        dir_name = os.path.join(self.get_base_path(),
                                self.model_package_name)

        # We do not assume default file names in case of the 'local' mode.
        # For now we rely on a hacky pattern matching approach:
        # There should be one and only one file named arch*.py under the dir.
        # There should be one and only one file named *.pth.tar under the dir.
        try:
            model_filename = glob.glob(f'{dir_name}/arch*.py')[0]
            checkpoint_filename = glob.glob(f'{dir_name}/*.pth.tar')[0]
        except IndexError:
            raise FileNotFoundError("Cannot find model architecture or checkpoint file")

        return model_filename, checkpoint_filename
