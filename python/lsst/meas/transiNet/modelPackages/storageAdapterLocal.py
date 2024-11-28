import os
import glob

from .storageAdapterBase import StorageAdapterBase

__all__ = ["StorageAdapterLocal"]


class StorageAdapterLocal(StorageAdapterBase):
    """An adapter for interfacing with ModelPackages stored in the
    'local' mode.

    Local mode means both the code and pretrained weights reside in
    the same repository as that of rbTransiNetInterface.
    """
    def __init__(self, model_package_name):
        super().__init__(model_package_name)

        self.fetch()
        self.model_filename, self.checkpoint_filename, self.metadata_filename = self.get_filenames()

    @staticmethod
    def get_base_path():
        """
        Return the base model packages storage path for this mode.

        Returns
        -------
        `str`
            The base path to the model packages storage.

        """
        try:
            base_path = os.environ['MEAS_TRANSINET_DIR']
        except KeyError:
            raise RuntimeError("The environment variable MEAS_TRANSINET_DIR is not set.")

        return os.path.join(base_path, 'model_packages')

    def get_filenames(self):
        """
        Find and return absolute paths to the architecture and checkpoint files

        Returns
        -------
        model_filename : `str`
            The full path to the .py file containing the model architecture.
        checkpoint_filename : `str`
            The full path to the file containing the saved checkpoint.

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
        # There should be one and only one file named *.pt under the dir.
        # There should be one and only one file named meta*.yaml under the dir.
        try:
            model_filenames = glob.glob(f'{dir_name}/arch*.py')
            checkpoint_filenames = glob.glob(f'{dir_name}/*.pt')
            metadata_filenames = glob.glob(f'{dir_name}/meta*.yaml')
        except IndexError:
            raise FileNotFoundError("Cannot find model architecture, checkpoint or metadata file.")

        # Check that there's only one file for each of the three categories.
        if len(model_filenames) != 1:
            raise RuntimeError(f"Found {len(model_filenames)} model files, "
                               f"expected 1 in {dir_name}.")
        if len(checkpoint_filenames) != 1:
            raise RuntimeError(f"Found {len(checkpoint_filenames)} checkpoint files, "
                               f"expected 1 in {dir_name}.")
        if len(metadata_filenames) != 1:
            raise RuntimeError(f"Found {len(metadata_filenames)} metadata files, "
                               f"expected 1 in {dir_name}.")

        return model_filenames[0], checkpoint_filenames[0], metadata_filenames[0]
