import nnModelPackageAdapterBase
import utils


class NNModelPackageAdapterLocal(nnModelPackageAdapterBase.NNModelPackageAdapterBase):
    """
    An adapter class for interfacing with ModelPackages stored in
    'local' mode: those of which both the code and pretrained weights
    reside in the same Git repository as that of rbTransiNetInterface
    """
    def __init__(self, model_package_name):
        super().__init__(self, model_package_name)

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

        """
        model_filename = ...
        checkpoint_filename = ...

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
        
    def load_weights(self):
        """
        Load and return a network checkpoint

        Parameters
        ----------

        Returns
        -------
        network_data : dict
        """

        network_data = torch.load(pretrained_file, map_location=device)
        return network_data
