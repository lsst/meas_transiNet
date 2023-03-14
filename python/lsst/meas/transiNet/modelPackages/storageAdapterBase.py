from . import utils
import torch


class StorageAdapterBase(object):
    """
    Base class for storage adapters.

    Parameters
    ----------
    model_package_name : `str`
        The name of the model package, e.g. "my_model".
    """

    model_package_name = None
    """Name of the model package (`str`).
    """

    def __init__(self, model_package_name):
        self.model_package_name = model_package_name

    def fetch(self):
        """
        Derived classes must implement any potentially
        needed fetching operation in this method.
        This is the place to implement any sort of task
        that needs to be done before loading the model and weights.
        Multiple calls to this method must not result in multiple
        fetches.
        """
        pass

    def load_arch(self, device):
        """
        Load and return the model architecture
        (no loading of pre-trained weights).

        Parameters
        ----------
        device : `torch.device`
            Device to load the model on.

        Returns
        -------
        model : `torch.nn.Module`
            The model architecture. The exact type of this object
            is model-specific.

        See Also
        --------
        load_weights
        """

        model = utils.import_model(self.model_filename).to(device)
        return model

    def load_weights(self, device):
        """
        Load and return a checkpoint of a neural network model.

        Parameters
        ----------
        device : `torch.device`
            Device to load the pretrained weights on.

        Returns
        -------
        network_data : `dict`
            Dictionary containing a saved network state in PyTorch format,
            composed of the trained weights, optimizer state, and other
            useful metadata.

        See Also
        --------
        load_arch
        """

        network_data = torch.load(self.checkpoint_filename, map_location=device)
        return network_data
