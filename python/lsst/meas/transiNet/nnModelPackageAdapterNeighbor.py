import abc


class NNModelPackageAdapterNeighbor(abc.ABC):
    """
    Abstract base class for NNModelPackageAdapter* adapters
    """
    def __init__(self, model_package_name):
        self.model_package_name = model_package_name

    @abc.abstractmethod
    def fetch(self):
        """
        Derived classes must implement any potentially
        needed fetching operation in this method.

        Parameters
        ----------

        Returns
        -------

        """
        return

    @abc.abstractmethod
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
        
    @abc.abstractmethod
    def load_weights(self):
        """
        Load and return a network checkpoint

        Parameters
        ----------

        Returns
        -------
        network_data : dict
        """
        
