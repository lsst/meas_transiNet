
from lsst.meas.transiNet.nnModelPackage import NNModelPackage
from lsst.meas.transiNet.nnModelPackageAdapterLocal import NNModelPackageAdapterLocal
from lsst.meas.transiNet.rbTransiNetInterface import RBTransiNetInterface
from lsst.meas.transiNet.transiNetTask import TransiNetTask

__all__ = ["NNModelPackageTest", "RBTransiNetInterfaceTest", "TransiNetTaskTest"]


class NNModelPackageTest(NNModelPackage):

    def __init__(self, model_package_name):
        self.model_package_name = model_package_name
        self.adapter = NNModelPackageAdapterLocal(model_package_name)


class RBTransiNetInterfaceTest(RBTransiNetInterface):
    """
    The interface between the LSST AP pipeline and a trained pytorch-based
    RBTransiNet neural network model.
    """

    def init_model(self):
        """Create and initialize an NN model
        """
        model_package = NNModelPackageTest(self.model_package_name)
        self.model = model_package.load(self.device)

        # Put the model in evaluation mode instead of training model.
        self.model.eval()


class TransiNetTaskTest(TransiNetTask):
    """Task for running TransiNet real/bogus classification on the output of
    the image subtraction pipeline.
    """
    _DefaultName = "transiNetTest"

    def __init__(self, **kwargs):
        super(TransiNetTask, self).__init__(**kwargs)

        self.interface = RBTransiNetInterfaceTest(self.config.modelPackageName)
