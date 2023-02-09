import os
import torch
import glob

from .nnModelPackageAdapterNeighbor import NNModelPackageAdapterNeighbor
from . import utils


class NNModelPackageTest(NNModelPackageAdapterNeighbor):

    def __init__(self, *args):
        super.__init__()
        self.adapter = NNModelPackageAdapterLocal(self.model_package_name)
