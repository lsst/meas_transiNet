# This file is part of meas_transiNet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["NNModelPackage"]

import enum

from .nnModelPackageAdapterLocal import NNModelPackageAdapterLocal
from .nnModelPackageAdapterNeighbor import NNModelPackageAdapterNeighbor

PackageStorageMode = enum.Enum('PackageStorageMode', ['local', 'neighbor'])  # ,'butler', ...


class NNModelPackage:
    """
    A class to abstract physical storage of network architecture &
    pretrained models out of clients' code.
    It handles all necessary required tasks, including fetching,
    decompression, etc. per need and creates a "Model Package"
    ready to use: a model architecture loaded with specific pretrained
    weights.
    """

    def __init__(self, model_package_name):
        self.model_package_name = model_package_name

    def storage_mode_from_path(self, path):
        """Infer (decode!) storage mode from path string.
        The storage mode is assumed to be encoded in the
        path name e.g as "local:///" for local storage or
        "neighbor:///" for neighbor data repository.

        Parameters
        ----------
        path : string
            Path pointing to a stored model package


        Returns
        -------
        storage_mode :
            Package storage mode

        """
        storage_mode = None  # TODO: replace with proper error handling by adding an 'else' below

        try:  # To catch non-standard paths
            token = path.split(':///')[0]
            if token.lower() == 'local':
                storage_mode = PackageStorageMode.local
            elif token.lower() == 'neighbor':
                storage_mode = PackageStorageMode.neighbor

        except Exception:
            pass  # TODO: replace with proper error handling

        return storage_mode

    def load(self, device):
        """Load model architecture and pretrained weights.
        This method handles all different modes of storages.


        Parameters
        ----------
        device : `str`
            Device to create the model on, e.g. 'cpu' or 'cuda:0'.

        Returns
        -------
        model :
            The neural network model, loaded with pretrained weights.
            It's type should be a subclass of nn.Module, defined by
            the architecture module.
        """

        # Parse storage mode out of the provided package name
        self.storage_mode = self.storage_mode_from_path(self.model_package_name)

        # Create a proper adapter based on the storage mode
        if self.storage_mode == PackageStorageMode.local:
            adapter = NNModelPackageAdapterLocal(self.model_package_name)
        elif self.storage_mode == PackageStorageMode.neighbor:
            adapter = NNModelPackageAdapterNeighbor(self.model_package_name)
        else:
            raise NotImplementedError

        # Load various components based on the storage mode
        model = adapter.load_model()
        network_data = adapter.load_weights(device)

        # Load pretrained weights into model
        model.load_state_dict(network_data['state_dict'], strict=True)

        return model
