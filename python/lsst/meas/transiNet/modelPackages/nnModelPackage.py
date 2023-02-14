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

from .storageAdapter import StorageAdapter


class NNModelPackage:
    """
    A class to abstract physical storage of network architecture &
    pretrained models out of clients' code.
    It handles all necessary required tasks, including fetching,
    decompression, etc. per need and returns a "Model Package"
    ready to use.
    """

    def __init__(self, model_package_name, package_storage_mode):
        self.model_package_name = model_package_name
        self.package_storage_mode = package_storage_mode

    def load(self, device):
        """Load model architecture and pretrained weights.
        This method handles all different modes of storages.


        Parameters
        ----------
        device : string
            device to create the model on.

        Returns
        -------
        model :
            The neural network model, loaded with pretrained weights.
            It's type should be a subclass of nn.Module, defined by
            the architecture module.
        """

        adapter = StorageAdapter.create(self.model_package_name, self.package_storage_mode)

        # Load various components based on the storage mode
        model = adapter.load_model()
        network_data = adapter.load_weights(device)

        # Load pretrained weights into model
        model.load_state_dict(network_data['state_dict'], strict=True)

        return model
