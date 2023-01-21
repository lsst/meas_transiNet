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
import torch

PackageStorageLocation = enum.Enum('PackageStorageLocation', ['local', 'neighbor'])  # ,'butler', ...


class NNModelPackage:
    """
    A class to abstract physical storage of network architecture &
    pretrained models out of clients' code.
    It handles all necessary required tasks, including fetching,
    decompression, etc. per need and returns a "Model Package"
    ready to use.
    """

    def __init__(self, model_package_name):
        pass

    def storage_location_from_path(self, path):
        """Infer (decode!) storage location from path string.
        The storage location is assumed to be encoded in the
        path name e.g as "file:///" for local storage or
        "neighbor:///" for neighbor data repository.

        Parameters
        ----------
        path : string
            Path pointing to a stored model package


        Returns
        -------
        storage_location :
            Package storage location

        """

        storage_location = None  # TODO: replace with proper error handling by adding an 'else' below

        try:  # To catch non-standard paths
            token = path.split(':///')[0]
            if token.lower() == 'file':
                storage_location = PackageStorageLocation.local
            elif token.lower() == 'neighbor':
                storage_location = PackageStorageLocation.neighbor

        except Exception:
            pass  # TODO: replace with proper error handling

        return storage_location

    def load(self, device):
        """Load model architecture and pretrained weights.
        This method handles all different locations of storages.


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

        network_data = torch.load(pretrained_file, map_location=device)
        self.model.load_state_dict(network_data['state_dict'], strict=True)
