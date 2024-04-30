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

from .storageAdapterFactory import StorageAdapterFactory

import torch


class NNModelPackage:
    """
    An interface to abstract physical storage of network architecture &
    pretrained models out of clients' code.

    It handles all necessary required tasks, including fetching,
    decompression, etc. per need and creates a "Model Package"
    ready to use: a model architecture loaded with specific pretrained
    weights.
    """

    def __init__(self, model_package_name, package_storage_mode, **kwargs):
        # Validate passed arguments.
        if (
            package_storage_mode
            not in StorageAdapterFactory.storageAdapterClasses.keys()
        ):
            raise ValueError("Unsupported storage mode: %s" % package_storage_mode)
        if None in (model_package_name, package_storage_mode):
            raise ValueError("None is not a valid argument")

        self.model_package_name = model_package_name
        self.package_storage_mode = package_storage_mode

        self.adapter = StorageAdapterFactory.create(
            self.model_package_name, self.package_storage_mode, **kwargs
        )

        self.metadata = self.adapter.load_metadata()

    def load(self, device):
        """Load model architecture and pretrained weights.
        This method handles all different modes of storages.


        Parameters
        ----------
        device : `str`
            Device to create the model on, e.g. 'cpu' or 'cuda:0'.

        Returns
        -------
        model : `torch.nn.Module`
            The neural network model, loaded with pretrained weights.
            Its type should be a subclass of nn.Module, defined by
            the architecture module.
        """

        # Check if the specified device is valid.
        if device not in ["cpu"] + [
            "cuda:%d" % i for i in range(torch.cuda.device_count())
        ]:
            raise ValueError("Invalid device: %s" % device)

        # Load various components.
        # Note that because of the way the StorageAdapterButler works,
        # the model architecture and the pretrained weights are loaded
        # into the cpu memory, and only then moved to the target device.
        model = self.adapter.load_arch(device="cpu")
        network_data = self.adapter.load_weights(device="cpu")

        # Load pretrained weights into model
        model.load_state_dict(network_data, strict=True)

        # Move model to the specified device, if it is not already there.
        if device != "cpu":
            model = model.to(device)

        return model

    def get_model_input_shape(self):
        """Return the input shape of the model.

        Returns
        -------
        input_shape : `tuple`
            The input shape of the model -- (height, width), ignores
            the other dimensions.

        Raises
        ------
        KeyError
            If the input shape is not found in the metadata.
        """
        return tuple(self.metadata["input_shape"])

    def get_input_scale_factors(self):
        """
        Return the scale factors to be applied to the input data.

        Returns
        -------
        scale_factors : `tuple`
            The scale factors to be applied to the input data.

        Raises
        ------
        KeyError
            If the scale factors are not found in the metadata.
        """
        return tuple(self.metadata["input_scale_factor"])

    def get_boost_factor(self):
        """
        Return the boost factor to be applied to the output data.

        If the boost factor is not found in the metadata, return None.
        It is the responsibility of the client to know whether this type
        of model requires a boost factor or not.

        Returns
        -------
        boost_factor : `float`
            The boost factor to be applied to the output data.

        Raises
        ------
        KeyError
            If the boost factor is not found in the metadata.
        """
        return self.metadata["boost_factor"]
