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

__all__ = ["RBTransiNetInterface", "CutoutInputs"]

import numpy as np
import dataclasses
import torch

from .modelPackages.nnModelPackage import NNModelPackage


@dataclasses.dataclass(frozen=True, kw_only=True)
class CutoutInputs:
    """Science/template/difference cutouts of a single object plus other
    metadata.
    """
    science: np.ndarray
    template: np.ndarray
    difference: np.ndarray

    label: bool = None
    """Known truth of whether this is a real or bogus object."""


class RBTransiNetInterface:
    """ The interface between the LSST AP pipeline and a trained pytorch-based
    RBTransiNet neural network model.

    Parameters
    ----------
    task : `lsst.meas.transiNet.RBTransiNetTask`
        The task that is using this interface: the 'left side'.
    model_package_name : `str`
        Name of the model package to load.
    package_storage_mode : {'local', 'neighbor'}
        Storage mode of the model package
    device : `str`
        Device to load and run the neural network on, e.g. 'cpu' or 'cuda:0'
    """

    def __init__(self, task, device='cpu'):
        self.task = task

        # in case the model package name is not set at this stage, it is not
        # needed (e.g. in butler mode).
        self.model_package_name = task.config.modelPackageName or 'N/A'

        self.package_storage_mode = task.config.modelPackageStorageMode
        self.device = device
        self.init_model()

    def init_model(self):
        """Create and initialize an NN model
        """

        if self.package_storage_mode == 'butler' and self.task.butler_loaded_package is None:
            raise RuntimeError("RBTransiNetInterface is trying to load a butler-mode NN model package, "
                               "but the RBTransiNetTask has not passed down a preloaded payload.")

        model_package = NNModelPackage(model_package_name=self.model_package_name,
                                       package_storage_mode=self.package_storage_mode,
                                       butler_loaded_package=self.task.butler_loaded_package)
        self.model = model_package.load(self.device)

        # Put the model in evaluation mode instead of training model.
        self.model.eval()

    def input_to_batches(self, inputs, batchSize):
        """Convert a list of inputs to a generator of batches.

        Parameters
        ----------
        inputs : `list` [`CutoutInputs`]
            Inputs to be scored.

        Returns
        -------
        batches : `generator`
            Generator of batches of inputs.
        """
        for i in range(0, len(inputs), batchSize):
            yield inputs[i:i + batchSize]

    def prepare_input(self, inputs):
        """Convert inputs from numpy arrays, etc. to a torch.tensor blob.

        Parameters
        ----------
        inputs : `list` [`CutoutInputs`]
            Inputs to be scored.

        Returns
        -------
        blob
            Prepared torch tensor blob to run the model on.
        labels
            Truth labels, concatenated into a single list.
        """
        # cutoutsList = []
        # labelsList = []
        # for inp in inputs:
        #     # Convert each cutout to a torch tensor
        #     template = torch.from_numpy(inp.template)
        #     science = torch.from_numpy(inp.science)
        #     difference = torch.from_numpy(inp.difference)

        #     # Stack the components to create a single blob
        #     singleBlob = torch.stack((template, science, difference), dim=0)

        #     # And append them to the temporary list
        #     cutoutsList.append(singleBlob)

        #     labelsList.append(inp.label)

        # blob = torch.stack(cutoutsList)
        # return blob, labelsList
        image_tensors = []
        labels_list = []
        
        for inp in inputs:
            # Convert each cutout to a torch tensor
            template_tensor = torch.from_numpy(inp.template)
            science_tensor = torch.from_numpy(inp.science)
            difference_tensor = torch.from_numpy(inp.difference)

            # Append them to the temporary list
            image_tensors.extend([template_tensor, science_tensor, difference_tensor])
            labels_list.extend([inp.label, inp.label, inp.label])  # Assuming label is same for all three images

        return image_tensors, labels_list

    def infer(self, inputs):
        """Return the score of this cutout.

        Parameters
        ----------
        inputs : `list` [`CutoutInputs`]
            Inputs to be scored.

        Returns
        -------
        scores : `numpy.array`
            Float scores for each element of ``inputs``.
        """

        # Convert the inputs to batches.
        # TODO: The batch size is set to 64 for now. Later when
        # deploying parallel instances of the task, memory limits
        # should be taken into account, if necessary.
        batches = self.input_to_batches(inputs, batchSize=64)

        # Loop over the batches
        for i, batch in enumerate(batches):
            torchBlob, labelsList = self.prepare_input(batch)

            # Run the model
            with torch.no_grad():
                output_ = self.model(torchBlob)
            output = torch.sigmoid(output_)

            # And append the results to the list
            if i == 0:
                scores = output
            else:
                scores = torch.cat((scores, output.cpu()), dim=0)

        npyScores = scores.detach().numpy().ravel()
        return npyScores