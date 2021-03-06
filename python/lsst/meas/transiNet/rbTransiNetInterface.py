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
    """
    The interface between the LSST AP pipeline and a trained pytorch-based
    RBTransiNet neural network model.
    """

    def __init__(self, model, pretrained_file=None, device='cpu'):
        self.model = model
        self.device = device
        self.init(pretrained_file)

    def init(self, pretrained_file):
        """Deferred (manual) initialization.

        Model initialization from a pretrained file can be slow.

        Parameters
        ----------
        pretrained_file : `str`
            Path to the trained model.
        """
        network_data = torch.load(pretrained_file, map_location=self.device)
        self.model.load_state_dict(network_data['state_dict'], strict=True)

        # Put the model in evaluation mode instead of training model.
        self.model.eval()

    def prepare_input(self, inputs):
        """
        Convert inputs from numpy arrays, etc. to a torch.tensor blob.

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
        cutoutsList = []
        labelsList = []
        for inp in inputs:
            # Convert each cutout to a torch tensor
            template = torch.from_numpy(inp.template)
            science = torch.from_numpy(inp.science)
            difference = torch.from_numpy(inp.difference)

            # Stack the components to create a single blob
            singleBlob = torch.stack((template, science, difference), dim=0)

            # And append them to the temporary list
            cutoutsList.append(singleBlob)

            labelsList.append(inp.label)

        torchBlob = torch.stack(cutoutsList)
        return torchBlob, labelsList

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
        blob, labels = self.prepare_input(inputs)
        result = self.model(blob)
        scores = torch.sigmoid(result)
        npyScores = scores.detach().numpy().ravel()

        return npyScores
