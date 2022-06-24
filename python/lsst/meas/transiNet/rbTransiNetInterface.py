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
    """A single cutout to be processed and put into a pytorch tensor object.
    """
    science: np.ndarray
    template: np.ndarray
    difference: np.ndarray

    label: bool = None
    """Known truth of whether this is a real or bogus object."""


class RBTransiNetInterface:
    """
    A class for interfacing between the LSST AP pipeline and
    an RBTransiNet model.
    """
    def __init__(self, model, pretrained_file=None, device='cpu'):
        """Constructor"""
        self.model = model
        self.device = device
        self.init(pretrained_file)

    def init(self, pretrained_file):
        """Deferred (manual) initialization.

        Normally takes a long time. So better be called only once
		and when there's time to wait!

        Parameters
        ----------
        pretrained_file : `str`
            Path to the trained model.
        """

        # --- Load pre-trained model from disk
        network_data = torch.load(pretrained_file, map_location=self.device)
        self.model.load_state_dict(network_data["state_dict"], strict=True)

        # --- put model in "eval" mode and stand by
        self.model.eval()

    def prepare_input(self, inputs):
        """
        Things like format conversion from afw.image.exposure to torch.tensor
        or stacking-up of images can happen here.

        Parameters
        ----------
        inputs : `list` [`CutoutInputs`]
            Inputs to be scored.

        Returns
        -------
        blob
            Prepared torch tensor blob to be passed to the network
        """

        if len(inputs) > 1:
            blob = np.concat() # stack inputs on top of eachother (in numpy)
        else:
            input = inputs[0]
            blob = np.concat(input.science, input.template, input.difference)

        # convert the numpy blob to Torch Tensor
        return blob, labels

    def infer(self, inputs):
    	"""Inference.
        It is the most frequently used method. Receives one or a batch of
        inputs, x, and returns corresponding scores.
        x is a list. It is intentionally defined loosely and the exact
        specifications of its contents are left for future versions.
        """

        Parameters
        ----------
        inputs : `list` [`CutoutInputs`]
            Inputs to be scored.

        Returns
        -------
        scores : `numpy.array`
            Float scores for each element of ``inputs``.
        """

        # --- Perform any required pre-processing and format conversion
        blob, labels = self.prepare_input(inputs)
        scores_ = self.model(blob)
        scores = result.detach().to_npy()

        return scores
