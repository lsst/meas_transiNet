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

from model import rbTransiNetModel
import torch


class rbTransiNetInterface:
    '''
    A class for interfacing between the LSST AP pipeline and
    an rbTransiNet model.
    '''

    def __init__(self, device='cpu'):
        self.model = rbTransiNetModel
        self.device = device

    def init(self, pretrained_file):

        # --- Load pre-trained model from disk
        network_data = torch.load(pretrained_file, map_location=self.device)
        self.model.load_state_dict(network_data['state_dict'], strict=True)

        # --- put model in "eval" mode and stand by
        self.model.eval()

    def prepare_input(self, x):
        '''
        Things like format conversion from afw.image.exposure to torch.tensor
        or stacking-up of images can happen here.
        '''
        x = x
        return x

    def infer(self, x):

        # --- Perform any required pre-processing and format conversion
        x = self.prepare_input(x)

        # --- Feed input to the network and get a score
        score = self.model(x)

        return score
