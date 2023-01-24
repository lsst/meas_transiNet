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

import unittest
import torch

from lsst.meas.transiNet import NNModelPackage


class TestModelPackageLocal(unittest.TestCase):
    def setUp(self):
        self.model_package_name = 'local:///dummy'

    def test_load(self):
        """Test loading of a local model package
        """
        model_package = NNModelPackage(self.model_package_name)
        model = model_package.load(device='cpu')

        weights = next(model.parameters())

        # test shape of loaded weights
        self.assertTupleEqual(weights.shape, (16, 3, 3, 3))

        # test weight values
        torch.testing.assert_close(weights[0][0],
                                   torch.tensor([[0.14145353, -0.10257456, 0.17189537],
                                                 [-0.03069756, -0.1093155, 0.15207087],
                                                 [0.06509985, 0.11900973, -0.16013929]]),
                                   rtol=1e-8, atol=1e-8)
