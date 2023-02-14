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

import numpy as np

from lsst.meas.transiNet import RBTransiNetInterface, CutoutInputs


class TestOneCutout(unittest.TestCase):
    def setUp(self):
        self.interface = RBTransiNetInterface("dummy", "local")

    def test_infer_empty(self):
        """Test running infer on images containing all zeros.
        """
        data = np.zeros((256, 256), dtype=np.single)
        inputs = CutoutInputs(science=data, difference=data, template=data)
        result = self.interface.infer([inputs])
        self.assertTupleEqual(result.shape, (1,))
        self.assertAlmostEqual(result[0], 0.5011908)  # Empricial meaningless value spit by this very model
