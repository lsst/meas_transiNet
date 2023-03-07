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
import os
import shutil

from lsst.meas.transiNet.modelPackages.nnModelPackage import NNModelPackage
from lsst.meas.transiNet.modelPackages.storageAdapterLocal import StorageAdapterLocal
from lsst.meas.transiNet.modelPackages.storageAdapterNeighbor import StorageAdapterNeighbor


class TestModelPackageLocal(unittest.TestCase):
    def setUp(self):
        self.model_package_name = 'dummy'
        self.package_storage_mode = 'local'

    def test_load(self):
        """Test loading of a local model package
        """
        model_package = NNModelPackage(self.model_package_name, self.package_storage_mode)
        model = model_package.load(device='cpu')

        weights = next(model.parameters())

        # Test shape of loaded weights.
        self.assertTupleEqual(weights.shape, (16, 3, 3, 3))

        # Test weight values.
        # Only test a single tensor, as the probability of randomly having
        # matching weights "only" in a single tensor is extremely low.
        torch.testing.assert_close(weights[0][0],
                                   torch.tensor([[0.14145353, -0.10257456, 0.17189537],
                                                 [-0.03069756, -0.1093155, 0.15207087],
                                                 [0.06509985, 0.11900973, -0.16013929]]),
                                   rtol=1e-8, atol=1e-8)

    def test_arch_weights_mismatch(self):
        """Test loading of a model package with mismatching architecture and
        weights.

        Does not use PyTorch's built-in serialization to be generic and
        independent of the backend.
        """
        model_package = NNModelPackage(self.model_package_name, self.package_storage_mode)

        # Create a fake architecture file.
        arch_f = os.path.basename(model_package.adapter.model_filename)
        model_filename_backup = model_package.adapter.model_filename
        model_package.adapter.model_filename = model_package.adapter.model_filename.replace(arch_f,
                                                                                            'fake_' + arch_f)

        with open(model_package.adapter.model_filename, 'w') as f:
            # Write a dummy 1-layer fully connected network into the file.
            f.write('__all__ = ["Net"]\n')
            f.write('import torch\n')
            f.write('import torch.nn as nn\n')
            f.write('class Net(nn.Module):\n')
            f.write('    def __init__(self):\n')
            f.write('        super(Net, self).__init__()\n')
            f.write('        self.fc1 = nn.Linear(3, 16)\n')
            f.write('    def forward(self, x):\n')
            f.write('        x = self.fc1(x)\n')
            f.write('        return x\n')

        # Now try to load the model.
        with self.assertRaises(RuntimeError):
            model_package.load(device='cpu')

        # Clean up.
        os.remove(model_package.adapter.model_filename)
        model_package.adapter.model_filename = model_filename_backup


class TestModelPackageNeighbor(unittest.TestCase):
    def setUp(self):
        # Create a dummy model package in the neighboring repository
        source_dir = os.path.join(StorageAdapterLocal.get_base_path(), 'dummy')
        self.temp_package_dir = os.path.join(StorageAdapterNeighbor.get_base_path(), 'dummy')

        try:
            shutil.copytree(source_dir, self.temp_package_dir)
        except FileExistsError:
            raise RuntimeError('Dummy model package in neighbor mode!')

        self.model_package_name = 'dummy'
        self.package_storage_mode = 'neighbor'

    def tearDown(self):
        # Remove the neighbor-mode dummy model package
        shutil.rmtree(self.temp_package_dir)

    def test_load(self):
        """Test loading of a model package of neighbor mode
        """
        model_package = NNModelPackage(self.model_package_name, self.package_storage_mode)
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
