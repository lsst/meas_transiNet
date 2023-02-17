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

from lsst.meas.transiNet.modelPackages.nnModelPackage import NNModelPackage
from lsst.meas.transiNet.modelPackages.storageAdapter import StorageAdapter
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

        # test shape of loaded weights
        self.assertTupleEqual(weights.shape, (16, 3, 3, 3))

        # test weight values
        torch.testing.assert_close(weights[0][0],
                                   torch.tensor([[0.14145353, -0.10257456, 0.17189537],
                                                 [-0.03069756, -0.1093155, 0.15207087],
                                                 [0.06509985, 0.11900973, -0.16013929]]),
                                   rtol=1e-8, atol=1e-8)


class TestModelPackageNeighbor(unittest.TestCase):
    def setUp(self):
        # Create a dummy model package in the neighboring repository
        source_dir = os.path.join(StorageAdapterLocal.get_base_path(), 'dummy')
        self.temp_package_dir = os.path.join(StorageAdapterNeighbor.get_base_path(), 'dummy')

        try:
            from shutil import copytree
            copytree(source_dir, self.temp_package_dir)
        except FileExistsError:
            raise RuntimeError('Dummy model package in neighbor mode!')

        self.model_package_name = 'dummy'
        self.package_storage_mode = 'neighbor'

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

    def tearDown(self):
        # Remove the neighbor-mode dummy model package
        from shutil import rmtree
        rmtree(self.temp_package_dir)


class TestStorageModeRestrictions(unittest.TestCase):
    """Test that the storage mode restrictions are enforced.
    This test relies on the fact that the standard storage modes
    expect model packages to be stored in one of the EUPS installation
    paths. If this is not the case, they throw a TypeError.
    """

    def setUp(self):
        # Get list of all eups installation paths
        eups_paths = self.get_all_eups_installation_paths()

        # Backup all enironment variables at once.
        self.env_vars = dict(os.environ)

        # Clear all environment variables which point at EUPS installation
        # paths.
        for key, value in os.environ.items():
            if value in eups_paths:
                del os.environ[key]

    def test_storage_modes(self):
        """Test that all storage modes fail in the absence of eups-related
        environment variables.
        This implies that all the standard storage modes throw a TypeError
        when they face an empty environment variable. This may change into
        a more more robust test in the future.
        """
        for mode in StorageAdapter.storageAdapterClasses.keys():
            print(f"Testing storage mode {mode}")
            with self.assertRaises(TypeError):
                StorageAdapter.create("dummy", mode)

    def get_all_eups_installation_paths(self):
        """Get all EUPS installation paths.

        Returns
        -------
        paths : `list` of `str`
        """
        import eups

        products = eups.Eups().getSetupProducts()
        return [product.dir for product in products]

    def tearDown(self):
        # Restore all environment variables at once.
        os.environ.update(self.env_vars)
