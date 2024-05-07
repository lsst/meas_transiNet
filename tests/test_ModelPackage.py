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
import os
import shutil
import tempfile

from lsst.meas.transiNet.modelPackages.nnModelPackage import NNModelPackage
from lsst.meas.transiNet.modelPackages.storageAdapterLocal import StorageAdapterLocal
from lsst.meas.transiNet.modelPackages.storageAdapterNeighbor import (
    StorageAdapterNeighbor,
)
from lsst.meas.transiNet.modelPackages.storageAdapterButler import StorageAdapterButler
from lsst.daf.butler import Butler
from lsst.daf.butler.registry._exceptions import ConflictingDefinitionError
import lsst.utils

try:
    neighborDirectory = lsst.utils.getPackageDir("rbClassifier_data")
except LookupError:
    neighborDirectory = None


def sanity_check_dummy_model(test, model):
    weights = next(model.parameters())

    # Test shape of loaded weights.
    test.assertTupleEqual(weights.shape, (16, 3, 3, 3))

    # Test weight values.
    # Only test a single tensor, as the probability of randomly having
    # matching weights "only" in a single tensor is extremely low.
    # torch.testing.assert_close(weights[0][0],
    #             torch.tensor([[0.14145353, -0.10257456, 0.17189537],
    #                             [-0.03069756, -0.1093155, 0.15207087],
    #                             [0.06509985, 0.11900973, -0.16013929]]),
    #             rtol=1e-8, atol=1e-8)


class TestModelPackageLocal(unittest.TestCase):
    def setUp(self):
        self.model_package_name = "dummy"
        self.package_storage_mode = "local"

    def test_load(self):
        """Test loading of a local model package"""
        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )
        model = model_package.load(device="cpu")
        sanity_check_dummy_model(self, model)

    def test_arch_weights_mismatch(self):
        """Test loading of a model package with mismatching architecture and
        weights.

        Does not use PyTorch's built-in serialization to be generic and
        independent of the backend.
        """
        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )

        # Create a fake architecture file.
        arch_f = os.path.basename(model_package.adapter.model_filename)
        # print(arch_f)
        model_filename_backup = model_package.adapter.model_filename
        # print(model_filename_backup)
        model_package.adapter.model_filename = (
            model_package.adapter.model_filename.replace(arch_f, "fake_" + arch_f)
        )

        try:
            with open(model_package.adapter.model_filename, "w") as f:
                # Write a dummy 1-layer fully connected network into the file.
                f.write('__all__ = ["Net"]\n')
                f.write("import torch\n")
                f.write("import torch.nn as nn\n")
                f.write("class Net(nn.Module):\n")
                f.write("    def __init__(self, img_size=(3,51,51)):\n")
                f.write("        super(Net, self).__init__()\n")
                f.write("        self.fc1 = nn.Linear(3, 16)\n")
                f.write("    def forward(self, x):\n")
                f.write("        x = self.fc1(x)\n")
                f.write("        return x\n")
        finally:
            # Now try to load the model.
            with self.assertRaises(RuntimeError):
                model_package.load(device="cpu")

            # Clean up.
            os.remove(model_package.adapter.model_filename)
            model_package.adapter.model_filename = model_filename_backup

    def test_invalid_inputs(self):
        """Test invalid and missing inputs
        (of NNModelPackage constructor, as well as the load method)
        """
        with self.assertRaises(ValueError):
            NNModelPackage("dummy", "invalid")

        with self.assertRaises(ValueError):
            NNModelPackage("invalid", None)

        with self.assertRaises(ValueError):
            NNModelPackage(None, "local")

        with self.assertRaises(ValueError):
            NNModelPackage(None, "invalid")

        with self.assertRaises(ValueError):
            NNModelPackage(None, None)

        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )

        with self.assertRaises(ValueError):
            model_package.load(device="invalid")

        with self.assertRaises(ValueError):
            model_package.load(device="gpu199")

        with self.assertRaises(ValueError):
            model_package.load(device=None)

    def test_metadata(self):
        """Test loading of metadata"""
        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )

        # Test whether the metadata object exists.
        # (it should be automatically loaded when the model package
        # is constructed)
        self.assertTrue(hasattr(model_package, "metadata"))

        # Test whether the metadata object is a dictionary.
        self.assertIsInstance(model_package.metadata, dict)

        # Test whether the metadata object contains the mandatory keys.
        self.assertListEqual(
            list(model_package.metadata.keys()),
            ["version", "description", "input_shape", "input_scale_factor"],
            msg="Metadata object does not contain the mandatory keys.",
        )

        # Test whether the metadata-related methods return the correct values
        # for the dummy model package.
        self.assertEqual(model_package.get_model_input_shape(), (51, 51, 3))
        self.assertEqual(
            model_package.get_input_scale_factors(), (1.0, 0.0033333333333333335, 1.0)
        )
        with self.assertRaises(KeyError):
            model_package.get_boost_factor()  # No boost factor for dummy

        # Test whether the number of scale factor elements matches the number
        # of input channels.
        self.assertEqual(
            len(model_package.get_input_scale_factors()),
            model_package.get_model_input_shape()[2],
        )


@unittest.skipIf(neighborDirectory is None, "rbClassifier_data not setup")
class TestModelPackageNeighbor(unittest.TestCase):
    def setUp(self):
        # Create a dummy model package in the neighboring repository
        source_dir = os.path.join(StorageAdapterLocal.get_base_path(), "dummy")
        self.temp_package_dir = os.path.join(
            StorageAdapterNeighbor.get_base_path(), "dummy"
        )

        try:
            shutil.copytree(source_dir, self.temp_package_dir)
        except FileExistsError:
            raise RuntimeError("Dummy model package in neighbor mode!")

        self.model_package_name = "dummy"
        self.package_storage_mode = "neighbor"

    def tearDown(self):
        # Remove the neighbor-mode dummy model package
        shutil.rmtree(self.temp_package_dir)

    def test_load(self):
        """Test loading of a model package of neighbor mode"""
        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )
        model = model_package.load(device="cpu")

        # test to make sure the model package is loading from the
        # neighbor repository.
        #
        # TODO: later if we move this test to the neighbor package itself, this
        # check needs to be updated.
        self.assertTrue(
            model_package.adapter.checkpoint_filename.startswith(
                lsst.utils.getPackageDir("rbClassifier_data")
            )
        )

        sanity_check_dummy_model(self, model)

    def test_metadata(self):
        """Test loading of metadata"""
        model_package = NNModelPackage(
            self.model_package_name, self.package_storage_mode
        )

        # Test whether the metadata object exists.
        # (it should be automatically loaded when the model package
        # is constructed)
        self.assertTrue(hasattr(model_package, "metadata"))

        # Test whether the metadata object is a dictionary.
        self.assertIsInstance(model_package.metadata, dict)

        # Test whether the metadata object contains the mandatory keys.
        self.assertListEqual(
            list(model_package.metadata.keys()),
            ["version", "description", "input_shape", "input_scale_factor"],
            msg="Metadata object does not contain the mandatory keys.",
        )

        # Test whether the metadata-related methods return the correct values
        # for the dummy model package.
        self.assertEqual(model_package.get_model_input_shape(), (51, 51, 3))
        self.assertEqual(
            model_package.get_input_scale_factors(), (1.0, 0.0033333333333333335, 1.0)
        )
        with self.assertRaises(KeyError):
            model_package.get_boost_factor()  # No boost factor for dummy

        # Test whether the number of scale factor elements matches the number
        # of input channels.
        self.assertEqual(
            len(model_package.get_input_scale_factors()),
            model_package.get_model_input_shape()[2],
        )


class TestModelPackageButler(unittest.TestCase):
    def setUp(self):
        self.model_package_name = "dummy"

        # Create a dummy butler repository (in a temporary directory).
        # Note that a test repo using makeTestRepo would not suffice
        # as we need to test the ingestion of a model package too.
        self.repo_root = tempfile.mkdtemp(prefix="butler_")
        Butler.makeRepo(root=self.repo_root)
        self.butler = Butler(self.repo_root, writeable=True)

    def tearDown(self):
        shutil.rmtree(self.repo_root)

    def ingest(self):
        # Load a local model package, to transfer/ingest to
        # the butler repository.
        local_model_package = NNModelPackage("dummy", "local")
        StorageAdapterButler.ingest(
            local_model_package, self.butler, model_package_name=self.model_package_name
        )

    def load_from_butler(self):
        # Load the model package from the butler repository.
        model_package = NNModelPackage(
            model_package_name=self.model_package_name,
            package_storage_mode="butler",
            butler=self.butler,
        )
        return model_package

    def test_double_ingest(self):
        """Test whether redundant ingestion of a model package to the butler
        repository fails as expected.
        """
        self.ingest()
        # assert that the second one raises ConflictingDefinitionError
        with self.assertRaises(ConflictingDefinitionError):
            self.ingest()

    def test_ingest_load(self):
        """Test ingesting and loading of a model package of butler mode"""
        self.ingest()
        model_package = self.load_from_butler()
        model = model_package.load(device="cpu")
        sanity_check_dummy_model(self, model)
