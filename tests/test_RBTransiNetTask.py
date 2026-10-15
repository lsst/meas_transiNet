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

import lsst.afw.table
from lsst.geom import Extent2I, Point2I, Point2D, Box2I
from lsst.images import Image
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.meas.transiNet import RBTransiNetTask


class TestRBTransiNetTask(lsst.utils.tests.TestCase):
    def create_sample_datasets(self):
        bbox = Box2I(Point2I(0, 0), Point2I(400, 400))
        dataset = lsst.meas.base.tests.TestDataset(bbox)
        dataset.addSource(5000, Point2D(50, 50.))
        # TODO: make one of these centered in a different corner of the pixel,
        # to test that the cutout is properly centered.
        dataset.addSource(10000, Point2D(100, 50.))
        dataset.addSource(20000, Point2D(1, 1))  # close-to-border source
        self.exposure, self.catalog = dataset.realize(10.0, dataset.makeMinimalSchema())
        self.image = Image.from_legacy(self.exposure.image)

    def setUp(self):
        self.create_sample_datasets()

        # Task configuration
        self.config = RBTransiNetTask.ConfigClass()
        self.config.modelPackageName = "dummy"
        self.config.modelPackageStorageMode = "local"
        self.config.cutoutSize = 256

    def test_make_cutouts(self):
        task = RBTransiNetTask(config=self.config)
        for record in self.catalog:
            result = task._make_cutouts(self.image, self.image, self.image, record)
            self._check_cutout(result.science, task.config.cutoutSize)
            self._check_cutout(result.template, task.config.cutoutSize)
            self._check_cutout(result.difference, task.config.cutoutSize)

            if record.getX() == 1 and record.getY() == 1:  # This is the "border"-source
                self._check_empty_cutout(result.science)
                self._check_empty_cutout(result.template)
                self._check_empty_cutout(result.difference)

    def test_cutout_pixels(self):
        """Test that a cutout holds the pixels of an afw cutout of the same
        box.
        """
        self.config.cutoutSize = 51
        task = RBTransiNetTask(config=self.config)
        record = self.catalog[0]
        box = Box2I.makeCenteredBox(record.getCentroid(), Extent2I(self.config.cutoutSize))
        expected = self.exposure.image.Factory(self.exposure.image, box).array
        result = task._make_cutouts(self.image, self.image, self.image, record)
        np.testing.assert_array_equal(result.science, expected)
        np.testing.assert_array_equal(result.template, expected)
        np.testing.assert_array_equal(result.difference, expected)

    def test_connections(self):
        """Test that the image inputs read the image component as
        ``ImageV2``.
        """
        config = RBTransiNetTask.ConfigClass()
        connections = config.connections.ConnectionsClass(config=config)
        self.assertEqual(connections.science.name, "calexp.image")
        self.assertEqual(connections.template.name, "deepDiff_templateExp.image")
        self.assertEqual(connections.difference.name, "deepDiff_differenceExp.image")
        for name in ("science", "template", "difference"):
            self.assertEqual(getattr(connections, name).storageClass, "ImageV2")

    def _check_cutout(self, image, size):
        """Test that the image cutout was made correctly.

        Parameters
        ----------
        image : `np.ndarray` (N, 2)
            Square cutout made from image.
        size : `int`
            Expected size of the cutout.
        """
        self.assertEqual(image.shape, (size, size))
        return

        # TODO: below test should be removed/fixed -- disabled for now.
        # It only passes with very specific cutout dimensions and in
        # very specific configurations.
        # See https://jira.lsstcorp.org/browse/DM-35635 for more info.

        max_index = np.unravel_index(image.argmax(), image.shape)
        # TODO: I'm not comfortable with this particular test: the exact
        # position of the max pixel depends on where in that pixel the
        # centroid is. We can assume Box2I.makeCenteredBox works correctly...
        self.assertEqual(max_index, ((size/2)-1, (size/2)-1))

    def _check_empty_cutout(self, cutout):
        """Test that the cutout is empty.

        Parameters
        ----------
        cutout : `np.ndarray` (N, 2)
            Square cutout made from image.
        """
        np.testing.assert_array_equal(cutout, np.zeros_like(cutout))

    def test_run(self):
        """Test that run passes an appropriate object to the interface.
        """
        task = RBTransiNetTask(config=self.config)
        result = task.run(self.image, self.image, self.image, self.catalog)
        self.assertIsInstance(result.classifications, lsst.afw.table.BaseCatalog)
        np.testing.assert_array_equal(self.catalog["id"], result.classifications["id"])
        np.testing.assert_equal("0.0", result.classifications[0]["version"])

    def test_config_butlerblock(self):
        config = RBTransiNetTask.ConfigClass()
        config.modelPackageName = "dummy"
        config.modelPackageStorageMode = "butler"
        with self.assertRaises(ValueError):
            config.validate()

    def test_config_butlerpass_none(self):
        config = RBTransiNetTask.ConfigClass()
        config.modelPackageName = None
        config.modelPackageStorageMode = "butler"
        # Should not raise
        config.validate()

    def test_config_butlerblock_empty(self):
        config = RBTransiNetTask.ConfigClass()
        config.modelPackageName = ""  # Want *only* None for butler packages
        config.modelPackageStorageMode = "butler"
        with self.assertRaises(ValueError):
            config.validate()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
