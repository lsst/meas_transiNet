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
from lsst.geom import Point2I, Point2D, Box2I
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.meas.transiNet import RBTransiNetTask


class TestRBTransiNetTask(lsst.utils.tests.TestCase):
    def setUp(self):
        bbox = Box2I(Point2I(0, 0), Point2I(400, 400))
        dataset = lsst.meas.base.tests.TestDataset(bbox)
        dataset.addSource(5000, Point2D(50, 50.))
        # TODO: make one of these centered in a different corner of the pixel,
        # to test that the cutout is properly centered.
        dataset.addSource(10000, Point2D(100, 50.))
        dataset.addSource(20000, Point2D(1, 1))  # close-to-border source
        self.exposure, self.catalog = dataset.realize(10.0, dataset.makeMinimalSchema())

        self.config = RBTransiNetTask.ConfigClass()
        self.config.modelPackageName = "dummy"
        self.config.modelPackageStorageMode = "local"

    def test_make_cutouts(self):
        task = RBTransiNetTask(config=self.config)
        for record in self.catalog:
            result = task._make_cutouts(self.exposure, self.exposure, self.exposure, record)
            self._check_cutout(result.science, task.config.cutoutSize)
            self._check_cutout(result.template, task.config.cutoutSize)
            self._check_cutout(result.difference, task.config.cutoutSize)

            if record.getX() == 1 and record.getY() == 1:  # This is the "border"-source
                self._check_empty_cutout(result.science)
                self._check_empty_cutout(result.template)
                self._check_empty_cutout(result.difference)

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
        """Test that run passes an appropriate object to the interface,
        mocking the interface infer step so we don't need to use pytorch.
        """
        scores = np.array([0.0, 1.0, 0.0])
        task = RBTransiNetTask(config=self.config)
        task.interface.infer = unittest.mock.Mock(task.interface.infer)
        with unittest.mock.patch.object(task.interface, "infer") as mock_infer:
            mock_infer.return_value = scores
            result = task.run(self.exposure, self.exposure, self.exposure, self.catalog)
            self.assertIsInstance(result.classifications, lsst.afw.table.BaseCatalog)
            np.testing.assert_array_equal(self.catalog["id"], result.classifications["id"])
            np.testing.assert_array_equal(scores, result.classifications["score"])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
