import lsst.afw.table
from lsst.geom import Point2I, Point2D, Box2I
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.meas.transiNet import RBTransiNetTask
import time


class ProfileRBTransiNetTask():
    def __init__(self, nSources):
        bbox = Box2I(Point2I(0, 0), Point2I(4000, 4000))
        dataset = lsst.meas.base.tests.TestDataset(bbox)

        # Add random sources to the dataset
        import random
        for i in range(nSources):
            dataset.addSource(5000, Point2D(random.uniform(0, 4000), random.uniform(0, 4000)))

        self.exposure, self.catalog = dataset.realize(10.0, dataset.makeMinimalSchema())

        self.config = RBTransiNetTask.ConfigClass()
        self.config.modelPackageName = "dummy"
        self.config.modelPackageStorageMode = "local"

    def test_run(self):
        task = RBTransiNetTask(config=self.config)
        task.run(self.exposure, self.exposure, self.exposure, self.catalog)


if __name__ == "__main__":

    nSources = 50
    profiler = ProfileRBTransiNetTask(nSources)
    start = time.time()
    profiler.test_run()
    end = time.time()
    print(f"It took {end - start} seconds to run {nSources} sources.")

    nSources = 500
    profiler = ProfileRBTransiNetTask(nSources)
    start = time.time()
    profiler.test_run()
    end = time.time()
    print(f"It took {end - start} seconds to run {nSources} sources.")
