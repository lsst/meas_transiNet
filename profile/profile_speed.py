import lsst.afw.table
from lsst.geom import Point2I, Point2D, Box2I
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.meas.transiNet import RBTransiNetTask
import time
import random
import numpy as np

class ProfileRBTransiNetTask():
    def init(self, nSources, modelPackageName="rbResnet50-DC2", modelPackageStorageMode="neighbor", device="cpu"):
        self.create_dummy_dataset(nSources)
        self.config = RBTransiNetTask.ConfigClass()
        self.config.modelPackageName = modelPackageName
        self.config.modelPackageStorageMode = modelPackageStorageMode
        self.config.computeDevice = device

    def create_dummy_dataset(self, nSources):
        '''Create a dummy dataset with nSources sources.
        This creates a dummy catalog and an *empty* exposure.
        We do not use TestDataset.AddSource() from meas_base since
        we cannot afford to wait for it to complete -- i.e. it is
        too slow for our purposes.
        '''
        bbox = Box2I(Point2I(0, 0), Point2I(4000, 4000))
        TD = lsst.meas.base.tests.TestDataset
        self.catalog = lsst.afw.table.SourceCatalog(TD.makeMinimalSchema())
        record = self.catalog.addNew()

        for i in range(nSources):
            record.set(TD.keys["instFlux"], 5000)
            record.set(TD.keys["instFluxErr"], 0)
            record.set(TD.keys["centroid"], Point2D(random.uniform(0, bbox.getWidth()), random.uniform(0, bbox.getHeight())))
            covariance = np.random.normal(0, 0.1, 4).reshape(2, 2)
            covariance[0, 1] = covariance[1, 0]  # CovarianceMatrixKey assumes symmetric x_y_Cov
            record.set(TD.keys["centroid_sigma"], covariance.astype(np.float32))
            record.set(TD.keys["isStar"], True)
            self.catalog.append(record)

        self.exposure = lsst.afw.image.ExposureF(bbox)

    def test_run(self):
        task = RBTransiNetTask(config=self.config)
        task.run(self.exposure, self.exposure, self.exposure, self.catalog)

    def profile(self, nSources=50, nTimes=1):
        for device in ["cpu", "cuda:0"]:
            for modelName in ["rbResnet50-DC2", "rbMobileNet"]:
                self.init(nSources, modelPackageName=modelName, device=device)
                start = time.time()
                for i in range(nTimes):
                    self.test_run()
                    end = time.time()
                    print(f"Time per run for {nSources} sources and {nTimes} runs with {modelName} \ton {device}: \t{(end-start)/nTimes:.2f} seconds")


if __name__ == "__main__":
    import sys
    profiler = ProfileRBTransiNetTask()
    profiler.profile(nSources=int(sys.argv[1]) if len(sys.argv) > 1 else 10,
                     nTimes=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
