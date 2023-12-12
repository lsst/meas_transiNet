""" A demo script to show how to ingest a new NNModelPackage into a
    butler repository.
"""

from lsst.meas.transiNet.modelPackages.nnModelPackage import NNModelPackage
from lsst.meas.transiNet.modelPackages.storageAdapterButler import StorageAdapterButler

from lsst.daf.butler import Butler
butler = Butler("./workspace/repo/", writeable=True)
local_model_package = NNModelPackage('dummy', 'local')
StorageAdapterButler.ingest(local_model_package, butler)
