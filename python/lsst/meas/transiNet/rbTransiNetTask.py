# This file is part of meas_transiNet.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["RBTransiNetTask", "RBTransiNetConfig"]

import lsst.geom
import lsst.pex.config
import lsst.pipe.base
from lsst.utils.timer import timeMethod
import numpy as np

from . import rbTransiNetInterface
from lsst.meas.transiNet.modelPackages.storageAdapterButler import StorageAdapterButler


class RBTransiNetConnections(lsst.pipe.base.PipelineTaskConnections,
                             dimensions=("instrument", "visit", "detector"),
                             defaultTemplates={"coaddName": "deep", "fakesType": ""}):
    # NOTE: Do we want the "ready to difference" template, or something
    # earlier? This one is warped, but not PSF-matched.
    template = lsst.pipe.base.connectionTypes.Input(
        doc="Input warped template to subtract.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp"
    )
    science = lsst.pipe.base.connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    difference = lsst.pipe.base.connectionTypes.Input(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    diaSources = lsst.pipe.base.connectionTypes.Input(
        doc="Detected sources on the difference image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}Diff_candidateDiaSrc",
    )
    pretrainedModel = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        doc="Pretrained neural network model (-package) for the RBClassifier.",
        dimensions=(),
        storageClass="NNModelPackagePayload",
        name=StorageAdapterButler.dataset_type_name,
    )

    # Outputs
    classifications = lsst.pipe.base.connectionTypes.Output(
        doc="Catalog of real/bogus classifications for each diaSource, "
            "element-wise aligned with diaSources.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Catalog",
        name="{fakesType}{coaddName}RealBogusSources",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if self.config.modelPackageStorageMode != "butler":
            del self.pretrainedModel


class RBTransiNetConfig(lsst.pipe.base.PipelineTaskConfig, pipelineConnections=RBTransiNetConnections):
    modelPackageName = lsst.pex.config.Field(
        optional=True,
        dtype=str,
        doc=("A unique identifier of a model package. ")
    )
    modelPackageStorageMode = lsst.pex.config.ChoiceField(
        dtype=str,
        doc=("A string that indicates _where_ and _how_ the model package is stored."),
        allowed={'local': 'packages stored in the meas_transiNet repository',
                 'neighbor': 'packages stored in the rbClassifier_data repository',
                 'butler': 'packages stored in the butler repository',
                 },
        default='neighbor',
    )
    cutoutSize = lsst.pex.config.Field(
        dtype=int,
        doc="Width/height of square cutouts to send to classifier.",
        default=51,
    )

    def validate(self):
        # if we are in the butler mode, the user should not set
        # a modelPackageName as a config field.
        if self.modelPackageStorageMode == "butler":
            if self.modelPackageName is not None:
                raise ValueError("In a _real_ run of a pipeline when the "
                                 "modelPackageStorageMode is 'butler', "
                                 "the modelPackageName cannot be specified "
                                 "as a config field. Pass it as a collection"
                                 "name in the command-line instead.")


class RBTransiNetTask(lsst.pipe.base.PipelineTask):
    """Task for running TransiNet real/bogus classification on the output of
    the image subtraction pipeline.
    """
    _DefaultName = "rbTransiNet"
    ConfigClass = RBTransiNetConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.butler_loaded_package = None

    @timeMethod
    def run(self, template, science, difference, diaSources, pretrainedModel=None):

        # Create the TransiNet interface object.
        # Note: assuming each quanta creates one instance of this task, this is
        # a proper place for doing this since loading of the model is run only
        # once. However, if in the future we come up with a design in which one
        # task instance is used for multiple quanta, this will need to be moved
        # somewhere else -- e.g. to the __init__ method, or even to runQuantum.
        self.butler_loaded_package = pretrainedModel  # This will be used by the interface
        self.interface = rbTransiNetInterface.RBTransiNetInterface(self)

        cutouts = [self._make_cutouts(template, science, difference, source) for source in diaSources]
        self.log.info("Extracted %d cutouts.", len(cutouts))
        scores = self.interface.infer(cutouts)
        self.log.info("Scored %d cutouts.", len(scores))
        schema = lsst.afw.table.Schema()
        schema.addField(diaSources.schema["id"].asField())
        schema.addField("score", doc="real/bogus score of this source", type=float)
        classifications = lsst.afw.table.BaseCatalog(schema)
        classifications.resize(len(scores))

        classifications["id"] = diaSources["id"]
        classifications["score"] = scores

        return lsst.pipe.base.Struct(classifications=classifications)

    def _make_cutouts(self, template, science, difference, source):
        """Return cutouts of each image centered at the source location.

        Parameters
        ----------
        template : `lsst.afw.image.ExposureF`
        science : `lsst.afw.image.ExposureF`
        difference : `lsst.afw.image.ExposureF`
            Exposures to cut images out of.
        source : `lsst.afw.table.SourceRecord`
            Source to make cutouts of.

        Returns
        -------
        cutouts, `lsst.meas.transiNet.CutoutInputs`
            Cutouts of each of the input images.
        """

        # Try to create cutouts, or simply return empty cutouts if
        # failed (most probably out-of-border box)
        extent = lsst.geom.Extent2I(self.config.cutoutSize)
        box = lsst.geom.Box2I.makeCenteredBox(source.getCentroid(), extent)

        if science.getBBox().contains(box):
            science_cutout = np.nan_to_num(science.Factory(science, box).image.array)
            template_cutout = np.nan_to_num(template.Factory(template, box).image.array)
            difference_cutout = np.nan_to_num(difference.Factory(difference, box).image.array)
        else:
            science_cutout = np.zeros((self.config.cutoutSize, self.config.cutoutSize), dtype=np.float32)
            template_cutout = np.zeros_like(science_cutout)
            difference_cutout = np.zeros_like(science_cutout)

        return rbTransiNetInterface.CutoutInputs(science=science_cutout,
                                                 template=template_cutout,
                                                 difference=difference_cutout)
