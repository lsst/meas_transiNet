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

__all__ = ["TransiNetTask", "TransiNetConfig"]

import lsst.geom
import lsst.pex.config
import lsst.pipe.base
import numpy as np

from . import utils
from . import rbTransiNetInterface


class TransiNetConnections(lsst.pipe.base.PipelineTaskConnections,
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
        name="{fakesType}{coaddName}Diff_diaSrc",
    )

    # Outputs
    classifications = lsst.pipe.base.connectionTypes.Output(
        doc="Catalog of real/bogus classifications for each diaSource, "
            "element-wise aligned with diaSources.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Catalog",
        name="{fakesType}{coaddName}RealBogusSources",
    )


class TransiNetConfig(lsst.pipe.base.PipelineTaskConfig, pipelineConnections=TransiNetConnections):
    modelFile = lsst.pex.config.ChoiceField(
        dtype=str,
        doc="TransiNet model to load. This is the name of a python file in the models/ "
            "directory that contains a single class that is a subclass of `torch.nn.Module`.",
        allowed={
            "testModel": "A very basic model, for testing the task and interface.",
        }
    )
    weightsFile = lsst.pex.config.Field(
        dtype=str,
        doc="Absolute path to pytorch weights file to load at init.",
        deprecated="This config is a placeholder while we sort out how to store and load large weight files."
    )
    cutoutSize = lsst.pex.config.Field(
        dtype=int,
        doc="Width/height of square cutouts to send to classifier.",
        default=40
    )


class TransiNetTask(lsst.pipe.base.PipelineTask):
    """Task for running TransiNet real/bogus classification on the output of
    the image subtraction pipeline.
    """
    _DefaultName = "transiNet"
    ConfigClass = TransiNetConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model = utils.import_model(self.config.modelFile)
        self.interface = rbTransiNetInterface.RBTransiNetInterface(model(), self.config.weightsFile)

    def run(self, template, science, difference, diaSources):
        cutouts = [self._make_cutouts(template, science, difference, source) for source in diaSources]
        scores = self.interface.infer(cutouts)

        schema = lsst.afw.table.Schema()
        schema.addField(diaSources.schema["id"].asField())
        schema.addField("score", doc="real/bogus score of this source", type=float)
        classifications = lsst.afw.table.BaseCatalog(schema)
        classifications.resize(len(scores))

        classifications["id"] = diaSources["id"]
        classifications["score"] = scores.ravel()

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
            science_cutout = science.Factory(science, box).image.array
            template_cutout = template.Factory(template, box).image.array
            difference_cutout = difference.Factory(difference, box).image.array
        else:
            science_cutout = np.zeros((self.config.cutoutSize, self.config.cutoutSize), dtype=np.single)
            template_cutout = np.zeros_like(science_cutout)
            difference_cutout = np.zeros_like(science_cutout)

        return rbTransiNetInterface.CutoutInputs(science=science_cutout,
                                                 template=template_cutout,
                                                 difference=difference_cutout)
