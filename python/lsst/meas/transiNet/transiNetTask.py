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

import numpy as np

import lsst.pex.config
import lsst.pipe.base


class TransiNetConnections(lsst.pipe.base.PipelineTaskConnections,
                           dimensions=("instrument", "visit", "detector"),
                           defaultTemplates={"coaddName": "deep", "fakesType": ""}):
    # NOTE: Do you want the "ready to difference" template, or something
    # earlier? this is warped, but not PSF-matched.
    template = lsst.pipe.base.connectionTypes.Input(
        doc="Input warped template to subtract.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_templateExp"
    )
    # NOTE: This is warped and PSF-matched.
    # matchedTemplate = lsst.pipe.base.connectionTypes.Input(
    #     doc="Warped and PSF-matched template used to create difference image.",
    #     dimensions=("instrument", "visit", "detector"),
    #     storageClass="ExposureF",
    #     name="{fakesType}{coaddName}Diff_matchedExp",
    # )
    science = lsst.pipe.base.connectionTypes.Input(
        doc="Input science exposure to subtract from.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}calexp"
    )
    # sources = lsst.pipe.base.connectionTypes.Input(
    #     doc="Sources measured on the science exposure; "
    #         "used to select sources for making the matching kernel.",
    #     dimensions=("instrument", "visit", "detector"),
    #     storageClass="SourceCatalog",
    #   7  name="{fakesType}src"
    # )
    difference = lsst.pipe.base.connectionTypes.Input(
        doc="Result of subtracting convolved template from science image.",
        dimensions=("instrument", "visit", "detector"),
        storageClass="ExposureF",
        name="{fakesType}{coaddName}Diff_differenceExp",
    )
    diaSources = lsst.pipe.base.connectionTypes.Input(
        doc="Detected sources on the difference image",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}Diff_diaSrc",
    )
    # This will need to come from an in-memory datastore, so that it only gets
    # loaded once; have to work with Middleware to test that.
    modelWeights = lsst.pipe.base.connectionTypes.PrerequisiteInput(
        doc="Trained model weights",
        dimensions=(),  # ????
        storageClass="PyTorchWeights",
        name="transiNetWeights",
    )

    # Outputs
    realBogusSources = lsst.pipe.base.connectionTypes.Output(
        doc="Catalog of real/bogus classifications for each diaSource",
        dimensions=("instrument", "visit", "detector"),
        storageClass="SourceCatalog",
        name="{fakesType}{coaddName}realBogusSources",
    )


class TransiNetConfig(lsst.pipe.base.PipelineTaskConfig, pipelineConnections=TransiNetConnections):
    modelFile = lsst.pex.config.ChoiceField(
        dtype=str,
        doc="TransiNet model to load.",
        allowed={
            "testModel": "A very basic model, mostly for testing",
            "ap_4_parameters": "A more complicated model"
        }
    )
    cutoutSize = lsst.pex.config.Field(
        dtype=int,
        doc="Width/height of square cutouts to send to classifier.",
        default=20
    )


class TransiNetTask(lsst.pipe.base.PipelineTask):
    """Task for running TransiNet real/bogus classification on difference
    images.
    """
    def run(self, template, science, difference, diaSources, weights):
        # The in-memory datastore should make this step fast.
        # TODO: does the in-memory representation depend on both the modelFile
        # and the weights data? If so, can we write *that* as the butler
        # dataset, so that the whole thing can be resident in-memory?
        model = transiNet.TransiNetModel(self.config.modelFile, weights)

        classifications = np.array.bool(len(diaSources))
        for i, source in enumerate(diaSources):
            cutouts = self._make_cutouts(template, science, difference, source)
            classifications[i] = model.infer(cutouts, source)

        return lsst.pipe.base.Struct(classifications=classifications)

    def _make_cutouts(self, template, science, difference, source):
        """Return cutouts of each image centered at the source location.
        """
        raise NotImplementedError("under construction")
