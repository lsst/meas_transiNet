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

__all__ = ["import_model"]

import importlib
import os.path
import sys

import torch.nn

import lsst.utils


def import_model(name):
    """Import a model from the models directory and return the class object.

    Parameters
    ----------
    name : `str`
        Name of the model file to be loaded.

    Returns
    -------
    model_class : `type`
        Model class to instantiate; a subclass of `torch.nn.Module`.

    Raises
    ------
    ImportError
        Raised if a valid pytorch model cannot be found in loaded module.
    """
    model_dir = os.path.join(lsst.utils.getPackageDir("meas_transiNet"), "models")
    model_path = os.path.join(model_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    if len(module.__all__) != 1:
        raise ImportError(f"Multiple entries in {module}: cannot find model class.")

    model = getattr(module, module.__all__[0])
    if torch.nn.Module not in model.__bases__:
        raise ImportError(f"Loaded class {model}, from {module}, is not a pytorch neural network module.")

    return model
