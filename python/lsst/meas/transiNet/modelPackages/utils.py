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
import importlib.abc
import importlib.machinery
import importlib.util
import torch.nn


def load_module_from_memory(file_like_object, name='model'):
    """Load a module from the specified file-like object.

    Parameters
    ----------
    file_like_object : `file-like object`
        The file-like object containing the module code.
    name : `str`, optional
        Name to give to the module. Default: 'model'.

    Returns
    -------
    module : `module`
        The module object.
    """

    class InMemoryLoader(importlib.abc.SourceLoader):
        def __init__(self, data):
            self.data = data

        def get_data(self, path):
            # In this context, 'path' is not used as data is already in memory
            return self.data

        def get_filename(self, fullname):
            # This method is required but the filename is not important here
            return '<in-memory>'

    content = file_like_object.getvalue()
    loader = InMemoryLoader(content)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    return module


def load_module_from_file(path, name='model'):
    """Load a module from the specified path and return the module object.

    Parameters
    ----------
    path : str
        Path to the module file.
    name : str, optional
        Name to give to the module. Default: 'model'.

    Returns
    -------
    module : module
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_model_from_module(module):
    """Import a pytorch neural network architecture from the specified module.

    Parameters
    ----------
    module : module
        The module containing the neural network architecture.

    Returns
    -------
    model : `torch.nn.Module`
        The model class object.
    """
    if len(module.__all__) != 1:
        raise ImportError(f"Multiple entries in {module}: cannot find model class.")

    model = getattr(module, module.__all__[0])
    if torch.nn.Module not in model.__bases__:
        raise ImportError(f"Loaded class {model}, from {module}, is not a pytorch neural network module.")

    return model()


def import_model(path):
    """Import a pytorch neural network architecture from the specified path.

    Parameters
    ----------
    path : `str`
        Path to the model file.

    Returns
    -------
    model : `torch.nn.Module`
        The model class object.

    Raises
    ------
    ImportError
        Raised if a valid pytorch model cannot be found in loaded module.
    """

    module = load_module_from_file(path)
    return import_model_from_module(module)
