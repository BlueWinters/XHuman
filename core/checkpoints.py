
import logging
import os
from . import XManager


def checkModelFiles(root=None):
    root = XManager.RootParameter if root is None else root
    info = 'current module list:\n\t'
    info_list = []
    for key, value in XManager.ModuleDict.items():
        if isinstance(value, tuple) is False:
            assert hasattr(value, 'getResources'), key
            resource_list = value.getResources()
            format_line = '{}\n\t\t{}'.format(key, '\n\t\t'.join(resource_list if len(resource_list) else ['None']))
            info_list.append(format_line)
            for resource in resource_list:
                full_path = '{}/{}'.format(root, resource)
                if os.path.exists(full_path) is False:
                    logging.error('missing checkpoints ({}, {})'.format(value.__name__, resource))
    info += '\n\t'.join(info_list)
    logging.warning(info)
