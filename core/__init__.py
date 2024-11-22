
import os
import logging
import importlib


class XManager:
    """
    """
    ModuleDict = {}  # module class dict
    ObjectContainer = dict()  # module objects dict
    RootParameter = 'checkpoints'  # parameters root
    CommonDevice = 'cuda:0'  # device (only support cuda and cpu)

    """
    import module
    """
    @staticmethod
    def getModuleFromClassDict():
        from .module import GlobalModuleDictClass
        XManager.ModuleDict = GlobalModuleDictClass
        assert len(XManager.ModuleDict) > 0

    @staticmethod
    def getModuleFromSource(module_name):
        def getModuleRootRelative():
            path_workdir = os.path.realpath(os.getcwd())
            path_file = os.path.split(os.path.realpath(__file__))[0]
            if path_workdir in path_file:
                path_name_list = path_file[len(path_workdir):].split(os.path.sep)
                path_father = '.'.join([one for one in path_name_list if len(one)])
                return path_father
            return ''

        from .module import GlobalModuleDictSource
        path_module_relative = getModuleRootRelative()
        source_module, class_name = GlobalModuleDictSource[module_name]
        path_module = '{}.{}'.format(path_module_relative, source_module)
        class_type = importlib.import_module(path_module).__getattribute__(class_name)
        if module_name in XManager.ModuleDict:
            logging.warning('try to import module {} again'.format(module_name))
        XManager.ModuleDict[module_name] = class_type
        logging.warning('import module {} from source'.format(module_name))

    """
    """
    @staticmethod
    def createEngine(config: dict):
        from .engine import createEngine
        return createEngine(config)

    """
    """
    @staticmethod
    def printModuleInfo():
        info = 'current module list:\n\t'
        info_list = []
        for key, value in XManager.ModuleDict.items():
            if isinstance(value, tuple) is False:
                if hasattr(value, 'getResources'):
                    info_list.append('{}\n\t\t{}'.format(key, '\n\t\t'.join(value.getResources())))
                else:
                    info_list.append('{}\n\t\tnot implement or no resources'.format(key))
        info += '\n\t'.join(info_list)
        logging.warning(info)

    @staticmethod
    def printResources():
        for name in XManager.ModuleDict:
            XManager.ObjectContainer[name] = XManager.createModule(name, dict())

    @staticmethod
    def checkImportAuto():
        for name in XManager.ModuleDict:
            XManager.ObjectContainer[name] = XManager.createModule(name, dict())
        XManager.releaseModules()
        logging.warning('check import finish...')

    @staticmethod
    def setRootParameters(root:str):
        path = os.path.realpath(root)
        assert os.path.exists(path), path
        XManager.RootParameter = path
        logging.warning('set parameter root {}...'.format(path))

    @staticmethod
    def setCommonDevice(device:str):
        assert 'cuda' in device or 'cpu' in device, device
        XManager.CommonDevice = device
        logging.warning('set common device {}...'.format(device))

    @staticmethod
    def createModule(name, options:dict, initialize:bool=True):
        if name not in XManager.ModuleDict:
            XManager.getModuleFromSource(name)
        if name not in XManager.ObjectContainer:
            # just create the object
            XManager.ObjectContainer[name] = XManager.ModuleDict[name](dict())
        if initialize is True:
            # initialize/re-initialize the engine
            XManager.ObjectContainer[name].initialize(**options)
        return XManager.ObjectContainer[name]

    @staticmethod
    def releaseModules(name:str=''):
        name_list = [name] if len(name) > 0 \
            else list(XManager.ObjectContainer.keys())
        for name in name_list:
            module = XManager.ObjectContainer.pop(name)
            del module
            logging.warning('release module {}'.format(name))

    @staticmethod
    def emptyTorchCache(verbose=True):
        from .utils.context import XContext
        @XContext.fromString(verbose, verbose, verbose)
        def _emptyTorchCache():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError as e:
                logging.error(str(e))

        _emptyTorchCache()

    @staticmethod
    def getModules(modules):
        """
        1.common method:
            names = ['name1', 'name2']
            module1, module2 = XManager.getModules(names)
            module3 = XManager.getModules('name3')
        2.specify method:
            TODO: option dict is only used for initialization
            modules_with_options = [
                ('name1', {'root': 'path_to_root', 'device':'cpu'}),
                ('name2', {'parameters': 'path_to_param', 'device':'cuda:0'}),
            ]
            module1, module2 = XManager.getModules(modules_with_options)
        """
        def unpack(module):
            options = dict(root=XManager.RootParameter, device=XManager.CommonDevice)
            if isinstance(module, str):
                return module, options
            if isinstance(module, (tuple, list)):
                assert len(module) == 2, module
                if isinstance(module[0], str) and isinstance(module[1], dict):
                    options.update(module[1])
                    return module[0], options
            raise Exception(str(module))

        object_list = []
        module_list = [modules] if isinstance(modules, str) else modules
        for module in module_list:
            object_list.append(XManager.createModule(*unpack(module)))
        return object_list[0] if isinstance(modules, str) else object_list


if len(XManager.ModuleDict) == 0:
    XManager.getModuleFromClassDict()
    # XManager.printModuleInfo()
