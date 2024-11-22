
import os
import platform
import logging


EngineClassDict = dict()
GlobalEngine = os.environ.get('ENGINE', None)


def import_engine_torch():
    from .xengine_th import XEngineTorch
    EngineClassDict['torch'] = XEngineTorch
    logging.info('import engine torch')

def import_engine_tonsorrt():
    from .xengine_trt import XEngineTensorRT
    EngineClassDict['tensorrt'] = XEngineTensorRT
    logging.info('import engine tensorrt')


if GlobalEngine is None:
    if platform.system().lower() == 'linux':
        import_engine_torch()
    if platform.system().lower() == 'windows':
        import_engine_torch()
else:
    if GlobalEngine.lower() == 'torch':
        import_engine_torch()
    if GlobalEngine.lower() == 'tensorrt':
        import_engine_tonsorrt()


def createEngine(engine_config:dict):
    if GlobalEngine is None:
        EngineClass = EngineClassDict[engine_config['type']]
        return EngineClass.create(config=engine_config)
    else:
        engine_config['type'] = str(GlobalEngine).lower()
        EngineClass = EngineClassDict[engine_config['type']]
        return EngineClass.create(config=engine_config)