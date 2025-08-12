
import os
import logging


EngineClassDict = dict()
EnableTensorRT = str(os.environ.get('ENABLE_TENSORRT', 'false').lower())


def import_engine_torch():
    from .xengine_th import XEngineTorch
    EngineClassDict['torch'] = XEngineTorch
    logging.info('import engine torch')


def import_engine_tonsorrt():
    from .xengine_trt import XEngineTensorRT
    EngineClassDict['tensorrt'] = XEngineTensorRT
    logging.info('import engine tensorrt')


# Torch is necessary
import_engine_torch()
# TensorRT is optional
if EnableTensorRT == 'true':
    import_engine_tonsorrt()


def createEngine(engine_config:dict):
    EngineClass = EngineClassDict[engine_config['type']]
    return EngineClass.create(config=engine_config)
