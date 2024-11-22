
import logging
import numpy as np
import os
import pycuda.driver as cuda
import tensorrt as trt
from .xengine import XEngine
from .xcontext import *





class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding):
        self.host = host_mem
        self.device = device_mem
        self.binding = binding

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()



class XEngineTensorRT(XEngine):
    """
    """
    # for implicit batch inference
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(self, config:dict):
        super(XEngineTensorRT, self).__init__(config)
        self.type = 'tensorrt'

    """
    """
    @staticmethod
    def create(*args, **kwargs):
        engine_config = kwargs['config']
        if 'option' in engine_config and engine_config['option'] is not None:
            if 'shape_optimize' in engine_config['option']:
                return LibEngineTensorRT_DynamicShape(config=engine_config)
            if 'dynamic_context' in engine_config['option']:
                return XEngineTensorRT_DynamicContext(config=engine_config)
        return XEngineTensorRT(config=engine_config)

    """
    """
    def _to_GiB(self, val):
        return val * 1 << 30

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def _do_inference(self, context, bindings, inputs, outputs, stream, batch_size:int=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return outputs

    """
    """
    @XContext.get_context()
    def initialize(self, *args, **kwargs):
        option = dict() if self.config['option'] is None else self.config['option']
        self.engine = self._build_engine(
            self.config['parameters'], **option)
        self.context = self.engine.create_execution_context()
        self.buffers = self.inputs, self.outputs, self.bindings = \
            self._allocate_buffers(self.engine, self.context)
        self.stream = cuda.Stream()

    def _build_engine(self, path_onnx:str, **kwargs):
        assert os.path.exists(path_onnx)
        serialize = False if 'serialize' not in kwargs \
            else bool(kwargs['serialize'])
        if serialize is True:
            engine = self._build_engine_trt(path_onnx, **kwargs)
        else:
            engine = self._build_engine_onnx(path_onnx, **kwargs)
        return engine

    def _build_engine_trt(self, path_onnx:str, **kwargs):
        builder = trt.Builder(XEngineTensorRT.TRT_LOGGER)
        path_trt = '{}.trt'.format(path_onnx[:-5])
        if os.path.exists(path_trt) is False:
            self.serialize_engine(path_onnx, path_trt, builder, **kwargs)
        return self.deserialize_engine(path_trt)

    def _build_engine_onnx(self, path_onnx, **kwargs):
        builder = trt.Builder(XEngineTensorRT.TRT_LOGGER)
        config = self._create_config(builder, **kwargs)
        network = self._parse_onnx_model(builder, path_onnx)
        return builder.build_engine(network, config)

    def _create_config(self, builder, **kwargs):
        config = builder.create_builder_config()
        workspace_size = 1 if 'max_workspace_size' not in kwargs \
            else kwargs['max_workspace_size']
        config.max_workspace_size = self._to_GiB(workspace_size)
        precision = None if 'precision' not in kwargs else kwargs['precision']
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            logging.info('set network precision to fp16')
            config.set_flag(trt.BuilderFlag.FP16)
        if precision == 'tf32' and builder.platform_has_tf32:
            logging.info('set network precision to tf32')
            config.clear_flag(trt.BuilderFlag.TF32)
        if precision == 'int8' and builder.platform_has_fast_int8:
            logging.info('set network precision to int8')
            # TODO: not finish, please ref to:
            # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8
        return config

    def _parse_onnx_model(self, builder, path_onnx):
        network = builder.create_network(XEngineTensorRT.EXPLICIT_BATCH)
        # parser for onnx
        parser = trt.OnnxParser(network, XEngineTensorRT.TRT_LOGGER)
        with open(path_onnx, 'rb') as model:
            if not parser.parse(model.read()):
                logging.critical('failed to parse the ONNX file: {}'.format(path_onnx))
                for error in range(parser.num_errors):
                    logging.error('\terror{}: {}'.format(error, parser.get_error(error)))
                return None
            logging.info('load model: ', path_onnx)
        return network

    def serialize_engine(self, path_onnx:str, path_trt:str, builder, **kwargs):
        config = self._create_config(builder, **kwargs)
        network = self._parse_onnx_model(builder, path_onnx)
        engine = builder.build_serialized_network(network, config)
        with open(path_trt, 'wb') as file:
            logging.info('serialize engine: {}'.format(path_trt))
            file.write(engine)

    def deserialize_engine(self, path_trt:str):
        trt.init_libnvinfer_plugins(XEngineTensorRT.TRT_LOGGER, '')
        runtime = trt.Runtime(XEngineTensorRT.TRT_LOGGER)
        with open(path_trt, 'rb') as file:
            logging.info('deserialize engine: {}'.format(path_trt))
            engine = runtime.deserialize_cuda_engine(file.read())
            return engine

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    def _allocate_buffers(self, engine, context):
        inputs = list()
        outputs = list()
        bindings = list()
        for binding in engine:
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            index = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(index)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem, binding))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem, binding))
        return inputs, outputs, bindings

    """
    """
    @XContext.get_context('maximum', 'maximum', 'maximum')
    def inference(self, inputs:list[np.ndarray], *args, **kwargs):
        self._assign(inputs, self.inputs)
        trt_outputs = self._do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        return [out.host for out in trt_outputs]

    def _assign(self, inputs:list[np.ndarray], hdm_list:list[HostDeviceMem]):
        assert len(inputs) == len(hdm_list)
        for array, hdm in zip(inputs, hdm_list):
            np.copyto(hdm.host, array.ravel())

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def _do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return outputs



class XEngineTensorRT_DynamicContext(XEngineTensorRT):
    def __init__(self, config:dict):
        super(XEngineTensorRT_DynamicContext, self).__init__(config)

    """
    """
    @XContext.get_context()
    def initialize(self, *args, **kwargs):
        self.engine = self._build_engine(
            self.config['parameters'], **self.config['option'])
        self.stream = cuda.Stream()
        self.context = None if bool(self.config['option']['dynamic_context']) \
            else self.engine.create_execution_context()

    """
    """
    @XContext.get_context('maximum', 'maximum', 'maximum')
    def inference(self, inputs:list[np.ndarray], *args, **kwargs):
        context = self._dynamic_context()
        buffers = self._allocate_buffers(self.engine, context)
        self._assign(inputs, buffers[0])
        trt_outputs = self._do_inference_v2(
            context, bindings=buffers[2], inputs=buffers[0], outputs=buffers[1], stream=self.stream)
        return [out.host for out in trt_outputs]

    def _dynamic_context(self):
        return self.context if self.context is not None else \
            self.engine.create_execution_context()



class LibEngineTensorRT_DynamicShape(XEngineTensorRT):
    def __init__(self, config:dict):
        super(LibEngineTensorRT_DynamicShape, self).__init__(config)

    def _create_config(self, builder, **kwargs):
        config = XEngineTensorRT._create_config(self, builder, **kwargs)
        profile = builder.create_optimization_profile()
        shape_optimize = kwargs['shape_optimize']
        assert isinstance(shape_optimize, dict)
        for binding, shape in shape_optimize.items():
            assert len(shape) == 3 # for: (min,opt,max)
            profile.set_shape(binding, *shape)
        config.add_optimization_profile(profile)
        return config

    """
    """
    @XContext.get_context()
    def initialize(self, *args, **kwargs):
        self.engine = self._build_engine(
            self.config['parameters'], **self.config['option'])
        self.stream = cuda.Stream()
        # dynamic context
        dynamic_context = None if 'dynamic_context' not in self.config['option'] \
            else self.config['option']['dynamic_context']
        self.context = None if bool(dynamic_context) \
            else self.engine.create_execution_context()

    """
    """
    @XContext.get_context('maximum', 'maximum', 'maximum')
    def inference(self, inputs:list[np.ndarray], *args, **kwargs):
        context = self._dynamic_context()
        for n, array in enumerate(inputs):
            context.set_binding_shape(n, array.shape)
        buffers = self._allocate_buffers(self.engine, context)
        # inference
        self._assign(inputs, buffers[0])
        trt_outputs = self._do_inference_v2(
            context, bindings=buffers[2], inputs=buffers[0], outputs=buffers[1], stream=self.stream)
        return [out.host for out in trt_outputs]

    def _dynamic_context(self):
        return self.context if self.context is not None else \
            self.engine.create_execution_context()