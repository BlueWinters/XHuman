
import logging
import numpy as np
import os
try:
    import tensorrt as trt
    from cuda import cuda, cudart
    logging.warning('tensorrt version: {}'.format(trt.__version__))
except ImportError as e:
    logging.error('tensorrt or cuda-python is not installed. please install tensorrt or cuda-python to use this module.')
    raise e
from .xengine import XEngine
from .. import XManager


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
        self.root = None
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

    """
    """
    @staticmethod
    def create(*args, **kwargs):
        engine_config = kwargs['config']
        return XEngineTensorRT(config=engine_config)

    """
    """
    @staticmethod
    def _to_GiB(val):
        return val * 1 << 30

    """
    """
    def initialize(self, *args, **kwargs):
        if self.root is None:
            self.root = kwargs['root'] if 'root' in kwargs else XManager.RootParameter
            path = '{}/{}'.format(self.root, self.config['parameters'])
            self.deserializeTRTEngine(path)
            self.allocateBuffers()

    def deserializeTRTEngine(self, path_trt):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(XEngineTensorRT.TRT_LOGGER, namespace='')
        # runtime = trt.Runtime(XEngineTensorRT.TRT_LOGGER)
        with open(path_trt, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
            logging.info('deserialize engine: {}'.format(path_trt))
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

    @staticmethod
    def cuda_call(call):
        err, res = call[0], call[1:]
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))
        if len(res) == 1:
            res = res[0]
        return res

    def allocateBuffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                s = s if s > 0 else 100
                size *= s
            # logging.info(name, shape, dtype, size)
            allocation = self.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "size": size,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    """
    """
    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]
    
    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            shape = [v if v > 0 else 100 for v in o["shape"]]
            specs.append((shape, o["dtype"]))
        return specs

    @staticmethod
    def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
        nbytes = host_arr.size * host_arr.itemsize
        XEngineTensorRT.cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

    @staticmethod
    def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
        # Wrapper for cudaMemcpy which infers copy size and does error checking
        nbytes = host_arr.size * host_arr.itemsize
        XEngineTensorRT.cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    def inference(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """

        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        self.memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(batch))

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            self.memcpy_device_to_host(outputs[o], self.outputs[o]["allocation"])
        return outputs

