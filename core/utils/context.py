
import logging
import os
import time


class XContextTemplate:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.value_max = 0
        self.value_cur = 0

    def __enter__(self):
        self.value = self.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        diff = self.record() - self.value
        logging.warning(self.format(diff, 'none'))

    def record(self):
        return 0

    def format(self, value, type):
        return ''

    def measure(self, beg, end):
        self.value_cur = end - beg
        self.value_max = max(self.value_cur, self.value_max)
        return self.value_cur



class XContextTimer(XContextTemplate):
    def __init__(self, verbose):
        super(XContextTimer, self).__init__(verbose)

    def record(self):
        return time.time()

    def format(self, eclipse, type):
        header = 'time running({}):'.format(type).ljust(30)
        return '{} {:.4f} ms'.format(header, eclipse * 1000)



class XContextRAMMemory(XContextTemplate):
    def __init__(self, verbose):
        super(XContextRAMMemory, self).__init__(verbose)

    def record(self):
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss

    def format(self, num_bytes, type):
        header = 'RAM memory changes({}):'.format(type).ljust(30)
        return '{} {:.4f} GB'.format(header, num_bytes / 1024 / 1024 / 1024)



class XContextGPUMemory(XContextTemplate):
    def __init__(self, verbose):
        super(XContextGPUMemory, self).__init__(verbose)

    def record(self):
        import pycuda.autoinit
        import pycuda.driver as driver
        free, total = driver.mem_get_info()
        return total - free

    def format(self, num_bytes, type):
        header = 'GPU memory changes({}):'.format(type).ljust(30)
        return '{} {:.4f} GB'.format(header, num_bytes / 1024 / 1024 / 1024)




class XContext:
    @staticmethod
    def printCodeInformation(function):
        code = function.__code__
        line = '{}::line_{}::{}'.format(
            code.co_filename, code.co_firstlineno, code.co_name)
        return line

    @staticmethod
    def measureContext(function, context, *args, **kwargs):
        pre = list(map(lambda ctx: ctx.record(), context))
        output = function(*args, **kwargs)
        aft = list(map(lambda ctx: ctx.record(), context))
        logging.warning(XContext.printCodeInformation(function))
        for ctx, p, a in zip(context, pre, aft):
            line = ctx.format(ctx.measure(p, a), 'current')
            if ctx.verbose is True:
                logging.warning('\t{}'.format(line))
        return output

    @staticmethod
    def fromString(timer=True, ram=True, gpu=True):
        return XContext(XContextTimer(timer), XContextRAMMemory(ram), XContextGPUMemory(gpu))

    """
    """
    def __init__(self, *contexts):
        self.contexts = contexts

    def __del__(self):
        for ctx in self.contexts:
            line = ctx.format(ctx.value_max, 'maximum')
            logging.warning('{}'.format(line))

    def __call__(self, function):
        self.function = function
        def call_wrapper(*args, **kwargs):
            return XContext.measureContext(self.function, self.contexts, *args, **kwargs)
        return call_wrapper