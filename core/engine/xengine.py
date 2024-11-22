


class XEngine:
    def __init__(self, config:dict):
        self.config = config
        self.type = None
        self.option = lambda name, default: self.config[name] \
            if name in self.config else default

    @staticmethod
    def create(*args, **kwargs):
        ...

    def initialize(self, *args, **kwargs):
        ...

    def inference(self, *args, **kwargs):
        ...
