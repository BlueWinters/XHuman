
import logging
import numpy as np
from .utils import formatBackground, estimateForeground, estimateComposite



class LibHumanMatting_Wrapper:
    """
    """
    DefaultBackground = (0, 0, 255)  # red

    """
    """
    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        pass

    def inference(self, *args, **kwargs):
        raise NotImplementedError(__file__)

    def _extractArgs(self, *args, **kwargs):
        if len(args) > 0:
            logging.warning('{} useless parameters in {}'.format(
                len(args), self.__class__.__name__))
        targets = kwargs.pop('targets', 'source')
        background = kwargs.pop('background', np.array(self.DefaultBackground, dtype=np.uint8))
        use_cf = kwargs.pop('use_cf', False)
        alpha = kwargs.pop('alpha', None)
        return targets, alpha, dict(background=background, use_cf=use_cf)

    def _returnResult(self, output, targets, bgr, background, use_cf):
        def _formatResult(target):
            alpha = output
            if target == 'source':
                foreground = estimateForeground(bgr, alpha, use_cf)
                format_background = formatBackground(bgr, background)
                composite = estimateComposite(alpha, foreground, format_background)
                return foreground, alpha, composite
            if target == 'foreground':
                foreground = estimateForeground(bgr, alpha, use_cf)
                return foreground
            if target == 'alpha':
                return alpha
            if target == 'composite':
                foreground = estimateForeground(bgr, alpha, use_cf)
                format_background = formatBackground(bgr, background)
                composite = estimateComposite(alpha, foreground, format_background)
                return composite
            raise Exception('no such return type {}'.format(target))

        if isinstance(targets, str):
            return _formatResult(targets)
        if isinstance(targets, list):
            return [_formatResult(target) for target in targets]
        raise Exception('no such return targets {}'.format(targets))

    def __call__(self, bgr, *args, **kwargs):
        targets, alpha, format_kwargs = self._extractArgs(*args, **kwargs)
        output = self.inference(bgr) if alpha is None else alpha
        return self._returnResult(output, targets, bgr, **format_kwargs)