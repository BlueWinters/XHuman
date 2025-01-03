
import numpy as np


class Rectangle:
    """
    """
    @staticmethod
    def assertValid(rectangle):
        assert isinstance(rectangle, Rectangle)
        assert rectangle.x_min < rectangle.x_max, (rectangle.x_min, rectangle.x_max)
        assert rectangle.y_min < rectangle.y_max, (rectangle.y_min, rectangle.y_max)

    """
    """
    def __init__(self, points):
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.points = None
        # assign
        self.fromPoints(points)

    @property
    def left(self):
        return self.x_min

    @property
    def right(self):
        return self.x_max

    @property
    def top(self):
        return self.y_min

    @property
    def bottom(self):
        return self.y_max

    @property
    def height(self):
        return self.y_max - self.y_min + 1

    @property
    def width(self):
        return self.x_max - self.x_min + 1

    def fromPoints(self, points):
        assert isinstance(points, np.ndarray)
        if self.points is None:
            if len(points.shape) == 1:
                assert points.shape[0] == 4, points.shape
                self.points = np.copy(points)
                self.x_min = points[0]  # left
                self.y_min = points[1]  # top
                self.x_max = points[2]  # right
                self.y_max = points[3]  # bottom
                Rectangle.assertValid(self)
                return
            if len(points.shape) == 2:
                assert points.shape[1] == 2, points.shape
                self.points = np.copy(points)
                self.x_min = np.min(points[:, 0])
                self.x_max = np.max(points[:, 0])
                self.y_min = np.min(points[:, 1])
                self.y_max = np.max(points[:, 1])
                Rectangle.assertValid(self)
                return
            raise NotImplementedError(points.shape)

    def area(self) -> float:
        return (self.y_max - self.y_min + 1) * (self.x_max - self.x_min + 1)

    def expand(self, ratio_x: float, ratio_y: float):
        assert ratio_x > 0 and ratio_y > 0, (ratio_x, ratio_y)
        w = self.width
        h = self.height
        self.x_min = self.x_min - w * ratio_x
        self.x_max = self.x_max + w * ratio_x
        self.y_min = self.y_min - h * ratio_y
        self.y_max = self.y_max + h * ratio_y
        return self

    def clip(self, xx_min, yy_min, xx_max, yy_max):
        self.x_min = max(xx_min, self.x_min)
        self.y_min = max(yy_min, self.y_min)
        self.x_max = min(xx_max, self.x_max)
        self.y_max = min(yy_max, self.y_max)
        return self

    def decouple(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def asInt(self):
        return [int(round(v)) for v in self.decouple()]

    def center(self):
        return (self.x_min+self.x_max)/2, (self.y_min+self.y_max)/2

    def toSquare(self):
        h, w = self.height, self.width
        center_x, center_y = self.center()
        self.x_min = center_x - w / 2
        self.x_max = center_x + w / 2
        self.y_min = center_y - h / 2
        self.y_max = center_y + h / 2
        return self
