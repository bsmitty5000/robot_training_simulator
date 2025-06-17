# sim/core.py
import math
from typing import NamedTuple, Sequence, Optional

class Vector2(NamedTuple):
    x: float
    y: float

    def __add__(self, other): return Vector2(self.x+other.x, self.y+other.y)
    def __sub__(self, other): return Vector2(self.x-other.x, self.y-other.y)
    def scale(self, k):      return Vector2(self.x*k, self.y*k)
    def length(self):        return math.hypot(self.x, self.y)
    def normalized(self):
        l = self.length() or 1.0
        return Vector2(self.x/l, self.y/l)

def rotate(v: Vector2, angle_deg: float) -> Vector2:
    r = math.radians(angle_deg)
    return Vector2(v.x*math.cos(r) - v.y*math.sin(r),
                   v.x*math.sin(r) + v.y*math.cos(r))

class Rect(NamedTuple):
    x: float; y: float; w: float; h: float

def line_rect_intersection(p1: Vector2, p2: Vector2, rect: Rect) -> Optional[Vector2]:
    # implement Liang–Barsky or Cohen–Sutherland...
    ...
