# sim/core.py
import math
from typing import NamedTuple, Sequence, Optional

class Vector2(NamedTuple):
    x: float
    y: float

    def __add__(self, other): 
        return Vector2(self.x+other.x, self.y+other.y)
    
    def __sub__(self, other): 
        return Vector2(self.x-other.x, self.y-other.y)
    
    def scale(self, k):      
        return Vector2(self.x*k, self.y*k)
    
    def length(self):        
        return math.hypot(self.x, self.y)
    
    def normalized(self):
        l = self.length() or 1.0
        return Vector2(self.x/l, self.y/l)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def angle(self):
        """Returns the angle in degrees of the vector from the positive x-axis."""
        return math.degrees(math.atan2(self.y, self.x))
    
    def distance_to(self, other):
        """Returns the Euclidean distance to another vector."""
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float):
        """Scalar multiplication."""
        return Vector2(self.x * scalar, self.y * scalar)
    
    def rotate(self, angle_deg: float):
        """Rotate the vector by a given angle in degrees."""
        r = math.radians(angle_deg)
        return Vector2(
            self.x * math.cos(r) - self.y * math.sin(r),
            self.x * math.sin(r) + self.y * math.cos(r)
        )

    def __repr__(self):
        return f"Vector2({self.x:.2f}, {self.y:.2f})"

class Rect(NamedTuple):
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.w
    
    @property
    def top(self) -> float:
        return self.y
    
    @property
    def bottom(self) -> float:
        return self.y + self.h

class Circle(NamedTuple):
    center: Vector2
    radius: float

class LineSegment(NamedTuple):
    start: Vector2
    end: Vector2

    def length(self) -> float:
        return self.start.distance_to(self.end)

    def direction(self) -> Vector2:
        return (self.end - self.start).normalized()

    def angle(self) -> float:
        return self.direction().angle()

def circle_rect_collision(circle : Circle, rect : Rect) -> bool:
    
    # Get the closest point on the rect to the robot's center
    closest = Vector2(
        max(rect.left, min(circle.center.x, rect.right)),
        max(rect.top, min(circle.center.y, rect.bottom))
    )

    # Check if the distance is less than the radius
    return circle.center.distance_to(closest) <= circle.radius

def clipline(line: LineSegment, rect: Rect) -> Optional[Vector2]:
    # Liang–Barsky algorithm from https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
    # Returns the coordinates of a line that is cropped to be completely inside the rectangle, 
    # otherwise None.

    p1 = -(line.end.x - line.start.x)
    p2 = -p1
    p3 = -(line.end.y - line.start.y)
    p4 = -p3

    q1 = line.start.x - rect.left
    q2 = rect.right - line.start.x
    q3 = line.start.y - rect.top
    q4 = rect.bottom - line.start.y

    exitParams = []
    entryParams = []
    exitIndex = 1
    entryIndex = 1
    exitParams.append(1)
    entryParams.append(0)

    if ((p1 == 0 and q1 < 0) or (p2 == 0 and q2 < 0) or (p3 == 0 and q3 < 0) or (p4 == 0 and q4 < 0)):
        # Line is outside the clipping window
        return None
    
    if (p1 != 0):
        r1 = q1 / p1
        r2 = q2 / p2
        if (p1 < 0):
            entryParams.append(r1)
            exitParams.append(r2)
        else :
            entryParams.append(r2)
            exitParams.append(r1)
    
    if (p3 != 0):
        r3 = q3 / p3
        r4 = q4 / p4
        if (p3 < 0):
            entryParams.append(r3)
            exitParams.append(r4)
        else :
            entryParams.append(r4)
            exitParams.append(r3)

    u1 = max(entryParams)
    u2 = min(exitParams)

    if (u1 > u2):
        return None

    return LineSegment(
        start=Vector2(
            line.start.x + (line.end.x - line.start.x) * u1,
            line.start.y + (line.end.y - line.start.y) * u1
        ),
        end=Vector2(
            line.start.x + (line.end.x - line.start.x) * u2,
            line.start.y + (line.end.y - line.start.y) * u2
        )
    )

def test_circle_rect_collision():
    rect = Rect(x=3, y=3, w=4, h=4)

    # 1. Circle completely outside, no collision
    circle1 = Circle(center=Vector2(0, 0), radius=1)
    print("Test 1 (no collision):", circle_rect_collision(circle1, rect))

    # 2. Circle just touching left edge
    circle2 = Circle(center=Vector2(2, 5), radius=1)
    print("Test 2 (touch left edge):", circle_rect_collision(circle2, rect))

    # 3. Circle overlapping top edge
    circle3 = Circle(center=Vector2(5, 2), radius=2)
    print("Test 3 (overlap top edge):", circle_rect_collision(circle3, rect))

    # 4. Circle completely inside rectangle
    circle4 = Circle(center=Vector2(5, 5), radius=1)
    print("Test 4 (inside):", circle_rect_collision(circle4, rect))

    # 5. Circle just touching bottom-right corner
    circle5 = Circle(center=Vector2(7, 7), radius=1)
    print("Test 5 (touch bottom-right corner):", circle_rect_collision(circle5, rect))

    # 6. Circle overlapping right edge
    circle6 = Circle(center=Vector2(8, 5), radius=2)
    print("Test 6 (overlap right edge):", circle_rect_collision(circle6, rect))

    # 7. Circle just outside, near top-left corner
    circle7 = Circle(center=Vector2(2, 2), radius=0.9)
    print("Test 7 (near top-left, no collision):", circle_rect_collision(circle7, rect))

def test_clipline():
    rect = Rect(x=3, y=3, w=4, h=4)

    # 1. Line completely outside, no intersection
    line1 = LineSegment(start=Vector2(0, 0), end=Vector2(1, 1))
    print("Test 1 (no intersection):", clipline(line1, rect))

    # 2. Line crosses rectangle diagonally
    line2 = LineSegment(start=Vector2(0, 0), end=Vector2(10, 10))
    print("Test 2 (diagonal cross):", clipline(line2, rect))

    # 3. Line starts inside, ends outside
    line3 = LineSegment(start=Vector2(4, 4), end=Vector2(10, 10))
    print("Test 3 (start inside):", clipline(line3, rect))

    # 4. Line starts outside, ends inside
    line4 = LineSegment(start=Vector2(0, 0), end=Vector2(4, 4))
    print("Test 4 (end inside):", clipline(line4, rect))

    # 5. Line completely inside rectangle
    line5 = LineSegment(start=Vector2(4, 4), end=Vector2(5, 5))
    print("Test 5 (inside):", clipline(line5, rect))

    # 6. Line coincides with one edge of the rectangle
    line6 = LineSegment(start=Vector2(3, 3), end=Vector2(7, 3))
    print("Test 6 (on edge):", clipline(line6, rect))

    # 7. Line touches rectangle at a corner
    line7 = LineSegment(start=Vector2(0, 0), end=Vector2(3, 3))
    print("Test 7 (touch corner):", clipline(line7, rect))

    # 8. Vertical line crossing rectangle
    line8 = LineSegment(start=Vector2(5, 0), end=Vector2(5, 10))
    print("Test 8 (vertical cross):", clipline(line8, rect))

    # 9. Horizontal line crossing rectangle
    line9 = LineSegment(start=Vector2(0, 5), end=Vector2(10, 5))
    print("Test 9 (horizontal cross):", clipline(line9, rect))

if __name__ == "__main__":
    test_clipline()
    
    test_circle_rect_collision()
