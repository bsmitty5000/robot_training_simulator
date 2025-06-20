# sim/core.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

class LineSegment(NamedTuple):
    start: Vector2
    end: Vector2

    def length(self) -> float:
        return self.start.distance_to(self.end)

    def direction(self) -> Vector2:
        return (self.end - self.start).normalized()

    def angle(self) -> float:
        return self.direction().angle()
    
class Shape(ABC):
    @abstractmethod
    def contains(self, point: Vector2) -> bool:
        """Check if the shape contains a point."""
        pass

    @abstractmethod
    def intersects(self, other: 'Shape') -> bool:
        """Check if this shape intersects with another shape."""
        pass

    @abstractmethod
    def clipline(self, other: LineSegment) -> Optional[LineSegment]:
        """Returns the coordinates of a line that is cropped to be 
        completely inside the Shape, otherwise None.
        """
        pass

    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape."""
        pass

    @abstractmethod
    def min_distance_to(self, point: Vector2) -> float:
        """ Calculates the distance from a point to the closest
            location on Shape
        """
        pass

@dataclass
class Rect(Shape):
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
    
    def contains(self, point: Vector2) -> bool:
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom
    
    def intersects(self, other: 'Shape') -> bool:
        if isinstance(other, Rect):
            return not (self.right < other.left or self.left > other.right or
                        self.bottom < other.top or self.top > other.bottom)
        elif isinstance(other, Circle):
            return circle_rect_collision(other, self)
        else:
            raise NotImplementedError("Intersection not implemented for this shape type.")

    def bounding_box(self) -> 'Rect':
        return self  # A rect's bounding box is itself
    
    def clipline(self, line: LineSegment) -> Optional[LineSegment]:
        # Liang–Barsky algorithm from https://en.wikipedia.org/wiki/Liang%E2%80%93Barsky_algorithm
        # Returns the coordinates of a line that is cropped to be completely inside the rectangle, 
        # otherwise None.

        dx = line.end.x - line.start.x
        dy = line.end.y - line.start.y

        t_enter = 0.0
        t_exit = 1.0

        p = [-dx, dx, -dy, dy]
        q = [line.start.x - self.left, self.right - line.start.x, line.start.y - self.top, self.bottom - line.start.y]

        for i in range(4):
            if p[i] == 0:  # Line parallel to boundary
                if q[i] < 0:  # Line is outside the boundary
                    return None
            else:
                t = q[i] / p[i]
                if p[i] < 0:  # Entering intersection
                    t_enter = max(t_enter, t)
                else:  # Exiting intersection
                    t_exit = min(t_exit, t)

        if t_enter > t_exit:  # Line is completely outside
            return None

        return LineSegment(
            start=Vector2(
                line.start.x + t_enter * dx,
                line.start.y + t_enter * dy
            ),
            end=Vector2(
                line.start.x + t_exit * dx,
                line.start.y + t_exit * dy
            )
        )

    def area(self):
        return self.w * self.h
    
    def min_distance_to(self, point: Vector2) -> float:
        closestPoint = Vector2( max(self.left, min(point.x, self.right)),
                                max(self.top, min(point.y, self.bottom)))
        
        return closestPoint.distance_to(point)

@dataclass
class Circle(Shape):
    center: Vector2
    radius: float

    def contains(self, point: Vector2) -> bool:
        return self.center.distance_to(point) <= self.radius
    
    def intersects(self, other: 'Shape') -> bool:
        if isinstance(other, Rect):
            return circle_rect_collision(self, other)
        elif isinstance(other, Circle):
            return self.center.distance_to(other.center) <= (self.radius + other.radius)
        else:
            raise NotImplementedError("Intersection not implemented for this shape type.")

    def bounding_box(self) -> Rect:
        return Rect(
            x=self.center.x - self.radius,
            y=self.center.y - self.radius,
            w=2 * self.radius,
            h=2 * self.radius
        )
    
    def clipline(self, line: LineSegment) -> Optional[LineSegment]:
        # Vector from start to end
        d = line.end - line.start
        f = line.start - self.center

        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - self.radius ** 2

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            # No intersection
            return None

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Find the segment of the line inside the circle
        t_start = max(0, min(t1, t2))
        t_end = min(1, max(t1, t2))

        if t_start > t_end or t_end < 0 or t_start > 1:
            # The intersection points are outside the segment
            return None

        clipped_start = Vector2(
            line.start.x + d.x * t_start,
            line.start.y + d.y * t_start
        )
        clipped_end = Vector2(
            line.start.x + d.x * t_end,
            line.start.y + d.y * t_end
        )

        return LineSegment(clipped_start, clipped_end)

    def area(self) -> float:
        return math.pi * self.radius ** 2
    
    def min_distance_to(self, point: Vector2) -> float:
        
        return self.clipline(LineSegment(point, self.center))[0]

def circle_rect_collision(circle : Circle, rect : Rect) -> bool:
    
    # Get the closest point on the rect to the robot's center
    closest = Vector2(
        max(rect.left, min(circle.center.x, rect.right)),
        max(rect.top, min(circle.center.y, rect.bottom))
    )

    # Check if the distance is less than the radius
    return circle.center.distance_to(closest) <= circle.radius

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

def test_rect_clipline():
    rect = Rect(x=3, y=3, w=4, h=4)

    # 1. Line completely outside, no intersection
    line1 = LineSegment(Vector2(0, 0), Vector2(1, 1))
    print("Rect Test 1 (no intersection):", rect.clipline(line1))

    # 2. Line crosses rectangle diagonally
    line2 = LineSegment(Vector2(0, 0), Vector2(10, 10))
    print("Rect Test 2 (diagonal cross):", rect.clipline(line2))

    # 3. Line starts inside, ends outside
    line3 = LineSegment(Vector2(4, 4), Vector2(10, 10))
    print("Rect Test 3 (start inside):", rect.clipline(line3))

    # 4. Line completely inside rectangle
    line4 = LineSegment(Vector2(4, 4), Vector2(5, 5))
    print("Rect Test 4 (inside):", rect.clipline(line4))

    # 5. Line coincides with one edge of the rectangle
    line5 = LineSegment(Vector2(3, 3), Vector2(7, 3))
    print("Rect Test 5 (on edge):", rect.clipline(line5))

    # 6. Line tangent to rectangle
    line6 = LineSegment(Vector2(0, 3), Vector2(10, 3))
    print("Rect Test 6 (tangent):", rect.clipline(line6))


def test_circle_clipline():
    circle = Circle(center=Vector2(5, 5), radius=2)

    # 1. Line completely outside, no intersection
    line1 = LineSegment(Vector2(0, 0), Vector2(1, 1))
    print("Circle Test 1 (no intersection):", circle.clipline(line1))

    # 2. Line passes through circle (diameter)
    line2 = LineSegment(Vector2(3, 5), Vector2(7, 5))
    print("Circle Test 2 (diameter):", circle.clipline(line2))

    # 3. Line starts inside, ends outside
    line3 = LineSegment(Vector2(5, 5), Vector2(10, 10))
    print("Circle Test 3 (start inside):", circle.clipline(line3))

    # 4. Line completely inside circle
    line4 = LineSegment(Vector2(5, 5), Vector2(6, 5))
    print("Circle Test 4 (inside):", circle.clipline(line4))

    # 5. Line tangent to circle
    line5 = LineSegment(Vector2(3, 7), Vector2(7, 7))
    print("Circle Test 5 (tangent):", circle.clipline(line5))

    # 6. Line passes through but segment is outside
    line6 = LineSegment(Vector2(8, 8), Vector2(10, 10))
    print("Circle Test 6 (outside, passes through):", circle.clipline(line6))

if __name__ == "__main__":
    
    test_rect_clipline()
    test_circle_clipline()
    test_circle_rect_collision()
