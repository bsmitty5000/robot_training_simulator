from abc import ABC, abstractmethod

from models.obstacle import Obstacle

class GridCoverageCourse(ABC):
    def __init__(self, width: int, height: int, cell_size: int = 10):
        self.width = width
        self.height = height
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        self.cell_size = cell_size
        self.coverage = [[False] * self.grid_height for _ in range(self.grid_width)]
        # Subclasses should call make_course in their __init__ to set up obstacles

    @abstractmethod
    def make_course(
                    self,
                    thickness: int = 10,
                    min_distance_between_obstacles_m: int = 0.3) -> list[Obstacle]:
        """Subclasses must implement this to create and return a list of obstacles."""
        pass

    def mark_visited(self, x: float, y: float) -> None:
        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            self.coverage[gx][gy] = True

    def coverage_ratio(self) -> float:
        visited = sum(cell for row in self.coverage for cell in row)
        total = self.grid_width * self.grid_height
        return visited / total if total > 0 else 0.0