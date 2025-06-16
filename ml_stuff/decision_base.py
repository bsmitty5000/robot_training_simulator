from abc import ABC, abstractmethod
from typing import Sequence, List

class DecisionBase(ABC):

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Number of outputs produced by this decision maker from decide."""
        pass

    @abstractmethod
    def decide(self, sensor_distances: Sequence[float]) -> List[float]:
        """
        Given a sequence of sensor distances, 
        return a list of control outputs (e.g., [left_pwm, right_pwm]).
        """
        pass