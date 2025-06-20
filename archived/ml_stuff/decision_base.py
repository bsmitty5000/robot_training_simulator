from abc import ABC, abstractmethod
from typing import Any, Sequence, List

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

    @staticmethod
    @abstractmethod
    def encode(*args, **kwargs) -> List[Any]:
        """
        Flatten the decision maker's parameters into a list (phenotype -> genotype).
        """
        pass
    
    @staticmethod
    @abstractmethod
    def from_genotype(genotype: List[Any], *args, **kwargs) -> Any:
        """
        Populate the decision maker's parameters from a genotype (flattened list)
        Uses layer_sizes to decode the genotype into weights and biases.
        """
        pass

    @abstractmethod
    def print_info(self) -> str:
        """
        Print the decision maker's internal structure or parameters.
        """
        pass