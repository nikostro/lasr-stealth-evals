from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """Abstract base class for all environments in the LASR stealth evaluations framework.

    This class defines the minimal interface that all environment implementations must follow.
    Currently only requires the tick() method which is used by the Manager to advance the environment.
    """

    @abstractmethod
    def tick(self) -> None:
        """Advance the environment by one time step.

        This method should update the environment's state according to its dynamics.
        """
        pass
