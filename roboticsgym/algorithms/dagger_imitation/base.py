

import abc

import stable_baselines3.common.logger as sb_logger


class BaseImitationAlgorithm(abc.ABC):
    """Base class for all imitation learning algorithms."""

    _logger: sb_logger
    """Object to log statistics and natural language messages to."""

    def __init__(
        self,
        *,
        custom_logger: sb_logger = None,
    ):
        """Creates an imitation learning algorithm.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self._logger = custom_logger

    @property
    def logger(self) -> sb_logger:
        return self._logger

    @logger.setter
    def logger(self, value: sb_logger) -> None:
        self._logger = value

    def __getstate__(self):
        state = self.__dict__.copy()
        # logger can't be pickled as it depends on open files
        del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # callee should modify self.logger directly if they want to override this
        self.logger = state.get("_logger")
