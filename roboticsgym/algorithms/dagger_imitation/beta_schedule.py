
import abc

class BetaSchedule(abc.ABC):
    """Computes beta (% of time demonstration action used) from training round."""

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Computes the value of beta for the current round.

        Args:
            round_num: the current round number. Rounds are assumed to be sequentially
                numbered from 0.

        Returns:
            The fraction of the time to sample a demonstrator action. Robot
                actions will be sampled the remainder of the time.
        """  # noqa: DAR202


class LinearBetaSchedule(BetaSchedule):
    """Linearly-decreasing schedule for beta."""

    def __init__(self, rampdown_rounds: int) -> None:
        """Builds LinearBetaSchedule.

        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta linearly decreasing from `1` to `0` between round `0` and
            `self.rampdown_rounds`. After that, it is 0.
        """
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


class ExponentialBetaSchedule(BetaSchedule):
    """Exponentially decaying schedule for beta."""

    def __init__(self, decay_probability: float):
        """Builds ExponentialBetaSchedule.

        Args:
            decay_probability: the decay factor for beta.

        Raises:
            ValueError: if `decay_probability` not within (0, 1].
        """
        if not (0 < decay_probability <= 1):
            raise ValueError("decay_probability lies outside the range (0, 1].")
        self.decay_probability = decay_probability

    def __call__(self, round_num: int) -> float:
        """Computes beta value.

        Args:
            round_num: the current round number.

        Returns:
            beta as `self.decay_probability ^ round_num`
        """
        assert round_num >= 0
        return self.decay_probability**round_num

