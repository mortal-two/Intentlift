class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class MiniSWERequestCancelled(RuntimeError):
    """Abort one in-flight agent request after another candidate has won."""

    def __init__(self, winner_branch_id: int) -> None:
        if not isinstance(winner_branch_id, int) or winner_branch_id < 1:
            raise ValueError("winner_branch_id must be positive")
        self.winner_branch_id = winner_branch_id
        super().__init__(
            f"candidate superseded by Gate1 winner branch {winner_branch_id}"
        )


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""
