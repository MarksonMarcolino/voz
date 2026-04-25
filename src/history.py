"""Per-connection conversation history for multi-turn context."""

from src.config import MAX_HISTORY_TURNS


class ConversationHistory:
    """Stores message history and builds the messages list for the LLM."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self._messages: list[dict] = []
        self._max_turns = max_turns

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def get_messages(self, system_prompt: str) -> list[dict]:
        """Return [system, ...history] ready for the LLM."""
        return [{"role": "system", "content": system_prompt}] + list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        """Keep at most max_turns pairs (2 messages per turn)."""
        max_msgs = self._max_turns * 2
        if len(self._messages) > max_msgs:
            self._messages = self._messages[-max_msgs:]
