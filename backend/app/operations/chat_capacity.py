from threading import Lock

from app.operations.limits import MAX_CONCURRENT_CHATS


class ChatAdmissionGate:
    """A process-local admission gate for the first-release chat envelope."""

    def __init__(self) -> None:
        self._active = 0
        self._lock = Lock()

    def try_admit(self) -> bool:
        with self._lock:
            if self._active >= MAX_CONCURRENT_CHATS:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def reset(self) -> None:
        with self._lock:
            self._active = 0


_chat_admission_gate = ChatAdmissionGate()


def get_chat_admission_gate() -> ChatAdmissionGate:
    return _chat_admission_gate
