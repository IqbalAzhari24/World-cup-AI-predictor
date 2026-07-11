"""In-process throttle for outbound API calls.

The app shares one API key across every visitor (see README), so many people
using the app at once could collectively exceed a provider's per-minute rate
limit even though no single user is doing anything wrong. This spaces calls
out across the whole process rather than just reacting to 429s after the fact.
"""
import threading
import time


class Throttle:
    """Blocks the caller until at least `min_interval` seconds have passed
    since the last call anywhere in this process (thread-safe)."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            remaining = self._last_call + self._min_interval - time.time()
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.time()
