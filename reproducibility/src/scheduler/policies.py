"""Priority queue implementations for FIFO and EDF real-time request scheduling."""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple
from .request import Request, RequestState


class PriorityQueue:
    """Abstract base priority queue interface."""

    def push(self, request: Request) -> None:
        raise NotImplementedError

    def pop(self) -> Request:
        raise NotImplementedError

    def peek(self) -> Optional[Request]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        return len(self) == 0

    def __len__(self) -> int:
        raise NotImplementedError

    def list_requests(self) -> List[Request]:
        raise NotImplementedError


class FIFOQueue(PriorityQueue):
    """First-In, First-Out queue ordering by arrival timestamp."""

    def __init__(self):
        # Entry format: (arrival_time_ms, insertion_counter, request_id, request)
        self._heap: List[Tuple[float, int, str, Request]] = []
        self._counter: int = 0

    def push(self, request: Request) -> None:
        request.state = RequestState.QUEUED
        heapq.heappush(self._heap, (request.arrival_time_ms, self._counter, request.request_id, request))
        self._counter += 1

    def pop(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty FIFOQueue")
        _, _, _, req = heapq.heappop(self._heap)
        return req

    def peek(self) -> Optional[Request]:
        if not self._heap:
            return None
        return self._heap[0][3]

    def list_requests(self) -> List[Request]:
        """Return requests in FIFO priority order without modifying queue."""
        sorted_entries = sorted(self._heap, key=lambda x: (x[0], x[1]))
        return [entry[3] for entry in sorted_entries]

    def __len__(self) -> int:
        return len(self._heap)


class EDFQueue(PriorityQueue):
    """Earliest Deadline First queue ordering by absolute deadline with deterministic tie-breaking."""

    def __init__(self):
        # Entry format: (absolute_deadline_ms, arrival_time_ms, insertion_counter, request_id, request)
        self._heap: List[Tuple[float, float, int, str, Request]] = []
        self._counter: int = 0

    def push(self, request: Request) -> None:
        request.state = RequestState.QUEUED
        heapq.heappush(
            self._heap,
            (request.absolute_deadline_ms, request.arrival_time_ms, self._counter, request.request_id, request),
        )
        self._counter += 1

    def pop(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty EDFQueue")
        _, _, _, _, req = heapq.heappop(self._heap)
        return req

    def peek(self) -> Optional[Request]:
        if not self._heap:
            return None
        return self._heap[0][4]

    def list_requests(self) -> List[Request]:
        """Return requests in EDF priority order without modifying queue."""
        sorted_entries = sorted(self._heap, key=lambda x: (x[0], x[1], x[2]))
        return [entry[4] for entry in sorted_entries]

    def __len__(self) -> int:
        return len(self._heap)
