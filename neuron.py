from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Tuple


Index3D = Tuple[int, int, int]
QueuedInput = Tuple[float, int]


@dataclass
class Connection:
    target_index: Index3D
    weight: float
    delay_ticks: int


@dataclass
class Neuron:
    index: Index3D
    connections: list[Connection]
    input_decay: float
    signal_delay_scale: int = 1
    fire_threshold: float = 1.0
    refractory_period: int = 1
    current_input: float = 0.0
    refractory_remaining: int = 0
    input_queue: Deque[QueuedInput] = field(default_factory=deque)

    def distance_to(self, other_index: Index3D) -> float:
        """Return the Euclidean distance to another neuron index."""
        return math.sqrt(
            (self.index[0] - other_index[0]) ** 2
            + (self.index[1] - other_index[1]) ** 2
            + (self.index[2] - other_index[2]) ** 2
        )

    def receive_input(self, amount: float, delay_ticks: int) -> None:
        """Queue an input signal with a precomputed connection delay."""
        self.input_queue.append((amount, delay_ticks))

    def update(self) -> bool:
        """Advance this neuron by one tick and return True if it fires."""
        if self.refractory_remaining > 0:
            self._advance_input_queue(accept_arrivals=False)
            self.refractory_remaining -= 1
            self.current_input = max(0.0, self.current_input - self.input_decay)
            return False

        self._advance_input_queue()

        if self.current_input >= self.fire_threshold:
            self.fire()
            return True

        self.current_input = max(0.0, self.current_input - self.input_decay)
        return False

    def fire(self) -> None:
        """Reset this neuron after firing and start its refractory period."""
        self.current_input = 0.0
        self.refractory_remaining = self.refractory_period

    def outgoing_connections(self) -> Iterable[Connection]:
        """Yield this neuron's sparse outgoing connections."""
        yield from self.connections

    def connection_to(self, target_index: Index3D) -> Connection | None:
        for connection in self.connections:
            if connection.target_index == target_index:
                return connection

        return None

    def _advance_input_queue(self, accept_arrivals: bool = True) -> None:
        remaining_inputs: Deque[QueuedInput] = deque()

        while self.input_queue:
            amount, time_until_arrival = self.input_queue.popleft()

            if time_until_arrival <= 1:
                if accept_arrivals:
                    self.current_input += amount
            else:
                remaining_inputs.append((amount, time_until_arrival - 1))

        self.input_queue = remaining_inputs
