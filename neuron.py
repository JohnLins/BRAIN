from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence, Tuple


Index3D = Tuple[int, int, int]
QueuedInput = Tuple[float, int]


@dataclass
class Neuron:
    index: Index3D
    weights: Sequence[Sequence[Sequence[float]]]
    input_decay: float
    signal_delay_scale: int = 1
    fire_threshold: float = 1.0
    refractory_period: int = field(default_factory=lambda: random.randint(1, 10))
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

    def receive_input(self, amount: float, source_index: Index3D) -> int:
        """Queue an input signal based on its distance from this neuron."""
        time_until_arrival = max(
            0,
            round(self.distance_to(source_index) * self.signal_delay_scale),
        )
        self.input_queue.append((amount, time_until_arrival))
        return time_until_arrival

    def update(self) -> bool:
        """Advance this neuron by one tick and return True if it fires."""
        self._advance_input_queue()

        if self.refractory_remaining > 0:
            self.refractory_remaining -= 1
            self.current_input = max(0.0, self.current_input - self.input_decay)
            return False

        if self.current_input >= self.fire_threshold:
            self.fire()
            return True

        self.current_input = max(0.0, self.current_input - self.input_decay)
        return False

    def fire(self) -> None:
        """Reset this neuron after firing and start its refractory period."""
        self.current_input = 0.0
        self.refractory_remaining = self.refractory_period

    def outgoing_connections(self) -> Iterable[Tuple[Index3D, float]]:
        """Yield every output neuron index and its connection weight."""
        for i, plane in enumerate(self.weights):
            for j, row in enumerate(plane):
                for k, weight in enumerate(row):
                    yield (i, j, k), weight

    def _advance_input_queue(self) -> None:
        remaining_inputs: Deque[QueuedInput] = deque()

        while self.input_queue:
            amount, time_until_arrival = self.input_queue.popleft()

            if time_until_arrival <= 1:
                self.current_input += amount
            else:
                remaining_inputs.append((amount, time_until_arrival - 1))

        self.input_queue = remaining_inputs
