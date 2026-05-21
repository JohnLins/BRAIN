from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from neuron import Index3D, Neuron


GRID_SIZE = 4
FRAME_INTERVAL_MS = 40
INPUT_DECAY = 0.01
SIGNAL_DELAY_SCALE = 8
STARTING_STIMULUS = 1.25
SHOW_INPUT_LABELS = True
SHOW_SIGNAL_DOTS = True
SIGNAL_DOT_SIZE = 1


@dataclass
class ActiveSignal:
    source_index: Index3D
    target_index: Index3D
    amount: float
    total_ticks: int
    remaining_ticks: int

    def position(self) -> tuple[float, float, float]:
        if self.total_ticks <= 0:
            return self.target_index

        progress = 1.0 - (self.remaining_ticks / self.total_ticks)
        progress = min(1.0, max(0.0, progress))

        return (
            self.source_index[0]
            + (self.target_index[0] - self.source_index[0]) * progress,
            self.source_index[1]
            + (self.target_index[1] - self.source_index[1]) * progress,
            self.source_index[2]
            + (self.target_index[2] - self.source_index[2]) * progress,
        )


def create_weights() -> list[list[list[float]]]:
    return [
        [
            [random.random() for _ in range(GRID_SIZE)]
            for _ in range(GRID_SIZE)
        ]
        for _ in range(GRID_SIZE)
    ]


def create_neurons() -> Dict[Index3D, Neuron]:
    neurons: Dict[Index3D, Neuron] = {}

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                index = (i, j, k)
                neurons[index] = Neuron(
                    index=index,
                    weights=create_weights(),
                    input_decay=INPUT_DECAY,
                    signal_delay_scale=SIGNAL_DELAY_SCALE,
                )

    return neurons


def apply_starting_stimulus(neurons: Dict[Index3D, Neuron]) -> None:
    starting_indexes: list[Index3D] = [
        (0, 0, 0),
        (3, 3, 3),
    ]

    for index in starting_indexes:
        neurons[index].current_input = STARTING_STIMULUS


def simulation_step(
    neurons: Dict[Index3D, Neuron],
    active_signals: list[ActiveSignal],
) -> list[Neuron]:
    fired_neurons = [
        neuron
        for neuron in neurons.values()
        if neuron.update()
    ]

    for source in fired_neurons:
        for target_index, weight in source.outgoing_connections():
            time_until_arrival = neurons[target_index].receive_input(
                weight,
                source.index,
            )
            if SHOW_SIGNAL_DOTS:
                active_signals.append(
                    ActiveSignal(
                        source_index=source.index,
                        target_index=target_index,
                        amount=weight,
                        total_ticks=time_until_arrival,
                        remaining_ticks=time_until_arrival,
                    )
                )

    return fired_neurons


def advance_active_signals(active_signals: list[ActiveSignal]) -> None:
    active_signals[:] = [
        signal
        for signal in active_signals
        if signal.remaining_ticks >= 0
    ]

    for signal in active_signals:
        signal.remaining_ticks -= 1


def signal_colors(active_signals: list[ActiveSignal]) -> list[tuple[float, float, float, float]]:
    return [
        (0.0, 0.25, 1.0, max(0.1, min(1.0, signal.amount)))
        for signal in active_signals
    ]


def signal_positions(
    active_signals: list[ActiveSignal],
) -> tuple[list[float], list[float], list[float]]:
    positions = [signal.position() for signal in active_signals]

    return (
        [position[0] for position in positions],
        [position[1] for position in positions],
        [position[2] for position in positions],
    )


def input_darkness(neurons: Dict[Index3D, Neuron]) -> list[float]:
    return [
        min(1.0, neuron.current_input / neuron.fire_threshold)
        for neuron in neurons.values()
    ]


def main() -> None:
    neurons = create_neurons()
    active_signals: list[ActiveSignal] = []
    apply_starting_stimulus(neurons)
    indexes = list(neurons.keys())
    xs = [index[0] for index in indexes]
    ys = [index[1] for index in indexes]
    zs = [index[2] for index in indexes]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(
        xs,
        ys,
        zs,
        c=input_darkness(neurons),
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        s=260,
        edgecolors="black",
    )
    signal_scatter = None
    if SHOW_SIGNAL_DOTS:
        signal_scatter = ax.scatter(
            [],
            [],
            [],
            c=[],
            s=SIGNAL_DOT_SIZE,
            depthshade=False,
        )
    input_labels = []
    if SHOW_INPUT_LABELS:
        input_labels = [
            ax.text(
                x + 0.08,
                y + 0.08,
                z + 0.08,
                f"{neurons[index].current_input:.2f}",
                fontsize=7,
            )
            for index, x, y, z in zip(indexes, xs, ys, zs)
        ]

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_zlim(-0.5, GRID_SIZE - 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def animate(tick: int) -> tuple[object]:
        if SHOW_SIGNAL_DOTS:
            advance_active_signals(active_signals)
        fired_neurons = simulation_step(neurons, active_signals)
        scatter.set_array(input_darkness(neurons))
        artists: list[object] = [scatter]
        if signal_scatter is not None:
            signal_scatter._offsets3d = signal_positions(active_signals)
            signal_scatter.set_color(signal_colors(active_signals))
            artists.append(signal_scatter)
        if SHOW_INPUT_LABELS:
            for label, index in zip(input_labels, indexes):
                label.set_text(f"{neurons[index].current_input:.2f}")
            artists.extend(input_labels)
        fired_indexes = [neuron.index for neuron in fired_neurons]
        ax.set_title(f"tick={tick} fired={fired_indexes}")
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        animate,
        interval=FRAME_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
