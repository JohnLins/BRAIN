from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from neuron import Connection, Index3D, Neuron


GRID_SIZE = 4
FRAME_INTERVAL_MS = 1
INPUT_DECAY = 0.01
SIGNAL_DELAY_SCALE = 8
INPUT_STIMULUS = 1.25
MIN_FIRE_THRESHOLD = 0.35
MAX_FIRE_THRESHOLD = 0.8
SHOW_SIGNAL_DOTS = True
SHOW_CONNECTION_LINES = True
SIGNAL_DOT_SIZE = 1
CONNECTION_LINE_WIDTH = 0.35
CONNECTION_LINE_ALPHA = 0.18
INITIAL_WEIGHT_MAX = 1.0
NETWORK_SEED = 7
TRAINING_SEED = 13
MIN_CONNECTIONS_PER_NEURON = 4
MAX_CONNECTIONS_PER_NEURON = 14
CONNECTION_DISTANCE_SCALE = 1.5
MIN_REFRACTORY_PERIOD = 1
MAX_REFRACTORY_PERIOD = 10
INPUT_NEURONS: list[Index3D] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 0, 3),
]
OUTPUT_NEURONS: list[Index3D] = [
    (3, 3, 0),
    (3, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
]
FIXED_INPUT_PATTERN = [1.0, 1.0, 0.0, 0.0]
FIXED_EXPECTED_OUTPUT = [0.0, 0.0, 0.0, 1.0]

WAIT_TICKS_BEFORE_MEASUREMENT = 30
OUTPUT_MEASUREMENT_TICKS = 60
WEIGHT_NOISE = 0.002
COFIRE_WINDOW_TICKS = 3
OUTPUT_ERROR_LEARNING_RATE = 0.12
OUTPUT_ERROR_CONNECTION_CREATE_PROBABILITY = 0.03
OUTPUT_ERROR_NEW_CONNECTION_WEIGHT = 0.2
OUTPUT_COMPETITION_CURRENT_DAMPING = 0.35
OUTPUT_COMPETITION_QUEUE_DAMPING = 0.5


EligibilityKey = tuple[Index3D, Index3D]


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


@dataclass
class TrainingTarget:
    input_pattern: list[float]
    expected_output: list[float]


@dataclass
class TrialState:
    trial_tick: int = 0
    output_counts: list[int] = field(default_factory=lambda: [0 for _ in OUTPUT_NEURONS])
    recent_fired: dict[Index3D, int] = field(default_factory=dict)
    eligibility: dict[EligibilityKey, float] = field(default_factory=dict)
    last_reward: float = 0.0
    last_reward_delta: float = 0.0
    last_output_distribution: list[float] = field(
        default_factory=lambda: l1_normalize([0.0 for _ in OUTPUT_NEURONS])
    )

    @property
    def trial_started(self) -> bool:
        return self.trial_tick > 0

    @property
    def measurement_started(self) -> bool:
        return self.trial_tick >= WAIT_TICKS_BEFORE_MEASUREMENT

    @property
    def measurement_finished(self) -> bool:
        return self.trial_tick >= WAIT_TICKS_BEFORE_MEASUREMENT + OUTPUT_MEASUREMENT_TICKS

    def start_next_trial(self) -> None:
        self.trial_tick = 0
        self.output_counts = [0 for _ in OUTPUT_NEURONS]
        self.recent_fired = {}
        self.eligibility = {}


def create_neuron_indexes() -> list[Index3D]:
    return [
        (i, j, k)
        for i in range(GRID_SIZE)
        for j in range(GRID_SIZE)
        for k in range(GRID_SIZE)
    ]


def distance_between(source_index: Index3D, target_index: Index3D) -> float:
    return (
        (source_index[0] - target_index[0]) ** 2
        + (source_index[1] - target_index[1]) ** 2
        + (source_index[2] - target_index[2]) ** 2
    ) ** 0.5


def connection_probability(distance: float) -> float:
    return math.exp(-distance / CONNECTION_DISTANCE_SCALE)


def seed_for_index(index: Index3D) -> int:
    return (
        NETWORK_SEED
        + index[0] * 73_856_093
        + index[1] * 19_349_663
        + index[2] * 83_492_791
    )


def weighted_sample_without_replacement(
    rng: random.Random,
    candidates: list[tuple[Index3D, float]],
    sample_size: int,
) -> list[Index3D]:
    selected_indexes: list[Index3D] = []
    remaining_candidates = candidates.copy()

    while remaining_candidates and len(selected_indexes) < sample_size:
        total_weight = sum(weight for _, weight in remaining_candidates)
        pick = rng.uniform(0.0, total_weight)
        cumulative_weight = 0.0

        for candidate_index, (target_index, weight) in enumerate(remaining_candidates):
            cumulative_weight += weight
            if cumulative_weight >= pick:
                selected_indexes.append(target_index)
                remaining_candidates.pop(candidate_index)
                break

    return selected_indexes


def create_connections(
    source_index: Index3D,
    all_indexes: list[Index3D],
    rng: random.Random,
) -> list[Connection]:
    target_count = rng.randint(MIN_CONNECTIONS_PER_NEURON, MAX_CONNECTIONS_PER_NEURON)
    candidates = []

    for target_index in all_indexes:
        if target_index == source_index:
            continue

        distance = distance_between(source_index, target_index)
        candidates.append((target_index, connection_probability(distance)))

    selected_targets = weighted_sample_without_replacement(
        rng,
        candidates,
        min(target_count, len(candidates)),
    )

    return [
        Connection(
            target_index=target_index,
            weight=rng.random() * INITIAL_WEIGHT_MAX,
            delay_ticks=max(
                1,
                round(distance_between(source_index, target_index) * SIGNAL_DELAY_SCALE),
            ),
        )
        for target_index in selected_targets
    ]


def create_neurons() -> Dict[Index3D, Neuron]:
    all_indexes = create_neuron_indexes()
    neurons: Dict[Index3D, Neuron] = {}

    for index in all_indexes:
        rng = random.Random(seed_for_index(index))
        neurons[index] = Neuron(
            index=index,
            connections=create_connections(index, all_indexes, rng),
            input_decay=INPUT_DECAY,
            fire_threshold=rng.uniform(MIN_FIRE_THRESHOLD, MAX_FIRE_THRESHOLD),
            refractory_period=rng.randint(MIN_REFRACTORY_PERIOD, MAX_REFRACTORY_PERIOD),
            signal_delay_scale=SIGNAL_DELAY_SCALE,
        )

    return neurons


def create_training_target() -> TrainingTarget:
    return TrainingTarget(
        input_pattern=FIXED_INPUT_PATTERN.copy(),
        expected_output=FIXED_EXPECTED_OUTPUT.copy(),
    )


def inject_input_pattern(neurons: Dict[Index3D, Neuron], pattern: list[float]) -> None:
    for index, value in zip(INPUT_NEURONS, pattern):
        neurons[index].current_input += value * INPUT_STIMULUS


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
        for connection in source.outgoing_connections():
            neurons[connection.target_index].receive_input(
                connection.weight,
                connection.delay_ticks,
            )
            if SHOW_SIGNAL_DOTS:
                active_signals.append(
                    ActiveSignal(
                        source_index=source.index,
                        target_index=connection.target_index,
                        amount=connection.weight,
                        total_ticks=connection.delay_ticks,
                        remaining_ticks=connection.delay_ticks,
                    )
                )

    return fired_neurons


def damp_input_queue(neuron: Neuron, damping: float) -> None:
    neuron.input_queue = type(neuron.input_queue)(
        (amount * damping, delay_ticks)
        for amount, delay_ticks in neuron.input_queue
    )


def apply_output_competition(
    neurons: Dict[Index3D, Neuron],
    fired_indexes: set[Index3D],
) -> None:
    fired_outputs = fired_indexes.intersection(OUTPUT_NEURONS)
    if not fired_outputs:
        return

    for fired_output in fired_outputs:
        for output_index in OUTPUT_NEURONS:
            if output_index == fired_output:
                continue

            output_neuron = neurons[output_index]
            output_neuron.current_input *= OUTPUT_COMPETITION_CURRENT_DAMPING
            damp_input_queue(output_neuron, OUTPUT_COMPETITION_QUEUE_DAMPING)


def advance_active_signals(active_signals: list[ActiveSignal]) -> None:
    active_signals[:] = [
        signal
        for signal in active_signals
        if signal.remaining_ticks >= 0
    ]

    for signal in active_signals:
        signal.remaining_ticks -= 1


def signal_colors(active_signals: list[ActiveSignal]) -> list[tuple[float, float, float, float]]:
    colors = []

    for signal in active_signals:
        strength = min(1.0, max(0.0, signal.amount))
        red_green = 0.65 * (1.0 - strength)
        colors.append((red_green, red_green, 1.0, max(0.08, strength)))

    return colors


def signal_positions(
    active_signals: list[ActiveSignal],
) -> tuple[list[float], list[float], list[float]]:
    positions = [signal.position() for signal in active_signals]

    return (
        [position[0] for position in positions],
        [position[1] for position in positions],
        [position[2] for position in positions],
    )


def connection_segments(
    neurons: Dict[Index3D, Neuron],
) -> list[tuple[Index3D, Index3D]]:
    return [
        (source.index, connection.target_index)
        for source in neurons.values()
        for connection in source.connections
    ]


def connection_colors(
    neurons: Dict[Index3D, Neuron],
) -> list[tuple[float, float, float, float]]:
    colors = []

    for source in neurons.values():
        for connection in source.connections:
            strength = min(1.0, max(0.0, connection.weight))
            shade = 1.0 - strength
            alpha = CONNECTION_LINE_ALPHA * (0.2 + 0.8 * strength)
            colors.append((shade, shade, shade, alpha))

    return colors


def input_darkness(neurons: Dict[Index3D, Neuron]) -> list[float]:
    return [
        min(1.0, neuron.current_input / neuron.fire_threshold)
        for neuron in neurons.values()
    ]


def neuron_colors(
    neurons: Dict[Index3D, Neuron],
    indexes: list[Index3D],
    fired_indexes: set[Index3D],
) -> list[tuple[float, float, float, float]]:
    colors = []

    for index in indexes:
        neuron = neurons[index]

        if index in fired_indexes:
            colors.append((0.0, 0.25, 1.0, 1.0))
        elif neuron.refractory_remaining > 0:
            colors.append((1.0, 0.0, 0.0, 0.85))
        else:
            darkness = min(1.0, neuron.current_input / neuron.fire_threshold)
            shade = 1.0 - darkness
            colors.append((shade, shade, shade, 1.0))

    return colors


def l1_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    total = sum(abs(value) for value in values)
    if total == 0:
        return [0.0 for _ in values]

    return [value / total for value in values]


def format_output_distribution(distribution: list[float]) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in distribution) + "]"


def reward_for_output(distribution: list[float], expected_output: list[float]) -> float:
    return sum(
        value * expected
        for value, expected in zip(distribution, expected_output)
    )


def reward_baseline(expected_output: list[float]) -> float:
    return sum(expected_output) / len(expected_output)


def clamp_weight(weight: float) -> float:
    return min(1.0, max(0.0, weight))


def update_eligibility(
    trial_state: TrialState,
    fired_indexes: set[Index3D],
) -> None:
    for target_index in fired_indexes:
        if (
            target_index not in OUTPUT_NEURONS
            or not trial_state.measurement_started
            or trial_state.measurement_finished
        ):
            continue

        for source_index, last_fire_tick in trial_state.recent_fired.items():
            if source_index == target_index:
                continue

            age = trial_state.trial_tick - last_fire_tick
            if 0 <= age <= COFIRE_WINDOW_TICKS:
                key = (source_index, target_index)
                trial_state.eligibility[key] = (
                    trial_state.eligibility.get(key, 0.0)
                    + 1.0 / (age + 1.0)
                )

    for index in fired_indexes:
        trial_state.recent_fired[index] = trial_state.trial_tick


def train_output_errors(
    neurons: Dict[Index3D, Neuron],
    source_indexes: set[Index3D],
    eligibility: dict[EligibilityKey, float],
    output_distribution: list[float],
    expected_output: list[float],
    rng: random.Random,
) -> int:
    created_connections = 0
    max_eligibility = max(eligibility.values()) if eligibility else 1.0

    for source_index in source_indexes:
        source = neurons[source_index]

        for output_slot, target_index in enumerate(OUTPUT_NEURONS):
            if source_index == target_index:
                continue

            error = expected_output[output_slot] - output_distribution[output_slot]
            if error == 0.0:
                continue

            eligibility_strength = eligibility.get((source_index, target_index))
            if eligibility_strength is None and error < 0.0:
                continue

            if eligibility_strength is None:
                eligibility_strength = max_eligibility

            normalized_eligibility = eligibility_strength / max_eligibility
            connection = source.connection_to(target_index)
            if connection is None:
                if error < 0.0:
                    continue

                if rng.random() > OUTPUT_ERROR_CONNECTION_CREATE_PROBABILITY:
                    continue

                connection = Connection(
                    target_index=target_index,
                    weight=OUTPUT_ERROR_NEW_CONNECTION_WEIGHT,
                    delay_ticks=max(
                        1,
                        round(distance_between(source_index, target_index) * SIGNAL_DELAY_SCALE),
                    ),
                )
                source.connections.append(connection)
                created_connections += 1

            noise = rng.uniform(-WEIGHT_NOISE, WEIGHT_NOISE)
            connection.weight = clamp_weight(
                connection.weight
                + OUTPUT_ERROR_LEARNING_RATE * error * normalized_eligibility
                + noise
            )

    return created_connections


def main() -> None:
    neurons = create_neurons()
    active_signals: list[ActiveSignal] = []
    training_rng = random.Random(TRAINING_SEED)
    training_target = create_training_target()
    trial_state = TrialState()
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
        c=neuron_colors(neurons, indexes, set()),
        s=260,
        edgecolors="black",
    )
    connection_lines = None
    if SHOW_CONNECTION_LINES:
        connection_lines = Line3DCollection(
            connection_segments(neurons),
            colors=connection_colors(neurons),
            linewidths=CONNECTION_LINE_WIDTH,
        )
        ax.add_collection3d(connection_lines)

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
    input_labels = [
        ax.text(
            index[0] - 0.85,
            index[1] - 0.55,
            index[2],
            f"input {input_index + 1}",
            fontsize=8,
            color="green",
        )
        for input_index, index in enumerate(INPUT_NEURONS)
    ]
    output_labels = [
        ax.text(
            index[0] + 0.65,
            index[1] + 0.45,
            index[2],
            f"output {output_index + 1}\ncount=0",
            fontsize=8,
            color="purple",
        )
        for output_index, index in enumerate(OUTPUT_NEURONS)
    ]

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_zlim(-0.5, GRID_SIZE - 0.5)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    def animate(tick: int) -> tuple[object]:
        if not trial_state.trial_started:
            inject_input_pattern(neurons, training_target.input_pattern)

        if SHOW_SIGNAL_DOTS:
            advance_active_signals(active_signals)
        fired_neurons = simulation_step(neurons, active_signals)
        fired_indexes = {neuron.index for neuron in fired_neurons}
        apply_output_competition(neurons, fired_indexes)

        update_eligibility(trial_state, fired_indexes)

        if trial_state.measurement_started and not trial_state.measurement_finished:
            for output_index, index in enumerate(OUTPUT_NEURONS):
                if index in fired_indexes:
                    trial_state.output_counts[output_index] += 1

        output_distribution = l1_normalize(
            [float(count) for count in trial_state.output_counts]
        )
        trial_state.last_output_distribution = output_distribution

        if trial_state.measurement_finished:
            trial_state.last_reward = reward_for_output(
                output_distribution,
                training_target.expected_output,
            )
            trial_state.last_reward_delta = (
                trial_state.last_reward
                - reward_baseline(training_target.expected_output)
            )
            training_source_indexes = set(trial_state.recent_fired)
            training_source_indexes.update(
                source_index
                for source_index, _ in trial_state.eligibility
            )
            created_connections = train_output_errors(
                neurons,
                training_source_indexes,
                trial_state.eligibility,
                output_distribution,
                training_target.expected_output,
                training_rng,
            )
            print(
                f"pattern={training_target.input_pattern} "
                f"expected={training_target.expected_output} "
                f"counts={trial_state.output_counts} "
                f"l1={format_output_distribution(output_distribution)} "
                f"reward={trial_state.last_reward:.3f} "
                f"reward_delta={trial_state.last_reward_delta:.3f} "
                f"eligible_pairs={len(trial_state.eligibility)} "
                f"created_output_connections={created_connections}"
            )
            trial_state.start_next_trial()
        else:
            trial_state.trial_tick += 1

        scatter.set_facecolors(neuron_colors(neurons, indexes, fired_indexes))
        scatter.set_edgecolors("black")
        artists: list[object] = [scatter]
        if connection_lines is not None:
            connection_lines.set_segments(connection_segments(neurons))
            connection_lines.set_color(connection_colors(neurons))
            artists.append(connection_lines)
        if signal_scatter is not None:
            signal_scatter._offsets3d = signal_positions(active_signals)
            signal_scatter.set_color(signal_colors(active_signals))
            artists.append(signal_scatter)
        for output_index, label in enumerate(output_labels):
            label.set_text(
                f"output {output_index + 1}\n"
                f"count={trial_state.output_counts[output_index]}"
            )
        artists.extend(input_labels)
        artists.extend(output_labels)
        if not trial_state.trial_started:
            phase = "stimulate"
        elif trial_state.measurement_started:
            phase = "measure"
        else:
            phase = "wait"
        # ax.set_title(
        #     f"tick={tick} phase={phase} "
        #     f"input={training_target.input_pattern} "
        #     f"expected={training_target.expected_output}\n"
        #     f"counts={trial_state.output_counts} "
        #     f"l1={format_output_distribution(trial_state.last_output_distribution)} "
        #     f"reward={trial_state.last_reward:.3f} "
        #     f"delta={trial_state.last_reward_delta:.3f}"
        # )
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
