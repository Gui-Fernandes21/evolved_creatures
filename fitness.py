import math


def distance_3d(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance_xy(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_fitness(positions, summit, spawn_position, airborne_steps) -> float:
    """
    Climbing fitness = weighted sum of:
      - exponential closeness to the summit (based on the FINAL position),
      - normalized height gained above the spawn height,
      - horizontal progress toward the summit,
      - minus a regression penalty for ending farther away than it started,
    all scaled by an airborne penalty multiplier. Creatures that spend more
    than AIRBORNE_THRESHOLD of their steps in the air, or that end below the
    spawn height, score 0 (anti-cheating: no flying to the top).
    """
    if not positions:
        return 0.0

    k = 4.0  # Higher = harsher penalty for being far from the summit
    CLOSENESS_WEIGHT = 1.5  # Weight for the closeness reward
    HEIGHT_GAIN_WEIGHT = 0.5  # Weight for the height gain reward
    AIRBORNE_THRESHOLD = 0.15  # Maximum allowed ratio of airborne steps
    HORIZONTAL_PROGRESS_WEIGHT = 1.5 # Weight for the horizontal progress reward

    spawn_height = spawn_position[2]
    final_z = positions[-1][2]

    initial_xy_dist = distance_xy(spawn_position, summit)
    final_xy_dist = distance_xy(positions[-1], summit)

    if final_z < spawn_height:
        return 0.0  # Penalize if the creature fell the simulation ground

    # HORIZONTAL PROGRESS REWARD
    horizontal_progress = max(
        0.0, (initial_xy_dist - final_xy_dist) / max(1.0, initial_xy_dist)
    )

    # CALCULATE MAXIMUM REWARD WHICH IS THE TOTAL DISTANCE FROM THE SPAWN POSITION TO THE SUMMIT
    spawn_to_summit_dist = distance_3d(spawn_position, summit)

    # Closeness is based on the creature's FINAL distance to the summit,
    # normalized by the spawn->summit distance so k behaves consistently
    # across different mountain sizes.
    normalized_distance = distance_3d(positions[-1], summit) / max(
        1.0, spawn_to_summit_dist
    )

    # Exponential closeness reward
    closeness_reward = math.exp(-k * normalized_distance)

    # Reward for ending higher than spawn
    height_gain = max(0.0, final_z - spawn_height)
    normalized_height_gain = height_gain / max(1.0, summit[2] - spawn_height)

    # height_bonus = 1.5 * normalized_height_gain

    # Penalize creatures that end farther from the summit than where they started.
    initial_distance_to_summit = spawn_to_summit_dist
    final_distance_to_summit = distance_3d(positions[-1], summit)
    summit_regression = max(0.0, final_distance_to_summit - initial_distance_to_summit)
    regression_penalty = summit_regression / max(1.0, initial_distance_to_summit)

    ## PENALTY FOR AIRBORNE STEPS
    # if creature is airborne for more than 15% of the time steps, it gets no reward
    # is the logic correct? I think it is, but I will double check
    #
    airborne_ratio = airborne_steps / len(positions)
    # print(airborne_ratio)
    if airborne_ratio > AIRBORNE_THRESHOLD:
        return 0.0
    else:
        airborne_penalty_multiplier = max(0.0, 1.0 - airborne_ratio)

    reward_func = (
        (CLOSENESS_WEIGHT * closeness_reward)
        + (HEIGHT_GAIN_WEIGHT * normalized_height_gain)
        + (HORIZONTAL_PROGRESS_WEIGHT * horizontal_progress)
        - regression_penalty
    ) * airborne_penalty_multiplier

    return max(0.0, reward_func)
