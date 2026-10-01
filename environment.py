"""
environment.py
Reusable builder for the Part B arena + mountain, adapted from cw-envt.py.

cw-envt.py is a GUI demo script; this module turns it into functions that
the GA simulation can call headlessly (DIRECT mode), once per simulation,
and that work with an explicit physicsClientId.

SCAFFOLD NOTE: the arena/mountain geometry here is lifted from the provided
cw-envt.py (template code). The summit-detection helper is new.
"""

import os
import pybullet as p

SHAPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes")
DEFAULT_MOUNTAIN_URDF = os.path.join(SHAPES_DIR, "gaussian_pyramid.urdf")
# DEFAULT_MOUNTAIN_URDF = os.path.join(SHAPES_DIR, "mountain_with_cubes.urdf")

# Same placement as cw-envt.py
MOUNTAIN_POSITION = (0, 0, -1.5) # x, y, z of the mountain base (not the summit)


# def make_arena(arena_size=20, wall_height=1, pid=0):
#     """Yellow floor + four grey walls. Straight from cw-envt.py, with
#     physicsClientId threaded through so it works in multi-sim setups."""
#     wall_thickness = 0.5
#     floor_col = p.createCollisionShape(
#         p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
#         physicsClientId=pid)
#     floor_vis = p.createVisualShape(
#         p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
#         rgbaColor=[1, 1, 0, 1], physicsClientId=pid)
#     p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_col,
#                       baseVisualShapeIndex=floor_vis,
#                       basePosition=[0, 0, -wall_thickness], physicsClientId=pid)

#     wall_col = p.createCollisionShape(
#         p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
#         physicsClientId=pid)
#     wall_vis = p.createVisualShape(
#         p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
#         rgbaColor=[0.7, 0.7, 0.7, 1], physicsClientId=pid)
#     p.createMultiBody(0, wall_col, wall_vis, [0, arena_size / 2, wall_height / 2],
#                       physicsClientId=pid)
#     p.createMultiBody(0, wall_col, wall_vis, [0, -arena_size / 2, wall_height / 2],
#                       physicsClientId=pid)


#     wall_col = p.createCollisionShape(
#         p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
#         physicsClientId=pid)
#     wall_vis = p.createVisualShape(
#         p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
#         rgbaColor=[0.7, 0.7, 0.7, 1], physicsClientId=pid)
#     p.createMultiBody(0, wall_col, wall_vis, [arena_size / 2, 0, wall_height / 2],
#                       physicsClientId=pid)
#     p.createMultiBody(0, wall_col, wall_vis, [-arena_size / 2, 0, wall_height / 2],
#                       physicsClientId=pid)
def make_arena(arena_size=20, wall_height=1, pid=0):
    """Yellow floor + four grey walls. Straight from cw-envt.py, with
    physicsClientId threaded through so it works in multi-sim setups."""

    wall_thickness = 0.5

    # -------------------------
    # Floor
    # -------------------------
    floor_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
        physicsClientId=pid,
    )

    floor_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, arena_size / 2, wall_thickness],
        rgbaColor=[1, 1, 0, 1],
        physicsClientId=pid,
    )

    floor_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_col,
        baseVisualShapeIndex=floor_vis,
        basePosition=[0, 0, -wall_thickness],
        physicsClientId=pid,
    )

    # Increase floor friction
    p.changeDynamics(
        floor_id,
        -1,  # -1 means base link
        lateralFriction=5,
        spinningFriction=1,
        rollingFriction=2,
        restitution=0.0,
        physicsClientId=pid,
    )

    # -------------------------
    # North/South walls
    # -------------------------
    wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
        physicsClientId=pid,
    )

    wall_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=pid,
    )

    wall_1 = p.createMultiBody(
        0, wall_col, wall_vis, [0, arena_size / 2, wall_height / 2], physicsClientId=pid
    )

    wall_2 = p.createMultiBody(
        0,
        wall_col,
        wall_vis,
        [0, -arena_size / 2, wall_height / 2],
        physicsClientId=pid,
    )

    # -------------------------
    # East/West walls
    # -------------------------
    wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
        physicsClientId=pid,
    )

    wall_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=pid,
    )

    wall_3 = p.createMultiBody(
        0, wall_col, wall_vis, [arena_size / 2, 0, wall_height / 2], physicsClientId=pid
    )

    wall_4 = p.createMultiBody(
        0,
        wall_col,
        wall_vis,
        [-arena_size / 2, 0, wall_height / 2],
        physicsClientId=pid,
    )

    return floor_id


def load_mountain(mountain_urdf=DEFAULT_MOUNTAIN_URDF, pid=0):
    """Load the mountain mesh at the same pose cw-envt.py uses.
    Returns the mountain body id."""
    orientation = p.getQuaternionFromEuler((0, 0, 0))
    mountain = p.loadURDF(mountain_urdf, MOUNTAIN_POSITION, orientation,
                          useFixedBase=1, physicsClientId=pid)
    return mountain


def get_summit_position(mountain_id, pid=0):
    """Compute the world-space summit of the loaded mountain.

    Uses the axis-aligned bounding box of the mountain body: the Gaussian
    pyramid peaks at its (x, y) centre and its highest z is the AABB top.
    Doing this at runtime (instead of hard-coding ~4.0) means the fitness
    target stays correct if you swap in a different mountain from
    prepare_shapes.py for the exceptional experiments.
    """
    aabb_min, aabb_max = p.getAABB(mountain_id, physicsClientId=pid)
    cx = (aabb_min[0] + aabb_max[0]) / 2.0
    cy = (aabb_min[1] + aabb_max[1]) / 2.0
    cz = aabb_max[2]
    return (cx, cy, cz)


def build_environment(arena_size=20, mountain_urdf=DEFAULT_MOUNTAIN_URDF, pid=0):
    """Build gravity + arena + mountain. Returns (mountain_id, summit_xyz, floor_id)."""
    p.setGravity(0, 0, -5, physicsClientId=pid) # gravity is weaker than Earth to make climbing easier
    floor_id = make_arena(arena_size=arena_size, pid=pid)
    mountain_id = load_mountain(mountain_urdf, pid=pid)
    summit = get_summit_position(mountain_id, pid=pid)
    return mountain_id, summit, floor_id
