import os
import math
import pybullet as p

import environment
import fitness


class MountainSimulation:
    def __init__(self, sim_id=0, gui=False, arena_size=20,
                 mountain_urdf=environment.DEFAULT_MOUNTAIN_URDF,
                 spawn_position=(6.0, 0.0, 0.8), sensing_enabled=True):
        self.sim_id = sim_id
        self.sensing_enabled = sensing_enabled  # False = sensor-off ablation
        self.pid = p.connect(p.GUI if gui else p.DIRECT)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.pid)

        self.arena_size = arena_size
        self.spawn_position = spawn_position
        self.mountain_id, self.summit, self.floor_id = environment.build_environment(
            arena_size=arena_size, mountain_urdf=mountain_urdf, pid=self.pid)

        self._creature_body = None  # id of the currently loaded creature

        # Increase mountain surface friction
        self._set_body_friction(
            self.mountain_id,
            lateral_friction=5.5,
            spinning_friction=0.05,
            rolling_friction=2,
        )

    def _set_body_friction(
        self,
        body_id,
        lateral_friction=2.0,
        spinning_friction=0.05,
        rolling_friction=0.02,
    ):
        """
        Apply friction settings to the base link and all child links.

        linkIndex = -1 means the base link.
        Child links use indices 0 to getNumJoints(body_id) - 1.
        """

        # Base link
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=lateral_friction,
            spinningFriction=spinning_friction,
            rollingFriction=rolling_friction,
            restitution=0.0,
            physicsClientId=self.pid,
        )

        # Child links
        for link_index in range(p.getNumJoints(body_id, physicsClientId=self.pid)):
            p.changeDynamics(
                body_id,
                link_index,
                lateralFriction=lateral_friction,
                spinningFriction=spinning_friction,
                rollingFriction=rolling_friction,
                restitution=0.0,
                physicsClientId=self.pid,
            )

    def run_creature(self, cr, iterations=2400):
        """Evaluate one creature. Stores cr.fitness and cr.tracked_positions."""
        pid = self.pid

        # Remove the previous creature instead of resetting the whole sim
        if self._creature_body is not None:
            p.removeBody(self._creature_body, physicsClientId=pid)
            self._creature_body = None

        xml_file = "temp" + str(self.sim_id) + ".urdf"
        with open(xml_file, "w") as f:
            f.write(cr.to_xml())

        try:
            cid = p.loadURDF(xml_file, physicsClientId=pid)
        except p.error:
            # Degenerate genome -> unloadable URDF. Score it 0, keep going.
            cr.tracked_positions = []
            cr.fitness = 0.0
            return

        self._creature_body = cid

        # Increase creature body friction too
        self._set_body_friction(
            self._creature_body,
            lateral_friction=5,
            spinning_friction=1,
            rolling_friction=2,
        )

        p.resetBasePositionAndOrientation(
            cid, list(self.spawn_position), [0, 0, 0, 1], physicsClientId=pid)

        positions = []  # full trajectory, every step — keeps distance calc clean
        airborne_steps = 0

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self.update_motors(cid, cr)

            pos, _orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            positions.append(pos)

            if len(p.getContactPoints(bodyA=cid, bodyB=self.mountain_id)) == 0:
                if len(p.getContactPoints(bodyA=cid, bodyB=self.floor_id)) == 0:
                    airborne_steps += 1

        cr.tracked_positions = positions
        cr.fitness = fitness.compute_fitness(
            positions,
            self.summit,
            self.spawn_position,
            airborne_steps
        )

    def update_motors(self, cid, cr):
        """
        Motor control with evolved summit-direction sensing.

        Each motor still uses its evolved waveform, but each motor also evolves
        how strongly it responds to the summit-direction sensor.
        """
        summit_dir = self._unit_vector_to_summit_xy(cid)
        motors = cr.get_motors()

        sensor = self._summit_heading_sensor(cid)

        for jin in range(p.getNumJoints(cid, physicsClientId=self.pid)):
            motor = motors[jin]

            base_output = motor.get_output()

            if self.sensing_enabled:
                direction_sensor = self._link_alignment_to_summit(cid, jin, summit_dir)
                evolved_sensor_bias = (
                    motor.get_sensor_weight()
                    * motor.get_sensor_sign()
                    * direction_sensor
                )
            else:
                evolved_sensor_bias = 0.0  # sensor-off ablation: pure evolved waveform

            target_velocity = base_output + evolved_sensor_bias

            # Alternate direction by joint index so the sensor creates turning,
            # not just identical motion in every joint.
            if jin % 2 == 0:
                sensor_bias = evolved_sensor_bias
            else:
                sensor_bias = -evolved_sensor_bias

            # Keep target velocity stable
            # target_velocity = base_output + sensor_bias
            target_velocity = max(-2.0, min(2.0, target_velocity))
            # target_velocity = max(-1.0, min(1.0, target_velocity))

            p.setJointMotorControl2(
                cid,
                jin,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=motor.get_force(),
                physicsClientId=self.pid
            )

    def eval_population(self, pop, iterations=2400):
        for cr in pop.creatures:
            self.run_creature(cr, iterations)

    def disconnect(self):
        p.disconnect(physicsClientId=self.pid)

    def _unit_vector_to_summit_xy(self, cid):
        base_pos, _orn = p.getBasePositionAndOrientation(
            cid,
            physicsClientId=self.pid
        )

        dx = self.summit[0] - base_pos[0]
        dy = self.summit[1] - base_pos[1]

        length = math.sqrt(dx * dx + dy * dy)

        if length < 1e-8:
            return (0.0, 0.0)

        return (dx / length, dy / length)

    def _base_forward_xy(self, cid):
        _pos, orn = p.getBasePositionAndOrientation(
            cid,
            physicsClientId=self.pid
        )

        rot = p.getMatrixFromQuaternion(orn)

        # Local x-axis of the creature in world coordinates
        fx = rot[0]
        fy = rot[3]

        length = math.sqrt(fx * fx + fy * fy)

        if length < 1e-8:
            return (0.0, 0.0)

        return (fx / length, fy / length)

    def _summit_heading_sensor(self, cid):
        """
        Returns a value in [-1, 1].

        +1 means the creature is facing the summit.
        0 means sideways.
        -1 means facing away from the summit.
        """
        summit_dir = self._unit_vector_to_summit_xy(cid)
        forward = self._base_forward_xy(cid)

        return (summit_dir[0] * forward[0]) + (summit_dir[1] * forward[1])
    
    def _link_alignment_to_summit(self, cid, link_index, summit_dir):
        base_pos, _orn = p.getBasePositionAndOrientation(
            cid,
            physicsClientId=self.pid
        )

        link_state = p.getLinkState(
            cid,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=self.pid
        )

        link_pos = link_state[0]

        lx = link_pos[0] - base_pos[0]
        ly = link_pos[1] - base_pos[1]

        link_len = math.sqrt(lx * lx + ly * ly)

        if link_len < 1e-8:
            return 0.0

        lx /= link_len
        ly /= link_len

        alignment = (lx * summit_dir[0]) + (ly * summit_dir[1])

        return max(-1.0, min(1.0, alignment))
