import argparse
import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID


class Phases(Enum):
    EMPTY = -1
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class Phase:
    def __init__(self, phase: Phases, flying_drone):
        self._phase = phase
        self._next_phase: Phase = self._get_initial_next_phase()
        self._drone = flying_drone

    def next_phase(self):
        return self._next_phase

    def start_phase(self):
        pass

    def update_local_position(self) -> bool:
        return False

    def update_velocity(self) -> bool:
        return False

    def update_state(self) -> bool:
        return False

    def _get_initial_next_phase(self):
        return EmptyPhase()


class EmptyPhase(Phase):
    def __init__(self):
        super().__init__(Phases.EMPTY, None)

    def _get_initial_next_phase(self):
        return None


class ManualPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.MANUAL, flying_drone)

    def update_state(self) -> bool:
        print("arming transition")
        self._drone.take_control()
        self._drone.arm()

        self._drone.set_home_position(self._drone.global_position[0],
                                      self._drone.global_position[1],
                                      self._drone.global_position[2])

        self._next_phase = ArmingPhase(self._drone)
        return True


class ArmingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.ARMING, flying_drone)

    def update_state(self) -> bool:
        if self._drone.armed:
            print("takeoff transition")
            target_altitude = 3.0
            self._drone.target_position[2] = target_altitude
            self._drone.takeoff(target_altitude)
            self._next_phase = TakeoffPhase(self._drone)
            return True
        return False


class TakeoffPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.TAKEOFF, flying_drone)

    def update_local_position(self) -> bool:
        altitude = -1.0 * self._drone.local_position[2]
        if altitude > 0.95 * self._drone.target_position[2]:
            self._next_phase = WaypointPhase(self._drone)
            return True
        return False


class DisarmingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.DISARMING, flying_drone)

    def update_state(self) -> bool:
        if not self._drone.armed:
            print("manual transition")
            self._drone.release_control()
            self._drone.stop()
            self._drone.in_mission = False
            self._next_phase = ManualPhase(self._drone)
            return True
        return False


class LandingPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.LANDING, flying_drone)

    def update_local_position(self) -> bool:
        if ((self._drone.global_position[2] - self._drone.global_home[2] < 0.1) and
                abs(self._drone.local_position[2]) < 0.01):
            self._disarm()
            return True
        return False

    def _disarm(self):
        print("disarm transition")
        self._drone.disarm()
        self._next_phase = DisarmingPhase(self._drone)


class WaypointPhase(Phase):
    def __init__(self, flying_drone: Drone):
        super().__init__(Phases.WAYPOINT, flying_drone)
        self._waypoint = None
        self._waypoints = self._drone.calculate_box()

    def start_phase(self):
        if self._waypoints:
            self._waypoint = self._waypoints.pop(0)
            self._drone.cmd_position(self._waypoint[0], self._waypoint[1], self._waypoint[2], 0)

    def update_local_position(self) -> bool:
        velocity = self._drone.local_velocity.copy()
        speed = np.linalg.norm(velocity)

        local_position = self._drone.local_position.copy()
        local_position[2] *= -1
        dist = np.linalg.norm(local_position - self._waypoint)

        if dist < 0.2 and speed < 0.1:
            if self._waypoints:
                self._waypoint = self._waypoints.pop(0)
                self._drone.cmd_position(self._waypoint[0], self._waypoint[1], self._waypoint[2], 0)
                return False
            else:
                self._drone.land()
                self._next_phase = LandingPhase(self._drone)
                return True

        # if dist < 3 and speed > 0.1:
        #     velocity *= 0.1
        #     self._drone.cmd_velocity(velocity[0], velocity[1], velocity[2], 0)
        return False


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self._target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = []
        self._in_mission = True
        self.check_state = {}

        # initial state
        self.current_phase: Phase = EmptyPhase()

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position

    @property
    def in_mission(self) -> bool:
        return self._in_mission

    @in_mission.setter
    def in_mission(self, value: bool):
        self._in_mission = value

    def local_position_callback(self):
        """
        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        if self.current_phase.update_local_position():
            self._start_next_phase()

    def velocity_callback(self):
        """
        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        if self.current_phase.update_velocity():
            self._start_next_phase()

    def state_callback(self):
        """
        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        if not self._in_mission:
            return

        if self.current_phase.update_state():
            self._start_next_phase()

    def calculate_box(self):
        """
        1. Return waypoints to fly a box
        """
        return [
            [15, 0, 3],
            [15, 15, 3],
            [0, 15, 3],
            [0, 0, 3]
        ]

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        self.current_phase = ManualPhase(self)

        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()

    def _start_next_phase(self):
        self.current_phase = self.current_phase.next_phase()
        self.current_phase.start_phase()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    # conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
