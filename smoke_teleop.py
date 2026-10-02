"""Short motion test using the existing official teleop_loop, recording OFF."""
import argparse
import json
import logging
from pathlib import Path

from collector import hardware_preflight, make_hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=Path, default=Path(__file__).with_name("devices.json"))
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        raise ValueError("Use a smoke duration between 0 and 60 seconds")
    config = json.loads(args.devices.read_text())
    robot, leader = make_hardware(config)
    hardware_preflight(robot, leader)
    # Import the installed official function. No new robot controller is introduced.
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import teleop_loop

    teleop_processor, action_processor, obs_processor = make_default_processors()
    logging.basicConfig(level=logging.INFO)
    print("Recording OFF. Gently move leader; observe follower. Ctrl+C stops immediately.", flush=True)
    try:
        leader.connect(calibrate=False)
        robot.connect(calibrate=False)
        teleop_loop(
            teleop=leader, robot=robot, fps=config["fps"],
            teleop_action_processor=teleop_processor,
            robot_action_processor=action_processor,
            robot_observation_processor=obs_processor,
            display_data=False, duration=args.seconds,
        )
    except KeyboardInterrupt:
        pass
    finally:
        for device in (robot, leader):
            if device.bus.is_connected:
                device.bus.disconnect(disable_torque=True)
        for camera in robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()


if __name__ == "__main__":
    main()
