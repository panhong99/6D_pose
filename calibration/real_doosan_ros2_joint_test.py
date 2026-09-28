"""Small ROS 2 joint-motion test for a real Doosan M1013.

This uses the Doosan ROS 2 Python API, not Isaac Sim.  By default it is a
dry-run and does not send a motion command.  Real motion requires both:

  --execute --target-deg J1 J2 J3 J4 J5 J6

Example (do not run until the robot and workspace are verified):
  python3 calibration/real_doosan_ros2_joint_test.py \
    --execute --target-deg 0 0 10 0 10 0
"""

import argparse
import sys

import rclpy
from rclpy.logging import get_logger


ROBOT_ID = "dsr01"
ROBOT_MODEL = "m1013"
MAX_SAFE_MOTION_RATE = 0.2
DSR_COMMON_IMP = "/home/panhong/pan/doosan-robot2/dsr_common2/imp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-deg",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="target joint angles in degrees",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=MAX_SAFE_MOTION_RATE,
        help="joint speed; capped at the conservative test limit",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=MAX_SAFE_MOTION_RATE,
        help="joint acceleration; capped at the conservative test limit",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually send the motion command; omitted means dry-run",
    )
    args = parser.parse_args()

    logger = get_logger("real_doosan_ros2_joint_test")
    if args.target_deg is None:
        parser.error("--target-deg J1 J2 J3 J4 J5 J6 is required")

    target = [float(v) for v in args.target_deg]
    if args.velocity <= 0.0 or args.acceleration <= 0.0:
        parser.error("--velocity and --acceleration must be positive")
    if args.velocity > MAX_SAFE_MOTION_RATE or args.acceleration > MAX_SAFE_MOTION_RATE:
        parser.error(
            f"velocity and acceleration are capped at {MAX_SAFE_MOTION_RATE}; "
            "edit the source only after an explicit safety review"
        )
    logger.info(f"robot_id={ROBOT_ID}, model={ROBOT_MODEL}")
    logger.info(f"target joint degrees={target}")
    logger.info(f"velocity={args.velocity}, acceleration={args.acceleration}")

    if not args.execute:
        logger.warning("DRY RUN: no motion command was sent")
        logger.info("Add --execute only after checking the target and safety area")
        return

    # Configure the Doosan ROS 2 Python API exactly as in the vendor examples.
    if DSR_COMMON_IMP not in sys.path:
        sys.path.insert(0, DSR_COMMON_IMP)
    import DR_init

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    rclpy.init(args=sys.argv)
    node = rclpy.create_node("real_doosan_ros2_joint_test", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import movej, posj, set_robot_mode
        from DSR_ROBOT2 import ROBOT_MODE_AUTONOMOUS

        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        logger.warning("Sending REAL Doosan movej command")
        movej(posj(*target), vel=args.velocity, acc=args.acceleration)
        logger.info("Doosan movej command returned")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
