import argparse
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def now_ns():
    return time.perf_counter_ns()


def ns_to_ms(delta_ns):
    return round(float(delta_ns) / 1_000_000.0, 6)


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def percentile_summary(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p99": round(float(np.percentile(arr, 99)), 6),
        "max": round(float(np.max(arr)), 6),
    }


class CsvEventLogger:
    fieldnames = [
        "timestamp_ns",
        "event",
        "step_id",
        "command_id",
        "state_seq",
        "payload_json",
    ]

    def __init__(self, output_dir, prefix):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = output_path / f"{prefix}_{timestamp}.csv"
        self.summary_path = output_path / f"{prefix}_{timestamp}_summary.json"
        self._fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._fh.flush()

    def log_event(
        self,
        event,
        step_id=None,
        command_id=None,
        state_seq=None,
        **payload,
    ):
        self._writer.writerow(
            {
                "timestamp_ns": now_ns(),
                "event": event,
                "step_id": "" if step_id is None else step_id,
                "command_id": "" if command_id is None else command_id,
                "state_seq": "" if state_seq is None else state_seq,
                "payload_json": json.dumps(
                    to_serializable(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        self._fh.flush()

    def write_summary(self, summary):
        self.summary_path.write_text(
            json.dumps(to_serializable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self):
        self._fh.close()


class ArmLatencyNode(Node):
    def __init__(self):
        super().__init__("x100_arm_latency_measure")
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.VOLATILE
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._lock = threading.Lock()
        self._command_seq = 0
        self._arm_state = {
            "left": {
                "seq": 0,
                "callback_ns": 0,
                "header_ns": 0,
                "name": [],
                "position": [],
                "velocity": [],
            },
            "right": {
                "seq": 0,
                "callback_ns": 0,
                "header_ns": 0,
                "name": [],
                "position": [],
                "velocity": [],
            },
        }

        self.left_arm_sub = self.create_subscription(
            JointState, "/cr100/left_arm_state", self._make_callback("left"), qos
        )
        self.right_arm_sub = self.create_subscription(
            JointState, "/cr100/right_arm_state", self._make_callback("right"), qos
        )
        self.left_arm_pub = self.create_publisher(
            JointState, "/cr100/left_arm/online_joint_command", 1
        )
        self.right_arm_pub = self.create_publisher(
            JointState, "/cr100/right_arm/online_joint_command", 1
        )

    @staticmethod
    def _stamp_to_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _make_callback(self, arm):
        def _callback(msg):
            callback_ns = now_ns()
            with self._lock:
                record = self._arm_state[arm]
                record["seq"] += 1
                record["callback_ns"] = callback_ns
                record["header_ns"] = self._stamp_to_ns(msg)
                record["name"] = list(msg.name)
                record["position"] = list(msg.position)
                record["velocity"] = list(msg.velocity)

        return _callback

    def get_arm_snapshot(self, arm):
        with self._lock:
            record = self._arm_state[arm]
            return {
                "seq": int(record["seq"]),
                "callback_ns": int(record["callback_ns"]),
                "header_ns": int(record["header_ns"]),
                "name": list(record["name"]),
                "position": list(record["position"]),
                "velocity": list(record["velocity"]),
            }

    def publish_arm_command(self, arm, positions, velocity):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        snapshot = self.get_arm_snapshot(arm)
        joint_names = snapshot["name"] or [f"{arm}_joint_{idx}" for idx in range(len(positions))]
        msg.name = joint_names
        msg.position = [float(value) for value in positions]
        msg.velocity = [float(velocity)]

        publish_ns = now_ns()
        with self._lock:
            self._command_seq += 1
            command_id = self._command_seq

        if arm == "left":
            self.left_arm_pub.publish(msg)
        else:
            self.right_arm_pub.publish(msg)

        return {
            "command_id": command_id,
            "publish_ns": publish_ns,
            "arm": arm,
            "positions": list(msg.position),
            "velocity": velocity,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone ROS2 arm latency measurement for X100 teleoperation."
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        default="left",
        help="Arm to measure.",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=0,
        help="0-based joint index on the selected arm.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.25,
        help="Step amplitude in radians.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.5,
        help="Nominal square-wave period in seconds.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=20,
        help="Number of high-low cycles to run.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=2.0,
        help="Seconds to wait after startup before the first step.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Per-step timeout in seconds.",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=30.0,
        help="How often to republish the current target in Hz.",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.0,
        help="Velocity value written into JointState.velocity.",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=5.0,
        help="How long to wait for the first arm state in seconds.",
    )
    parser.add_argument(
        "--log-dir",
        default="teleop/latency_logs",
        help="Directory used to store CSV and summary JSON outputs.",
    )
    return parser.parse_args()


def wait_for_initial_state(node, arm, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = node.get_arm_snapshot(arm)
        if snapshot["seq"] > 0 and snapshot["position"]:
            return snapshot
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {arm} arm state.")


def build_step_result(step, timeout=False):
    return {
        "step_id": step["step_id"],
        "command_id": step["command_id"],
        "phase": step["phase"],
        "start_joint_value": step["start_joint_value"],
        "target_joint_value": step["target_joint_value"],
        "latency_10_ms": step["threshold_hits"].get("10"),
        "latency_50_ms": step["threshold_hits"].get("50"),
        "latency_90_ms": step["threshold_hits"].get("90"),
        "settled_ms": step.get("settled_ms"),
        "timeout": timeout,
    }


def maybe_record_state_progress(step, snapshot, joint_index, logger):
    events = []
    state_seq = snapshot["seq"]
    if state_seq <= step["last_state_seq"]:
        return events

    step["last_state_seq"] = state_seq
    current_joint = float(snapshot["position"][joint_index])
    target_delta = step["target_joint_value"] - step["start_joint_value"]
    if abs(target_delta) <= 1e-9:
        return events

    progress = abs(current_joint - step["start_joint_value"]) / abs(target_delta)
    progress = max(0.0, progress)

    for threshold in (0.1, 0.5, 0.9):
        threshold_key = str(int(threshold * 100))
        if threshold_key in step["threshold_hits"]:
            continue
        if progress >= threshold:
            latency_ms = ns_to_ms(snapshot["callback_ns"] - step["command_ns"])
            step["threshold_hits"][threshold_key] = latency_ms
            logger.log_event(
                "threshold_hit",
                step_id=step["step_id"],
                command_id=step["command_id"],
                state_seq=state_seq,
                threshold=threshold,
                latency_ms=latency_ms,
                joint_value=current_joint,
                progress=progress,
            )

    settle_tolerance = max(0.01, abs(target_delta) * 0.05)
    if (
        step.get("settled_ms") is None
        and abs(current_joint - step["target_joint_value"]) <= settle_tolerance
    ):
        step["settled_ms"] = ns_to_ms(snapshot["callback_ns"] - step["command_ns"])
        events.append(build_step_result(step, timeout=False))

    return events


def main():
    args = parse_args()
    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than 0.")
    if args.publish_rate <= 0:
        raise ValueError("--publish-rate must be greater than 0.")
    if args.period <= 0:
        raise ValueError("--period must be greater than 0.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")

    logger = CsvEventLogger(args.log_dir, f"{args.arm}_arm_latency")
    logger.log_event("session_start", config=vars(args))

    node = None
    spin_thread = None
    rclpy.init(args=None)
    node = ArmLatencyNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    results = []
    step = None
    try:
        initial_snapshot = wait_for_initial_state(node, args.arm, args.state_timeout)
        num_joints = len(initial_snapshot["position"])
        if not 0 <= args.joint < num_joints:
            raise ValueError(
                f"--joint must be in [0, {num_joints - 1}] for the current {args.arm} arm state."
            )

        baseline = np.asarray(initial_snapshot["position"], dtype=float)
        low_target = baseline.copy()
        high_target = baseline.copy()
        high_target[args.joint] += args.amplitude
        current_target = low_target.copy()

        half_period_ns = max(1, int(args.period * 1_000_000_000 / 2.0))
        publish_interval_ns = max(1, int(1_000_000_000 / args.publish_rate))
        timeout_ns = int(args.timeout * 1_000_000_000)
        next_switch_earliest_ns = now_ns() + int(args.settle_time * 1_000_000_000)
        next_publish_ns = now_ns()
        transitions_target = args.cycles * 2
        transition_count = 0

        logger.log_event(
            "initialized",
            arm=args.arm,
            joint=args.joint,
            baseline_joint_value=float(baseline[args.joint]),
            high_joint_value=float(high_target[args.joint]),
            num_joints=num_joints,
            csv_path=str(logger.csv_path),
            summary_path=str(logger.summary_path),
        )

        print(
            f"[Latency] start: arm={args.arm}, joint={args.joint}, "
            f"baseline={baseline[args.joint]:.6f}, target={high_target[args.joint]:.6f}"
        )

        while True:
            current_ns = now_ns()
            snapshot = node.get_arm_snapshot(args.arm)
            if snapshot["seq"] <= 0 or not snapshot["position"]:
                time.sleep(0.002)
                continue

            if step is not None:
                finished = maybe_record_state_progress(step, snapshot, args.joint, logger)
                if finished:
                    result = finished[0]
                    results.append(result)
                    logger.log_event(
                        "step_complete",
                        step_id=result["step_id"],
                        command_id=result["command_id"],
                        state_seq=snapshot["seq"],
                        **result,
                    )
                    print(
                        f"[Latency] step {result['step_id']} done: "
                        f"50%={result['latency_50_ms']} ms, "
                        f"90%={result['latency_90_ms']} ms, "
                        f"settled={result['settled_ms']} ms"
                    )
                    step = None
                elif current_ns - step["command_ns"] > timeout_ns:
                    result = build_step_result(step, timeout=True)
                    results.append(result)
                    logger.log_event(
                        "step_timeout",
                        step_id=result["step_id"],
                        command_id=result["command_id"],
                        state_seq=snapshot["seq"],
                        **result,
                    )
                    print(f"[Latency] step {result['step_id']} timeout")
                    step = None

            if step is None and transition_count >= transitions_target:
                break

            if step is None and current_ns >= next_switch_earliest_ns:
                if np.array_equal(current_target, low_target):
                    current_target = high_target.copy()
                    phase = "high"
                else:
                    current_target = low_target.copy()
                    phase = "low"

                transition_count += 1
                command_record = node.publish_arm_command(
                    args.arm,
                    current_target,
                    args.velocity,
                )
                next_publish_ns = command_record["publish_ns"] + publish_interval_ns
                next_switch_earliest_ns = command_record["publish_ns"] + half_period_ns
                step = {
                    "step_id": transition_count,
                    "command_id": command_record["command_id"],
                    "command_ns": command_record["publish_ns"],
                    "phase": phase,
                    "start_joint_value": float(snapshot["position"][args.joint]),
                    "target_joint_value": float(current_target[args.joint]),
                    "threshold_hits": {},
                    "settled_ms": None,
                    "last_state_seq": snapshot["seq"] - 1,
                }
                logger.log_event(
                    "command_sent",
                    step_id=step["step_id"],
                    command_id=step["command_id"],
                    state_seq=snapshot["seq"],
                    phase=phase,
                    target_positions=current_target,
                    start_joint_value=step["start_joint_value"],
                    target_joint_value=step["target_joint_value"],
                    publish_ns=command_record["publish_ns"],
                )
                print(
                    f"[Latency] step {step['step_id']} command sent: "
                    f"{step['start_joint_value']:.6f} -> {step['target_joint_value']:.6f}"
                )

            if current_ns >= next_publish_ns:
                node.publish_arm_command(args.arm, current_target, args.velocity)
                next_publish_ns = current_ns + publish_interval_ns

            time.sleep(0.002)

        successful = [result for result in results if not result.get("timeout")]
        summary = {
            "arm": args.arm,
            "joint": args.joint,
            "amplitude": args.amplitude,
            "cycles": args.cycles,
            "period_s": args.period,
            "num_steps": len(results),
            "num_success": len(successful),
            "num_timeout": len(results) - len(successful),
            "latency_10_ms": percentile_summary(
                [result["latency_10_ms"] for result in successful if result["latency_10_ms"] is not None]
            ),
            "latency_50_ms": percentile_summary(
                [result["latency_50_ms"] for result in successful if result["latency_50_ms"] is not None]
            ),
            "latency_90_ms": percentile_summary(
                [result["latency_90_ms"] for result in successful if result["latency_90_ms"] is not None]
            ),
            "settled_ms": percentile_summary(
                [result["settled_ms"] for result in successful if result["settled_ms"] is not None]
            ),
            "results": results,
        }
        logger.log_event("session_summary", summary=summary)
        logger.write_summary(summary)

        print("[Latency] summary")
        for key in ("latency_10_ms", "latency_50_ms", "latency_90_ms", "settled_ms"):
            metric = summary[key]
            if metric is None:
                continue
            print(
                f"  {key}: mean={metric['mean']} ms, p50={metric['p50']} ms, "
                f"p90={metric['p90']} ms, p99={metric['p99']} ms, max={metric['max']} ms"
            )
        print(f"  csv: {logger.csv_path}")
        print(f"  summary: {logger.summary_path}")

    except Exception as exc:
        logger.log_event("session_error", error=str(exc))
        raise
    finally:
        logger.close()
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
