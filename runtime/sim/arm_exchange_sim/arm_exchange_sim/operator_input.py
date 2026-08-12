from __future__ import annotations

import os
import select
import struct

import rclpy
from arm_exchange_interfaces.msg import OperatorInputState
from rclpy.node import Node


EV_KEY = 0x01
KEY_RELEASE = 0
KEY_PRESS = 1
KEY_REPEAT = 2
EXIT_KEY_CODE = 45

INPUT_EVENT = struct.Struct("llHHI")

KEY_CODE_MAP = {
    1: "esc",
    2: "key_1",
    3: "key_2",
    4: "key_3",
    5: "key_4",
    6: "key_5",
    15: "tab",
    16: "q",
    17: "w",
    18: "e",
    23: "i",
    28: "enter",
    30: "a",
    31: "s",
    32: "d",
    36: "j",
    37: "k",
    38: "l",
    46: "c",
    35: "h",
}


class OperatorInputNode(Node):
    def __init__(self):
        super().__init__("operator_input_node")
        self.declare_parameter("keyboard_device", "")
        self.declare_parameter("publish_rate", 100.0)
        self.keyboard_device = str(self.get_parameter("keyboard_device").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        if not self.keyboard_device:
            raise ValueError("operator_input_node requires keyboard_device=/dev/input/eventX")
        if publish_rate <= 0.0:
            raise ValueError("operator_input_node publish_rate must be positive")

        self.state_pub = self.create_publisher(
            OperatorInputState,
            "/operator/input_state",
            10,
        )
        self.keyboard_fd = os.open(self.keyboard_device, os.O_RDONLY | os.O_NONBLOCK)
        self.pressed: set[str] = set()
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_input_state)
        self.get_logger().info(
            f"operator input snapshot: keyboard={self.keyboard_device}; "
            f"publish_rate={publish_rate:.1f}Hz"
        )

    def _publish_input_state(self) -> None:
        self._poll_keyboard()

        msg = OperatorInputState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "operator_input"
        for key in KEY_CODE_MAP.values():
            setattr(msg, key, key in self.pressed)
        self.state_pub.publish(msg)

    def _poll_keyboard(self) -> None:
        while select.select([self.keyboard_fd], [], [], 0.0)[0]:
            payload = os.read(self.keyboard_fd, INPUT_EVENT.size)
            if len(payload) != INPUT_EVENT.size:
                raise RuntimeError(f"incomplete input_event read: {len(payload)} bytes")
            _, _, event_type, code, value = INPUT_EVENT.unpack(payload)
            if event_type != EV_KEY:
                continue
            if code == EXIT_KEY_CODE and value == KEY_PRESS:
                raise KeyboardInterrupt
            key = KEY_CODE_MAP.get(code)
            if key is None:
                continue
            if value == KEY_PRESS:
                self.pressed.add(key)
            elif value == KEY_RELEASE:
                self.pressed.discard(key)
            elif value == KEY_REPEAT:
                continue
            else:
                raise ValueError(f"unexpected EV_KEY value: {value}")

    def close(self) -> None:
        os.close(self.keyboard_fd)


def main(args=None):
    rclpy.init(args=args)
    node = OperatorInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
