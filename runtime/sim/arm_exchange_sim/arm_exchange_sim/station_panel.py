from __future__ import annotations

import curses
from dataclasses import dataclass

import rclpy
from arm_exchange_interfaces.msg import ExchangeStationState
from rclpy.node import Node


@dataclass(frozen=True)
class FieldSpec:
    name: str
    lower: float
    upper: float
    step: float


FIELDS = (
    FieldSpec("x", -0.1, 0.0, 0.005),
    FieldSpec("y", 0.1, 0.3, 0.005),
    FieldSpec("z", 0.5, 0.7, 0.005),
    FieldSpec("alpha", -0.7854, 0.7854, 0.01),
    FieldSpec("theta", -1.5708, 1.5708, 0.01),
    FieldSpec("phi", 0.0, 1.5708, 0.01),
)
FIELD_FIRST_ROW = 5

DEFAULT_STATE = {
    "x": -0.05,
    "y": 0.2,
    "z": 0.6,
    "alpha": 0.0,
    "theta": 0.0,
    "phi": 0.0,
}


class StationPanelNode(Node):
    def __init__(self):
        super().__init__("station_panel")
        self.state_pub = self.create_publisher(
            ExchangeStationState,
            "/debug/scene/exchange_station/set_state",
            10,
        )

    def publish_station_state(self, values: dict[str, float]) -> None:
        msg = ExchangeStationState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mujoco_world"
        for field in FIELDS:
            setattr(msg, field.name, float(values[field.name]))
        self.state_pub.publish(msg)


class StationPanel:
    def __init__(self, node: StationPanelNode, screen):
        self.node = node
        self.screen = screen
        self.values = dict(DEFAULT_STATE)
        self.selected = 0
        self.step_scale = 1.0
        self.status = "Ready"
        self.publish_button_rect = (0, 0, 0, 0)

    def run(self) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        self.screen.keypad(True)
        self.screen.timeout(50)
        while rclpy.ok():
            self._draw()
            key = self.screen.getch()
            if key == -1:
                rclpy.spin_once(self.node, timeout_sec=0.0)
                continue
            if key in (ord("q"), ord("Q")):
                return
            self._handle_key(key)
            rclpy.spin_once(self.node, timeout_sec=0.0)

    def _handle_key(self, key: int) -> None:
        if key == curses.KEY_UP:
            self.selected = max(0, self.selected - 1)
        elif key == curses.KEY_DOWN:
            self.selected = min(len(FIELDS) - 1, self.selected + 1)
        elif key == curses.KEY_LEFT:
            self._adjust_selected(-1.0)
        elif key == curses.KEY_RIGHT:
            self._adjust_selected(1.0)
        elif key in (ord("["), ord("{")):
            self.step_scale = max(0.1, self.step_scale / 2.0)
        elif key in (ord("]"), ord("}")):
            self.step_scale = min(10.0, self.step_scale * 2.0)
        elif key in (ord("r"), ord("R")):
            self.values = dict(DEFAULT_STATE)
            self.status = "Reset to default values"
        elif key == ord(" "):
            self._publish()
        elif key == curses.KEY_MOUSE:
            self._handle_mouse()

    def _handle_mouse(self) -> None:
        _, x, y, _, event = curses.getmouse()
        if not (event & curses.BUTTON1_PRESSED):
            return
        first_row = FIELD_FIRST_ROW
        last_row = first_row + len(FIELDS) - 1
        if first_row <= y <= last_row:
            self.selected = y - first_row
            return
        x0, y0, x1, y1 = self.publish_button_rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            self._publish()

    def _adjust_selected(self, direction: float) -> None:
        field = FIELDS[self.selected]
        value = self.values[field.name] + direction * field.step * self.step_scale
        self.values[field.name] = min(field.upper, max(field.lower, value))
        self.status = f"{field.name}={self.values[field.name]:.4f}"

    def _publish(self) -> None:
        self.node.publish_station_state(self.values)
        self.status = "Published ExchangeStationState"

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 18 or width < 76:
            self.screen.addstr(0, 0, "Terminal too small. Need at least 76x18.")
            self.screen.refresh()
            return

        self.screen.addstr(0, 0, "Arm Exchange Station Panel", curses.A_BOLD)
        self.screen.addstr(
            1,
            0,
            "Arrows: up/down select field; left/right adjust its local value.",
        )
        self.screen.addstr(
            2,
            0,
            "Space: publish current values to simulation; Enter: unused.",
        )
        self.screen.addstr(3, 0, "[ / ]: step scale; R: reset values; Q: quit panel.")
        self.screen.addstr(4, 0, "Field        Value        Range                 Step")

        for index, field in enumerate(FIELDS):
            row = FIELD_FIRST_ROW + index
            selected = index == self.selected
            prefix = ">" if selected else " "
            style = curses.A_REVERSE if selected else curses.A_NORMAL
            text = (
                f"{prefix} {field.name:<8} "
                f"{self.values[field.name]:>9.4f}   "
                f"[{field.lower:>7.4f}, {field.upper:>7.4f}]   "
                f"{field.step * self.step_scale:>7.4f}"
            )
            self.screen.addstr(row, 0, text, style)

        button_row = FIELD_FIRST_ROW + len(FIELDS) + 2
        button_text = "[ Space: Publish ]"
        self.publish_button_rect = (0, button_row, len(button_text) - 1, button_row)
        self.screen.addstr(button_row, 0, button_text, curses.A_BOLD)
        self.screen.addstr(button_row, 20, "[ R: Reset ]   [ Q: Quit ]")

        status_row = button_row + 2
        self.screen.addstr(status_row, 0, f"step_scale={self.step_scale:.2f}  status: {self.status}")
        self.screen.addstr(status_row + 1, 0, "Topic: /debug/scene/exchange_station/set_state")
        self.screen.refresh()


def _run_panel(screen, node: StationPanelNode) -> None:
    StationPanel(node, screen).run()


def main(args=None):
    rclpy.init(args=args)
    node = StationPanelNode()
    try:
        curses.wrapper(_run_panel, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
