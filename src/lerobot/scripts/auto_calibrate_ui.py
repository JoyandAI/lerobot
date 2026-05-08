#!/usr/bin/env python3

"""PyQt-based UI for SO auto calibration."""

from __future__ import annotations

import glob
import inspect
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from serial.tools import list_ports

from lerobot.motors import auto_calibrate as auto_calibrate_module
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import make_robot_from_config, so101_follower  # noqa: F401
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators import make_teleoperator_from_config, so101_leader  # noqa: F401
from lerobot.teleoperators.so101_leader import SO101LeaderConfig

try:
    from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal as Signal
    from PyQt5.QtGui import QColor, QFont, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    QT_LIB = "PyQt5"
except ImportError:
    from PySide6.QtCore import QObject, QThread, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    QT_LIB = "PySide6"


logger = logging.getLogger(__name__)

AutoCalibrateConfig = auto_calibrate_module.AutoCalibrateConfig
auto_calibrate_connected_device = auto_calibrate_module.auto_calibrate_connected_device
explore_literal_limit = auto_calibrate_module.explore_literal_limit
get_joint_behavior = auto_calibrate_module.get_joint_behavior
is_calibration_paused = getattr(auto_calibrate_module, "is_calibration_paused", lambda: False)
set_calibration_paused = getattr(auto_calibrate_module, "set_calibration_paused", lambda paused: None)

IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
LINUX_PORT_PREFIX = "/dev/ttyACM"

GRIPPER_CHECK_TORQUE = 300
GRIPPER_CHECK_VELOCITY = 300

DEVICE_CONFIG_FACTORIES = {
    "tele": SO101LeaderConfig,
    "robot": SO101FollowerConfig,
}

DEVICE_FACTORIES = {
    "tele": make_teleoperator_from_config,
    "robot": make_robot_from_config,
}

AUTO_CALIBRATION_DEFAULTS = {
    "tele": {
        "try_torque": 400,
        "max_torque": 500,
        "torque_step": 50,
        "explore_velocity": 600,
        "wait_time_s": 0.5,
        "velocity_threshold": 4,
        "position_tolerance": 4000,
    },
    "robot": {
        "try_torque": 600,
        "max_torque": 1000,
        "torque_step": 50,
        "explore_velocity": 800,
        "wait_time_s": 0.5,
        "velocity_threshold": 4,
        "position_tolerance": 4000,
    },
}

STATUS_IDLE = "空闲中"
STATUS_PORT_DETECTED = "检测到有外接端口"
STATUS_CALIBRATING = "标定中"
STATUS_PAUSED = "标定已暂停"
STATUS_FINISHED = "标定完毕，请拔掉USB"
STATUS_FAILED = "标定失败"
STATUS_AUTHORIZING = "端口授权中"
STATUS_CHECKING_ARM = "机械臂检测中"

ACTIVE_STATUSES = {STATUS_IDLE, STATUS_PORT_DETECTED}
BUSY_STATUSES = {STATUS_CALIBRATING, STATUS_PAUSED, STATUS_AUTHORIZING, STATUS_CHECKING_ARM}
TERMINAL_STATUSES = {STATUS_FINISHED, STATUS_FAILED}


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str

    @property
    def label(self) -> str:
        return f"{self.device}  |  {self.description}"


class LogEmitter(QObject):
    message_emitted = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.emitter = LogEmitter()

    def emit(self, record: logging.LogRecord):
        self.emitter.message_emitted.emit(self.format(record))


def run_auto_calibration_for_ui(device, config: AutoCalibrateConfig, output_path: Path) -> Path:
    """Run auto calibration across old/new helper signatures and return the saved file path."""

    calibration_func = auto_calibrate_connected_device
    save_calibration_to_file = getattr(auto_calibrate_module, "save_calibration_to_file", None)

    try:
        parameters = inspect.signature(calibration_func).parameters
    except (TypeError, ValueError):
        parameters = {"calibration_path": None}

    if "calibration_path" in parameters:
        result = calibration_func(device, config, calibration_path=output_path)
    elif "save" in parameters:
        result = calibration_func(device, config, save=False)
    else:
        result = calibration_func(device, config)

    saved_path = getattr(result, "calibration_path", None)
    calibration_dict = getattr(result, "calibration_dict", None)
    if calibration_dict is None and isinstance(result, dict):
        calibration_dict = result

    if saved_path is not None and Path(saved_path) == output_path:
        return output_path

    if calibration_dict is not None and save_calibration_to_file is not None:
        save_calibration_to_file(calibration_dict, output_path)
        return output_path

    if saved_path is not None:
        return Path(saved_path)

    return output_path


class WorkerBase(QObject):
    finished = Signal()
    succeeded = Signal(str)
    failed = Signal(str)
    status_changed = Signal(str, str)


class CalibrationWorker(WorkerBase):
    def __init__(self, device_type: str, port: str, filename: str):
        super().__init__()
        self.device_type = device_type
        self.port = port
        self.filename = filename

    def run(self):
        device = None
        try:
            self.status_changed.emit(STATUS_CALIBRATING, f"正在连接 {self.port} 并开始标定。")
            config_kwargs = {
                "port": self.port,
                "id": Path(self.filename).stem or "my_so101",
            }
            device_config = DEVICE_CONFIG_FACTORIES[self.device_type](**config_kwargs)
            auto_calib_config = AutoCalibrateConfig(
                robot=device_config,
                **AUTO_CALIBRATION_DEFAULTS[self.device_type],
            )

            device = DEVICE_FACTORIES[self.device_type](device_config)
            device.connect(calibrate=False)

            output_path = Path(device.calibration_fpath).with_name(self.ensure_json_suffix(self.filename))
            final_path = run_auto_calibration_for_ui(device, auto_calib_config, output_path)
            self.status_changed.emit(STATUS_FINISHED, f"标定完成，文件已保存到：{final_path}")
            self.succeeded.emit(str(final_path))
        except Exception as error:
            logger.exception("Auto calibration failed.")
            self.status_changed.emit(STATUS_FAILED, str(error))
            self.failed.emit(str(error))
        finally:
            if device is not None:
                try:
                    device.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect calibration device cleanly.")
            self.finished.emit()

    @staticmethod
    def ensure_json_suffix(filename: str) -> str:
        cleaned = filename.strip() or "my_so101"
        return cleaned if cleaned.lower().endswith(".json") else f"{cleaned}.json"


class LinuxPermissionWorker(WorkerBase):
    def __init__(self, password: str):
        super().__init__()
        self.password = password

    def run(self):
        try:
            self.status_changed.emit(STATUS_AUTHORIZING, "正在给 /dev/ttyACM* 设置读写权限。")
            matched_ports = sorted(glob.glob(f"{LINUX_PORT_PREFIX}*"))
            if not matched_ports:
                raise RuntimeError("未找到 /dev/ttyACM* 设备，请先连接机械臂。")

            result = subprocess.run(
                ["sudo", "-S", "/bin/sh", "-c", "chmod 666 /dev/ttyACM*"],
                input=f"{self.password}\n",
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                error = (result.stderr or result.stdout).strip()
                raise RuntimeError(error or "端口授权失败，请确认密码是否正确。")

            inaccessible = [port for port in matched_ports if not os.access(port, os.R_OK | os.W_OK)]
            if inaccessible:
                raise RuntimeError(f"以下端口仍不可读写：{', '.join(inaccessible)}")

            message = f"端口授权成功：{', '.join(matched_ports)}"
            self.status_changed.emit(STATUS_PORT_DETECTED, message)
            self.succeeded.emit(message)
        except Exception as error:
            logger.exception("Failed to grant Linux port permissions.")
            self.status_changed.emit(STATUS_FAILED, str(error))
            self.failed.emit(str(error))
        finally:
            self.finished.emit()


class ArmCheckWorker(WorkerBase):
    def __init__(self, device_type: str, port: str):
        super().__init__()
        self.device_type = device_type
        self.port = port

    def run(self):
        device = None
        try:
            self.status_changed.emit(STATUS_CHECKING_ARM, f"正在检测机械臂，端口：{self.port}")
            device_config = DEVICE_CONFIG_FACTORIES[self.device_type](port=self.port, id="arm_check")
            device = DEVICE_FACTORIES[self.device_type](device_config)
            device.connect(calibrate=False)

            bus = getattr(device, "bus", None)
            if bus is None:
                raise RuntimeError("当前设备未暴露 bus，无法执行机械臂检测。")

            motor_name = self.find_gripper_motor_name(bus)
            self.move_motor_to_limits(bus, motor_name)

            message = f"{motor_name} 已完成双向边界检测。"
            self.status_changed.emit(STATUS_PORT_DETECTED, message)
            self.succeeded.emit(message)
        except Exception as error:
            logger.exception("Arm check failed.")
            self.status_changed.emit(STATUS_FAILED, str(error))
            self.failed.emit(str(error))
        finally:
            if device is not None:
                try:
                    device.disconnect()
                except Exception:
                    logger.exception("Failed to disconnect after arm check.")
            self.finished.emit()

    @staticmethod
    def find_gripper_motor_name(bus) -> str:
        if "gripper" in bus.motors:
            return "gripper"
        for motor_name, motor in bus.motors.items():
            if getattr(motor, "id", None) == 6:
                return motor_name
        raise RuntimeError("未找到 6 号舵机或 gripper 电机。")

    @staticmethod
    def move_motor_to_limits(bus, motor_name: str):
        original_mode = bus.read("Operating_Mode", motor_name, normalize=False)
        original_torque_limit = bus.read("Torque_Limit", motor_name, normalize=False)
        original_torque_enable = bus.read("Torque_Enable", motor_name, normalize=False)

        behavior = get_joint_behavior(motor_name)
        check_config = AutoCalibrateConfig(
            robot=None,
            try_torque=GRIPPER_CHECK_TORQUE,
            max_torque=GRIPPER_CHECK_TORQUE,
            torque_step=50,
            explore_velocity=GRIPPER_CHECK_VELOCITY,
            wait_time_s=0.2,
            velocity_threshold=4,
            position_tolerance=4000,
        )

        try:
            bus.write("Operating_Mode", motor_name, OperatingMode.VELOCITY.value, normalize=False)
            bus.write("Torque_Limit", motor_name, GRIPPER_CHECK_TORQUE, normalize=False)
            bus.write("Torque_Enable", motor_name, 1, normalize=False)

            logger.info("Arm check: exploring first limit for %s", motor_name)
            explore_literal_limit(bus, motor_name, behavior.first_direction, check_config)
            time.sleep(check_config.wait_time_s)
            logger.info("Arm check: exploring reverse limit for %s", motor_name)
            explore_literal_limit(bus, motor_name, behavior.second_direction, check_config)
        finally:
            try:
                bus.write("Goal_Velocity", motor_name, 0, normalize=False)
            finally:
                bus.write("Operating_Mode", motor_name, original_mode, normalize=False)
                bus.write("Torque_Limit", motor_name, original_torque_limit, normalize=False)
                bus.write("Torque_Enable", motor_name, original_torque_enable, normalize=False)


class AutoCalibrateWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread: QThread | None = None
        self.worker: WorkerBase | None = None
        self.current_status = STATUS_IDLE
        self.last_output_path = ""
        self._log_handler: QtLogHandler | None = None

        self.setWindowTitle("SO 自动标定")
        self.resize(1040, 760)
        self.setMinimumSize(920, 680)

        self._setup_logging()
        self._build_ui()
        self._apply_styles()
        self._setup_port_timer()
        self.refresh_ports()

    def _setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        self._log_handler = QtLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        )
        self._log_handler.emitter.message_emitted.connect(self.append_log)
        root_logger.addHandler(self._log_handler)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(18)

        header_layout = QVBoxLayout()
        title_label = QLabel("自动标定控制台")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(
            f"基于 {QT_LIB} 的标定页面，支持串口检测、Linux 端口授权、机械臂检测和自动标定。"
        )
        subtitle_label.setObjectName("subtitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(18)
        main_layout.addLayout(top_layout)

        config_group = QGroupBox("标定配置")
        config_layout = QGridLayout(config_group)
        config_layout.setHorizontalSpacing(14)
        config_layout.setVerticalSpacing(14)

        self.device_type_combo = QComboBox()
        self.device_type_combo.addItem("tele", "tele")
        self.device_type_combo.addItem("robot", "robot")
        self.device_type_combo.currentIndexChanged.connect(self.on_device_type_changed)

        self.port_combo = QComboBox()
        self.port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.port_combo.currentIndexChanged.connect(self.on_port_changed)

        self.refresh_button = QPushButton("刷新串口")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.clicked.connect(self.refresh_ports)

        self.filename_input = QLineEdit("my_so101")
        self.filename_input.setPlaceholderText("例如：tele_calibration 或 robot_01.json")
        self.filename_input.textChanged.connect(self.on_filename_changed)

        filename_hint = QLabel("只修改文件名，不修改保存目录。最终仍保存到设备默认 calibration 路径。")
        filename_hint.setObjectName("hintLabel")

        config_layout.addWidget(QLabel("设备类型"), 0, 0)
        config_layout.addWidget(self.device_type_combo, 0, 1, 1, 2)
        config_layout.addWidget(QLabel("串口"), 1, 0)
        config_layout.addWidget(self.port_combo, 1, 1)
        config_layout.addWidget(self.refresh_button, 1, 2)
        config_layout.addWidget(QLabel("输出文件名"), 2, 0)
        config_layout.addWidget(self.filename_input, 2, 1, 1, 2)
        config_layout.addWidget(filename_hint, 3, 0, 1, 3)

        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(12)

        self.status_badge = QLabel(STATUS_IDLE)
        self.status_badge.setObjectName("statusBadge")

        self.status_detail_label = QLabel("等待检测串口。")
        self.status_detail_label.setWordWrap(True)
        self.status_detail_label.setObjectName("statusDetailLabel")

        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(10)
        info_layout.addWidget(self.status_badge)
        info_layout.addWidget(self.status_detail_label)

        self.output_hint_label = QLabel("输出路径：未生成")
        self.output_hint_label.setWordWrap(True)
        self.output_hint_label.setObjectName("outputHintLabel")

        status_layout.addWidget(info_card)
        status_layout.addWidget(self.output_hint_label)
        status_layout.addStretch(1)

        top_layout.addWidget(config_group, 3)
        top_layout.addWidget(status_group, 2)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        main_layout.addLayout(button_layout)

        self.start_button = QPushButton("开始标定")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_calibration)

        self.pause_button = QPushButton("暂停标定")
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.clicked.connect(self.toggle_pause_calibration)

        self.recalibrate_button = QPushButton("重新标定")
        self.recalibrate_button.setObjectName("secondaryButton")
        self.recalibrate_button.clicked.connect(self.restart_calibration)

        self.arm_check_button = QPushButton("检测机械臂")
        self.arm_check_button.setObjectName("secondaryButton")
        self.arm_check_button.clicked.connect(self.start_arm_check)

        self.permission_button = QPushButton("Linux 端口授权")
        self.permission_button.setObjectName("secondaryButton")
        self.permission_button.clicked.connect(self.grant_linux_permissions)
        self.permission_button.setVisible(IS_LINUX)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.recalibrate_button)
        button_layout.addWidget(self.arm_check_button)
        if IS_LINUX:
            button_layout.addWidget(self.permission_button)
        button_layout.addStretch(1)

        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(4000)
        self.log_output.setObjectName("logOutput")
        self.log_output.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_output)
        main_layout.addWidget(log_group, 1)

        self.append_log(f"UI 已启动，正在检测本机可用串口。当前图形库：{QT_LIB}")
        self.update_action_buttons()
        self.update_output_preview()

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f1ea;
                color: #1f2933;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #d5ccc1;
                border-radius: 16px;
                margin-top: 12px;
                padding-top: 16px;
                background: #fbfaf7;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #6f4e37;
            }
            QLabel#titleLabel {
                font-size: 28px;
                font-weight: 700;
                color: #173f35;
            }
            QLabel#subtitleLabel {
                color: #5b6b73;
                font-size: 13px;
            }
            QLabel#hintLabel, QLabel#outputHintLabel, QLabel#statusDetailLabel {
                color: #5f6b66;
            }
            QLineEdit, QComboBox, QPushButton {
                min-height: 40px;
                border-radius: 10px;
                border: 1px solid #cfc4b7;
                padding: 0 12px;
                background: #ffffff;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #1f8a70;
            }
            QPushButton {
                background: #fffaf4;
                font-weight: 600;
                color: #344054;
            }
            QPushButton:hover {
                border: 1px solid #1f8a70;
                background: #f5efe6;
            }
            QPushButton:disabled {
                background: #ddd6cb;
                color: #8a847a;
                border: 1px solid #c7beb3;
            }
            QPushButton#primaryButton {
                background: #1f8a70;
                color: white;
                border: 1px solid #1f8a70;
            }
            QPushButton#primaryButton:hover {
                background: #19715c;
            }
            QPushButton#primaryButton:disabled {
                background: #9fcabc;
                color: #eef7f3;
                border: 1px solid #9fcabc;
            }
            QPushButton#pauseButton {
                background: #fff3e8;
                color: #9a3412;
                border: 1px solid #fdba74;
            }
            QPushButton#pauseButton:hover {
                background: #ffedd5;
                border: 1px solid #fb923c;
            }
            QPushButton#pauseButton:disabled {
                background: #f1e3d5;
                color: #b9a18d;
                border: 1px solid #e3d1bf;
            }
            QPushButton#secondaryButton {
                background: #f8f5ef;
                color: #475467;
                border: 1px solid #d7cec2;
            }
            QPushButton#secondaryButton:hover {
                background: #f1ece4;
                border: 1px solid #b9ab98;
            }
            QPushButton#secondaryButton:disabled {
                background: #e7e1d8;
                color: #9b948a;
                border: 1px solid #d1c8bc;
            }
            QFrame#infoCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #f7efe6, stop: 1 #eef6f2
                );
                border-radius: 18px;
                border: 1px solid #dfd6ca;
            }
            QLabel#statusBadge {
                font-size: 22px;
                font-weight: 700;
                color: #173f35;
            }
            QPlainTextEdit#logOutput {
                background: #18232d;
                color: #d8efe6;
                border-radius: 14px;
                border: 1px solid #25333f;
                padding: 12px;
                selection-background-color: #2e6f62;
            }
            """
        )

    def _setup_port_timer(self):
        self.port_timer = QTimer(self)
        self.port_timer.setInterval(1500)
        self.port_timer.timeout.connect(self.refresh_ports)
        self.port_timer.start()

    def closeEvent(self, event):
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)

    def append_log(self, message: str):
        self.log_output.appendPlainText(message)
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_output.setTextCursor(cursor)

    def set_status(self, status: str, detail: str):
        self.current_status = status
        self.status_badge.setText(status)
        self.status_detail_label.setText(detail)
        self.update_status_badge_color(status)
        self.update_action_buttons()

    def update_status_badge_color(self, status: str):
        color_map = {
            STATUS_IDLE: "#6b7280",
            STATUS_PORT_DETECTED: "#1d7a5f",
            STATUS_CALIBRATING: "#a05a00",
            STATUS_PAUSED: "#b45309",
            STATUS_FINISHED: "#0f766e",
            STATUS_FAILED: "#b42318",
            STATUS_AUTHORIZING: "#7c3aed",
            STATUS_CHECKING_ARM: "#2563eb",
        }
        self.status_badge.setStyleSheet(f"color: {QColor(color_map.get(status, '#173f35')).name()};")

    def on_device_type_changed(self):
        suggested_name = "tele_calibration" if self.selected_device_type() == "tele" else "robot_calibration"
        if not self.filename_input.text().strip() or self.filename_input.text() in {
            "my_so101",
            "tele_calibration",
            "robot_calibration",
        }:
            self.filename_input.setText(suggested_name)
        self.update_output_preview()
        self.update_action_buttons()

    def on_port_changed(self):
        self.update_output_preview()
        self.update_action_buttons()

    def on_filename_changed(self):
        self.update_output_preview()
        self.update_action_buttons()

    def list_available_ports(self) -> list[SerialPortInfo]:
        ports: list[SerialPortInfo] = []
        for port in list_ports.comports():
            device_name = str(port.device)
            if IS_WINDOWS and not device_name.upper().startswith("COM"):
                continue
            if IS_LINUX and LINUX_PORT_PREFIX not in device_name:
                continue
            ports.append(SerialPortInfo(device=device_name, description=port.description or "Unknown device"))
        ports.sort(key=lambda item: item.device)
        return ports

    def refresh_ports(self):
        if self.current_status in BUSY_STATUSES:
            return

        current_port = self.selected_port()
        ports = self.list_available_ports()

        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for port in ports:
            self.port_combo.addItem(port.label, port.device)

        if ports:
            target_index = 0
            if current_port:
                for index, port in enumerate(ports):
                    if port.device == current_port:
                        target_index = index
                        break
            self.port_combo.setCurrentIndex(target_index)
        self.port_combo.blockSignals(False)

        if self.current_status not in TERMINAL_STATUSES:
            if ports:
                self.set_status(STATUS_PORT_DETECTED, f"当前检测到 {len(ports)} 个可用串口。")
            else:
                if IS_LINUX:
                    self.set_status(STATUS_IDLE, "未检测到 /dev/ttyACM* 串口，请连接设备后重试。")
                else:
                    self.set_status(STATUS_IDLE, "未检测到可用串口，请连接设备后重试。")

        self.update_output_preview()
        self.update_action_buttons()

    def selected_device_type(self) -> str:
        return self.device_type_combo.currentData() or "tele"

    def selected_port(self) -> str:
        return self.port_combo.currentData() or ""

    def normalized_filename(self) -> str:
        text = self.filename_input.text().strip()
        if not text:
            return ""
        return text if text.lower().endswith(".json") else f"{text}.json"

    def resolve_output_path_preview(self) -> Path | None:
        filename = self.normalized_filename()
        port = self.selected_port()
        if not filename or not port:
            return None

        try:
            device_type = self.selected_device_type()
            device_config = DEVICE_CONFIG_FACTORIES[device_type](port=port, id=Path(filename).stem or "my_so101")
            device = DEVICE_FACTORIES[device_type](device_config)
            return Path(device.calibration_fpath).with_name(filename)
        except Exception:
            return None

    def update_output_preview(self):
        filename = self.normalized_filename()
        if not filename:
            self.output_hint_label.setText("输出路径：请先输入文件名")
            return

        if self.last_output_path:
            preview_path = Path(self.last_output_path).with_name(filename)
            self.output_hint_label.setText(f"输出路径：{preview_path}")
            return

        preview_path = self.resolve_output_path_preview()
        if preview_path is not None:
            self.output_hint_label.setText(f"输出路径：{preview_path}")
            return

        self.output_hint_label.setText(f"输出路径：{filename} | 连接设备后可显示完整保存位置")

    def can_start_calibration(self) -> bool:
        return bool(self.selected_port() and self.filename_input.text().strip())

    def has_running_calibration(self) -> bool:
        return isinstance(self.worker, CalibrationWorker)

    def start_worker(self, worker: WorkerBase):
        self.worker_thread = QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_changed.connect(self.set_status)
        self.worker.succeeded.connect(self.on_worker_success)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.start()
        self.update_action_buttons()

    def update_action_buttons(self):
        ready = self.can_start_calibration()
        is_busy = self.current_status in BUSY_STATUSES
        has_running_calibration = self.has_running_calibration()

        self.start_button.setEnabled(ready and not is_busy and self.current_status in ACTIVE_STATUSES)
        self.pause_button.setEnabled(has_running_calibration and self.current_status in {STATUS_CALIBRATING, STATUS_PAUSED})
        self.pause_button.setText("继续标定" if is_calibration_paused() else "暂停标定")
        self.recalibrate_button.setEnabled(ready and not is_busy and self.current_status in TERMINAL_STATUSES)
        self.arm_check_button.setEnabled(bool(self.selected_port()) and not is_busy)
        self.refresh_button.setEnabled(not is_busy)
        self.device_type_combo.setEnabled(not is_busy)
        self.port_combo.setEnabled(not is_busy)
        self.filename_input.setEnabled(not is_busy)
        if IS_LINUX:
            self.permission_button.setEnabled(not is_busy)

    def start_calibration(self):
        if not self.can_start_calibration():
            QMessageBox.warning(self, "输入不完整", "请选择设备类型、串口，并填写输出文件名。")
            return

        selected_port = self.selected_port()
        selected_type = self.selected_device_type()
        filename = self.filename_input.text().strip()

        set_calibration_paused(False)
        self.append_log("")
        self.append_log("=" * 72)
        self.append_log(
            f"准备开始标定 | device_type={selected_type} | port={selected_port} | file={filename}"
        )
        self.start_worker(CalibrationWorker(selected_type, selected_port, filename))

    def toggle_pause_calibration(self):
        if not self.has_running_calibration():
            return

        if is_calibration_paused():
            set_calibration_paused(False)
            self.append_log("继续标定。")
            self.set_status(STATUS_CALIBRATING, "标定已继续执行。")
        else:
            set_calibration_paused(True)
            self.append_log("暂停标定，当前舵机保持锁住状态。")
            self.set_status(STATUS_PAUSED, "标定已暂停，当前动作会安全停下，舵机保持锁住，等待继续。")

    def restart_calibration(self):
        self.append_log("收到重新标定请求，准备重新开始。")
        self.start_calibration()

    def start_arm_check(self):
        selected_port = self.selected_port()
        if not selected_port:
            QMessageBox.warning(self, "未选择串口", "请先选择串口。")
            return

        self.append_log("")
        self.append_log("=" * 72)
        self.append_log(f"准备检测机械臂 | device_type={self.selected_device_type()} | port={selected_port}")
        self.start_worker(ArmCheckWorker(self.selected_device_type(), selected_port))

    def grant_linux_permissions(self):
        if not IS_LINUX:
            return

        password, ok = QInputDialog.getText(
            self,
            "Linux 端口授权",
            "请输入 sudo 密码：",
            QLineEdit.Password,
        )
        if not ok:
            return
        if not password:
            QMessageBox.warning(self, "密码为空", "请输入 sudo 密码。")
            return

        self.append_log("")
        self.append_log("=" * 72)
        self.append_log("准备执行 Linux 端口授权。")
        self.start_worker(LinuxPermissionWorker(password))

    def on_worker_success(self, message: str):
        if self.current_status == STATUS_FINISHED:
            self.last_output_path = message
            self.update_output_preview()
        self.append_log(message)

    def on_worker_failed(self, error_message: str):
        log_title, dialog_title = classify_worker_failure(self.worker)
        self.append_log(f"{log_title}：{error_message}")
        QMessageBox.critical(self, dialog_title, error_message)

    def on_worker_finished(self):
        set_calibration_paused(False)
        self.worker = None
        self.worker_thread = None
        self.update_action_buttons()


def main():
    app = QApplication(sys.argv)
    window = AutoCalibrateWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
