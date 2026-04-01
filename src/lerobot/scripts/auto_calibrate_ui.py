#!/usr/bin/env python3

"""PyQt-based UI for SO101 auto calibration."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from serial.tools import list_ports

from lerobot.motors.auto_calibrate import AutoCalibrateConfig, auto_calibrate_connected_device
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators import make_teleoperator_from_config, so_leader  # noqa: F401
from lerobot.teleoperators.so_leader import SO101LeaderConfig

try:
    from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal as Signal
    from PyQt5.QtGui import QColor, QFont, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
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
    from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
    from PySide6.QtGui import QColor, QFont, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
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
STATUS_FINISHED = "标定完毕，请拔掉USB"
STATUS_FAILED = "标定失败"

ACTIVE_STATUSES = {STATUS_IDLE, STATUS_PORT_DETECTED}
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
        message = self.format(record)
        self.emitter.message_emitted.emit(message)


class CalibrationWorker(QObject):
    finished = Signal()
    succeeded = Signal(str)
    failed = Signal(str)
    status_changed = Signal(str, str)

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
            result = auto_calibrate_connected_device(
                device,
                auto_calib_config,
                calibration_path=output_path,
            )

            final_path = result.calibration_path or output_path
            self.status_changed.emit(STATUS_FINISHED, f"标定完成，文件已保存到: {final_path}")
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
        cleaned = filename.strip()
        if not cleaned:
            cleaned = "my_so101"
        if cleaned.lower().endswith(".json"):
            return cleaned
        return f"{cleaned}.json"


class AutoCalibrateWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker_thread: QThread | None = None
        self.worker: CalibrationWorker | None = None
        self.current_status = STATUS_IDLE
        self.last_output_path = ""
        self._log_handler: QtLogHandler | None = None

        self.setWindowTitle("SO101 Auto Calibrate")
        self.resize(980, 720)
        self.setMinimumSize(900, 640)

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
        subtitle_label = QLabel(f"基于 {QT_LIB} 的 SO101 标定页面，支持日志输出、串口检测和重新标定。")
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

        device_type_label = QLabel("设备类型")
        self.device_type_combo = QComboBox()
        self.device_type_combo.addItem("tele", "tele")
        self.device_type_combo.addItem("robot", "robot")
        self.device_type_combo.currentIndexChanged.connect(self.on_device_type_changed)

        port_label = QLabel("COM 端口")
        self.port_combo = QComboBox()
        self.port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.port_combo.currentIndexChanged.connect(self.update_action_buttons)
        self.refresh_button = QPushButton("刷新端口")
        self.refresh_button.clicked.connect(self.refresh_ports)

        filename_label = QLabel("输出文件名")
        self.filename_input = QLineEdit("my_so101")
        self.filename_input.setPlaceholderText("例如: tele_calibration 或 robot_01.json")
        self.filename_input.textChanged.connect(self.update_action_buttons)

        filename_hint = QLabel("最终会保存为同一路径下的 JSON 文件，只改文件名，不改保存目录。")
        filename_hint.setObjectName("hintLabel")

        config_layout.addWidget(device_type_label, 0, 0)
        config_layout.addWidget(self.device_type_combo, 0, 1, 1, 2)
        config_layout.addWidget(port_label, 1, 0)
        config_layout.addWidget(self.port_combo, 1, 1)
        config_layout.addWidget(self.refresh_button, 1, 2)
        config_layout.addWidget(filename_label, 2, 0)
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

        self.output_hint_label = QLabel("输出文件: 未生成")
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
        self.start_button.clicked.connect(self.start_calibration)
        self.start_button.setObjectName("primaryButton")

        self.recalibrate_button = QPushButton("重新标定")
        self.recalibrate_button.clicked.connect(self.restart_calibration)
        self.recalibrate_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.recalibrate_button)
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

        self.append_log("UI 已启动，正在检测本机可用 COM 端口。")
        self.update_action_buttons()

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
                background: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #1f8a70;
            }
            QPushButton:disabled {
                background: #ece8e0;
                color: #9aa5ad;
                border: 1px solid #d8d1c8;
            }
            QPushButton#primaryButton {
                background: #1f8a70;
                color: white;
                border: 1px solid #1f8a70;
            }
            QPushButton#primaryButton:hover {
                background: #19715c;
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
            STATUS_FINISHED: "#0f766e",
            STATUS_FAILED: "#b42318",
        }
        color = color_map.get(status, "#173f35")
        self.status_badge.setStyleSheet(f"color: {QColor(color).name()};")

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

    def list_available_ports(self) -> list[SerialPortInfo]:
        ports: list[SerialPortInfo] = []
        for port in list_ports.comports():
            if sys.platform.startswith("win") and not str(port.device).upper().startswith("COM"):
                continue
            ports.append(SerialPortInfo(device=port.device, description=port.description or "Unknown device"))
        ports.sort(key=lambda item: item.device)
        return ports

    def refresh_ports(self):
        if self.current_status == STATUS_CALIBRATING:
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
        if text.lower().endswith(".json"):
            return text
        return f"{text}.json"

    def update_output_preview(self):
        filename = self.normalized_filename()
        if not filename:
            self.output_hint_label.setText("输出文件: 请先输入文件名")
            return

        if self.last_output_path:
            preview_path = Path(self.last_output_path).with_name(filename)
            self.output_hint_label.setText(f"输出文件: {preview_path}")
            return

        self.output_hint_label.setText(f"输出文件名: {filename}  |  保存目录将沿用当前标定代码默认路径")

    def can_start_calibration(self) -> bool:
        return bool(self.selected_port() and self.filename_input.text().strip())

    def update_action_buttons(self):
        ready = self.can_start_calibration()
        is_busy = self.current_status == STATUS_CALIBRATING

        self.start_button.setEnabled(ready and not is_busy and self.current_status in ACTIVE_STATUSES)
        self.recalibrate_button.setEnabled(ready and not is_busy and self.current_status in TERMINAL_STATUSES)
        self.refresh_button.setEnabled(not is_busy)
        self.device_type_combo.setEnabled(not is_busy)
        self.port_combo.setEnabled(not is_busy)
        self.filename_input.setEnabled(not is_busy)

    def start_calibration(self):
        if not self.can_start_calibration():
            QMessageBox.warning(self, "输入不完整", "请选择设备类型、COM 端口，并填写输出文件名。")
            return

        selected_port = self.selected_port()
        selected_type = self.selected_device_type()
        filename = self.filename_input.text().strip()

        self.append_log("")
        self.append_log("=" * 72)
        self.append_log(f"准备开始标定 | device_type={selected_type} | port={selected_port} | file={filename}")

        self.set_status(STATUS_CALIBRATING, f"正在对 {selected_type} 设备进行标定，请勿断开 USB。")

        self.worker_thread = QThread(self)
        self.worker = CalibrationWorker(selected_type, selected_port, filename)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_changed.connect(self.set_status)
        self.worker.succeeded.connect(self.on_calibration_success)
        self.worker.failed.connect(self.on_calibration_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.on_worker_finished)
        self.worker_thread.start()

        self.update_action_buttons()

    def restart_calibration(self):
        self.append_log("收到重新标定请求，准备重新开始。")
        self.start_calibration()

    def on_calibration_success(self, output_path: str):
        self.last_output_path = output_path
        self.update_output_preview()
        self.append_log(f"标定成功，输出文件: {output_path}")

    def on_calibration_failed(self, error_message: str):
        self.append_log(f"标定失败: {error_message}")
        QMessageBox.critical(self, "标定失败", error_message)

    def on_worker_finished(self):
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
