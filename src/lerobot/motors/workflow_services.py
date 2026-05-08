"""Workflow helpers for auto calibration UI."""

from __future__ import annotations

import logging

from lerobot.motors.probe_soarm_role import DEFAULT_BAUDRATE, DEFAULT_PROTOCOL_VERSION, probe_many

logger = logging.getLogger(__name__)


def detect_soarm_device_type(port: str) -> str:
    results = probe_many(
        port=port,
        baudrate=DEFAULT_BAUDRATE,
        protocol_version=DEFAULT_PROTOCOL_VERSION,
        ids=None,
    )
    if not results:
        raise RuntimeError("No soarm101 servo found; cannot detect leader/follower.")

    first_result = sorted(results, key=lambda item: item.servo_id)[0]
    logger.info(
        "Address 14 probe result: id=%s max_input_voltage=%s role=%s",
        first_result.servo_id,
        first_result.max_input_voltage,
        first_result.role,
    )
    return "so101_follower" if first_result.max_input_voltage == 140 else "so101_leader"
