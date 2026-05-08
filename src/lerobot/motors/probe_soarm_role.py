#!/usr/bin/env python

"""Probe Feetech servo registers and infer soarm leader/follower role."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.motors.feetech.feetech import patch_setPacketTimeout

logger = logging.getLogger(__name__)

DEFAULT_BAUDRATE = 1_000_000
DEFAULT_PROTOCOL_VERSION = 0
MAX_ID = 253

FIRMWARE_MAJOR_ADDR = 0
SERVO_ID_ADDR = 5
MAX_INPUT_VOLTAGE_ADDR = 14


@dataclass
class ProbeResult:
    servo_id: int
    firmware_major: int
    eeprom_id: int
    max_input_voltage: int

    @property
    def role(self) -> str:
        return "follower" if self.max_input_voltage == 140 else "leader"


def _build_sdk(port: str, protocol_version: int):
    import scservo_sdk as scs

    port_handler = scs.PortHandler(port)
    port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(port_handler, scs.PortHandler)
    packet_handler = scs.PacketHandler(protocol_version)
    return scs, port_handler, packet_handler


def _read_1b(packet_handler, port_handler, servo_id: int, address: int) -> int:
    value, comm, error = packet_handler.read1ByteTxRx(port_handler, servo_id, address)
    if comm != 0:
        raise RuntimeError(f"read1ByteTxRx failed for id={servo_id}, address={address}: comm={comm}")
    if error != 0:
        raise RuntimeError(f"read1ByteTxRx returned servo error for id={servo_id}, address={address}: error={error}")
    return int(value)


def _iter_found_ids(scs, packet_handler, port_handler, protocol_version: int) -> list[int]:
    if protocol_version == 0 and hasattr(packet_handler, "broadcastPing"):
        logger.info("Using SDK broadcastPing to scan servo IDs.")
        found = packet_handler.broadcastPing(port_handler)
        if found is not None:
            return sorted(int(id_) for id_ in found.keys())

    logger.info("Falling back to sequential ping scan for servo IDs.")
    found_ids: list[int] = []
    for servo_id in range(MAX_ID + 1):
        _model_number, comm, error = packet_handler.ping(port_handler, servo_id)
        if comm == scs.COMM_SUCCESS and error == 0:
            found_ids.append(servo_id)
    return found_ids


def probe_one(packet_handler, port_handler, servo_id: int) -> ProbeResult:
    firmware_major = _read_1b(packet_handler, port_handler, servo_id, FIRMWARE_MAJOR_ADDR)
    eeprom_id = _read_1b(packet_handler, port_handler, servo_id, SERVO_ID_ADDR)
    max_input_voltage = _read_1b(packet_handler, port_handler, servo_id, MAX_INPUT_VOLTAGE_ADDR)
    return ProbeResult(
        servo_id=servo_id,
        firmware_major=firmware_major,
        eeprom_id=eeprom_id,
        max_input_voltage=max_input_voltage,
    )


def probe_many(
    port: str,
    baudrate: int,
    protocol_version: int,
    ids: Iterable[int] | None,
) -> list[ProbeResult]:
    scs, port_handler, packet_handler = _build_sdk(port, protocol_version)

    if not port_handler.openPort():
        raise ConnectionError(f"Failed to open port: {port}")
    try:
        if not port_handler.setBaudRate(baudrate):
            raise ConnectionError(f"Failed to set baudrate {baudrate} on port: {port}")

        target_ids = list(ids) if ids is not None else _iter_found_ids(scs, packet_handler, port_handler, protocol_version)
        if not target_ids:
            raise RuntimeError("No servos found on the bus.")

        return [probe_one(packet_handler, port_handler, servo_id) for servo_id in target_ids]
    finally:
        port_handler.closePort()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Feetech servo raw registers for soarm role detection.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM6 or /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"Baudrate, default {DEFAULT_BAUDRATE}")
    parser.add_argument(
        "--protocol-version",
        type=int,
        default=DEFAULT_PROTOCOL_VERSION,
        choices=[0, 1],
        help=f"Feetech protocol version, default {DEFAULT_PROTOCOL_VERSION}",
    )
    parser.add_argument(
        "--id",
        type=int,
        action="append",
        dest="ids",
        help="Servo ID to probe. Repeat for multiple IDs. If omitted, scan first.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = probe_many(
        port=args.port,
        baudrate=args.baudrate,
        protocol_version=args.protocol_version,
        ids=args.ids,
    )

    for result in results:
        print(
            f"id={result.servo_id} "
            f"firmware_major@0={result.firmware_major} "
            f"eeprom_id@5={result.eeprom_id} "
            f"max_input_voltage@14={result.max_input_voltage} "
            f"role={result.role}"
        )


if __name__ == "__main__":
    main()
