"""Robotiq 2F-85 over the PC's USB-RS485 adapter (Modbus RTU), system Python + pyserial.

  /usr/bin/python3 calibration/gripper.py status              read-only: activation, position, object detection, fault
  /usr/bin/python3 calibration/gripper.py activate            (re)activation: the fingers travel fully open/closed once
                                                              at the gripper's own activation speed (not settable)
  /usr/bin/python3 calibration/gripper.py open | close
  /usr/bin/python3 calibration/gripper.py goto --pos 120      0 = fully open (85 mm) .. 255 = fully closed

The gripper is powered by the Doosan tool flange but commanded only from this PC (the pendant cannot jog it).
Every motion command asks for confirmation (or --yes) and always uses the SLOWEST speed (Human rule: speed 0 =
about 20 mm/s); the speed is deliberately not an option.  Force: config.yaml gripper.force (0..255).
Register map: Robotiq 2F-85 manual, Modbus RTU (slave 9, 115200 8N1; write 0x03E8, read 0x07D0).
"""
import argparse
import sys
import time

import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOWEST = 0                                             # rSP: 0 = minimum speed (~20 mm/s)
STATUS_TEXT = {0: 'reset / not activated', 1: 'activating', 2: 'not used', 3: 'activated'}
FAULT_TEXT = {0x00: 'none', 0x05: 'action delayed, activation must finish first', 0x07: 'activation bit must be set first',
              0x08: 'over temperature', 0x09: 'no communication for >= 1 s (minor; normal between one-shot commands, does not block motion)',
              0x0A: 'under voltage', 0x0B: 'automatic release in progress', 0x0C: 'internal fault',
              0x0D: 'activation fault', 0x0E: 'over current', 0x0F: 'automatic release completed'}
OBJECT_TEXT = {0: 'moving', 1: 'stopped on an object while opening', 2: 'stopped on an object while closing',
               3: 'at the requested position (no object)'}


def crc16(data):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, crc >> 8])


class Gripper:
    def __init__(self, port, slave=9, baud=115200):
        import serial
        self.slave = slave
        self.ser = serial.Serial(port, baud, bytesize=8, parity='N', stopbits=1, timeout=0.2)

    def _xfer(self, frame, reply_len):
        frame = bytes([self.slave]) + frame
        self.ser.reset_input_buffer()
        self.ser.write(frame + crc16(frame))
        reply = self.ser.read(reply_len)
        if len(reply) != reply_len or crc16(reply[:-2]) != reply[-2:]:
            raise IOError(f'no / bad Modbus reply ({reply.hex() or "nothing"}): check power, port, slave id')
        return reply

    def read(self):
        """FC04, 3 input registers from 0x07D0 -> dict."""
        r = self._xfer(bytes([0x04, 0x07, 0xD0, 0x00, 0x03]), 3 + 6 + 2)[3:9]
        return dict(activated=bool(r[0] & 0x01), going=bool(r[0] & 0x08), status=(r[0] >> 4) & 0x3,
                    object=(r[0] >> 6) & 0x3, fault=r[2], pos_request=r[3], pos=r[4], current=r[5])

    def write(self, action, pos=0, force=0):
        """FC16, 3 holding registers from 0x03E8: [action, 0, 0, pos, speed, force]."""
        payload = bytes([action, 0, 0, pos, SLOWEST, force])
        self._xfer(bytes([0x10, 0x03, 0xE8, 0x00, 0x03, 0x06]) + payload, 8)

    def wait(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.read()
            if s['fault']:
                return s
            if s['status'] == 3 and s['object'] != 0:
                return s
            time.sleep(0.1)
        raise TimeoutError('gripper did not finish within %.0f s' % timeout)


def show(s):
    mm = 85.0 * (1 - s['pos'] / 255.0)
    print(f"status   : {STATUS_TEXT[s['status']]} (activated bit {int(s['activated'])}, go {int(s['going'])})")
    print(f"position : {s['pos']} / 255  (~{mm:.0f} mm opening; request {s['pos_request']})")
    print(f"object   : {OBJECT_TEXT[s['object']]}")
    print(f"current  : {s['current'] * 10} mA   fault: 0x{s['fault']:02X} {FAULT_TEXT.get(s['fault'], 'see manual')}")


def confirm(what, yes):
    if yes:
        return
    if input(f'{what}: the fingers will move (slowest speed). Hands clear? type yes: ').strip() != 'yes':
        raise SystemExit('cancelled')


def main():
    cfg = yaml.safe_load((HERE / 'config.yaml').read_text())['gripper']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('status', 'activate', 'open', 'close', 'goto'))
    p.add_argument('--pos', type=int, help='goto: 0 (open) .. 255 (closed)')
    p.add_argument('--force', type=int, default=cfg['force'], help='0..255 (default config.yaml gripper.force)')
    p.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    a = p.parse_args()
    g = Gripper(cfg['port'], cfg['slave'], cfg['baud'])
    if a.mode == 'status':
        return show(g.read())
    force = max(0, min(255, a.force))
    if a.mode == 'activate':
        confirm('activate (full open/close stroke)', a.yes)
        g.write(0x00)                                   # clear rACT, then set it
        time.sleep(0.5)
        g.write(0x01)
        t0 = time.time()
        while g.read()['status'] != 3:
            if time.time() - t0 > 20:
                raise TimeoutError('activation did not finish')
            time.sleep(0.2)
        return show(g.read())
    s = g.read()
    if s['status'] != 3:
        raise SystemExit('gripper not activated: run `activate` first')
    pos = {'open': 0, 'close': 255}.get(a.mode, a.pos)
    if pos is None or not 0 <= pos <= 255:
        raise SystemExit('goto needs --pos 0..255')
    confirm(f'{a.mode} to {pos}/255, force {force}', a.yes)
    g.write(0x09, pos, force)                           # rACT | rGTO
    show(g.wait())


SYSTEM_PY = '/usr/bin/python3'

if __name__ == '__main__':
    if sys.executable != SYSTEM_PY and Path(SYSTEM_PY).is_file():   # (10/20 보스) conda python has a different 'serial'
        import os                                                    # package (no serial.Serial): rerun in system Python
        os.execv(SYSTEM_PY, [SYSTEM_PY, str(Path(__file__).resolve()), *sys.argv[1:]])
    try:
        main()
    except (IOError, TimeoutError) as e:
        sys.exit(f'gripper: {e}')
