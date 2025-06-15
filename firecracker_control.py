#!/usr/bin/env python3
"""
extended_firecracker_parallel.py – spin up multiple Firecracker VMs in parallel,
run a vsock echo-test inside each one, report success, and tear everything
back down – leaving **no** artefacts on the host.

This is an evolution of *extended_firecracker.py*:
•  **FirecrackerVM** now accepts *per-instance* paths for the API socket,
   vsock backend socket and log files, defaulting to unique temp-files so
   concurrent VMs never clash.
•  A **threaded** `main()` boots 256 VMs concurrently, validates that a
   host→guest vsock round-trip returns the taskd test banner and prints
   `VM-<idx>: OK` when it does.
"""
from __future__ import annotations

import os
import pty
import re
import selectors
import signal
import socket
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from threading import Thread
from typing import Optional

import requests_unixsocket
import json

# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────
FC_BINARY = "./vm/firecracker/firecracker"
ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
CMD_END_MARKER = "# __CMD_DONE__"


def clean_output(text: str) -> str:
    """Strip ANSI escape sequences from *text*."""
    return ANSI_ESCAPE_RE.sub("", text)


def read_line(sock: socket.socket, max_bytes: int = 256, timeout: float = 2.0) -> bytes:
    """Read a single line (up to *max_bytes*) from *sock*, honouring *timeout*."""
    sock.settimeout(timeout)
    data = b""
    try:
        while not data.endswith(b"\n") and len(data) < max_bytes:
            chunk = sock.recv(1)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        pass
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Firecracker wrapper class
# ─────────────────────────────────────────────────────────────────────────────
class FirecrackerVM:
    """Context-manager that guarantees graceful shutdown & on-disk cleanup.

    Paths (API socket, vsock backend socket and log files) can be supplied
    explicitly.  If omitted a dedicated temporary directory is created so that
    multiple VMs can run at the same time without any filename collisions.
    """

    def __init__(
        self,
        kernel: str,
        rootfs: str,
        mem_mib: int = 128,
        api_socket: Optional[Path] = None,
        vsock_socket: Optional[Path] = None,
        serial_log: Optional[Path] = None,
        fc_log: Optional[Path] = None,
    ):
        self.kernel = kernel
        self.rootfs = rootfs
        self.mem_mib = mem_mib

        # Create an isolated work-dir when any of the paths are unspecified
        if None in (api_socket, vsock_socket, serial_log, fc_log):
            self._tmpdir = Path(tempfile.mkdtemp(prefix="fcvm_"))
            api_socket = api_socket or self._tmpdir / "firecracker.socket"
            vsock_socket = vsock_socket or self._tmpdir / "vsock.sock"
            serial_log = serial_log or self._tmpdir / "serial.out"
            fc_log = fc_log or self._tmpdir / "fc_log.log"
        else:
            # Caller provided everything – no temp-dir lifecycle to manage
            self._tmpdir = None

        self.api_socket: Path = api_socket  # type: ignore
        self.vsock_socket: Path = vsock_socket  # type: ignore
        self.serial_log: Path = serial_log  # type: ignore
        self.fc_log: Path = fc_log  # type: ignore

        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.session = requests_unixsocket.Session()

    # ───────── Low-level helpers ────────────────────────────────────────────
    def _api(self, method: str, path: str, json=None):
        """Thin wrapper around the Firecracker REST API."""
        url = f"http+unix://{str(self.api_socket).replace('/', '%2F')}{path}"
        return self.session.request(
            method, url, headers={"Content-Type": "application/json"}, json=json
        )

    def _send_serial(self, data: str):
        os.write(self.master_fd, data.encode())  # type: ignore[arg-type]

    def _send_cmd(self, cmd: str):
        self._send_serial(cmd + "\n" + CMD_END_MARKER + "\n")

    def _read_until(self, marker: str = CMD_END_MARKER, timeout: float = 3.0) -> str:
        output, start = "", time.time()
        sel = selectors.DefaultSelector()
        try:
            sel.register(self.master_fd, selectors.EVENT_READ)
        except KeyError:
            return ""  # FD not valid or already closed

        try:
            while time.time() - start < timeout:
                events = sel.select(timeout=0.1)
                for key, _ in events:
                    try:
                        chunk = os.read(key.fd, 1024).decode(errors="ignore")
                        output += chunk
                        if marker in output:
                            raise StopIteration
                    except (OSError, ValueError):  # FD may be closed or invalid
                        break
        except StopIteration:
            pass
        finally:
            try:
                sel.unregister(self.master_fd)
            except KeyError:
                pass
            sel.close()

        cleaned = clean_output(output)
        lines = cleaned.splitlines()
        for i, line in enumerate(reversed(lines)):
            if marker in line:
                return "\n".join(lines[: len(lines) - i - 1]).strip()
        return cleaned.strip()

    # ───────── Context-manager hooks ────────────────────────────────────────
    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    # ───────── Public API ───────────────────────────────────────────────────
    def run(self, cmd: str, timeout: float = 3.0) -> str:
        """Run *cmd* inside the VM via the serial console and return stdout."""
        self._send_cmd(cmd)
        return self._read_until(timeout=timeout)

    # ───────── Start / shutdown internals ──────────────────────────────────
    def _start(self):
        # Ensure a clean slate – delete any left-overs at *our* paths only.
        for path in (
            self.api_socket,
            self.vsock_socket,
            self.serial_log,
            self.fc_log,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        # Hand-rolled PTY so we can drive the guest shell via the serial port
        self.master_fd, self.slave_fd = pty.openpty()
        slave_path = os.ttyname(self.slave_fd)

        # Launch Firecracker
        self.proc = subprocess.Popen(
            [FC_BINARY, "--api-sock", str(self.api_socket)],
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

        # Give Firecracker up to 2 s to create the API socket
        for _ in range(20):
            if self.api_socket.exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Firecracker API socket was not created")

        # Configure the micro-VM
        self._api(
            "PUT",
            "/logger",
            {
                "log_path": str(self.fc_log),
                "level": "Info",
                "show_level": True,
                "show_log_origin": True,
            },
        )
        self._api("PUT", "/machine-config", {"vcpu_count": 1, "mem_size_mib": self.mem_mib})
        self._api(
            "PUT",
            "/boot-source",
            {
                "kernel_image_path": self.kernel,
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
            },
        )
        self._api(
            "PUT",
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": self.rootfs,
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        self._api(
            "PUT",
            "/vsock/0",
            {"guest_cid": 3, "uds_path": str(self.vsock_socket)},
        )
        self._api("PUT", "/actions", {"action_type": "InstanceStart"})
        time.sleep(2)  # allow the guest to finish booting

    def shutdown(self):
        """Attempt graceful shutdown, escalate if needed and clean disk files."""
        if not self.proc:
            return

        try:
            resp = self._api("PUT", "/actions", {"action_type": "InstanceShutdown"})
            if resp.status_code == 204:
                for _ in range(30):
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.1)
        except Exception:
            pass  # Firecracker pre-1.7 or other problem – will escalate below

        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

        if self.master_fd:
            os.close(self.master_fd)
        if self.slave_fd:
            os.close(self.slave_fd)

        # Clean up files we created
        for path in (
            self.api_socket,
            self.vsock_socket,
            self.serial_log,
            self.fc_log,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        # Remove temp-dir (if one was made)
        if self._tmpdir and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# Thread-worker for the parallel test
# ─────────────────────────────────────────────────────────────────────────────




def vm_worker(idx: int, kernel: str, rootfs: str):
    """Boot one VM, handshake with taskd, dispatch setup + verify recipes."""
    try:
        with FirecrackerVM(kernel=kernel, rootfs=rootfs) as vm:
            # Wait for taskd to be ready inside the guest
            result = "NO"
            while "task done" not in result:
                result = vm.run("/bin/taskd 52; echo \"task\"\" done\"")
                if "task done" not in result:
                    time.sleep(0.2)
            time.sleep(0.1)

            # Step 1: Handshake (first connection)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(str(vm.vsock_socket))
                s.sendall(b"CONNECT 52\n")
                banner = read_line(s).decode().strip()

                handshake = json.dumps({ "hello": "firecracker", "version": 1 }).encode('utf-8') + b'\x00'
                s.sendall(handshake)
                reply = read_line(s).decode().strip()
                try:
                    response = json.loads(reply)
                    if response.get("status") != 0:
                        print(f"VM-{idx}: handshake failed → {response}")
                        return
                except Exception as e:
                    print(f"VM-{idx}: invalid handshake reply → {reply!r} ({e})")
                    return

            # Step 2: Send recipes, one per connection
            recipe_paths = [
                Path("sm_recipes/file_copy/setup.json"),
                Path("sm_recipes/file_copy/verify.json"),
            ]

            for path in recipe_paths:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(str(vm.vsock_socket))
                    s.sendall(b"CONNECT 52\n")
                    banner = read_line(s).decode().strip()

                    # Send handshake again for each connection (as per server expectations)
                    handshake = json.dumps({ "hello": "firecracker", "version": 1 }).encode('utf-8') + b'\x00'
                    s.sendall(handshake)
                    reply = read_line(s).decode().strip()
                    try:
                        response = json.loads(reply)
                        if response.get("status") != 0:
                            print(f"VM-{idx}: handshake (re)failed → {response}")
                            return
                    except Exception as e:
                        print(f"VM-{idx}: invalid handshake reply (again) → {reply!r} ({e})")
                        return

                    # Send recipe
                    recipe = json.loads(path.read_text())
                    recipe_data = json.dumps(recipe).encode('utf-8') + b'\x00'
                    s.sendall(recipe_data)
                    #time.sleep(0.1)

                    # Wait for response
                    try:
                        final_reply = s.recv(1024).decode().strip()
                        if final_reply:
                            print(f"VM-{idx}: report → {final_reply}")
                        else:
                            print(f"VM-{idx}: no report received")
                    except Exception as e:
                        print(f"VM-{idx}: error receiving report → {e}")

                    result = vm.run("cp -r /home/jessica/Documents/important/* /home/jessica/Desktop/")
                    #print(f"VM-{idx}: contents → {result}")
                    result = vm.run("ls /home/jessica/Desktop")
                    #print(f"VM-{idx}: contents → {result}")

            print(f"VM-{idx}: OK")

    except Exception as e:
        print(f"VM-{idx}: FAILED → {e}")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    KERNEL = "vm/vmlinux-6.1.134"
    ROOTFS = "vm/ubuntu-24.04.ext4"

    threads: list[Thread] = []
    for i in range(32):
        t = Thread(target=vm_worker, args=(i, KERNEL, ROOTFS), daemon=True)
        t.start()
        threads.append(t)
        #time.sleep(0.05)  # slight delay to avoid startup storm

    # Wait for all workers to finish
    for t in threads:
        t.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ensure every VM still tears down correctly
        pass

