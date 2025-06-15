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
from typing import Optional
import asyncio

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

    def start_from_snapshot(self, snapshot_path: Path):
        """Launch Firecracker and restore from *snapshot_path*."""

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

        self.master_fd, self.slave_fd = pty.openpty()
        self.proc = subprocess.Popen(
            [FC_BINARY, "--api-sock", str(self.api_socket)],
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

        for _ in range(20):
            if self.api_socket.exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Firecracker API socket was not created")

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

        mem_path = snapshot_path.with_suffix(snapshot_path.suffix + ".mem")
        resp = self._api(
            "PUT",
            "/snapshot/load",
            {
                "snapshot_path": str(snapshot_path),
                "mem_file_path": str(mem_path),
                "resume_vm": True,
            },
        )
        if resp.status_code != 204:
            raise RuntimeError("Failed to load snapshot")

        time.sleep(1)

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
# Async worker for VM interaction
# ─────────────────────────────────────────────────────────────────────────────


def snapshot_vm(
    kernel: str,
    rootfs: str,
    snapshot_path: Path,
    *,
    vsock_socket: Optional[Path] = None,
) -> bool:
    """Boot a VM, wait for readiness then snapshot to *snapshot_path*.

    The snapshot consists of *snapshot_path* for the VM state and a companion
    memory file at ``snapshot_path.with_suffix(snapshot_path.suffix + '.mem')``.
    Returns ``True`` on success, ``False`` otherwise.
    """

    vm = FirecrackerVM(kernel=kernel, rootfs=rootfs, vsock_socket=vsock_socket)
    try:
        vm._start()

        ready = False
        for _ in range(50):
            output = vm.run('echo "VM" "ready"')
            if 'VM ready' in output:
                ready = True
                break
            time.sleep(0.2)
        if not ready:
            return False

        mem_path = snapshot_path.with_suffix(snapshot_path.suffix + '.mem')

        resp = vm._api('PATCH', '/vm', {'state': 'Paused'})
        #print(resp.status_code, resp.text)
        resp = vm._api(
            'PUT',
            '/snapshot/create',
            {
                'snapshot_type': 'Full',
                'snapshot_path': str(snapshot_path),
                'mem_file_path': str(mem_path),
            },
        )
        #print(resp.status_code, resp.text)
        resp = vm._api('PATCH', '/vm', {'state': 'Resumed'})

        return resp.status_code == 204
    except Exception:
        return False
    finally:
        vm.shutdown()




async def vm_worker(
    idx: int,
    kernel: str,
    rootfs: str,
    *,
    setup_recipe: Path,
    verify_recipe: Path,
    prompt_format: str,
    command_q: asyncio.Queue[str],
    result_q: asyncio.Queue[str],
    snapshot_dir: Path = Path("snapshots"),
):
    """Boot one VM, execute setup & verify recipes and relay commands asynchronously."""

    def _send_recipe(vm: FirecrackerVM, recipe_path: Path) -> str:
        recipe = json.loads(recipe_path.read_text())
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(vm.vsock_socket))
            s.sendall(b"CONNECT 52\n")
            read_line(s)
            handshake = (
                json.dumps({"hello": "firecracker", "version": 1}).encode("utf-8")
                + b"\x00"
            )
            s.sendall(handshake)
            read_line(s)
            s.sendall(json.dumps(recipe).encode("utf-8") + b"\x00")
            return s.recv(4096).decode().strip()

    snap_path = snapshot_dir / f"vm-{idx}.snap"
    vsock_path = snapshot_dir / f"vsock-{idx}.sock"
    api_sock = snapshot_dir / f"api-{idx}.sock"
    serial_log = snapshot_dir / f"serial-{idx}.log"
    fc_log = snapshot_dir / f"fc-{idx}.log"

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if not snap_path.exists():
        ok = await asyncio.to_thread(
            snapshot_vm,
            kernel,
            rootfs,
            snap_path,
            vsock_socket=vsock_path,
        )
        if not ok:
            await result_q.put("snapshot failed")
            return

    vm = FirecrackerVM(
        kernel=kernel,
        rootfs=rootfs,
        api_socket=api_sock,
        vsock_socket=vsock_path,
        serial_log=serial_log,
        fc_log=fc_log,
    )
    await asyncio.to_thread(vm.start_from_snapshot, snap_path)

    try:
        result = "NO"
        while "task done" not in result:
            result = await asyncio.to_thread(
                # Do NOT merge "task" and "done" into a single string
                # The terminal command *will* appear in the PTY output
                # VERBATIM, we're relying on echo combining the strings
                # into "task done" for robust verification.
                vm.run, "/bin/taskd 52; echo \"task\" \"done\""
            )
            if "task done" not in result:
                await asyncio.sleep(0.2)
        await asyncio.sleep(0.1)

        setup_reply = await asyncio.to_thread(_send_recipe, vm, setup_recipe)
        try:
            values = json.loads(setup_reply)[0].get("values", [])
        except Exception:
            values = []
        try:
            prompt = prompt_format.format(*values)
        except Exception:
            prompt = prompt_format
        await result_q.put(prompt)

        while True:
            cmd = await command_q.get()
            if cmd is None:
                break
            output = await asyncio.to_thread(vm.run, cmd)
            verify_reply = await asyncio.to_thread(_send_recipe, vm, verify_recipe)
            report = json.dumps({"output": output, "verify": verify_reply})
            await result_q.put(report)
    finally:
        await asyncio.to_thread(vm.shutdown)



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Example entry-point that boots a single async worker."""

    KERNEL = "vm/vmlinux-6.1.134"
    ROOTFS = "vm/ubuntu-24.04.ext4"

    cmd_q: asyncio.Queue[str] = asyncio.Queue()
    result_q: asyncio.Queue[str] = asyncio.Queue()

    worker = asyncio.create_task(
        vm_worker(
            0,
            KERNEL,
            ROOTFS,
            setup_recipe=Path("tasks/file_copy/setup.json"),
            verify_recipe=Path("tasks/file_copy/verify.json"),
            prompt_format="Copy all files from {0} into {1}.",
            command_q=cmd_q,
            result_q=result_q,
        )
    )

    prompt = await result_q.get()
    print(f"Prompt → {prompt}")

    cmd_q.put_nowait("echo hello")
    report = await result_q.get()
    print(f"Report → {report}")

    cmd_q.put_nowait(None)
    await worker


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ensure every VM still tears down correctly
        pass

