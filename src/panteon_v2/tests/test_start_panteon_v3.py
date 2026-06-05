from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_launcher():
    root = Path(__file__).resolve().parents[3]
    path = root / "Start_panteon_v3.py"
    spec = importlib.util.spec_from_file_location("start_panteon_v3", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_launcher_enables_mexc_and_bitget_by_default():
    launcher = _load_launcher()

    assert launcher.EXCHANGES["MEXC"] == "ON"
    assert launcher.EXCHANGES["BITGET"] == "ON"
    assert launcher._enabled_exchanges() == ("MEXC", "BITGET")


def test_v3_launcher_skips_exchange_when_instance_lock_is_held(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_mexc.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=True,
    ), patch.object(launcher, "_spawn_child") as spawn:
        result = launcher._launch_exchange("MEXC", dry_run=False)

    spawn.assert_not_called()
    assert result.status == "already_running"
    assert result.exchange == "MEXC"


def test_v3_launcher_spawns_itself_as_exchange_worker(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_bitget.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.object(launcher, "_spawn_child", return_value=12345) as spawn:
        result = launcher._launch_exchange(
            "BITGET",
            dry_run=False,
            python_executable="python-test",
        )

    spawn.assert_called_once_with("BITGET", "python-test")
    assert result.status == "launched"
    assert result.pid == 12345


def test_v3_worker_args_do_not_use_v2_entrypoint_scripts():
    launcher = _load_launcher()

    assert launcher._worker_args("MEXC") == (
        "Start_panteon_v3.py",
        "--worker",
        "MEXC",
    )
    assert launcher._worker_args("BITGET") == (
        "Start_panteon_v3.py",
        "--worker",
        "BITGET",
    )


def test_v3_launcher_sets_default_genetics_specialists_manifest(tmp_path):
    launcher = _load_launcher()
    manifest = tmp_path / "genetics_specialists_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with patch.object(launcher, "DEFAULT_GENETICS_SPECIALISTS_MANIFEST", manifest), patch.dict(
        os.environ,
        {launcher.GENETICS_SPECIALISTS_ENV: ""},
    ):
        os.environ.pop(launcher.GENETICS_SPECIALISTS_ENV, None)
        launcher._configure_default_genetics_manifests()

        assert os.environ[launcher.GENETICS_SPECIALISTS_ENV] == str(manifest)


def test_v3_launcher_reports_spawn_failure(tmp_path):
    launcher = _load_launcher()
    lock_path = tmp_path / "panteon_v2_mexc.lock"

    with patch.object(launcher, "_lock_path", return_value=lock_path), patch.object(
        launcher,
        "_lock_is_held",
        return_value=False,
    ), patch.object(launcher, "_spawn_child", side_effect=RuntimeError("boom")):
        result = launcher._launch_exchange("MEXC", python_executable="python-test")

    assert result.status == "failed"
    assert "boom" in result.message


def test_v3_windows_spawn_uses_nonblocking_start_process(tmp_path):
    launcher = _load_launcher()
    out_path = tmp_path / "worker.out.log"
    err_path = tmp_path / "worker.err.log"

    with patch.object(
        launcher.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="4242\n", stderr=""),
    ) as run:
        pid = launcher._spawn_child_windows(
            python_executable="python-test",
            worker_args=("Start_panteon_v3.py", "--worker", "MEXC"),
            out_path=out_path,
            err_path=err_path,
        )

    command = run.call_args.args[0][-1]
    assert pid == 4242
    assert "-FilePath 'python-test'" in command
    assert "-WindowStyle Hidden -PassThru" in command
    assert "-RedirectStandardOutput" not in command
    assert "-RedirectStandardError" not in command


def test_v3_windows_spawn_starts_base_python_with_venv_context(tmp_path):
    launcher = _load_launcher()
    out_path = tmp_path / "worker.out.log"
    err_path = tmp_path / "worker.err.log"

    with patch.object(
        launcher,
        "_windows_python_process_context",
        return_value=(
            "base-python",
            {
                "__PYVENV_LAUNCHER__": "venv-python",
                "PATH": "venv-scripts;system-path",
                "VIRTUAL_ENV": "venv-root",
            },
        ),
    ), patch.object(
        launcher.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout="4242\n", stderr=""),
    ) as run:
        pid = launcher._spawn_child_windows(
            python_executable="venv-python",
            worker_args=("Start_panteon_v3.py", "--worker", "BITGET"),
            out_path=out_path,
            err_path=err_path,
        )

    command = run.call_args.args[0][-1]
    assert pid == 4242
    assert "$env:__PYVENV_LAUNCHER__ = 'venv-python'" in command
    assert "$env:VIRTUAL_ENV = 'venv-root'" in command
    assert "$env:PATH = 'venv-scripts;system-path'" in command
    assert "Remove-Item Env:PYTHONHOME" in command
    assert "-FilePath 'base-python'" in command


def test_v3_windows_venv_context_launches_base_python_without_stub(tmp_path):
    launcher = _load_launcher()
    venv_root = tmp_path / ".venv"
    scripts_dir = venv_root / "Scripts"
    scripts_dir.mkdir(parents=True)
    venv_python = scripts_dir / "python.exe"
    venv_python.write_text("", encoding="utf-8")
    base_python = tmp_path / "Python312" / "python.exe"
    base_python.parent.mkdir()
    base_python.write_text("", encoding="utf-8")
    (venv_root / "pyvenv.cfg").write_text(
        f"base-executable = {base_python}\n",
        encoding="utf-8",
    )

    with patch.object(launcher.os, "name", "nt"):
        executable, env = launcher._windows_python_process_context(str(venv_python))

    assert executable == str(base_python)
    assert env["__PYVENV_LAUNCHER__"] == str(venv_python)
    assert env["VIRTUAL_ENV"] == str(venv_root)
    assert str(scripts_dir) in env["PATH"].split(launcher.os.pathsep)[:1]
    assert "PYTHONHOME" not in env
