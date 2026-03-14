from __future__ import annotations

import os
import tempfile
import traceback
from dataclasses import replace
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing import shared_memory
from pathlib import Path

from ..libllaisys import DeviceType
from ..utils.nvtx import nvtx_range
from .batch_plan import BatchPlan
from .config import EngineConfig
from .model_registry import ModelRegistry, create_default_registry
from .runtime_factory import select_tp_device_ids
from .scheduler import SchedulerOutputs
from .shm_batch_plan import SharedBatchPlanBuffer, estimate_shared_batch_plan_bytes
from .worker import Worker


def _default_init_method(tag: str) -> str:
    tmp_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return f"file://{(tmp_dir / f'llaisys_{tag}_{os.getpid()}.id').resolve()}"


def _mp_worker_loop(
    config: EngineConfig,
    model_registry: ModelRegistry,
    conn: Connection,
    shm_name: str,
    shm_size: int,
) -> None:
    worker = None
    plan_buf = None
    try:
        plan_buf = SharedBatchPlanBuffer(size_bytes=int(shm_size), name=str(shm_name))
        worker = Worker(config=config, model_registry=model_registry)
        conn.send(("ready", None))
        while True:
            cmd, payload = conn.recv()
            if cmd == "execute_model_plan":
                if payload == "shm":
                    worker.execute_model_plan(plan_buf.read())
                else:
                    worker.execute_model_plan(payload)
                conn.send(("ok", None))
            elif cmd == "free_request":
                worker.free_request(int(payload))
                conn.send(("ok", None))
            elif cmd == "reset_prefix_cache":
                runner = worker.model_runner
                reset_fn = getattr(runner, "kv_reset_prefix_cache", None)
                rc = int(reset_fn()) if callable(reset_fn) else 0
                conn.send(("ok", rc))
            elif cmd == "health_check":
                conn.send(("ok", True))
            elif cmd == "close":
                worker.close()
                conn.send(("ok", None))
                return
            else:
                raise RuntimeError(f"unsupported mp worker command: {cmd}")
    except EOFError:
        return
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass
        if plan_buf is not None:
            try:
                plan_buf.close(unlink=False)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class MPExecutor:
    def __init__(self, config: EngineConfig, model_registry: ModelRegistry | None = None):
        if config.device != DeviceType.NVIDIA:
            raise RuntimeError("mp executor currently requires NVIDIA device")
        if int(config.tensor_parallel_size) <= 1:
            raise RuntimeError("mp executor requires tensor_parallel_size > 1")
        self._model_registry = model_registry if model_registry is not None else create_default_registry()
        self._config = replace(config)
        self._config.tensor_parallel_size = max(1, int(self._config.tensor_parallel_size))
        self._config.tensor_parallel_device_ids = tuple(
            int(v) for v in select_tp_device_ids(self._config.tensor_parallel_size, self._config.tensor_parallel_device_ids)
        )
        if not self._config.tp_init_method:
            self._config.tp_init_method = _default_init_method(f"tp{self._config.tensor_parallel_size}_mp")
        self._plan_shm = SharedBatchPlanBuffer(
            size_bytes=estimate_shared_batch_plan_bytes(self._config),
        )

        self._driver_cfg = replace(self._config, tp_rank=0, tp_local_rank=0)
        self._worker = None
        self._ctx = mp.get_context("spawn")
        self._children: list[tuple[int, mp.Process, Connection]] = []
        try:
            for rank in range(1, int(self._config.tensor_parallel_size)):
                child_cfg = replace(self._config, tp_rank=int(rank), tp_local_rank=int(rank))
                parent_conn, child_conn = self._ctx.Pipe()
                proc = self._ctx.Process(
                    target=_mp_worker_loop,
                    args=(
                        child_cfg,
                        self._model_registry,
                        child_conn,
                        self._plan_shm.name,
                        int(self._plan_shm.size_bytes),
                    ),
                    daemon=False,
                )
                proc.start()
                child_conn.close()
                self._children.append((rank, proc, parent_conn))
            self._worker = Worker(config=self._driver_cfg, model_registry=self._model_registry)
            for rank, proc, conn in self._children:
                status, payload = conn.recv()
                if status != "ready":
                    raise RuntimeError(f"mp worker rank={rank} failed to initialize: {payload}")
        except Exception:
            self.close()
            raise

    @property
    def worker(self) -> Worker:
        if self._worker is None:
            raise RuntimeError("mp executor worker is not initialized")
        return self._worker

    def _broadcast(self, cmd: str, payload) -> list[object]:
        for _, _, conn in self._children:
            conn.send((cmd, payload))
        replies: list[object] = []
        for rank, proc, conn in self._children:
            status, data = conn.recv()
            if status != "ok":
                raise RuntimeError(f"mp worker rank={rank} command={cmd} failed:\n{data}")
            if proc.exitcode not in (None, 0):
                raise RuntimeError(f"mp worker rank={rank} exited unexpectedly with code={proc.exitcode}")
            replies.append(data)
        return replies

    def _send_only(self, cmd: str, payload) -> None:
        for _, _, conn in self._children:
            conn.send((cmd, payload))

    def _recv_all(self, cmd: str) -> list[object]:
        replies: list[object] = []
        for rank, proc, conn in self._children:
            status, data = conn.recv()
            if status != "ok":
                raise RuntimeError(f"mp worker rank={rank} command={cmd} failed:\n{data}")
            if proc.exitcode not in (None, 0):
                raise RuntimeError(f"mp worker rank={rank} exited unexpectedly with code={proc.exitcode}")
            replies.append(data)
        return replies

    def execute_scheduler_step(self, scheduler_outputs: SchedulerOutputs) -> list[int] | None:
        with nvtx_range("py/executor/execute_scheduler_step"):
            if not scheduler_outputs.scheduled_seqs:
                return None
            batch_plan = self.worker.build_batch_plan(scheduler_outputs)
            with nvtx_range("py/executor/mp_execute_model"):
                self._plan_shm.write(batch_plan)
                self._send_only("execute_model_plan", "shm")
                self.worker.execute_model_plan(batch_plan)
                self._recv_all("execute_model_plan")
            with nvtx_range("py/executor/worker_sample_tokens"):
                return self.worker.sample_tokens()

    def free_request(self, seq_id: int) -> None:
        self._broadcast("free_request", int(seq_id))
        self.worker.free_request(int(seq_id))

    def reset_prefix_cache(self) -> int:
        child_rcs = self._broadcast("reset_prefix_cache", None)
        runner = self.worker.model_runner
        reset_fn = getattr(runner, "kv_reset_prefix_cache", None)
        rc = int(reset_fn()) if callable(reset_fn) else 0
        for child_rc in child_rcs:
            if int(child_rc) != 0:
                return int(child_rc)
        return rc

    def check_health(self) -> None:
        self._broadcast("health_check", None)

    def close(self) -> None:
        for _, proc, conn in list(self._children):
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for rank, proc, conn in list(self._children):
            try:
                if conn.poll(2.0):
                    _ = conn.recv()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        self._children.clear()
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if getattr(self, "_plan_shm", None) is not None:
            self._plan_shm.close(unlink=True)
