"""Run independent analysis steps on several CPU cores.

The heavy steps (per-column statistics and distribution fits, association tests, charts) are
CPU-bound Python, so threads would not help. Workers are forked processes: the DataFrame and
everything computed so far are inherited copy-on-write when the pool starts, never pickled.
Only the small task keys go to the workers and only their results come back.

Forked workers are used on Linux only: Windows has no fork, and on macOS forking a process that
has loaded the system's numerical libraries can crash the children (Python itself no longer forks
there by default). There, the chunks of a big file, which are self-contained, are reduced by fresh
``spawn`` processes (:func:`run_spawned`); the other steps run in this process, with identical results.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Iterable, Iterator

from .config import StoneConfig

_SHARED: dict[str, Any] = {}
_IN_WORKER = "ELESSAR_WORKER"


class TaskError:
    """Result of a task that raised; carries the error text instead of stopping the run."""

    def __init__(self, error: str) -> None:
        self.error = error

    def __repr__(self) -> str:
        return f"TaskError({self.error!r})"


def shared() -> dict[str, Any]:
    """The state the current pool was started with (inside a task)."""
    return _SHARED


def can_fork() -> bool:
    return sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods() and not os.environ.get(_IN_WORKER)


def n_workers(cfg: StoneConfig, n_cells: int, n_tasks: int) -> int:
    """How many processes are worth starting: 1 for small work, otherwise up to one per core.

    ``cfg.n_jobs``: ``None`` decides automatically, ``1`` forces one process, ``-1`` all cores,
    ``k`` at most k processes.
    """
    if n_tasks < 2 or not can_fork():
        return 1
    cores = os.cpu_count() or 1
    if cfg.n_jobs is None:
        if n_cells < cfg.parallel_min_cells:
            return 1
        wanted = cores
    elif cfg.n_jobs < 0:
        wanted = max(1, cores + 1 + cfg.n_jobs)
    else:
        wanted = cfg.n_jobs
    return max(1, min(wanted, n_tasks, cores))


def n_spawn_workers(cfg: StoneConfig, n_tasks: int) -> int:
    """Processes for self-contained tasks (chunks of a big file) where fork is not available or not
    safe (macOS, Windows): fresh ``spawn`` processes, which only pay off for heavy, independent work."""
    if n_tasks < 2 or can_fork() or os.environ.get(_IN_WORKER) or cfg.n_jobs == 1:
        return 1
    cores = os.cpu_count() or 1
    wanted = cores if cfg.n_jobs is None else (max(1, cores + 1 + cfg.n_jobs) if cfg.n_jobs < 0 else cfg.n_jobs)
    return max(1, min(wanted, n_tasks, cores))


@contextlib.contextmanager
def _main_hidden() -> Iterator[None]:
    """Start spawned processes without re-running the caller's script: a spawned child imports the
    ``__main__`` module, so a script calling ``elessar.see`` without an ``if __name__ == "__main__"``
    guard would run again in every child. The tasks only need elessar, so the children skip it."""
    main = sys.modules.get("__main__")
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if main is not None and k in main.__dict__}
    try:
        for k in saved:
            if k == "__spec__":
                main.__spec__ = None  # type: ignore[union-attr]
            else:
                del main.__dict__[k]  # type: ignore[union-attr]
        yield
    finally:
        if main is not None:
            main.__dict__.update(saved)


def _call(fn: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - a failing task is reported, never fatal
        return TaskError(repr(exc))


def run_spawned(fn: Callable[..., Any], args: list[tuple[Any, ...]], workers: int) -> list[Any]:
    """``[fn(*a) for a in args]`` on ``workers`` spawned processes, in order. ``fn`` and its arguments
    must be picklable (module-level functions, plain objects). If the pool cannot start or dies, the
    remaining tasks run in this process."""
    if workers <= 1 or len(args) < 2:
        return [_call(fn, a) for a in args]
    try:
        ctx = mp.get_context("spawn")
        with _main_hidden():
            pool = ctx.Pool(workers, initializer=_init_worker)  # every process starts here, main hidden
    except Exception:  # noqa: BLE001 - frozen application, no permission to start processes...
        return [_call(fn, a) for a in args]
    try:
        return pool.starmap(_call, [(fn, a) for a in args], chunksize=1)
    except Exception:  # noqa: BLE001 - unpicklable task or dead worker: finish here
        return [_call(fn, a) for a in args]
    finally:
        pool.terminate()
        pool.join()


def _init_worker() -> None:
    os.environ[_IN_WORKER] = "1"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one BLAS thread per worker: the cores are already busy
    try:  # the BLAS was loaded before the fork: limit its running thread pools too
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 - threadpoolctl missing or unsupported BLAS: just slower
        pass


def _run(fn: Callable[..., Any], item: Any) -> Any:
    try:
        return fn(item)
    except Exception as exc:  # noqa: BLE001 - a failing task is reported, never fatal
        return TaskError(repr(exc))


def run_tasks(fn: Callable[[Any], Any], items: Iterable[Any], workers: int, state: dict[str, Any] | None = None) -> list[Any]:
    """``[fn(item) for item in items]`` on ``workers`` processes, in order.

    ``fn`` must be a module-level function; it reads large inputs through :func:`shared`, which
    returns ``state``. A task that raises yields a :class:`TaskError`. If the pool itself dies
    (e.g. a worker killed for lack of memory), the remaining tasks run in this process.
    """
    items = list(items)
    previous = dict(_SHARED)
    _SHARED.clear()
    _SHARED.update(state or {})
    try:
        if workers <= 1 or len(items) < 2 or not can_fork():
            return [_run(fn, item) for item in items]
        results: list[Any] = [None] * len(items)
        done = [False] * len(items)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)  # "fork() in a multi-threaded process"
                ctx = mp.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as pool:
                    futures = [pool.submit(_run, fn, item) for item in items]
                    for i, fut in enumerate(futures):
                        try:
                            results[i] = fut.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as exc:  # noqa: BLE001 - e.g. a result that cannot be pickled
                            results[i] = TaskError(repr(exc))
                        done[i] = True
        except (BrokenProcessPool, OSError):
            for i, item in enumerate(items):
                if not done[i]:
                    results[i] = _run(fn, item)
        return results
    finally:
        _SHARED.clear()
        _SHARED.update(previous)
