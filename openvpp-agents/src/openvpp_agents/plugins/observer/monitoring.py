import asyncio
import time

from asyncio import coroutine

import numpy as np

from openvpp_agents.core.observer.monitoring import Monitoring

import psutil


class PerformanceMonitoring(Monitoring):
    def __init__(self, dbfile):
        super().__init__(dbfile)
        self._stop_monitor_run = True
        self._task_monitor_run = None
        self._loop = asyncio.get_event_loop()

    def setup(self, date):
        """Setup performance monitoring for a new run / negotiation.
        This also triggers the setup of the parent monitoring class.

        :param date: The begin date of the target_schedule for the
        negotitation.
        """
        super().setup(date)

        # Monitoring setup
        self._stop_monitor_run = False
        self._task_monitor_run = self._loop.run_in_executor(
            None, self._monitor_run)

    @coroutine
    def flush(self, target_schedule, weights, solution):
        # performance monitoring
        self._stop_monitor_run = True
        perf_data = yield from self._task_monitor_run

        self._store_perf_data(self._topgroup, perf_data)
        # self._db.flush()

        # general monitoring flushes database
        yield from super().flush(target_schedule, weights, solution)

    def _monitor_run(self):
        def add_children(proc):
            for c in proc.children():
                procs.append(c)
                add_children(c)

        procs = [psutil.Process()]
        add_children(procs[0])

        mem_bytes = psutil.virtual_memory().total
        mem_percent = mem_bytes / 100
        data = []
        try:
            while not self._stop_monitor_run:
                cpu = sum(p.cpu_percent() for p in procs)
                mem = sum(p.memory_percent() for p in procs) * mem_percent
                t = time.monotonic()
                data.append((t, cpu, mem))
                time.sleep(0.01)
        except psutil.AccessDenied:
            # May happen when the monitored procs terminated while we slept
            pass

        return data

    def _store_perf_data(self, group, perf_data):
        dtype = np.dtype([
            ('t', 'float64'),
            ('cpu_percent', 'float32'),
            ('mem_bytes', 'uint64'),
        ])
        perf_data = np.array(perf_data, dtype=dtype)
        group.create_dataset('perf_data', data=perf_data)
