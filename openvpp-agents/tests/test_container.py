from asyncio import coroutine
import asyncio
import os.path
import subprocess
import sys

import aiomas
import arrow
import numpy as np
import pytest

from openvpp_agents import container, util


class MyAgent(aiomas.Agent):
    def __init__(self, container, name):
        super().__init__(container)
        self._name = name

    @aiomas.expose
    def method(self, array):
        return (self._name, array)

    @aiomas.expose
    def get_time(self):
        return self.container.clock.time()


@pytest.fixture
def start():
    return arrow.get()


@pytest.yield_fixture
def proc(start):
    cmd = [
        sys.executable, container.__file__,
        '--start-date=%s' % start,
        '--log-level=debug',
        '127.0.0.1:5556',
    ]
    # Update ENV with PYTHONPATH so that the remote container can find and
    # import our MyAgent class:
    env = os.environ.copy()
    pp = env.get('PYTHONPATH', '')
    if pp:
        pp += ':'
    env['PYTHONPATH'] = pp + os.path.dirname(__file__)

    proc = subprocess.Popen(cmd, env=env, universal_newlines=True)

    yield proc
    if proc.poll() is None:
        proc.terminate()
        proc.wait(0.1)


@pytest.yield_fixture
def c(start, event_loop):
    c = aiomas.Container.create(('127.0.0.1', 5555),
                                **util.get_container_kwargs(start))
    yield c
    c.shutdown()


@pytest.mark.asyncio
def test_container(proc, start, c):
    yield from asyncio.sleep(1)
    mgr = yield from c.connect('tcp://127.0.0.1:5556/0', timeout=5)

    ta, ta_addr = yield from mgr.spawn('%s:MyAgent' % __name__, 'spam')
    assert str(ta) == "Proxy(('127.0.0.1', 5556), 'agents/1')"
    assert ta_addr == 'tcp://127.0.0.1:5556/1'

    # Test if the agent received its named and the serializers were set up:
    ret = yield from ta.method(np.arange(3))
    assert ret[0] == 'spam'
    assert np.array_equal(ret[1], np.arange(3))

    # Test setting the clock
    t = yield from ta.get_time()
    assert t == -1
    yield from mgr.set_time(42)
    t = yield from ta.get_time()
    assert t == 42

    yield from mgr.stop()
    proc.wait(timeout=0.1) == 0
