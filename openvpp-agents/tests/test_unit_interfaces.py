import arrow
import numpy as np
import pytest

from openvpp_agents import unit_interfaces
from openvpp_agents.util import TimeSeries as TS


@pytest.fixture
def mif(ua):
    ua.unit = unit_interfaces.MosaikInterface(ua, 'a-0', 'u-0')
    return ua.unit


def test_mosaik_interface_udpate_state(mif):
    state = object()
    mif.update_state(state)
    # state is stored in deque as (timestamp, state) tuple
    assert mif.state[-1][1] is state


def test_mosaik_interface_set_schedule(mif):
    start = arrow.get('2015-01-01')
    res = 60

    schedules = {
        0: TS(start.replace(minutes=1), res, np.array([0, 2])),
        1: TS(start.replace(minutes=0), res, np.array([1, 3, 5])),
        2: TS(start.replace(minutes=2), res, np.array([0, 2, 3])),
    }
    mif._agent.model.get_schedule = lambda i: schedules[i]

    mif.set_schedule(0)
    assert mif._schedule == TS(start.replace(minutes=1), res, [0, 2])

    # Schedule 1 starts earlier then mif._schedule, so its first value should
    # get stripped:
    mif.set_schedule(1)
    assert mif._schedule == TS(start.replace(minutes=1), res, [3, 5])

    mif.set_schedule(2)
    assert mif._schedule == TS(start.replace(minutes=1), res, [3, 0, 2, 3])


def test_mosaik_interface_get_setpoint(mif):
    start = arrow.get('2015-01-01')
    sp = mif.get_setpoint(start)
    assert sp is None

    mif._schedule = TS(start.replace(days=1), 60, [1, 1, 0, 0, 2, 3])

    sp = mif.get_setpoint(start)
    assert sp is None

    sp = mif.get_setpoint(start.replace(days=1, minutes=0))
    assert sp == ('a-0', 'u-0', 1)

    sp = mif.get_setpoint(start.replace(days=1, minutes=1))
    assert sp is None

    sp = mif.get_setpoint(start.replace(days=1, minutes=2))
    assert sp == ('a-0', 'u-0', 0)

    sp = mif.get_setpoint(start.replace(days=1, minutes=3))
    assert sp is None

    # We skip "minutes=4" on purpose!

    sp = mif.get_setpoint(start.replace(days=1, minutes=5))
    assert sp == ('a-0', 'u-0', 3)

    sp = mif.get_setpoint(start.replace(days=1, minutes=6))
    assert sp is None
