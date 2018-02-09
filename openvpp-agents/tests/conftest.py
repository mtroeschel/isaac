from asyncio import coroutine
import asyncio
import unittest.mock as mock

from aiomas import expose
import aiomas
import numpy as np
import pytest

from openvpp_agents import controller, observer, planning, unit, util


@pytest.yield_fixture
def containers(event_loop):
    containers = []
    for i in range(2):
        c = aiomas.Container.create(
            ('127.0.0.1', 5555 + i), codec=aiomas.MsgPack,
            extra_serializers=util.get_extra_serializers())
        containers.append(c)

    yield containers

    for c in containers:
        c.shutdown()


@pytest.fixture
def ctrl(ctrl_obs):
    return ctrl_obs[0]
    # cleanup is done in ctrl_obs fixture

@pytest.fixture
def ctrl_mock(event_loop, containers):
    ctrl = ControllerAgentMock(containers[1])
    return ctrl


@pytest.fixture
def obs(ctrl_obs):
    return ctrl_obs[1]
    # cleanup is done in ctrl_obs fixture

@pytest.yield_fixture
def ctrl_obs(containers):
    # ctrl = controller.ControllerAgent(containers[0])
    ctrl_kwargs = {}
    obs_kwargs = {}
    ctrl, obs  =  aiomas.run(controller.ControllerAgent.factory(
        containers[0], ctrl_kwargs, observer.ObserverAgent, obs_kwargs))
    # obs = observer.ObserverAgent(containers[0], ctrl.addr)
    yield (ctrl, obs)
    ctrl.stop()
    obs.stop()



@pytest.fixture
def obs_mock(event_loop, containers):
    obs = ObserverAgentMock(containers[1])
    return obs


@pytest.yield_fixture
def ua(containers, ctrl_mock, obs_mock, event_loop):
    ua = aiomas.run(unit.UnitAgent.factory(
        containers[0],
        ctrl_agent_addr=ctrl_mock.addr,
        obs_agent_addr=obs_mock.addr,
        unit_model=(__name__ + ':UnitModelMock', {}),
        unit_if=(__name__ + ':UnitIfMock', {}),
        planner=(__name__ + ':PlannerMock', {}),
        sleep_before_connect=False))
    yield ua
    ua.stop()


@pytest.fixture
def ua_mock(ua_mocks):
    """Generate a single ua mock w/o connection to a ctrl."""
    return ua_mocks[0]


@pytest.fixture
def ua_mocks(event_loop, containers, ctrl_obs):
    """Generate multiple ua mocks connected to the ctrl."""
    ctrl, obs = ctrl_obs
    ua_mocks = [UnitAgentMock(containers[1]) for i in range(3)]
    solution = planning.Candidate(
        ua_mocks[0].addr, {a.addr: i for i, a in enumerate(ua_mocks)},
        np.arange(12, dtype=float).reshape(4, 3), ['s0', 's4', 's8'], 1)

    futs = []
    for ua in ua_mocks:
        futs.append(ua.setup(ctrl.addr, obs.addr))
        ua.solution = solution

    event_loop.run_until_complete(asyncio.gather(*futs))

    return ua_mocks


class ControllerAgentMock(aiomas.Agent):
    def __init__(self, c):
        super().__init__(c)
        self.registered = []

    @expose
    def register(self, agent_proxy, addr):
        self.registered.append((agent_proxy, addr))

    @expose
    def update_stats(self, agent, t, perf, n_os, msgs_in, msgs_out, msgs_sent):
        self.update_stats_called.set_result(
            (agent, t, perf, n_os, msgs_in, msgs_out, msgs_sent))


class ObserverAgentMock(aiomas.Agent):
    def __init__(self, c):
        super().__init__(c)
        self.registered = []
        self.update_stats_called = asyncio.Future()
        self.final_cand = None

    @expose
    def update_stats(self, agent, t, perf, n_os, msgs_in, msgs_out, msgs_sent):
        self.update_stats_called.set_result(
            (agent, t, perf, n_os, msgs_in, msgs_out, msgs_sent))
    @expose
    def register(self, agent_proxy, addr):
        self.registered.append((agent_proxy, addr))

    @expose
    def update_final_cand(self, candidate):
        self.final_cand = candidate


class UnitAgentMock(aiomas.Agent):
    def __init__(self, c):
        super().__init__(c)
        self.planner = PlannerMock(self)
        self.model = UnitModelMock()
        self.unit = UnitIfMock(self)
        self.ctrl_agent = None
        self.solution = None
        self.obs_agent = None
        self.update_forecast_called = asyncio.Future()
        self.init_negotiation_called = asyncio.Future()
        self.stop_negotiation_called = asyncio.Future()
        self.set_schedule_called = asyncio.Future()
        self.update_called = asyncio.Future()

    @coroutine
    def setup(self, ctrl_addr, obs_addr):
        self.ctrl_agent = yield from self.container.connect(ctrl_addr)
        yield from self.ctrl_agent.register(self, self.addr)
        self.obs_agent = yield from self.container.connect(obs_addr)
        yield from self.obs_agent.register(self, self.addr)

    @expose
    def update_forecast(self, fc):
        self.update_forecast_called.set_result(fc)

    @expose
    def init_negotiation(self, neighbors, start, res, ts, weights, send_wm):
        res = (list(sorted(neighbors)), start, res, list(ts), list(weights),
               send_wm)
        self.init_negotiation_called.set_result(res)

    @expose
    def stop_negotiation(self):

        self.stop_negotiation_called.set_result(None)
        yield from self.obs_agent.update_final_cand(self.solution)
        return self.solution

    @expose
    def set_schedule(self, schedule_id):
        self.set_schedule_called.set_result(schedule_id)

    @expose
    def update(self, sysconf, candidate):
        self.update_called.set_result((sysconf, candidate))


UnitModelMock = mock.create_autospec(unit.UnitModel, spec_set=True)
UnitIfMock = mock.create_autospec(unit.UnitInterface, spec_set=True)
PlannerMock = mock.create_autospec(unit.Planner, spec_set=True)
