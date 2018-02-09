import asyncio
import logging
from asyncio import coroutine

import aiomas
from aiomas import expose

from openvpp_agents.core.controller.management import TopologyManager
from openvpp_agents.core.controller.scheduling import DayAheadScheduler

logger = logging.getLogger(__name__)

# TODO: maybe simplify the module structure? is 'management' useful?
DEFAULT_TOPMNGR = 'openvpp_agents.core.controller.management:TopologyManager'


class ControllerAgent(aiomas.Agent):
    """Controller for system of unit agents.

    Create pairs of (controller, observer) via the factory :meth:`factory`.

    Tasks:

    - Initiate a new planning process if triggered by the UserAgent

    - Trigger Observer to monitor the negotiation process

    - Broadcast negotiation results, i.e. new schedules, to UnitAgents

    """
    @classmethod
    @coroutine
    def factory(cls, container, ctrl_kwargs, obs_cls, obs_kwargs):
        """Factory for general setup of controller-observer system.

        As in a controller-observer system these agents have to know
        eachother before any other process in the system starts, this factory
        creates and returns both agents after the setup is finished.

        The observer class has to be provided and thus can be modified,
        if needed. In the given setup, both agents reside in the same
        container.

        """
        ctrl = cls(container, **ctrl_kwargs)
        ctrl_proxy = yield from container.connect(ctrl.addr)
        obs = obs_cls(container, ctrl_proxy, **obs_kwargs)
        # register obs at ctrl
        ctrl_proxy.register_observer(obs)
        return (ctrl, obs)

    def __init__(self, container, *,
                 n_agents=None,
                 negotiation_single_start=True,
                 negotiation_timeout=15 * 60,
                 topology_manager=DEFAULT_TOPMNGR,
                 topology_phi=1,
                 topology_seed=None,
                 scheduling_res=15 * 60,
                 scheduling_period=48 * 3600):
        assert scheduling_period % scheduling_res == 0

        super().__init__(container)

        self._agents = {}
        self._n_agents = n_agents
        self._agents_registered = asyncio.Future()
        if n_agents is None:
            self._agents_registered.set_result(True)

        self._observer = None
        self._observer_registered = asyncio.Future()

        # init topology management
        cls = aiomas.util.obj_from_str(topology_manager)
        self._topology_manager = cls(topology_phi, topology_seed)
        assert isinstance(self._topology_manager, TopologyManager)

        # scheduling / negotiation
        self._da_scheduler = DayAheadScheduler(self,
                                               scheduling_res,
                                               scheduling_period)

        self._neg_single_start = negotiation_single_start
        self._neg_timeout = negotiation_timeout
        self._task_negotiation = None
        self._neg_done = None

    @expose
    def stop(self):
        if not self._task_negotiation.done():
            self._task_negotiation.cancel()
            try:
                yield from self._task_negotiation
            except asyncio.CancelledError:
                pass  # Because we just cancelled it!

    @expose
    def register(self, agent_proxy, addr):
        """Register the agent represented by *agent_proxy* with the address
        *addr*."""
        logger.info('Agent registered: %s', addr)
        self._agents[agent_proxy] = addr
        if self._n_agents is not None and len(self._agents) == self._n_agents:
            self._agents_registered.set_result(True)

    @expose
    def register_observer(self, agent_proxy):
        """Register the observer agent represented by *agent_proxy*
        with the address *addr*."""
        logger.info('Observer agent registered: %s', str(agent_proxy))
        self._observer = agent_proxy
        self._observer_registered.set_result(True)

    @expose
    @coroutine
    def update_thermal_demand_forecast(self, target_schedule, weights, fc):
        # process logic for update is implemented in DayAheadScheduler
        yield from self._da_scheduler.update_thermal_demand_forecast(
            target_schedule, weights, fc)
        # process management for negotations and other tasks is kept locally
        self._task_negotiation = aiomas.async(self._run_negotiation(fc.start))

    # REFACTOR: This coroutine is part of day-ahead-scheduling
    @coroutine
    def _run_negotiation(self, start):
        # initialize local tasks / futures
        self._neg_done = asyncio.Future()

        # build topology (as topology may be negotiation-specific)
        topology = self._topology_manager.make_topology(self._agents)
        conn_data = self._topology_manager.topology_as_list()

        # init scheduler
        res, ts, weights = self._da_scheduler.init_negotiation(start)

        # begin observation of negotiation
        if (self._observer is None):
            raise RuntimeError('Observer not registered yet!')

        yield from self._observer.start_observation(conn_data, start,
                                                    ts, weights)

        # start negotiation by informing unit agents
        send_wm = True
        for a in self._agents:
            aiomas.async(a.init_negotiation(tuple(topology[a]),
                                            start, res, ts, weights, send_wm))
            # Sets "send_wm" to False for all remaining agents if only
            # a single agent should start sending its WM.
            send_wm = not self._neg_single_start

        # Wait until agents finish (observer informs controller via
        # :meth:negotiation_finished) or timeout is reached.
        try:
            yield from asyncio.wait_for(self._neg_done, self._neg_timeout)
        except asyncio.TimeoutError:
            pass  # we handle this error by stopping the negotiation below

        # stop negotiation
        futs = [a.stop_negotiation() for a in self._agents]
        yield from asyncio.gather(*futs)

        # get solution from observer
        solution = yield from self._observer.pass_solution()

        # TODO wording: solution = EInsatzplan = cluster schedule
        yield from self._broadcast_solution(solution)

        # finalize negotiation by updating scheduler
        self._da_scheduler.finalize_negotiation(solution)

    @expose
    def negotiation_finished(self):
        """Called by Observer after termination of negotiation has been
        detected.

        """
        logger.info('Negotiation finished info rcvd by observer.')
        self._neg_done.set_result(True)

    @coroutine
    def _broadcast_solution(self, solution):
        futs = []
        print("Solution perf:", solution.sids, solution.perf)
        for a, addr in self._agents.items():
            schedule_id = solution.sids[solution.idx[addr]]
            futs.append(a.set_schedule(schedule_id))
        yield from asyncio.gather(*futs)

    @expose
    @coroutine
    def get_day_ahead_plan(self, dap_start, dap_end):
        """Return the day ahead plan (which is the result of a negotiation).

        Called by external parties (like the mosaik agent).

        """
        # local process control
        yield from self._task_negotiation

        # extract dap
        dap = self._da_scheduler.get_day_ahead_plan(dap_start, dap_end)
        return dap
