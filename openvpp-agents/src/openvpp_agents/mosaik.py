from asyncio import coroutine
import asyncio
import io
import json
import logging
import lzma
import multiprocessing
import numpy as np
import os
import time

from aiomas import expose
import aiomas
import arrow
import click

from openvpp_agents import (
    controller,
    observer,
    util,
)


logger = logging.getLogger(__name__)


@click.command()
@click.option('--log-level', '-l', default='info', show_default=True,
              type=click.Choice(['debug', 'info', 'warning', 'error',
                                 'critical']),
              help='Log level for the MAS')
@click.argument('addr', metavar='HOST:PORT', callback=util.validate_addr)
def main(addr, log_level):
    """Open VPP multi-agent system."""
    try:
        aiomas.run(until=run(addr, log_level))
    finally:
        asyncio.get_event_loop().close()


@coroutine
def run(addr, log_level):
    mosaik_api = MosaikAPI(log_level)
    try:
        # Create an RPC connection to mosaik. It will handle all incoming
        # requests until one of them sets a result for "self.stopped".
        logger.debug('Connecting to %s:%s ...' % addr)
        mosaik_con = yield from aiomas.rpc.open_connection(
            addr, rpc_service=mosaik_api, codec=aiomas.JSON)
        mosaik_api.mosaik = mosaik_con.remote

        def on_connection_reset_cb(exc):
            # Gets called if the remote side closes the connection
            # (e.g., because it dies).
            if not mosaik_api.stopped.done():
                # If the remote side stopped, we also want to stop ...
                mosaik_api.stopped.set_result(True)

        mosaik_con.on_connection_reset(on_connection_reset_cb)

        # Wait until mosaik asks us to stop
        logger.debug('Waiting for mosaik requests ...')
        yield from mosaik_api.stopped
    except KeyboardInterrupt:
        logger.info('Execution interrupted by user')
    finally:
        logger.debug('Closing socket and terminating ...')
        yield from mosaik_con.close()
        yield from mosaik_api.finalize()


class MosaikAPI:
    """Interface to mosaik."""
    router = aiomas.rpc.Service()

    def __init__(self, log_level, host='localhost', port_start=10000):
        self.log_level = log_level
        self.host = host
        self.port_start = port_start
        self.step_size = 60  # seconds

        self.meta = {
            'api_version': '2.2',
            'models': {
                'Agent': {
                    'public': True,
                    'params': ['model_conf'],
                    'attrs': ['chp_p_el', 'storage_e_th'],
                },
            },
        }
        # This future is gonna be triggered by "self.stop()".  "run()" waits
        # for it and stops the main event loop when it is done.
        self.stopped = asyncio.Future()

        # Set in run()
        self.mosaik_con = None  # Rpc connection to mosaik
        self.mosaik = None  # Proxy object for mosaik

        # Set in init()
        self.sid = None
        self.n_agents = None
        self.config = None
        self.container = None  # Container for all agents (all local)
        self.start_date = None

        self.ua = None  # UserAgent
        self.ca = None  # ControllerAgent
        self.ma = None  # MosaikAgent

        self.agent_containers = []  # Proxies to agent containers
        self.container_procs = []  # Popen instances for agent containers

        # Set/updated in create()/setup_done()
        self._aids = []
        self._model_conf = {}
        self._agents = {}  # (UnitAgent, PlanningAgent) pairs by aid
        self._t_last_step = None

    @coroutine
    def finalize(self):
        """Stop all agents, containers and subprocesses when the simulation
        finishes."""
        futs = [a.stop() for a in self._agents.values()]
        futs += [c.stop() for c in self.agent_containers]
        yield from asyncio.gather(*futs)
        if self.ca:
            yield from self.ca.stop()
        if self.oa:
            yield from self.oa.stop()
        if self.ua:
            yield from self.ua.stop()
        if self.ma:
            # remember to handle as coro if you change it to be one!
            assert self.ma.stop() is None

        # Shutdown sub-processes
        for p in self.container_procs:
            yield from p.wait()
            print('Container process terminated')

        yield from self.container.shutdown(as_coro=True)

    @expose
    @coroutine
    def init(self, sid, *, start_date, n_agents, config):
        """Create a local agent container and the mosaik agent."""
        self.sid = sid
        self.n_agents = n_agents
        self.config = config

        # In-proc container for UserAgent, ControllerAgent and MosaikAgent
        addr = (self.host, self.port_start)
        container_kwargs = util.get_container_kwargs(start_date)
        container_kwargs.update(as_coro=True)  # Behave like a coro
        self.container = yield from aiomas.Container.create(
            addr, **container_kwargs)
        self.start_date = arrow.get(start_date).to('utc')

        ca_conf = config['ControllerAgent']
        oa_conf = config['ObserverAgent']

        self.ca, self.oa = yield from controller.ControllerAgent.factory(
            self.container, ca_conf, observer.ObserverAgent, oa_conf)

        ua_conf = config['UserAgent']
        self.ua = UserAgent(self.container, self.ca.addr, **ua_conf)

        ma_conf = config['MosaikAgent']
        self.ma = MosaikAgent(self.container, **ma_conf)

        # Remote containers for Unit- and PlanningAgents
        c, p = yield from self._start_containers(
            self.host, self.port_start + 1, start_date, self.log_level)
        self.agent_containers = c
        self.container_procs = p

        return self.meta

    @expose
    @coroutine
    def create(self, num, model, **model_conf):
        # We do not yet want to instantiate the agents, b/c we don't know
        # to which units they will be connected.  So we just store the model
        # conf. for the agents and start them later.  Mosaik will get a list
        # of entities nonetheless.
        entities = []

        # Number of agents we "started" so far:
        n_agents = len(self._model_conf)

        # Store agent config.  They get instantiate in setup_done().
        self._model_conf.update(model_conf['model_conf'])

        assert len(self._model_conf) == n_agents + num, \
            'Some units from "model_conf" are already configured.'

        for i in range(n_agents, n_agents + num):
            eid = 'Agent_%s' % i
            entities.append({'eid': eid, 'type': model})
            aid = '%s.%s' % (self.sid, eid)
            self._aids.append(aid)

        return entities

    @expose
    @coroutine
    def setup_done(self):
        if len(self._model_conf) != self.n_agents:
            raise RuntimeError('%d agents required but only %d instantiated.' %
                               (self.n_agents, len(self._model_conf)))

        relations = yield from self.mosaik.get_related_entities(self._aids)
        n_containers = len(self.agent_containers)
        futs = []
        for i, (aid, units) in enumerate(relations.items()):
            # in debug mode, start all unit and planning agents in same
            # container to acchieve determinism in messages
            if self.log_level == 'debug':
                c = self.agent_containers[0]
            else:
                c = self.agent_containers[i % n_containers]
            assert len(units) == 1
            uid, _ = units.popitem()
            model_conf = self._model_conf.pop(uid)
            futs.append(self._spawn_ua(c, aid, uid, model_conf))

        assert not self._model_conf  # Should have used all items
        results = yield from asyncio.gather(*futs)
        self._agents = {aid: agent for aid, agent in results}
        self.ma.agents = self._agents
        self._t_last_step = time.monotonic()

    @expose
    @coroutine
    def step(self, t, inputs):
        # Update the time for the agents
        yield from self._set_time(t)

        # Prepare input data and forward it to the agents
        data = {}
        for eid, attrs in inputs.items():
            input_data = {}
            for attr, values in attrs.items():
                assert len(values) == 1  # b/c we're only connected to 1 unit
                _, value = values.popitem()
                input_data[attr] = value
                data['%s.%s' % (self.sid, eid)] = input_data
        yield from self.ma.update_agents(data)

        # Check if we got new schedules and send them to mosaik
        futures = asyncio.gather(self.ua.step_done(), self.ma.step_done())
        timeout = self._t_last_step + self.step_size - time.monotonic()
        # TODO: "wait_for" raises a asyncio.TimeoutError if a timeout occurs.
        # Do we just want to ignore it or shall we crash?  What's the behavior
        # of ua.step_done()?  Should we shield() the call do "step_done()"?
        yield from asyncio.wait_for(futures, timeout=timeout)

        self._t_last_step = time.monotonic()

        t_next = t + self.step_size

        setpoints = yield from self.ma.get_setpoints(
            self.start_date.replace(seconds=t_next))
        inputs = {aid: {uid: {'chp_p_el': setpoint}}
                  for aid, uid, setpoint in setpoints}
        if inputs:
            yield from self.mosaik.set_data(inputs)

        return t_next

    @expose
    def stop(self):
        self.stopped.set_result(True)

    @coroutine
    def _start_containers(self, host, start_port, start_date, log_level):
        addrs = []
        procs = []
        for i in range(multiprocessing.cpu_count()):
            addr = (host, start_port + i)
            addrs.append('tcp://%s:%s/0' % addr)
            cmd = ['openvpp-container',
                   '--start-date=%s' % start_date,
                   '--log-level=%s' % log_level,
                   '%s:%s' % addr]
            procs.append(asyncio.async(asyncio.create_subprocess_exec(*cmd)))

        procs = yield from asyncio.gather(*procs)
        futs = [self.container.connect(a, timeout=10) for a in addrs]
        containers = yield from asyncio.gather(*futs)
        return containers, procs

    @coroutine
    def _spawn_ua(self, container, aid, uid, model_conf):
        """Configure agents and connect simulated entities from mosaik with
        unit agents and unit agents with planning agents.

        """
        unit_agent, ua_addr = yield from container.spawn(
            'openvpp_agents.unit:UnitAgent.factory',
            ctrl_agent_addr=self.ca.addr,
            obs_agent_addr=self.oa.addr,
            unit_model=(
                'openvpp_agents.unit_models:ChpModel',
                model_conf,
            ),
            unit_if=(
                'openvpp_agents.unit_interfaces:MosaikInterface',
                {'agent_id': aid, 'unit_id': uid},
            ),
            planner=(
                'openvpp_agents.planning:Planner',
                self.config['Planner'],
            ),
        )

        return aid, unit_agent

    @coroutine
    def _set_time(self, time):
        self.container.clock.set_time(time)
        futs = [c.set_time(time) for c in self.agent_containers]
        yield from asyncio.gather(*futs)


class UserAgent(aiomas.Agent):
    """This agents is the user interface to the MAS.

    Tasks:

    - Feed new thermal forcasts to CA every 15min
    - Extract day-ahead schedule every day at 0900 and send it to ...?

    """
    def __init__(self, container, dca_addr, *,
                 demand_file,
                 target_dir,
                 result_dir=None,
                 tz='Europe/Berlin',
                 dap_start={'hour': 9},
                 dap_res=900,
                 dap_planning_horizon={'days': 1}):
        super().__init__(container)

        # TODO: Refactor dca's name
        self._dca_addr = dca_addr  # dca = (d)ap (c)ontroller (a)gent
        self._tz = tz
        self._dap_start = dap_start
        self._dap_res = dap_res
        self._dap_planning_horizon = dap_planning_horizon

        self._demand_gen = self._init_demand_gen(demand_file)
        self._target_gens = self._init_target_gens(target_dir)
        self._result_dir = result_dir
        self._fc = None  # thermal forecast
        self._ts = None  # target schedule
        self._weights = None

        self._task_run = aiomas.async(self.run())
        self._task_dap = None
        self._stop = False

    @coroutine
    def stop(self):
        self._stop = True
        tasks = []
        if self._task_run is not None and not self._task_run.done():
            self._task_run.cancel()
            tasks.append(self._task_run)
        if self._task_dap is not None and not self._task_dap.done():
            self._task_dap.cancel()
            tasks.append(self._task__dap)

        # Wait for tasks ignoring their exceptions (CancelledErrors):
        yield from asyncio.gather(*tasks, return_exceptions=True)

    @coroutine
    def run(self):
        clock = self.container.clock
        # TODO: Rename dca to ctrl!
        dca = yield from self.container.connect(self._dca_addr)

        while not self._stop:
            # Wait until next planning begins
            next_date = self._get_next_date(**self._dap_start)
            yield from clock.sleep_until(next_date)

            # Get new dap
            self._task_dap = aiomas.async(
                self._get_dap(clock, next_date, dca))
            dap = yield from self._task_dap
            self._task_dap = None

            self._store_results(dap, next_date)

    @coroutine
    def step_done(self):
        # TODO: catch and ignore CancelledError
        if self._task_dap is None:
            return
        if self._task_dap.done():
            return self._task_dap.result()
        yield from asyncio.shield(self._task_dap)

    def _init_demand_gen(self, demand_file):
        open = lzma.open if demand_file.endswith('.xz') else io.open
        demand_gen = open(demand_file, 'rt')
        demand_meta = json.loads(next(demand_gen).strip())
        demand_start = arrow.get(demand_meta['start_time']).to('utc')
        demand_res = demand_meta['interval_minutes'] * 60
        return (demand_start, demand_res, demand_gen)

    def _init_target_gens(self, target_dir):
        """Return a dict with key=start_time and val=(target resolution,
        target file generator) for all target files in target dir.
        Target files are expected a json meta data string followed by
        csv lines in format p_el, weight"""
        target_gens = {}
        file_list = []
        for f in os.listdir(target_dir):
            file_list.append(os.path.join(target_dir, f))

        for target_file in file_list:
            open = lzma.open if target_file.endswith('.xz') else io.open
            target_gen = open(target_file, 'rt')
            target_meta = json.loads(next(target_gen).strip())
            target_start = arrow.get(target_meta['start_time']).to('utc')
            target_res = target_meta['interval_minutes'] * 60
            target_gens[target_start] = (target_res, target_gen)
        return target_gens

    def _get_next_date(self, hour=0, minute=0):
        """Return a UTC date for the next occurence of *hour:minute* o'clock
        in local time."""
        now = self.container.clock.utcnow().to(self._tz)
        if now.format('HH:mm:ss,SSS') > '%02d:%02d' % (hour, minute):
            days = 1  # Next occurrence is tomorrow
        else:
            days = 0  # Next occurrence is today

        next_date = now.replace(days=days, hour=hour, minute=minute, second=0,
                                microsecond=0).to('utc')
        return next_date

    @coroutine
    def _get_dap(self, clock, next_date, dca):
        fc = self._get_thermal_demand_forecast(next_date, hours=48)
        (ts, weights) = self._get_target(next_date)

        yield from dca.update_thermal_demand_forecast(ts, weights, fc)
        start, end = self._get_dap_dates(clock)

        # print('UserAgent/_get_dap: start: %s, end: %s' % (start, end))

        dap = yield from dca.get_day_ahead_plan(start, end)
        return dap

    def _get_thermal_demand_forecast(self, date, hours):
        """Return a :class:`~util.TimeSeries` with a thermal demand forecast
        for the period from *date* for *hours* hours."""
        # The forecast usually covers a period of 48h and we request a new
        # forecast every 24 hours.
        #
        # The forecast data comes from a large file.  We don't want to keep the
        # full content of that file in memory.
        #
        # So in order to create a new forecast, we
        # 1. reuse the old one,
        # 2. strip all data from before the new start date, and
        # 3. extend it with new data from the data file
        #

        demand_start, demand_res, demand_gen = self._demand_gen
        if self._fc is None:
            # Create an initial, empty forecast if there is none
            self._fc = util.TimeSeries(date, demand_res, [])

        # Strip old data from the forecast
        self._fc.lstrip(date)

        # We need data for [start_date, end_date) from the data file
        # (in the first run, we load 2 days, but on second day, we only
        # have to add the data for the second, as the first has been
        # read already)
        period = hours * 3600 - self._fc.period
        start_date = self._fc.end
        end_date = start_date.replace(seconds=period)

        # Skip lines until we reach "start_date"
        start_diff = util.get_intervals_between(start_date, demand_start,
                                                demand_res)
        [next(demand_gen) for i in range(start_diff)]

        # Get the data from the file and extend the forecast
        n_lines = util.get_intervals_between(end_date, start_date, demand_res)
        fc_data = []
        for i in range(n_lines):
            data = next(demand_gen).strip().split(',')
            p_el, p_th_heat, p_th_water = map(float, data)
            sum_p_th = p_th_heat + p_th_water
            fc_data.append(sum_p_th)
        new_data = util.TimeSeries(start_date, demand_res, fc_data)
        self._fc.extend(new_data)

        self._demand_gen = (end_date, demand_res, demand_gen)

        return self._fc

    def _get_target(self, date):
        """Sets and returns self._ts and self._weights, both TimeSeries,
        depending on the given date. If no target has been given for date,
        zero target with zero weights is given.

        Raises a RunTimeError if more than one target is passed for one day,
        as this has to be handled by the intraday routines."""

        # get appropriate target
        start_keys = []
        for start_key in self._target_gens:
            date_diff = start_key - date
            if date_diff.days >= 0 and date_diff.days < 1:
                start_keys.append(start_key)

        if len(start_keys) > 1:
            raise RuntimeError('More than 1 target given for day %s' % date)

        # TODO FIXME: We have hardcoded stuff here for dap far too much
        if len(start_keys) == 0:
            logger.info('UserAgent:_get_target: no target for day %s,'
                  'taking zero ts' % date)
            target_res = 15*60  # TODO FIXME: hardcoded default resolution
            # day-ahead-planning relies on targets starting at midnight local
            # time, so we have to use this for zero target as well
            dap_target_start = util.get_tomorrow(date, 'local')
            self._ts = util.TimeSeries(dap_target_start,
                                       target_res,
                                       [0 for i in range(96)])
            self._weights = util.TimeSeries(dap_target_start,
                                            target_res,
                                            [0 for i in range(96)])
        else:
            # date_str = start_keys[0].format('YYYY-MM-DD HH:mm:ss ZZ')
            logger.info('UserAgent:_get_target: starting with target %s'
                        % start_keys[0])
            target_start = start_keys[0]
            target_res, target_gen = self._target_gens[target_start]

            # read all target and weights data entries from file
            ts_data = []
            weights_data = []
            for line in target_gen:
                data = line.strip().split(',')
                val, weight = map(float, data)
                ts_data.append(val)
                weights_data.append(weight)

            self._ts = util.TimeSeries(target_start, target_res, ts_data)
            self._weights = util.TimeSeries(target_start, target_res,
                                            weights_data)

        # target has to be dap-compatible:
        assert self._ts.start.to('local').hour == 0
        return (self._ts, self._weights)

    def _get_dap_dates(self, clock):
        start = self._ts.start
        end = start.to('local').replace(days=1)
        return start.to('utc'), end.to('utc')

    def _store_results(self, dap, start):
        print("---- finished scheduling for ", str(start), "---")
        if self._result_dir is None:
            logger.info("Scheduling results will not be stored. "
                        "Pass result_dir to UserAgent to change.")
            return

        print("Writing scheduling results to directory ",
              str(self._result_dir))
        os.makedirs(self._result_dir, exist_ok=True)
        rf = os.path.join(self._result_dir, str(self._fc.start))
        # with open(rf, 'w') as f:
        np.savetxt(rf, dap, delimiter='\n', fmt='%1.3f')
        print("-----------------------------------------------------------")


class MosaikAgent(aiomas.Agent):
    """Bridge between UserAgent and simulated units."""
    def __init__(self, container):
        super().__init__(container)
        self.agents = None  # aid: UnitInterface proxy, set by MosaikAPI

    def stop(self):
        pass

    @coroutine
    def update_agents(self, data):
        """Update the unit agents with new data from mosaik."""
        futs = [self.agents[aid].unit.update_state(input_data)
                for aid, input_data in data.items()]
        yield from asyncio.gather(*futs)

    @coroutine
    def step_done(self):
        # There's nothing yet to wait for ...
        return True

    def get_setpoints(self, time):
        futs = [a.unit.get_setpoint(time) for a in self.agents.values()]
        setpoints = yield from asyncio.gather(*futs)
        return [s for s in setpoints if s is not None]  # Filter "None" vals
