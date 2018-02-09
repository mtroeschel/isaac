from asyncio import coroutine
import asyncio
import random

from aiomas import expose
import aiomas


class UnitAgent(aiomas.Agent):
    """Create a new unit agent representing *unit_model*.

    It talks to the actual unit via its unit interface *unit_if*.

    The *forecast* instance is used to make forecasts.

    """
    @classmethod
    @coroutine
    def factory(cls, container, *, ctrl_agent_addr, obs_agent_addr, unit_model,
                unit_if, planner, sleep_before_connect=True):
        # Sleep a short time to avoid that all agents try to connect to the
        # ctrl_agent at the same time.
        if sleep_before_connect:
            yield from asyncio.sleep(
                float(sleep_before_connect) * random.random())

        ctrl_agent = yield from container.connect(ctrl_agent_addr)
        obs_agent = yield from container.connect(obs_agent_addr)
        agent = cls(container, ctrl_agent, obs_agent,
                    unit_model, unit_if, planner)
        yield from ctrl_agent.register(agent, agent.addr)
        yield from obs_agent.register(agent, agent.addr)

        return agent

    def __init__(self, container, ctrl_agent, obs_agent, unit_model,
                 unit_if, planner):
        super().__init__(container)

        self.ctrl_agent = ctrl_agent
        self.obs_agent = obs_agent

        clsname, config = unit_model
        cls = aiomas.util.obj_from_str(clsname)
        self.model = cls(**config)
        assert isinstance(self.model, UnitModel)

        clsname, config = unit_if
        cls = aiomas.util.obj_from_str(clsname)
        self.unit = cls(self, **config)
        assert isinstance(self.unit, UnitInterface)

        clsname, config = planner
        cls = aiomas.util.obj_from_str(clsname)
        self.planner = cls(self, **config)
        assert isinstance(self.planner, Planner)

        # Expose/alias functions for the ControllerAgent
        self.update_forecast = self.model.update_forecast
        self.init_negotiation = self.planner.init_negotiation
        self.stop_negotiation = self.planner.stop_negotiation
        self.set_schedule = self.unit.set_schedule

        # Expose/alias function for other UnitAgents
        self.update = self.planner.update

    @expose  # Management agent (e.g., mosaik API)
    def stop(self):
        self.planner.stop()


class UnitModel:
    def __init__(self, **config):
        pass

    def get_schedule(self, schedule_id):
        raise NotImplementedError

    @expose
    def update_forecast(self, fc):
        raise NotImplementedError

    def generate_schedules(self, start, res, intervals, state):
        """Generate new schedule for the period specified by *start*, *res*
        and *intervals* based on the current unit *state*.

        :param start: The start of the schedule
        :param res: Temporal resolution of the schedule (might be different
                    from the model's internal resolution).
        :param intervals: Number of intervals in the schedule.
        :param state: The current state of the unit.
        :return: A tuple *(id, utility, data)*.

        """
        raise NotImplementedError


class UnitInterface:
    router = aiomas.rpc.Service()

    def __init__(self, agent, **config):
        pass

    @property
    def state(self):
        raise NotImplementedError

    @expose
    def update_state(self, data):
        raise NotImplementedError

    @expose
    def set_schedule(self, schedule_id):
        raise NotImplementedError

    @expose
    def get_setpoint(self, time):
        pass


class Planner:
    def __init__(self, agent, **config):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    @expose
    @coroutine
    def init_negotiation(self, neighbors, start, res, target_schedule, weights,
                         send_wm):
        raise NotImplementedError

    @expose
    def stop_negotiation(self):
        raise NotImplementedError

    @expose
    def update(self, sysconf_other, candidate_other):
        raise NotImplementedError
