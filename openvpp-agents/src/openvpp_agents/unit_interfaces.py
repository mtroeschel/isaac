import collections

from aiomas import expose
import aiomas

from openvpp_agents.unit import UnitInterface


STATE_BUFSIZE = 15  # We want to store that many state updates from our unit


class MosaikInterface(UnitInterface):
    router = aiomas.rpc.Service()

    def __init__(self, agent, agent_id, unit_id):
        self._agent = agent
        self._utcnow = agent.container.clock.utcnow
        self._aid = agent_id
        self._uid = unit_id
        self._state = collections.deque(maxlen=STATE_BUFSIZE)
        self._schedule = None
        self._last_setpoint = None
        # Register the interface's router as subrouter of the agent:
        agent.router.set_sub_router(self.router, 'unit')

    @property
    def state(self):
        return self._state

    @expose
    def update_state(self, data):
        row = (self._utcnow(), data)
        self._state.append(row)

    @expose
    def set_schedule(self, schedule_id):
        schedule = self._agent.model.get_schedule(schedule_id)
        if self._schedule is None:
            self._schedule = schedule
        else:
            schedule.lstrip(self._schedule.start)  # Strip values in the past
            self._schedule.extend(schedule)  # Update/extend existing schedule

    @expose
    def get_setpoint(self, time):
        """Return the setpoint (*P_el* in W) for a given *time*."""
        if not self._schedule:
            return None

        if time < self._schedule.start:
            return None

        # The following code roughly does this:
        #   setpoint = self._schedule[time]  # Get sp for current time
        #   self._schedule.lstrip(time, inclusive=True)  # Strip old stuff
        sp = self._schedule.lstrip(time, inclusive=True)
        setpoint = sp.data[-1]

        if setpoint == self._last_setpoint:
            return None

        self._last_setpoint = setpoint
        return (self._aid, self._uid, setpoint)
