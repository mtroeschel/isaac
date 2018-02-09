import asyncio
import functools


class TerminationDetector:
    """Abstract base class for termination detectors.

    """
    def __init__(self, num_agents=0):
        self.terminated = None
        self._num_agents = num_agents

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def detect(self):
        raise NotImplementedError


class MessageCounter(TerminationDetector):
    """Detects termination of a negotiation by counting ingoing and
    outgoing messages in an agent system.

    """
    def __init__(self, num_agents=0):
        super().__init__(num_agents)
        self._agent_msgs = None

    def reset(self):
        """Reset termination detection.

        """
        self._agent_msgs = {}
        self.terminated = asyncio.Future()

    def update(self, agent, msgs_in, msgs_out):
        """Update internal data and trigger termination detection.

        :param agent: ID of agent
        :param msgs_in: number of ingoing messages of *agent*
        :param msgs_out: number of outgoing messages of *agent*
        """
        assert self._agent_msgs is not None
        self._agent_msgs[agent] = (msgs_in, msgs_out)
        self.detect()

    def detect(self):
        """Detect termination of a negotiation by counting and comparing
        ingoing and outgoing messages in an agent system.

        """
        assert self._agent_msgs is not None
        if len(self._agent_msgs) == self._num_agents:
            # If we've got updates from all agents, check if the negotiation
            # terminated.
            # In the lambda, "x" is the aggregated result so far, "y" is the
            # (msg_in, msgs_out) tuple for the current agent:
            sums = functools.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]),
                                    self._agent_msgs.values())
            if sums[0] == sums[1]:
                # If the sum of incoming and outgoing messages for the whole
                # system is equal, we are done:
                assert not self.terminated.done()
                self.terminated.set_result(True)
