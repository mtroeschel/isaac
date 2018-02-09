import random


class TopologyManager:
    """Builds and manages the small world topology of a COHDA-based MAS.

    """
    def __init__(self, phi=1, seed=None):
        self._agents = None
        self._topology = None
        self._topology_phi = phi
        self._topology_seed = seed

    def make_topology(self, agents):
        # TODO: Maybe we should refer to the controller here instead of
        # using a separate list of agents...
        self._agents = agents
        assert self._agents is not None
        # All connections must be symetric and irreflexive.
        # That means, for each connection A -> B:
        # - A != B (no connection to self)
        # - If A connects to B, than B connects to A
        n_agents = len(self._agents)

        # Build list of agent proxies sorted by their address
        agent_list = [a for a, _ in sorted(self._agents.items(),
                                           key=lambda x: x[1])]
        self._topology = {agent: set() for agent in self._agents}
        if n_agents == 1:
            return self._topology

        self._build_ring(agent_list, n_agents)
        self._add_random_topology(agent_list)
        return self._topology

    def _build_ring(self, agent_list, n_agents):
        for i, agent in enumerate(agent_list):
            # Get left and right agent
            agent_left = agent_list[(i - 1) % n_agents]
            agent_right = agent_list[(i + 1) % n_agents]

            # Add addresses of left and right to the agents connect-set
            self._topology[agent].add(self._agents[agent_left])
            self._topology[agent].add(self._agents[agent_right])

    def _add_random_topology(self, agent_list):
        rnd = random.Random(self._topology_seed)

        # Add some random connections ("small world")
        for _ in range(int(len(agent_list) * self._topology_phi)):
            # We'll get *at most* n_agent * phi connections.

            agent_a = rnd.choice(agent_list)
            agent_b = rnd.choice(agent_list)

            if agent_a is agent_b:
                continue

            self._topology[agent_a].add(self._agents[agent_b])
            self._topology[agent_b].add(self._agents[agent_a])

    def topology_as_list(self):
        """Return the topology as list of tuples (agent_1, agent_2), to
        be understood as bidirectional connections.

        """
        assert self._topology is not None
        assert self._agents is not None

        connections = set()
        for agent_proxy, remotes in self._topology.items():
            # Convert proxy to str
            agent = self._agents[agent_proxy]
            for other in remotes:  # remotes is already list of str
                # Merge two directed connections into one bidirectional one
                con = (agent, other) if agent < other else (other, agent)
                connections.add(con)
        topology_list = sorted(connections)
        return topology_list
