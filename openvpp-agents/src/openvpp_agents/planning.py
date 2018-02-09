"""
Planning module

See :file:`docs/planning.rst` for a detailed description of our planning
algorithm.

"""
from asyncio import coroutine
from collections import namedtuple
import asyncio
import logging
import random
import time

from aiomas import expose
import aiomas
import numpy as np

from openvpp_agents import unit


logger = logging.getLogger(__name__)


@aiomas.codecs.serializable
class SystemConfig:
    """Immutable data structure that holds the system configuration."""
    Data = namedtuple('Data', 'os, sid, count')

    def __init__(self, idx, cs, sids, cnt):
        self._idx = idx
        self._cs = cs
        self._cs.setflags(write=False)  # Make the NumPy array read-only
        self._sids = tuple(sids)
        self._cnt = tuple(cnt)

    def __eq__(self, other):
        return (
            self._idx == other._idx and
            self._sids == other._sids and
            self._cnt == other._cnt and
            np.array_equal(self._cs, other._cs)
        )

    @property
    def idx(self):
        """Mapping from agent names to indices of the corresponding agents
        within the remaining attributes."""
        return self._idx

    @property
    def cs(self):
        """Cluster schedule; list of operational schedules as 2D NumPy array.
        """
        return self._cs

    @property
    def sids(self):
        """List of schedule IDs for each OS in the cluster schedule."""
        return self._sids

    @property
    def cnt(self):
        """Counter values for each OS selection in the cluster schedule."""
        return self._cnt

    @classmethod
    def merge(cls, sysconf_i, sysconf_j):
        """Merge *sysconf_i* and *sysconf_j* and return the result.

        If *sysconf_j* does not lead to modifications to *sysconf_i*, return
        the original instance of *sysconf_i* (unchanged).

        """
        modified = False
        keyset_i = set(sysconf_i.idx)
        keyset_j = set(sysconf_j.idx)

        idx_map = {}
        cs = []
        sids = []
        cnt = []

        # Keep agents sorted to that all agents build the same index map
        for i, a in enumerate(sorted(keyset_i | keyset_j)):
            os = None
            count = -1

            # An a might be in keyset_i, keyset_j or in both!
            if a in keyset_i:
                # Use data of this a if it exists
                os, sid, count = sysconf_i.data(a)
            if a in keyset_j:
                # Use data of other a if it exists ...
                data_j = sysconf_j.data(a)
                if data_j.count > count:
                    # ... and if it is newer:
                    modified = True
                    os, sid, count = data_j

            idx_map[a] = i
            cs.append(os)
            sids.append(sid)
            cnt.append(count)

        if modified:
            sysconf = cls(idx=idx_map, cs=np.array(cs), sids=sids, cnt=cnt)
        else:
            sysconf = sysconf_i

        return sysconf

    def data(self, agent):
        """Return a tuple *(os, sid, count)* for *agent*."""
        idx = self._idx[agent]
        os = self._cs[idx]
        sid = self._sids[idx]
        count = self._cnt[idx]
        return self.Data(os, sid, count)

    def update(self, agent, os, sid):
        """Clone the current instance with an updated operation schedule *os*
        (with ID *sid*) for *agent*.  Also increase the corresponding counter
        value."""
        idx = self._idx.copy()
        i = idx[agent]
        cs = self._cs.copy()
        cs[i] = os
        sids = list(self._sids)
        sids[i] = sid
        cnt = list(self._cnt)
        cnt[i] += 1
        return self.__class__(idx=idx, cs=cs, sids=sids, cnt=cnt)


@aiomas.codecs.serializable
class Candidate:
    """Document me please.
    """
    Data = namedtuple('Data', 'os, sid')

    def __init__(self, agent, idx, cs, sids, perf):
        self._agent = agent
        self._idx = idx
        self._cs = cs
        self._cs.setflags(write=False)
        self._sids = tuple(sids)
        self._perf = perf

    def __eq__(self, other):
        return (
            self._agent == other._agent and
            self._idx == other._idx and
            self._sids == other._sids and
            self._perf == other._perf and
            np.array_equal(self._cs, other._cs)
        )

    @property
    def agent(self):
        """Name of the agent that created this candidate."""
        return self._agent

    @property
    def idx(self):
        """Mapping from agent names to indices of the corresponding agents
        within the remaining attributes."""
        return self._idx

    @property
    def cs(self):
        """Cluster schedule; list of operational schedules as 2D NumPy array.
        """
        return self._cs

    @property
    def sids(self):
        """List of schedule IDs for each OS in the cluster schedule."""
        return self._sids

    @property
    def perf(self):
        """Performance of this candidate."""
        return self._perf

    @classmethod
    def merge(cls, candidate_i, candidate_j, agent, perf_func):
        """Return a new candidate for *agent* based on the agent's
        *candidate_i* or the new *candidate_j*.

        If *candidate_j* does *not* lead to modifications in *candidate_i*,
        return the original *candidate_i* instance (unchanged).

        """
        keyset_i = set(candidate_i.idx)
        keyset_j = set(candidate_j.idx)
        candidate = candidate_i  # Default candidate is *i*

        if keyset_i < keyset_j:
            # Use *j* if *K_i* is a true subset of *K_j*
            candidate = candidate_j
        elif keyset_i == keyset_j:
            # Compare the performance if the keyset is equal
            if candidate_j.perf > candidate_i.perf:
                # Choose *j* if it performs better
                candidate = candidate_j
            elif candidate_j.perf == candidate_i.perf:
                # If both perform equally well, order them by name
                if candidate_j.agent < candidate_i.agent:
                    candidate = candidate_j
        elif keyset_j - keyset_i:
            # If *K(j)* shares some entries with *K(i)*, update *candidate_i*
            idx_map = {}
            cs_buf = []
            sids = []
            # Index should be sorted by agent name (because determinism)
            for i, a in enumerate(sorted(keyset_i | keyset_j)):
                idx_map[a] = i
                if a in keyset_i:
                    data = candidate_i.data(a)
                else:
                    data = candidate_j.data(a)
                cs_buf.append(data.os)
                sids.append(data.sid)

            cs = np.array(cs_buf)
            perf = perf_func(cs)
            candidate = Candidate(agent, idx_map, cs, sids, perf)

        # If "candidate" and "candidate_i" are equal,
        # they must also have the same identity
        assert (candidate == candidate_i) == (candidate is candidate_i)

        return candidate

    def data(self, agent):
        """Return a tuple *(os, sid)* for *agent*."""
        idx = self._idx[agent]
        os = self._cs[idx]
        sid = self._sids[idx]
        return self.Data(os, sid)

    def update(self, agent, os, sid, perf_func):
        """Clone the current instance with an updated operation schedule *os*
        (with ID *sid*) for *agent*.  Also evaluate the performance of the
        new candidate using *perf_func*."""
        agent = agent
        idx = dict(self._idx)
        cs = self._cs.copy()
        sids = list(self._sids)
        i = idx[agent]
        cs[i] = os
        sids[i] = sid
        perf = perf_func(cs)
        return self.__class__(agent=agent, idx=idx, cs=cs, sids=sids,
                              perf=perf)


class WorkingMemory:
    """Stores all negotiation related state."""
    def __init__(self, neighbors, start, res, intervals, ts, weights, ps,
                 sysconf, candidate, msgs_in=0, msgs_out=0):
        self.neighbors = neighbors
        self.start = start
        self.res = res
        self.intervals = intervals
        self.ts = ts
        self.weights = weights
        self.ps = ps

        self.sysconf = sysconf
        self.candidate = candidate
        self.msgs_in = msgs_in
        self.msgs_out = msgs_out

    def __eq__(self, other):
        ret = (
            self.neighbors == other.neighbors and
            self.start == other.start and
            self.res == other.res and
            self.intervals == other.intervals and
            np.array_equal(self.ts, other.ts) and
            np.array_equal(self.weights, other.weights) and
            self.sysconf == other.sysconf and
            self.candidate == other.candidate and
            self.msgs_in == other.msgs_in and
            self.msgs_out == other.msgs_out
        )
        # The list of possible schedules is an ugly beast
        ret = ret and (len(self.ps) == len(other.ps))
        for ps_i, ps_j in zip(self.ps, other.ps):
            ret = ret and (ps_i[:2] == ps_j[:2])
            ret = ret and np.array_equal(ps_i[2], ps_j[2])

        return ret

    def objective_function(self, cluster_schedule):
        # Return the negative(!) sum of all deviations, because bigger scores
        # mean better plans (e.g., -1 is better then -10).
        sum_cs = cluster_schedule.sum(axis=0)
        diff = np.abs(self.ts - sum_cs)
        w_diff = diff * self.weights
        result = -np.sum(w_diff)
        return result


class Planner(unit.Planner):
    def __init__(self, agent, check_inbox_interval=0.01):
        self.agent = agent
        self.name = agent.addr
        self.check_inbox_interval = check_inbox_interval

        self.task_negotiation = None  # Task for negotiation
        self.task_negotiation_stop = False
        self.inbox = []
        self.wm = None

    def stop(self):
        if self.task_negotiation and not self.task_negotiation.done():
            self.task_negotiation.cancel()

    @expose
    @coroutine
    def init_negotiation(self, neighbors, start, res, target_schedule, weights,
                         send_wm):
        # TODO: rename "send_wm" to "send_update"?

        logger.debug('%s init_negotiation @%s' % (self.agent, time.monotonic()))

        futs = [self.agent.container.connect(n) for n in neighbors]
        neighbors = yield from asyncio.gather(*futs)

        intervals = len(target_schedule)
        possible_schedules = self.agent.model.generate_schedules(
            start, res, intervals, self.agent.unit.state)

        self.wm = WorkingMemory(neighbors=neighbors,
                                start=start, res=res, intervals=intervals,
                                ts=target_schedule, weights=weights,
                                ps=possible_schedules,
                                sysconf=None, candidate=None)

        # Take the first possible OS and ignore its utility (not important yet)
        schedule_id, _, os = possible_schedules[0]

        sysconf = SystemConfig(
            idx={self.name: 0},
            cs=np.array([os], dtype=float),
            sids=[schedule_id],
            cnt=[0])
        self.wm.sysconf = sysconf

        perf = self.wm.objective_function(sysconf.cs)
        candidate = Candidate(
            agent=self.name,
            idx={self.name: 0},
            cs=np.array([os], dtype=float),
            sids=[schedule_id],
            perf=perf)
        self.wm.candidate = candidate

        self.task_negotiation_stop = False
        self.task_negotiation = aiomas.async(self.process_inbox())

        if send_wm:
            for neighbor in self.wm.neighbors:
                self.wm.msgs_out += 1
                logger.debug('%s sending message %d', self.agent,
                             self.wm.msgs_out)

                aiomas.async(neighbor.update(sysconf, candidate))

        logger.debug('%s updating obs @%s from init_negotiation.' % (self.agent, time.monotonic()))
        yield from self._update_obs_agent(send_wm)
        # yield from self._update_ctrl_agent(send_wm)

    @expose
    def update(self, sysconf_other, candidate_other):
        logger.debug('%s received message @%s' % (self.agent, time.monotonic()))
        self.inbox.append((sysconf_other, candidate_other))

    @coroutine
    def process_inbox(self):
        while not self.task_negotiation_stop:
            if type(self.check_inbox_interval) is tuple:
                a, b = self.check_inbox_interval
                diff = b - a
                t = a + random.random() * diff
            else:
                t = self.check_inbox_interval
            yield from asyncio.sleep(t)

            if not self.inbox:
                continue

            wm = self.wm
            sysconf = wm.sysconf
            candidate = wm.candidate
            inbox, self.inbox = self.inbox, []

            for sysconf_other, candidate_other in inbox:
                self.wm.msgs_in += 1
                # Update initial sysconf/candidate with each message
                sysconf, candidate = self._perceive(
                    sysconf, sysconf_other,
                    candidate, candidate_other)

            state_changed = (sysconf is not wm.sysconf or
                             candidate is not wm.candidate)

            if state_changed:
                # sysconf or candidate changed.  Check if we can do better.
                sc, cand = self._decide(sysconf, candidate)
                wm.sysconf = sc
                wm.candidate = cand

                self._act()

            # yield from self._update_ctrl_agent(state_changed)
            logger.debug('%s updating obs @%s.' % (self.agent, time.monotonic()))
            yield from self._update_obs_agent(state_changed)

    @expose
    def stop_negotiation(self):
        logger.debug('%s stop_negotiation', self.agent)
        self.task_negotiation_stop = True
        yield from self.task_negotiation
        candidate = self.wm.candidate
        self.inbox = []
        self.wm = None
        # print(self.agent, ' stop neg cand ', candidate)
        yield from self.agent.obs_agent.update_final_cand(candidate)
        # return candidate

    def _perceive(self, sysconf, sysconf_other, candidate, candidate_other):
        """Merge the system configuration and candidates from *self* and
        *other*.

        """
        # It's important to *not* update the WorkingMemory here!  We want to
        # keep our original sysconf/candidate until we know if and which new
        # sysconf/candidate we choose.
        sysconf = SystemConfig.merge(sysconf, sysconf_other)
        candidate = Candidate.merge(candidate, candidate_other, self.name,
                                    self.wm.objective_function)
        return (sysconf, candidate)

    def _decide(self, sysconf, candidate):
        """Try to find a schedule that leads to a better candidate."""
        name = self.name
        current_sid = sysconf.data(name).sid
        best = candidate.data(name)
        best_os, best_sid = best.os, best.sid  # Expand "best"
        new_os = self._get_new_os(current_sid, candidate, name)

        if new_os is not None:
            # We have new os that is locally better then the old one. Check if
            # it is also globally better.
            new_os, new_sid = new_os  # new_os is actually a tuple!
            new_candidate = candidate.update(name, new_os, new_sid,
                                             self.wm.objective_function)
            if new_candidate.perf > candidate.perf:
                # We found a new candidate
                candidate = new_candidate
                best_os = new_os
                best_sid = new_sid

                # print("new schedule: ", best_sid)

        if current_sid != best_sid:
            # We need a new counter value if
            # - we create a new, better candidate
            # - the updated candidate contains a different schedule then we
            #   stored in our current sysconf.
            #
            # -> We need a new count if the os in the candidate is different
            #    from the os in the sysconf.
            sysconf = sysconf.update(name, best_os, best_sid)

        return sysconf, candidate

    def _act(self):
        """Broadcast new sysconf and candidate to all neighbors."""
        wm = self.wm
        for neighbor in wm.neighbors:
            wm.msgs_out += 1
            logger.debug('%s sending message %d', self.agent, wm.msgs_out)
            aiomas.async(neighbor.update(wm.sysconf, wm.candidate))

    def _get_new_os(self, current_sid, candidate, name):
        """Return a new *os* from the list of possible schedules *ps* if we
        find one that's better then the *current_sid*.

        Return ``None`` if we don't find one.

        """
        best_perf = float('-inf')
        new_os = None
        new_sid = None
        for sid, _, os in self.wm.ps:
            # print(sid, os)
            # Currently, we use the "global" check here, but this might change
            # so don't return the candidate directly.
            new_c = candidate.update(name, os, sid, self.wm.objective_function)
            if new_c.perf > best_perf:
                best_perf = new_c.perf
                new_os = os
                new_sid = sid

        if current_sid == new_sid:
            return None

        return new_os, new_sid

    # def _update_ctrl_agent(self, msgs_sent):
    #     wm = self.wm
    #     return self.agent.ctrl_agent.update_stats(
    #         agent=self.name,
    #         t=time.monotonic(),
    #         perf=wm.candidate.perf,
    #         n_os=len(wm.candidate.cs),  # Number of known op. scheds.
    #         msgs_in=wm.msgs_in,
    #         msgs_out=wm.msgs_out,
    #         msgs_sent=msgs_sent)

    def _update_obs_agent(self, msgs_sent):
        wm = self.wm
        return self.agent.obs_agent.update_stats(
            agent=self.name,
            t=time.monotonic(),
            perf=wm.candidate.perf,
            n_os=len(wm.candidate.cs),  # Number of known op. scheds.
            msgs_in=wm.msgs_in,
            msgs_out=wm.msgs_out,
            msgs_sent=msgs_sent)
