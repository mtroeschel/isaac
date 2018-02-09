from asyncio import coroutine
import asyncio

import arrow
import aiomas.rpc
import numpy as np
import pytest

from openvpp_agents import planning


@pytest.fixture
def planner(ua):
    p = planning.Planner(ua, check_inbox_interval=0.01)
    ua.planner = p
    return p


@pytest.fixture
def wm():
    return planning.WorkingMemory(
        neighbors=['0', '1'],
        start=arrow.get('2015-01-01'),
        res=15,
        intervals=4,
        ts=np.array([10, 10, 10, 10]),
        weights=np.array([1, 1, 1, 1]),
        ps=[('s0', 1, np.arange(4)), ('s100', 0, np.arange(100, 104))],
        sysconf=Fixtures.sysconf_a,
        candidate=Fixtures.candidate_a)


@pytest.fixture
def model(ua):
    def generate_schedules(start, res, intervals, state):
        ua.generate_schedule_args = (start, res, intervals, state)
        return [('s0', 1, np.arange(intervals)),
                ('s100', 0, np.arange(100, 100+intervals))]

    ua.model.generate_schedules = generate_schedules
    ua.unit.state = {
        'chp_p_el': 23,
        'storage_e_th': 42,
    }

    return ua.model


class Fixtures:
    # A: {0, 1}
    sysconf_a = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        cnt=(1, 1),
    )
    candidate_a = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        perf=-12,
    )

    # B: {0, 1} (Same as A, but different agents choose the candidate)
    sysconf_b = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        cnt=(1, 1),
    )
    candidate_b = planning.Candidate(
        agent='tcp://127.0.0.1:5555/0',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        perf=-12,
    )

    # C: {0, 1}  (Same as A, but but better performance)
    sysconf_c = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(1, 9).reshape(2, 4),
        sids=('s1', 's5'),
        cnt=(0, 2),
    )
    candidate_c = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(1, 9).reshape(2, 4),
        sids=('s1', 's5'),
        perf=-8,
    )

    # D: {0, 1}  (Same as A, but but different sysconf)
    sysconf_d = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        cnt=(1, 2),
    )
    candidate_d = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        perf=-12,
    )

    # E: {1, 2}  # Overlaps with A
    sysconf_e = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/1': 0, 'tcp://127.0.0.1:5555/2': 1},
        cs=np.arange(1, 9).reshape(2, 4),
        sids=('s1', 's5'),
        cnt=[0, 1],
    )
    candidate_e = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/1': 0, 'tcp://127.0.0.1:5555/2': 1},
        cs=np.arange(8).reshape(2, 4),
        sids=('s0', 's4'),
        perf=-12,
    )

    # F: {1}  # Subset of A
    sysconf_f = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/1': 0},
        cs=np.arange(4, 8).reshape(1, 4),
        sids=('s4'),
        cnt=[1],
    )
    candidate_f = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/1': 0},
        cs=np.arange(4, 8).reshape(1, 4),
        sids=('s4'),
        perf=-18,
    )

    # G: {0, 1, 2}  # Super set of A
    sysconf_g = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1,
             'tcp://127.0.0.1:5555/2': 2},
        cs=np.arange(1, 13).reshape(3, 4),
        cnt=[0, 1, 1],
        sids=('s1', 's5', 's9'),
    )
    candidate_g = planning.Candidate(
        agent='tcp://127.0.0.1:5555/2',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1,
             'tcp://127.0.0.1:5555/2': 2},
        cs=np.arange(0, 12).reshape(3, 4),
        sids=('s0', 's4', 's8'),
        perf=-26,
    )

    # A|A merged
    sysconf_aa = sysconf_a
    candidate_aa = candidate_a

    # A|B merged
    sysconf_ab = sysconf_a
    candidate_ab = candidate_b

    # A|C merged
    sysconf_ac = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.array([[0, 1, 2, 3], [5, 6, 7, 8]]),
        sids=('s0', 's5'),
        cnt=(1, 2),
    )
    candidate_ac = candidate_c
    # Merged sysconf after perceive() *and* decide().
    # Sysconf_c contains an older (but better) OS for 5555/0. Thus its count is
    # also raised from 1 to 2 (compared to the sysconf above).
    sysconf_ac2 = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1},
        cs=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        sids=('s1', 's5'),
        cnt=(2, 2),
    )
    candidate_ac2 = candidate_c

    # C|A merged
    sysconf_ca = sysconf_ac
    candidate_ca = candidate_c

    # A|D merged
    sysconf_ad = sysconf_d
    candidate_ad = candidate_a

    # A|E merged
    sysconf_ae = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1,
             'tcp://127.0.0.1:5555/2': 2},
        cs=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [5, 6, 7, 8]]),
        sids=('s0', 's4', 's5'),
        cnt=[1, 1, 1],
    )
    candidate_ae = planning.Candidate(
        agent='tcp://127.0.0.1:5555/1',
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1,
             'tcp://127.0.0.1:5555/2': 2},
        cs=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]]),
        sids=('s0', 's4', 's4'),
        perf=-14,
    )

    # A|F merged
    sysconf_af = sysconf_a
    candidate_af = candidate_a

    # A|G merged
    sysconf_ag = planning.SystemConfig(
        idx={'tcp://127.0.0.1:5555/0': 0, 'tcp://127.0.0.1:5555/1': 1,
             'tcp://127.0.0.1:5555/2': 2},
        cs=np.array([[0, 1, 2, 3], [4, 5, 6, 7], [9, 10, 11, 12]]),
        sids=('s0', 's4', 's9'),
        cnt=[1, 1, 1],
    )
    candidate_ag = candidate_g


def test_with_repr():
    assert str(Fixtures.sysconf_a) == repr(Fixtures.sysconf_a)
    assert str(Fixtures.sysconf_a).startswith('SystemConfig')

    assert str(Fixtures.candidate_a) == repr(Fixtures.candidate_a)
    assert str(Fixtures.candidate_a).startswith('Candidate')


def test_sysconf_equal():
    a = planning.SystemConfig({'1': 0}, np.array(1), [1], [2])
    b = planning.SystemConfig({'1': 0}, np.array(1), [1], [2])
    assert a == b


@pytest.mark.parametrize('b', [
    planning.SystemConfig({'1': 1}, np.array(1), [1], [2]),
    planning.SystemConfig({'1': 0}, np.array(2), [1], [2]),
    planning.SystemConfig({'1': 0}, np.array(1), [2], [2]),
    planning.SystemConfig({'1': 0}, np.array(1), [1], [3]),
])
def test_sysconf_not_equal(b):
    a = planning.SystemConfig({'1': 0}, np.array(1), [1], [2])
    assert a != b


def test_sysconf_update():
    idx = {str(i): i for i in range(3)}
    cs = np.arange(12).reshape(3, 4)
    sids = ['s0'] * 3
    cnt = [0 for i in range(3)]
    sysconf_i = planning.SystemConfig(dict(idx), cs, sids, cnt)
    sysconf_j = sysconf_i.update('1', [0, 0, 0, 0], 's1')

    # Assert that a deep copy was made
    assert sysconf_j is not sysconf_i
    assert sysconf_j.idx is not sysconf_i.idx
    assert sysconf_j.cs is not sysconf_i.cs
    assert sysconf_j.sids is not sysconf_i.sids
    assert sysconf_j.cnt is not sysconf_i.cnt

    # Assert that the original values are unchanged
    expected = planning.SystemConfig(idx,
                                     np.array([[0, 1, 2, 3], [4, 5, 6, 7],
                                               [8, 9, 10, 11]]),
                                     ['s0', 's0', 's0'],
                                     [0, 0, 0])
    assert sysconf_i == expected

    # Assert that the new values are correct
    expected = planning.SystemConfig(idx,
                                     np.array([[0, 1, 2, 3], [0, 0, 0, 0],
                                               [8, 9, 10, 11]]),
                                     ['s0', 's1', 's0'],
                                     [0, 1, 0])
    assert sysconf_j == expected


@pytest.mark.parametrize(('i', 'j', 'expected'), [
    (Fixtures.sysconf_a, Fixtures.sysconf_a, Fixtures.sysconf_aa),
    (Fixtures.sysconf_a, Fixtures.sysconf_b, Fixtures.sysconf_ab),
    (Fixtures.sysconf_a, Fixtures.sysconf_c, Fixtures.sysconf_ac),
    (Fixtures.sysconf_a, Fixtures.sysconf_d, Fixtures.sysconf_ad),
    (Fixtures.sysconf_a, Fixtures.sysconf_e, Fixtures.sysconf_ae),
    (Fixtures.sysconf_a, Fixtures.sysconf_f, Fixtures.sysconf_af),
    (Fixtures.sysconf_a, Fixtures.sysconf_g, Fixtures.sysconf_ag),
    (Fixtures.sysconf_c, Fixtures.sysconf_a, Fixtures.sysconf_ca),
])
def test_sysconf_merge(i, j, expected):
    res = planning.SystemConfig.merge(i, j)
    if expected is i:
        assert res is expected
    assert res == expected


def test_candidate_equal():
    a = planning.Candidate('a', {'1': 0}, np.array(1), [1], 2)
    b = planning.Candidate('a', {'1': 0}, np.array(1), [1], 2)
    assert a == b


@pytest.mark.parametrize('b', [
    planning.Candidate('b', {'1': 0}, np.array(1), [1], 2),
    planning.Candidate('a', {'1': 1}, np.array(1), [1], 2),
    planning.Candidate('a', {'1': 0}, np.array(2), [1], 2),
    planning.Candidate('a', {'1': 0}, np.array(1), [2], 2),
    planning.Candidate('a', {'1': 0}, np.array(1), [1], 3),
])
def test_candidate_not_equal(b):
    a = planning.Candidate('a', {'1': 0}, np.array(1), [1], 2)
    assert a != b


def test_candidate_update():
    agent = '0'
    idx = {str(i): i for i in range(3)}
    cs = np.arange(12).reshape(3, 4)
    sids = ['s0'] * 3
    perf = -23
    candidate_i = planning.Candidate(agent, dict(idx), cs, sids, perf)
    candidate_j = candidate_i.update('1', [0, 0, 0, 0], 's1',
                                     lambda x: x.sum())

    # Assert that a deep copy was made
    assert candidate_j is not candidate_i
    assert candidate_j.idx is not candidate_i.idx
    assert candidate_j.cs is not candidate_i.cs
    assert candidate_j.sids is not candidate_i.sids

    # Assert that the original values are unchanged
    assert candidate_i.idx == idx
    assert np.array_equal(candidate_i.cs, np.array([[0, 1, 2, 3],
                                                    [4, 5, 6, 7],
                                                    [8, 9, 10, 11]]))
    assert candidate_i == planning.Candidate('0', idx,
                                             np.array([[0, 1, 2, 3],
                                                       [4, 5, 6, 7],
                                                       [8, 9, 10, 11]]),
                                             ['s0'] * 3, -23)

    # Assert that the new values are correct
    assert candidate_j == planning.Candidate('1', idx,
                                             np.array([[0, 1, 2, 3],
                                                       [0, 0, 0, 0],
                                                       [8, 9, 10, 11]]),
                                             ['s0', 's1', 's0'], 44)


@pytest.mark.parametrize(('i', 'j', 'expected'), [
    (Fixtures.candidate_a, Fixtures.candidate_a, Fixtures.candidate_aa),
    (Fixtures.candidate_a, Fixtures.candidate_b, Fixtures.candidate_ab),
    (Fixtures.candidate_a, Fixtures.candidate_c, Fixtures.candidate_ac),
    (Fixtures.candidate_a, Fixtures.candidate_d, Fixtures.candidate_ad),
    (Fixtures.candidate_a, Fixtures.candidate_e, Fixtures.candidate_ae),
    (Fixtures.candidate_a, Fixtures.candidate_f, Fixtures.candidate_af),
    (Fixtures.candidate_a, Fixtures.candidate_g, Fixtures.candidate_ag),
    (Fixtures.candidate_c, Fixtures.candidate_a, Fixtures.candidate_ca),
])
def test_candidate_merge(i, j, expected, planner, wm):
    planner.wm = wm
    res = planning.Candidate.merge(i, j, 'tcp://127.0.0.1:5555/1',
                                   wm.objective_function)
    if expected is i:
        assert res is expected
    assert res == expected


def test_wm_equal(wm):
    wm == wm


@pytest.mark.parametrize(['attr', 'val'], [
    ('neighbors', []),
    ('start', arrow.get()),
    ('res', 23),
    ('intervals', 42),
    ('ts', np.zeros(4)),
    ('weights', np.zeros(4)),
    ('ps', [('s0', 1, np.arange(4))]),
    ('ps', [('s0', 1, np.arange(4)), ('s100', 1, np.arange(100, 104))]),
    ('ps', [('s0', 1, np.arange(4)), ('s100', 0, np.arange(4))]),
    ('sysconf', Fixtures.sysconf_c),
    ('candidate', Fixtures.candidate_c),
    ('msgs_in', 1),
    ('msgs_out', 1),
])
def test_wm_not_equal(wm, attr, val):
    wm2 = planning.WorkingMemory(**wm.__dict__)
    setattr(wm2, attr, val)
    assert wm2 != wm


def test_planner_perceive(planner, wm):
    planner.wm = wm
    res_sc, res_c = planner._perceive(wm.sysconf, Fixtures.sysconf_b,
                                      wm.candidate, Fixtures.candidate_b)
    assert res_sc is Fixtures.sysconf_ab
    assert res_c is Fixtures.candidate_ab


@pytest.mark.parametrize(['agent', 'sysconf', 'candidate', 'ps', 'cs', 'cnt'],
                         [
    # 0’s POV, 1’s candidate chosen
    # no new candidate, new counter
    ('0', Fixtures.sysconf_ac, Fixtures.candidate_ac, [(0, 1, [1, 2, 3, 4])], Fixtures.candidate_ca.cs, (2, 2)),  # NOQA
    # new but worse candidate, new counter
    ('0', Fixtures.sysconf_ac, Fixtures.candidate_ac, [(0, 1, [0, 0, 0, 0])], Fixtures.candidate_ca.cs, (2, 2)),  # NOQA
    # new and better candidate, new counter
    ('0', Fixtures.sysconf_ac, Fixtures.candidate_ac, [(0, 1, [4, 3, 2, 1])], np.array([[4, 3, 2, 1], [5, 6, 7, 8]]), (2, 2)),  # NOQA

    # 1’s POV, 1’s candidate chosen
    # no new candidate, no new counter
    ('1', Fixtures.sysconf_ca, Fixtures.candidate_ca, [(0, 1, [1, 2, 3, 4])], Fixtures.candidate_ca.cs, (1, 2)),  # NOQA
    # new but worse candidate, no new counter
    ('1', Fixtures.sysconf_ca, Fixtures.candidate_ca, [(0, 1, [0, 0, 0, 0])], Fixtures.candidate_ca.cs, (1, 2)),  # NOQA
    # new and better candidate, new counter
    ('1', Fixtures.sysconf_ca, Fixtures.candidate_ca, [(0, 1, [9, 8, 7, 6])], np.array([[1, 2, 3, 4], [9, 8, 7, 6]]), (1, 3)),  # NOQA
])
def test_planner_decide(planner, wm, agent, sysconf, candidate, ps, cs, cnt):
    planner.wm = wm
    planner.wm.ps = ps
    planner.name = 'tcp://127.0.0.1:5555/%s' % agent

    new_sc, new_cand = planner._decide(sysconf, candidate)

    assert np.array_equal(new_cand.cs, cs)
    assert np.array_equal(new_sc.cs[int(agent)], new_cand.cs[int(agent)])
    assert new_sc.cnt == cnt


@pytest.mark.asyncio
def test_planner_init_negotiation(planner, wm, model, ua_mock, ctrl_mock, obs_mock):
    # ua_mocks[0] is associated with the planner, the others are neighbors:
    neighbors = [ua_mock.addr]
    wm.sysconf = planning.SystemConfig(
        {planner.name: 0}, np.array([[0, 1, 2, 3]]), ['s0'], [0])
    wm.candidate = planning.Candidate(
        planner.name, {planner.name: 0}, np.array([[0, 1, 2, 3]]), ['s0'], -34)

    # Run test
    yield from planner.init_negotiation(neighbors, wm.start, wm.res,
                                        wm.ts, wm.weights, True)

    sent_sc, sent_cand = yield from ua_mock.update_called

    assert len(planner.wm.neighbors) == 1
    assert 'UnitAgentMockProxy' in str(planner.wm.neighbors[0])

    wm.neighbors = planner.wm.neighbors
    wm.msgs_out = 1
    assert planner.wm == wm

    assert sent_sc == planner.wm.sysconf
    assert sent_cand == planner.wm.candidate

    assert planner.inbox == []

    # Check update_stats in ctrl_mock
    stats = yield from obs_mock.update_stats_called

    assert stats[0] == 'tcp://127.0.0.1:5555/0'
    assert stats[1] > 0
    assert stats[2:] == (-34, 1, 0, 1, True)

    # Check update_stats in obs_mock
    stats = yield from obs_mock.update_stats_called
    assert stats[0] == 'tcp://127.0.0.1:5555/0'
    assert stats[1] > 0
    assert stats[2:] == (-34, 1, 0, 1, True)


@pytest.mark.asyncio
def test_planner_init_negotiation_no_send_wm(planner, wm, model, ua_mock,
                                             ctrl_mock):
    neighbors = [ua_mock.addr]
    yield from planner.init_negotiation(neighbors, wm.start, wm.res,
                                        wm.ts, wm.weights, False)
    with pytest.raises(asyncio.TimeoutError):
        yield from asyncio.wait_for(ua_mock.update_called, 0.01)


def test_planner_update(planner, wm):
    planner.wm = wm
    assert wm.msgs_in == 0
    assert planner.inbox == []
    planner.update(Fixtures.sysconf_a, Fixtures.candidate_a)
    assert wm.msgs_in == 0  # Raise couner in process_inbox()
    assert planner.inbox == [(Fixtures.sysconf_a, Fixtures.candidate_a)]


@pytest.mark.asyncio
def test_planner_process_inbox_no_action(planner, wm, model, ua,
                                         ctrl_mock, obs_mock, ua_mock):
    """Other's sc and candidate don't add anything new, so the planner won't send
    new messages out."""
    wm.neighbors = [(yield from ua.container.connect(ua_mock.addr))]
    planner.wm = wm
    planner.inbox = [(Fixtures.sysconf_a, Fixtures.candidate_a)]
    ua.planner = planner

    task = aiomas.async(planner.process_inbox())

    # Check update_stats in ctrl_mock
    stats = yield from obs_mock.update_stats_called
    assert stats[0] == 'tcp://127.0.0.1:5555/0'
    assert stats[1] > 0
    assert stats[2:] == (-12, 2, 1, 0, False)

    planner.task_negotiation_stop = True
    yield from task

    assert not ua_mock.update_called.done()


@pytest.mark.parametrize(['local', 'other', 'exp'], [
    ('a', 'b', 'ab'),  # Sysconf unchanged, new candidate
    ('a', 'c', 'ac2'),  # New sysconf and candidate
    ('a', 'd', 'ad'),  # New Sysconf, candidate unchanged
])
@pytest.mark.asyncio
def test_planner_process_inbox(planner, wm, model, ua, ctrl_mock, obs_mock, ua_mock,
                               local, other, exp):
    """Sysconf and/or candidate changed, but there is no new local or global
    plan."""
    sysconf_l = getattr(Fixtures, 'sysconf_%s' % local)
    sysconf_o = getattr(Fixtures, 'sysconf_%s' % other)
    candidate_l = getattr(Fixtures, 'candidate_%s' % local)
    candidate_o = getattr(Fixtures, 'candidate_%s' % other)
    expected_sc = getattr(Fixtures, 'sysconf_%s' % exp)
    expected_cand = getattr(Fixtures, 'candidate_%s' % exp)

    wm.neighbors = [(yield from ua.container.connect(ua_mock.addr))]
    wm.sysconf = sysconf_l
    wm.candidate = candidate_l
    planner.inbox = [(sysconf_o, candidate_o)]
    planner.wm = wm
    ua.planner = planner

    task = aiomas.async(planner.process_inbox())

    res_sc, res_cand = yield from ua_mock.update_called
    assert res_sc == expected_sc
    assert res_cand == expected_cand

    # Check update_stats in ctrl_mock
    stats = yield from obs_mock.update_stats_called
    assert stats[0] == 'tcp://127.0.0.1:5555/0'
    assert stats[1] > 0
    assert stats[2] < 0
    assert stats[3:] == (2, 1, 1, True)

    planner.task_negotiation_stop = True
    yield from task


@pytest.mark.asyncio
def test_planner_stop_negotiation(planner, wm):
    planner.wm = wm
    planner.task_negotiation = asyncio.Future()
    planner.task_negotiation.set_result(None)
    yield from planner.stop_negotiation()
    assert planner.task_negotiation_stop is True
    assert planner.wm is None
