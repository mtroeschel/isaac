import numpy as np

from chpsim.sim import CHP, ChpConfig


def test_set_setpoint():
    """Test setting a list of setpoints under the following conditions:

    - No setpoint given, setpoint unchanged
    - CHP is off, setpoint > 0, min_off_time not reached -> remain off
    - CHP is off, setpoint > 0, min_off_time reached -> turn on
    - CHP is off, setpoint == 0 -> remain off
    - CHP is on, setpoint > 0 -> adjust power level accordingly
    - CHP is on, setpoint == 0 -> turn off, reset off-time counter

    """
    p_levels = [(0, 0), (1, 2), (2, 4)]
    chp = CHP([
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
    ])
    chp.chp_setpoint = np.array([1, 0, 0, 0, 1, 1], int)
    chp.chp_p_el = chp.chp_p_levels[range(len(chp.chp_setpoint)),
                                    chp.chp_setpoint,
                                    0]
    chp.chp_p_th = chp.chp_p_levels[range(len(chp.chp_setpoint)),
                                    chp.chp_setpoint,
                                    1]
    chp.chp_off_since = np.array([10, 9, 10, 10, 10, 10], int)

    chp.set_setpoint([None, 2, 2, 0, 2, 0])

    assert list(chp.chp_off_since) == [10, 9, 10, 10, 10, 0]
    assert list(chp.chp_setpoint) == [1, 0, 2, 0, 2, 0]
    assert list(chp.chp_p_el) == [1, 0, 2, 0, 2, 0]
    assert list(chp.chp_p_th) == [2, 0, 4, 0, 4, 0]


def test_chp_status():
    """Test if the controllers turns on/off the CHP according to the current
    storage energy:

    - storage full, on -> off
    - storage full, off -> (remain) off
    - storage empty, on -> no change
    - storage empty, off -> on
    - else -> no change

    """
    p_levels = [(0, 0), (1, 2), (2, 4)]
    chp = CHP([
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.5),
        ChpConfig(p_levels, 10, 1, 0.0),
        ChpConfig(p_levels, 10, 1, 0.0),
        ChpConfig(p_levels, 10, 1, 1.0),
        ChpConfig(p_levels, 10, 1, 1.0),
    ])
    chp.chp_setpoint = np.array([1, 0, 1, 0, 1, 0], int)

    def set_setpoint(setpoints):
        assert list(setpoints) == [None, None, None, 2, 0, 0]

    chp.set_setpoint = set_setpoint

    chp.check_chp_status()


def test_step_basic():
    chp = CHP([ChpConfig(
        chp_p_levels=[(0, 0), (1, 60), (2, 120)],
        chp_min_off_time=10,
        storage_e_th_max=30,
        storage_e_th_init=2,
    )])
    setpoints = {
        10: [1],
        12: [2],
        35: [0],
        43: [1],
        44: [2],
        47: [0],
    }

    data = []
    for i in range(50):
        setpoint = setpoints.get(i, None)
        chp.step(60, setpoint)
        data.append((
            i,
            chp.chp_setpoint[0],
            chp.chp_off_since[0],
            chp.storage_e_th[0],
        ))

    assert data == [
        (0,  0, 11, 1.0),
        (1,  0, 12, 0.0),
        (2,  2, 13, 1.0),  # Storage empty, engine turned on
        (3,  2, 14, 2.0),
        (4,  2, 15, 3.0),
        (5,  2, 16, 4.0),
        (6,  2, 17, 5.0),
        (7,  2, 18, 6.0),
        (8,  2, 19, 7.0),
        (9,  2, 20, 8.0),
        (10, 1, 21, 8.0),  # Reduce power output for a while
        (11, 1, 22, 8.0),
        (12, 2, 23, 9.0),  # Reset setpoint to max.
        (13, 2, 24, 10.0),
        (14, 2, 25, 11.0),
        (15, 2, 26, 12.0),
        (16, 2, 27, 13.0),
        (17, 2, 28, 14.0),
        (18, 2, 29, 15.0),
        (19, 2, 30, 16.0),
        (20, 2, 31, 17.0),
        (21, 2, 32, 18.0),
        (22, 2, 33, 19.0),
        (23, 2, 34, 20.0),
        (24, 2, 35, 21.0),
        (25, 2, 36, 22.0),
        (26, 2, 37, 23.0),
        (27, 2, 38, 24.0),
        (28, 2, 39, 25.0),
        (29, 2, 40, 26.0),
        (30, 2, 41, 27.0),
        (31, 2, 42, 28.0),
        (32, 2, 43, 29.0),
        (33, 2, 44,  30.0),
        (34, 0, 1,  29.0),  # Storage full, engine turned off
        (35, 0, 2,  28.0),
        (36, 0, 3,  27.0),
        (37, 0, 4,  26.0),
        (38, 0, 5,  25.0),
        (39, 0, 6,  24.0),
        (40, 0, 7,  23.0),
        (41, 0, 8,  22.0),
        (42, 0, 9,  21.0),
        (43, 0, 10, 20.0),  # Cannot turn engine on yet
        (44, 2, 11, 21.0),  # Now we can
        (45, 2, 12, 22.0),
        (46, 2, 13, 23.0),
        (47, 0, 1,  22.0),  # And we turn it off again
        (48, 0, 2,  21.0),
        (49, 0, 3,  20.0),
    ]
