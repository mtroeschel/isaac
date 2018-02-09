def test_exposed(ua):
    """Test if the UnitAgent exposes all required methods."""
    assert ua.update_forecast is ua.model.update_forecast
    assert ua.init_negotiation is ua.planner.init_negotiation
    assert ua.stop_negotiation is ua.planner.stop_negotiation
    assert ua.set_schedule is ua.unit.set_schedule

    assert ua.update is ua.planner.update


def test_registered(ua, ctrl_mock):
    """Test if the UnitAgents registers with the ControllerAgent."""
    assert len(ctrl_mock.registered) == 1
    print(ctrl_mock.registered)
    assert ctrl_mock.registered[0][0]._path[-1] == ua.addr[-1]  # Test proxy
    assert ctrl_mock.registered[0][1] == ua.addr
