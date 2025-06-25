import pytest

from thesis import (
    get_transition_probs,
    stable_unstable,
    determin_next_F,
    determin_next_G,
    pick_lyapunov,
    run_sim,
    run_sim_1,
    run_sim_2,
    run_sim_3,
    find_most_common,
    V,
    has_converged
)

@pytest.mark.parametrize("st", [0, 1, 5, 10])
def test_get_transition_probs(st):
    probs = get_transition_probs(st)
    assert isinstance(probs, dict)
    assert set(probs.keys()) == {0, 1, 2}
    assert all(0 <= v <= 1 for v in probs.values())

@pytest.mark.parametrize("st,dt", [(0, 0), (0, 1), (5, 2)])
def test_stable_unstable_determinism(st, dt):
    prs = get_transition_probs(st)
    result = stable_unstable(st, dt, prs)
    assert isinstance(result, int)
    assert result >= 0

@pytest.mark.parametrize("st", [0, 1, 5])
def test_determin_next_F(st):
    action, g_values = determin_next_F(st)
    assert action in {0, 1, 2}
    assert isinstance(g_values, dict)
    assert len(g_values) == 3

@pytest.mark.parametrize("st", [0, 1, 5])
def test_determin_next_G(st):
    action, g_values = determin_next_G(st)
    assert action in {0, 1, 2}
    assert isinstance(g_values, dict)
    assert len(g_values) == 3

@pytest.mark.parametrize("st,a,b,c", [(0, 0.5, 2, 0), (3, 0.5, 2, 1)])
def test_pick_lyapunov(st, a, b, c):
    action, prs = pick_lyapunov(st, a, b, c)
    assert action in {0, 1, 2}
    assert isinstance(prs, dict)

@pytest.mark.parametrize("arr,expected", [
    ([0, 0, 1, 2, 2, 2, 1], 2),
    ([5, 5, 3, 3, 5], 5)
])
def test_find_most_common(arr, expected):
    assert find_most_common(arr) == expected

@pytest.mark.parametrize("x,a,b,c,h,expected", [
    (0, 1, 2, 0, 1, 0),
    (1, 2, 2, 1, 1, 3),
    (2, 0.5, 2, -1, 1, 1.0)
])
def test_V_function(x, a, b, c, h, expected):
    assert V(x, a, b, c, h) == pytest.approx(expected)

