from src import config


def test_config_values():
    assert config.H == 21
    assert config.SEED == 42
    assert config.LOOKBACKS == (21, 63, 126)
    assert config.START_DATE == "2010-01-01"


def test_set_seeds_is_deterministic():
    import numpy as np

    config.set_seeds()
    a = np.random.rand(3)
    config.set_seeds()
    assert (a == np.random.rand(3)).all()
