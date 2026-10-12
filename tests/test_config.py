from macro_agent.config import ASSETS, HORIZONS, RULES, Settings, api_key


def test_strategy_grid_size():
    s = Settings()
    assert len(s.assets) * len(s.rules) * len(s.horizons) == 48
    assert set(s.assets) == set(ASSETS) and s.rules == RULES and s.horizons == HORIZONS


def test_paths_live_under_data_dir():
    s = Settings()
    assert s.db_path.parent == s.data_dir and s.raw_dir.parent == s.data_dir


def test_api_key_prefers_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert api_key("TAVILY_API_KEY") == "from-env"
