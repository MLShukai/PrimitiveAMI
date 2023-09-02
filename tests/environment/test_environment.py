from src.environment.environment import Environment


class EnvironmentImple(Environment):
    def setup():
        return

    def observe(self):
        return {}

    def affect(self, action):
        return

    def teardown(self):
        return


class TestEnvironment:
    def test_is_abstract(self):
        assert Environment.__abstractmethods__ == frozenset({"observe", "affect", "setup", "teardown"})

    def test_step(self):
        env = EnvironmentImple()
        assert env.step(None) == {}
