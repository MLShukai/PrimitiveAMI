from src.models.aggregations.neural_networks import NeuralNetworks


class TestNeuralNetworks:
    def test_is_abstract(self):
        assert NeuralNetworks.__abstractmethods__ == frozenset({"build_agent_models"})
