import pytest
import torch
from torch.distributions.distribution import Distribution

from src.models.components.policy.normal_stochastic_policy import NormalStochasticPolicy
from src.models.components.policy_value_common_net import PolicyValueCommonNet
from src.models.components.utils import SmallConvNet
from src.models.components.value.fully_connect_value import FullyConnectValue


@pytest.mark.parametrize(
    """
    batch,
    height,
    width,
    dim_hidden,
    dim_dist,
    """,
    [
        (4, 123, 345, 623, 432),
        (32, 234, 345, 64, 123),
        (17, 234, 241, 345, 56),
    ],
)
def test_policy_value_common_net(batch, height, width, dim_hidden, dim_dist):
    policy = NormalStochasticPolicy(dim_hidden, dim_dist)
    value = FullyConnectValue(dim_hidden)
    base_model = SmallConvNet(height, width, dim_hidden)
    pvc = PolicyValueCommonNet(base_model, policy, value, height, width, dim_hidden)
    input = torch.randn(batch, 3, height, width)
    dist, value = pvc(input)
    assert dist.rsample().size() == (batch, dim_dist)
    assert isinstance(dist, Distribution)
    assert value.size() == (batch, 1)
