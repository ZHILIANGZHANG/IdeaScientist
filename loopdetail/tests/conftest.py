import pytest
import torch

from loopdetail.models.registry import load_adapter, model_spec
from loopdetail.utils import set_threads

set_threads(4)


@pytest.fixture(scope="session")
def fake_ouro():
    return load_adapter(model_spec("fake-ouro"), "cpu", "float32")


@pytest.fixture(scope="session")
def fake_huginn():
    return load_adapter(model_spec("fake-huginn"), "cpu", "float32")


@pytest.fixture(params=["fake_ouro", "fake_huginn"])
def looped(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def ids(looped):
    return torch.tensor([looped.tok.encode("Box AB contains the apple. Box CD contains the hammer. Box AB")])
