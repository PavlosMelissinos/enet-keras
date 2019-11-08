from src.models.enet_unpooling import decoder as sut


def test_bottleneck():
    assert sut.bottleneck(None, None) is None
