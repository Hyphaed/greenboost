"""gb_cnn_tune: CPU-only checks (memory format conversion keeps the numbers)."""
import torch
import torch.nn as nn

import gb_cnn_tune as ct


def _net():
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.Conv2d(8, 3, 3, padding=1)).eval()


def test_channels_last_keeps_the_output():
    net, x = _net(), torch.randn(1, 3, 20, 20)
    with torch.no_grad():
        ref = net(x)
    ct.apply(net)
    assert net[0].weight.is_contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        out = net(ct.to_input(x))
    assert torch.allclose(out, ref, atol=1e-5) and not torch.backends.cudnn.benchmark


def test_env_switches(monkeypatch):
    monkeypatch.setenv("GB_CNN_CHANNELS_LAST", "0")
    net = ct.apply(_net())
    assert not net[0].weight.is_contiguous(memory_format=torch.channels_last)
    assert not ct.to_input(torch.randn(1, 3, 4, 4)).is_contiguous(memory_format=torch.channels_last)
    monkeypatch.setenv("GB_CNN_CUDNN_BENCHMARK", "1")
    ct.apply(_net())
    assert torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False


def test_models_without_2d_convs_are_left_alone():
    lin = nn.Sequential(nn.Linear(4, 4))
    assert ct.apply(lin) is lin
