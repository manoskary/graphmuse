from unittest import TestCase
import torch
from graphmuse.nn import MusGConv


class TestConvolutionalModels(TestCase):
    def test_musgconv(self):
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 100))
        edge_attr = torch.randn(100, 32)
        conv = MusGConv(16, 32, in_edge_channels=32, norm_msg=True)
        out = conv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (10, 32))