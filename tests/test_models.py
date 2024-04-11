from unittest import TestCase
import torch
from graphmuse.nn import MusGConv, MetricalConvLayer


class TestConvolutionalModels(TestCase):
    def test_musgconv(self):
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 100))
        edge_attr = torch.randn(100, 32)
        conv = MusGConv(16, 32, in_edge_channels=32, norm_msg=True)
        out = conv(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (10, 32))

    def test_metrical_conv_layer(self):
        note_feats = torch.randn(40, 16)
        metrical_feats = torch.randn(10, 16)
        edge_index_src = torch.randint(0, 40, (1, 100))
        edge_index_dst = torch.randint(0, 10, (1, 100))
        edge_index = torch.cat((edge_index_src, edge_index_dst), dim=0)
        batch = torch.randint(0, 4, (10, )).sort()[0]
        conv = MetricalConvLayer(16, 32)
        out = conv(metrical_feats, note_feats, edge_index, batch)
        self.assertEqual(out.shape, (40, 32))
