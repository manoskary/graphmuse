import unittest
import os
import partitura as pt
import numpy as np
import graphmuse.samplers as sam
from graphmuse.utils.graph import edges_from_note_array


class TestGraphMuse(unittest.TestCase):
    """
    Tests for graphmuse graph creation.
    """
    def test_edge_list(self):
        score_path = os.path.join(os.path.dirname(__file__), "samples", "wtc1f01.musicxml")
        score = pt.load_score(score_path)
        note_array = score.note_array()
        edges_python = np.sort(edges_from_note_array(note_array))


        edge_list, edge_types = sam.compute_edge_list(note_array['onset_div'].astype(np.int32), note_array['duration_div'].astype(np.int32))

        edges_c = np.sort(edge_list)

        self.assertTrue((edges_c==edges_python).all())