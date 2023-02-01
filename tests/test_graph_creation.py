import unittest
import os
import partitura as pt
from graphmuse.utils.graph import edges_from_note_array


class TestGraphMuse(unittest.TestCase):
    """
    Tests for graphmuse graph creation.
    """
    def test_edge_list(self):
        score_path = os.path.join(os.path.dirname(__file__), "samples", "wtc1f01.musicxml")
        score = pt.load_score(score_path)
        note_array = score.note_array()
        edges_python = edges_from_note_array(note_array)
        edges_c =
