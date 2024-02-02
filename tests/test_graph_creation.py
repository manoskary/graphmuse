import unittest
import os
import partitura as pt
import numpy as np
import graphmuse.samplers as sam
from graphmuse.utils import create_score_graph, create_random_music_graph
import time


def edges_from_note_array(note_array):
    '''Turn note_array to list of edges.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.

    Returns
    -------
    edg_src : np.array
        The edges in the shape of (2, num_edges).
    '''

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]))[0]: #& (note_array["id"] != x["id"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where((x["onset_div"] < note_array["onset_div"]) & (x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    end_times = note_array["onset_div"] + note_array["duration_div"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_div"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_div"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)

    edges = np.array([edg_src, edg_dst])
    return edges


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
        self.assertTrue(edges_c.shape==edges_python.shape)
        self.assertTrue((edges_c==edges_python).all())
        print("Edgle list creation assertions passed")

    def test_graph_creation(self):
        part = pt.load_score(pt.EXAMPLE_MUSICXML)[0]
        note_array = part.note_array()
        measures = part.measures
        features = np.random.rand(len(note_array), 10)
        graph = create_score_graph(features, note_array, measures=measures, add_beats=True)
        self.assertTrue(graph["note"].num_nodes == len(note_array))
        self.assertTrue(graph["beat"].num_nodes == int(note_array["onset_beat"].max()) + 1)
        self.assertTrue(graph["measure"].num_nodes == len(measures))
        self.assertTrue(graph.num_edges == 23)

    def test_edge_creation_speed(self):
        # create a random graph
        max_nodes = 10000
        num_notes_per_voice = max_nodes // 4
        dur = np.random.randint(2, 16, size=(4, num_notes_per_voice))
        ons = np.cumsum(np.concatenate((np.zeros((4, 1)), dur), axis=1), axis=1)[:, :-1]
        dur = dur.flatten()
        ons = ons.flatten()

        pitch = np.row_stack((np.random.randint(70, 80, size=(1, num_notes_per_voice)),
                              np.random.randint(50, 60, size=(1, num_notes_per_voice)),
                              np.random.randint(60, 70, size=(1, num_notes_per_voice)),
                              np.random.randint(40, 50, size=(1, num_notes_per_voice)))).flatten()

        beats = ons / 4
        note_array = np.vstack((ons, dur, pitch, beats))
        # transform to structured array
        note_array = np.core.records.fromarrays(note_array, names='onset_div,duration_div,pitch,onset_beat')

        # create features array of shape (num_nodes, num_features)
        time_baseline = time.time()
        _ = edges_from_note_array(note_array)
        time_baseline = time.time() - time_baseline

        time_c = time.time()
        _ = sam.compute_edge_list(note_array['onset_div'].astype(np.int32),
                                                      note_array['duration_div'].astype(np.int32))
        time_c = time.time() - time_c
        print("Time for python edge list creation: ", time_baseline)
        print("Time for C edge list creation: ", time_c)
        self.assertTrue(time_c < time_baseline)



