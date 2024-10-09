from .features import get_score_features
from .graph import create_score_graph
import numpy as np
import torch


def remove_ties_acros_barlines(score):
    """Remove ties that are across barlines.

    This function don't return anything since the score will be modified in place.
    """
    for part in score.parts:
        measure_map = part.measure_number_map
        for note in part.notes:
            if note.tie_next is not None:
                if measure_map(note.start.t) != measure_map(note.tie_next.start.t):
                    note.tie_next.tie_prev = None
                    note.tie_next = None


def get_polyphonic_truth_edges(note_array, original_ids):
    """
    Extract ground truth voice edges for polyphonic voice separation.

    Add a tuple (ind_start,ind_end) to truth_edges list, where ind_start, ind_end
    are indices in the note_array, if the respective notes are consecutive with
    with the same voice and staff.
    Consecutive means that there is not another note with the same staff and voice whose onset is
    between the two notes.

    Parameters
    ----------
    note_array: structured array
        The note array of the score. Should contain staff, voice, onset_div, and duration_div.
    original_ids: list
        The original ids of the full note array. Necessary because this function is usualy called in a subset.

    Returns
    -------
    truth_edges: list
        Ground truth edges.
    """
    truth_edges = list()

    # build a square boolean matrix (n,n) to check the same voice
    voice_2dmask = note_array["voice"][:, np.newaxis] == note_array["voice"][np.newaxis, :]
    # the same for staff
    staff_2dmask = note_array["staff"][:, np.newaxis] == note_array["staff"][np.newaxis, :]
    # check if one note is after the other
    after_2dmask = (note_array["onset_div"] + note_array["duration_div"])[:, np.newaxis] <= note_array["onset_div"][
                                                                                            np.newaxis, :]
    # find the intersection of all masks
    voice_staff_after_2dmask = np.logical_and(np.logical_and(voice_2dmask, staff_2dmask), after_2dmask)

    # check if there is no other note before, i.e., there is not note with index ind_middle such as
    # note_array["onset"][ind_middle] < note_array["onset"][ind_end]
    # since the notes are only after by previous checks, this check that there are not notes in the middle
    for start_id, start_note in enumerate(note_array):
        # find the possible end notes left from the previous filtering
        possible_end_notes_idx = np.argwhere(voice_staff_after_2dmask[start_id])[:, 0]
        # find the notes with the smallest onset
        if len(possible_end_notes_idx) != 0:
            possible_end_notes_idx = possible_end_notes_idx[
                note_array[possible_end_notes_idx]["onset_div"] == note_array[possible_end_notes_idx][
                    "onset_div"].min()]
            # add the all couple (start_id, end_id) where end_id is in possible_end_notes_idx to truth_edges
            truth_edges.extend([(original_ids[start_id], original_ids[end_id]) for end_id in possible_end_notes_idx])

    return truth_edges


def sanitize_staff_voices(note_array):
    # case when there are two parts, and each one have only one staff.
    if len(np.unique(note_array["staff"])) == 1:
        # id is in the form "P01_something", extract the number before the underscore and after P
        staff_from_id = np.char.partition(np.char.partition(note_array["id"], sep="_")[:, 0], sep="P")[:, 2].astype(int)
        note_array["staff"] = staff_from_id + 1 # staff is 1-indexed
    # check if only two staves exist
    if len(np.unique(note_array["staff"]))!=2:
        raise Exception("After sanitizing, the score has",len(np.unique(note_array["staff"])), "staves but it must have only 2.")
    # sometimes staff numbers are shifted. Shift them back to 1-2
    if note_array["staff"].min()!=1:
        note_array["staff"] = note_array["staff"] - note_array["staff"].min() +1
    # check if they are between 1 and 2
    if note_array["staff"].min()!=1:
        raise Exception(f"After sanitizing, the minimum staff is {note_array['staff'].min()} but it should be 1")
    if note_array["staff"].max()!=2:
        raise Exception(f"After sanitizing, the maximum staff is {note_array['staff'].max()} but it should be 2")
    # check that there are no None voice values.
    if np.any(note_array["voice"] == None):
        raise Exception("Note array contains None voice values.")
    return note_array


def get_measurewise_truth_edges(note_array, note_measure):
    """Create groundtruth edges for polyphonic music.

    The function creates edges between consecutive notes in the same voice.

    The groundtruth is restricted per bar (measure) by the score typesetting.
    Voices can switch between bars.

    Parameters
    ----------
    note_array : np.structured array
        The partitura note array

    measure_notes: np.array
        A array with the measure number for each note in the note_array

    Returns
    -------
    edges : np.array (2, n)
        Ground truth edges.
    """
    # bring all note arrays to a common format, with a single part and 2 staves
    note_array = sanitize_staff_voices(note_array)

    edges = list()
    # split note_array to measures where every onset_div is within range of measure start and end
    measurewise_na_list = np.split(note_array, np.where(np.diff(note_measure))[0] + 1)
    measurewise_original_ids = np.split(np.arange(len(note_array)), np.where(np.diff(note_measure))[0] + 1)
    for measurewise_na, original_id in zip(measurewise_na_list, measurewise_original_ids):
        bar_edges = get_polyphonic_truth_edges(measurewise_na, original_id)
        edges.extend(bar_edges)
    return np.array(edges).T


def get_measurewise_pot_edges(note_array, measure_notes):
    """Get potential edges for the polyphonic voice separation dataset.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
    measure_notes : np.array
        The measure number of each note.
    Returns
    -------
    pot_edges : np.array (2, n)
        Potential edges.
    """
    pot_edges = list()
    for un in np.unique(measure_notes):
        # Sort indices which are in the same voice.
        voc_inds = np.sort(np.where(measure_notes == un)[0])
        # edge indices between all pairs of notes in the same measure (without self loops). size of (2, n)
        edges = np.vstack((np.repeat(voc_inds, len(voc_inds)), np.tile(voc_inds, len(voc_inds))))
        # # remove self loops
        # self_loop_mask = edges[0] != edges[1]
        # remove edges whose end onset is before start offset
        not_after_mask = note_array["onset_div"][edges[1]] >= note_array["onset_div"][edges[0]] + note_array["duration_div"][edges[0]]
        # # remove edges that are not during the same note
        # during_mask = (note_array["onset_div"]+note_array["duration_div"])[edges[0]] < note_array["onset_div"][edges[1]]
        # apply all masks
        edges = edges[:, not_after_mask]
        pot_edges.append(edges)
    pot_edges = np.hstack(pot_edges)
    return pot_edges


def get_pot_chord_edges(note_array, onset_edges):
    """Get edges connecting notes with same onset and duration"""
    # checking only the duration, the same onset is already checked in the onset edges
    same_duration_mask = note_array[onset_edges[0]]["duration_div"] == note_array[onset_edges[1]]["duration_div"]
    pot_edges = onset_edges[:, same_duration_mask]
    # remove self loops
    return pot_edges[:, pot_edges[0] != pot_edges[1]]


def get_truth_chords_edges(note_array, pot_chord_edges):
    """Get edges connecting notes with same onset and duration and voice and staff"""
    same_voice_mask = note_array[pot_chord_edges[0]]["voice"] == note_array[pot_chord_edges[1]]["voice"]
    same_staff_mask = note_array[pot_chord_edges[0]]["staff"] == note_array[pot_chord_edges[1]]["staff"]
    return pot_chord_edges[:, same_voice_mask & same_staff_mask]


def create_piano_svsep_graph(score, add_measures_to_graph=False, sort_graph=True, add_beats_to_graph=False):
    """
    Create a piano single-voice separation graph.

    This function processes a musical score to create a graph representation suitable for single-voice separation tasks.

    Parameters
    ----------
    score : partitura.Score
        The musical score to process.
    add_measures_to_graph : bool, optional
        Whether to add measure information to the graph (default is False).
    sort_graph : bool, optional
        Whether to sort the graph (default is True).
    add_beats_to_graph : bool, optional
        Whether to add beat information to the graph (default is False).

    Returns
    -------
    graph : torch_geometric.data.HeteroData
        The constructed graph with various edge types and node features.
    """
    remove_ties_acros_barlines(score)

    features, feat_names = get_score_features(score)
    note_array = score.note_array(
        include_time_signature=True,
        include_grace_notes=True,
        include_staff=True,
    )

    measures = score[0].measures if add_measures_to_graph else None

    graph = create_score_graph(features, note_array, sort=sort_graph, measures=measures, add_beats=add_beats_to_graph)

    # get the measure number for each note in the note array
    mn_map = score[np.array([p._quarter_durations[0] for p in score]).argmax()].measure_number_map
    note_measures = mn_map(note_array["onset_div"])

    # Compute the output graph
    truth_edges = get_measurewise_truth_edges(note_array, note_measures)
    pot_edges = get_measurewise_pot_edges(note_array, note_measures)
    pot_chord_edges = get_pot_chord_edges(note_array, graph["note", "onset", "note"].edge_index.numpy())
    truth_chords_edges = get_truth_chords_edges(note_array, pot_chord_edges)
    graph["note"].staff = torch.from_numpy(note_array["staff"].copy()).long()
    graph["note", "true_voice", "note"].edge_index = torch.from_numpy(truth_edges).long()
    graph["note", "pot_voice", "note"].edge_index = torch.from_numpy(pot_edges).long()
    graph["note", "pot_chord", "note"].edge_index = torch.from_numpy(pot_chord_edges).long()
    graph["note", "true_chord", "note"].edge_index = torch.from_numpy(truth_chords_edges).long()
    return graph
