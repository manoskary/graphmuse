import numpy as np
from typing import List, Tuple


def get_pc_one_hot(note_array):
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["pc_{:02d}".format(i) for i in range(12)]


def get_octave_one_hot(note_array):
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["octave_{:02d}".format(i) for i in range(10)]


def get_score_features(score) -> Tuple[np.ndarray, List]:
    """
    Returns features Voice Detection features.

    Parameters
    ----------
    score: partitura.Score

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    note_array = score.note_array(include_time_signature=True)
    octave_oh, octave_names = get_octave_one_hot(note_array)
    pc_oh, pc_names = get_pc_one_hot(note_array)
    # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_beat"]/note_array["ts_beats"])))-0.5)*2, 1)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"] / note_array["ts_beats"]), 1)
    onset_feature = np.expand_dims(
        np.remainder(note_array["onset_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    is_down_beat = np.expand_dims(np.remainder(note_array["onset_beat"], 1) == 0, 1)
    dur_names = ["bar_exp_duration", "onset_bar_norm", "is_down_beat"]
    names = dur_names + pc_names + octave_names
    out = np.hstack((duration_feature, onset_feature, is_down_beat, pc_oh, octave_oh))
    return out, names