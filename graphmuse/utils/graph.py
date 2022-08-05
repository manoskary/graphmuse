import numpy as np
from cython_graph import GraphFromAdj


def graph_from_note_array(note_array, rest_array=None, norm2bar=True):
    '''Turn note_array to homogeneous graph dictionary.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    '''

    edg_src = list()
    edg_dst = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where((np.isclose(note_array["onset_beat"], x["onset_beat"], rtol=1e-04, atol=1e-04) == True) & (note_array["pitch"] != x["pitch"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

        for j in np.where(np.isclose(note_array["onset_beat"], x["onset_beat"] + x["duration_beat"], rtol=1e-04, atol=1e-04) == True)[0]:
            edg_src.append(i)
            edg_dst.append(j)


        for j in np.where((x["onset_beat"] < note_array["onset_beat"]) & (x["onset_beat"] + x["duration_beat"] > note_array["onset_beat"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)

    end_times = note_array["onset_beat"] + note_array["duration_beat"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_beat"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_beat"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)

    edges = np.array([edg_src, edg_dst])
    return edges


def graph_from_notearray(note_array, cython=True):
    if cython:
        edges = GraphFromAdj(
            np.ascontiguousarray(note_array["onset_beat"], np.float32),
            np.ascontiguousarray(note_array["duration_beat"], np.float32),
            np.ascontiguousarray(note_array["pitch"], np.float32),
            4).process()
    else:
        edges = graph_from_note_array(note_array)
    return edges


if __name__ == '__main__':
    import partitura as pt
    from timeit import default_timer as timer

    score_dir = "/home/manos/Desktop/JKU/data/mozart_piano_sonatas/K279-1.musicxml"
    note_array = pt.load_score(score_dir).note_array()

    # lst = []
    # for i in range(10):
    #     start = timer()
    #     graph_from_notearray(note_array, cython=False)
    #     end = timer()
    #     lst.append(end-start)
    # print("Python Code: ", sum(lst) / len(lst))

    lst = []
    for i in range(10):
        start = timer()
        graph_from_notearray(note_array, cython=True)
        end = timer()
        lst.append(end - start)
    print("Cython Code: ", sum(lst) / len(lst))