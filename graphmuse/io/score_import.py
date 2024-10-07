import partitura as pt
from graphmuse.utils import create_score_graph, get_score_features
from urllib.parse import urlparse
import urllib.request
from mido import MidiFile
import os.path as osp
import io


def load_data_from_url(url):
    response = urllib.request.urlopen(url)
    data = response.read()
    mid = MidiFile(file=io.BytesIO(data))
    return mid


def is_url(input):
    try:
        result = urlparse(input)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def load_midi_to_graph(path):
    if is_url(path):
        path = load_data_from_url(path)
    else:
        if not osp.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        else:
            assert path.endswith('.mid'), 'File must be a MIDI file, ending in .mid'
    score = pt.load_score_midi(path)
    features, f_names = get_score_features(score)
    graph = create_score_graph(features, score.note_array())
    return graph