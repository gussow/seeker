"""
Model classes and functions for viral or bacterial sequence prediction.
"""
# Imports --------------------------------------------------------------------------------------------------------------
import os
import re
import sys

import numpy as np
from pkg_resources import resource_filename
from collections import OrderedDict

import random

# Minimize TF logs, has to be done in this order
# logging.info("Importing Tensorflow")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import get_logger, autograph
get_logger().setLevel('ERROR')
autograph.set_verbosity(0)

# Import what's needed from tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Constants ------------------------------------------------------------------------------------------------------------
DEFAULT_NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G"])}
NUCLEOTIDES = sorted([x for x in DEFAULT_NUC_ORDER.keys()])
SEGMENT_LENGTH = 1000
PHAGE_THRESHOLD = 0.5


# Functions ------------------------------------------------------------------------------------------------------------
def load_lstm_model(model_path):
    """
    Loads Keras model.
    :param model_path: Path to H5 model.
    :return: Keras model.
    """
    model_loaded = load_model(model_path)
    return model_loaded


def random_base(match):
    """
    Generate a random base.
    :return: Random base.
    """
    return random.choice(NUCLEOTIDES)


def handle_non_ATGC(sequence):
    """
    Handle non ATGCs.
    :param sequence: String input.
    :return: String output (only ATCGs), with randomly assigned bp to non-ATGCs.
    """
    ret = re.sub('[^ATCG]', random_base, sequence)
    assert len(ret) == len(sequence)
    return ret


def pad_sequence(sequence, source_sequence, length=SEGMENT_LENGTH):
    """
    Pad sequence by repeating it.
    :param sequence: Segmented sequence to pad.
    :param source_sequence: Original, complete sequence.
    :param length: Length of sequence to pad up to.
    :return: Padded sequence of lenth length.
    """
    assert len(sequence) < length, len(sequence)
    assert sequence == source_sequence or len(source_sequence) > length
    if len(source_sequence) > length:
        ret = source_sequence[len(source_sequence)-len(sequence)-(length-len(sequence)):
                              len(source_sequence)-len(sequence)]+sequence

    else:
        assert sequence == source_sequence
        ret = (source_sequence * (int(length / len(sequence)) + 1))[:length]
    assert len(ret) == length
    return ret


def frag2matrix(fragment, frag_max=SEGMENT_LENGTH):
    """
    Convert sequence fragment to one-hot binary matrix to use in model converted from Matlab.
    :param fragment: Sequence fragment to convert.
    :param frag_max: Fragment should be at most this length.
    :return: Numpy matrix representing one-hot encoded sequence.
    """
    assert len(fragment) <= frag_max

    # Create a dataframe to represent the sequence
    ret = np.zeros((frag_max, 4))

    # Set each base in the matrix
    for idx, base in enumerate(fragment):
        ret[idx, DEFAULT_NUC_ORDER[base]] = 1

    return np.array(ret)


def frags2matrices(fragments, frag_max=SEGMENT_LENGTH):
    """
    Convert list of sequence fragments to one-hot binary matrices to use in model converted from Matlab.
    :param fragments: List of sequences to convert.
    :param frag_max: Fragments should be at most this length.
    :return: Array of numpy matrices representing one-hot encoded  sequence.
    """
    matrices = []

    for frag in fragments:
        matrices.append(frag2matrix(frag, frag_max))

    return np.array(pad_sequences(matrices, maxlen=frag_max, dtype='float', padding="post"))


def encode_sequence(sequence, nuc_order=None):
    """
    Encode a sequence to integers for use in Python LSTM model.
    :param sequence: Sequence to encode.
    :param nuc_order: Order of nucleotides for encoding.
    :return: Encoded sequence as integers.
    """
    if nuc_order is None:
        nuc_order = DEFAULT_NUC_ORDER

    sequence = sequence.upper()
    accepted_nucleotides = "".join(nuc_order.keys())

    assert re.match('^[{}]+$'.format(accepted_nucleotides), sequence) is not None, \
        "Only {} allowed".format(accepted_nucleotides)

    encoded_seq = [(nuc_order[x] + 1) for x in sequence]
    encoded_seq = np.array([encoded_seq])

    return encoded_seq


def encode_sequences(sequences, nuc_order=None, segment_length=SEGMENT_LENGTH):
    """
    Encode a sequence to integers for use in model.
    :param sequences: List of sequences to encode.
    :param nuc_order: Order of nucleotides for encoding.
    :param segment_length: Segments should be at most this length.
    :return: Encoded sequence as integers.
    """
    encoded_seqs = []
    for sequence in sequences:
        assert len(sequence) <= segment_length
        encoded_seqs.append(encode_sequence(sequence, nuc_order)[0])

    return np.array(pad_sequences(encoded_seqs, maxlen=segment_length, padding="post"))


def segment_sequence(sequence, segment_length=SEGMENT_LENGTH):
    """
    Convert a sequence into list of equally sized segments.
    :param sequence: Sequence of A, C, G, T.
    :param segment_length: Length of segment to divide into.
    :return: List of segments of size segment_length.
    """
    assert len(sequence) > 200, "Sequence is fewer than 200 bases, minimum input is 200 bases"
    ret = []

    for start_idx in range(0, len(sequence), segment_length):
        fragment = sequence[start_idx:(start_idx + segment_length)]
        assert len(fragment) <= segment_length
        if len(fragment) < segment_length:
            fragment = pad_sequence(fragment, sequence, segment_length)
        ret.append(fragment.upper())

    return ret


def segment_fasta(fasta, segment_lengths=SEGMENT_LENGTH):
    """
    Parse Fasta into segments.
    :param fasta: File handle in Fasta format.
    :param segment_lengths: Length of segments for model.
    :return: Dictionary of Fasta name -> list of segments.
    """
    ret = OrderedDict()
    seq_name = None
    seq_value = None

    count = 0
    for line in fasta:
        line = line.strip()
        if line.startswith(">"):
            # Reached a new sequence in Fasta, if this is not the first sequence let's save the previous one
            random.seed(243789)
            if seq_name is not None:  # This is not the first sequence
                assert seq_value is not None, seq_name

                # Save the sequence to a dictionary
                ret[seq_name] = segment_sequence(seq_value, segment_lengths)
                count += 1
                print("Read Fasta entry {}, total {} entries read".format(seq_name, count), file=sys.stderr)

            seq_name = line.lstrip(">").split()[0]
            seq_value = ""
        else:
            seq_value += handle_non_ATGC(line)

    # Write last entry
    ret[seq_name] = segment_sequence(seq_value)
    count += 1
    print("Read Fasta entry {}, total {} entries read".format(seq_name, count), file=sys.stderr)

    return ret


def convolute_scores(score, window_size=None):
    """
    Returns a convolution of the score with an average moving window.
    :param score: scores to convolute.
    :param window_size: Window size for convolution.
    :return: Convolution in list format.
    """
    if window_size is None:
        window_size = 10 if len(score>20) else 3

    cv = [np.mean(score[i:i+window_size]) for i in range(len(score))]

    return cv


def get_prophage_coords(sc, seq, phage_threshold=0.5, skip=50, edge=15000,
                        windsz=10, interval=SEGMENT_LENGTH, plenMN=0):
    """
    Given a list of scores, return predicted prophage coordinates.
    Prophage parameters should be set according to users needs.

    Parameters to modify sensitivity of detection:
    :param sc: list of scores assigned to segments of the sequence.
    :param seq: Complete sequence.
    :param phage_threshold: Seeker threshold for phage selection
    :param skip: segments to skip to concatenate high confidence regions
    :param edge: sequence edges to add to high confidence region
    :param windsz: Window size
    :param interval: intervals of segments
    :param plenMN: minimal length of detected prophage to consider
    :return: List of prophage coordinates.
    """
    EPS = 0.1
    if np.mean(sc) > (phage_threshold-EPS/2):
        sc2 = sc
        sc = [max(i-(np.mean(sc)-phage_threshold+EPS),0) for i in sc2]

    addL = edge+5000
    addR = edge+10000
    CNT = 0; ToStop = 0; WS = windsz
    try:
        while ToStop == 0:
            covsc = convolute_scores(sc, WS-CNT)
            phlocs = [j for j, i in enumerate(covsc) if i > phage_threshold]

            starts = []
            ends = []
            prev = phlocs[0]
            for i in phlocs:
                if len(starts) > len(ends):
                    if i > prev + skip:
                        ends.append(prev * interval)
                        starts.append(i * interval)
                elif len(starts) == len(ends):
                    starts.append(prev * interval)
                prev = i
            if len(starts) > len(ends):
                ends.append(prev * interval)

            ret = zip(*[(max(starts[i]-addL, 0), min(ends[i]+addR, len(seq)))
                        for i, j in enumerate(np.subtract(ends,starts)) if j>=plenMN])
            NumPred = len(starts)

            if (1 <= NumPred <= 10) or CNT < 1:
                ToStop=1
            if NumPred<=1:
                CNT += 1
            if NumPred>=10:
                CNT -= 1

    except IndexError:  # No prophages detected
        ret = [], []

    return ret


# Classes --------------------------------------------------------------------------------------------------------------
class SeekerModel:
    """
    An instance of a sequence predictor that can differentiate bacterial DNA from phage DNA using python LSTM trained
    model.
    """
    def __init__(self, LSTM_type="matlab", model_path=None,):
        """
        Initialize instance of Seeker. Loads model and sets sequence prep function.
        :param LSTM_type: Which model to use. Python, Matlab or prophage.
        :param model_path: Supply path to model. Defaults to precompiled models in package.
        """
        LSTM_type = LSTM_type.lower()
        assert LSTM_type in {"python", "matlab","prophage"}, \
            "LSTM type must be python, matlab, or prophage not {}".format(LSTM_type)

        # Based on the LSTM type we can set the model and which function to use for sequence prep
        if LSTM_type == 'python' and model_path is None:
            model_path = resource_filename(__name__, "models/model.h5")
            self.__seq_prep = encode_sequences
        elif LSTM_type == 'matlab' and model_path is None:
            model_path = resource_filename(__name__, "models/MatModel0.h5")
            self.__seq_prep = frags2matrices
        elif LSTM_type == 'prophage' and model_path is None:
            model_path = resource_filename(__name__, "models/MatModelPRO.h5")
            self.__seq_prep = frags2matrices

        self.LSTM_type = LSTM_type
        self.model = load_lstm_model(model_path)

    def _score_fragments(self, fragments):
        """
        Assigns scores for a list of all 1000 nucleotide fragments denoting whether it is bacteria or phage.
        :param fragments: List of all 1000 nucleotide fragment, IE several strings of A, C, G, T with length 1000.
        :return: Scores between 0 and 1, with 1 indicating phage and 0 indicating bacteria.
        """
        assert {len(x) for x in fragments} == {1000}
        return self.model.predict(self.__seq_prep(fragments))[:, 1]

    def score_fasta_segments(self, segments_dict):
        """
        Score sequences from a dictionary where each value is a list of sequence fragments.
        :param segments_dict: Dictionary of sequence name -> list of fragment sequences.
        :return: Returns a dictionary of sequence name -> list of fragment scores.
        """
        ret = {}

        for name, segments in segments_dict.items():
            print("Scoring Fasta sequence {}".format(name), file=sys.stderr)
            scores = self._score_fragments(segments)
            ret[name] = scores

        return ret

    def score_fasta(self, fasta_path):
        """
        Score sequences from a Fasta file.
        :param fasta_path: Path to Fasta to score.
        :return: Returns a dictionary of sequence name -> list of fragment scores.
        """
        with open(fasta_path) as fasta_file:
            segments_dict = segment_fasta(fasta_file)

        return self.score_fasta_segments(segments_dict)


class SeekerFasta:
    """
    A Fasta file to process with Seeker.
    """
    def __init__(self, path_or_str, LSTM_type="python", seeker_model=None, load_seqs=True, is_fasta_str=False):
        """
        Initialize SeekerFasta by loading and processing fasta at path.
        :param path_or_str: Either a path to a Fasta or a Fasta string.
        :param LSTM_type: Which LSTM implementation to use. Options are "python", "matlab", "prophage".
        :param seeker_model: Supply path to model. Defaults to precompiled models in package.
        :param load_seqs: If true, all sequences are saved to memory.
        :param is_fasta_str: Set to True if path_or_str is a Fasta string instead of a path.
        """
        if seeker_model is None:
            seeker_model = SeekerModel(LSTM_type)

        if not is_fasta_str:
            assert os.path.isfile(path_or_str), "{} needs to be a path to a Fasta".format(path_or_str)
            self.path = path_or_str

            if load_seqs:
                self._seqs = segment_fasta(open(path_or_str))
                self.scores = seeker_model.score_fasta_segments(self._seqs)
            else:
                self._seqs = None
                self.scores = seeker_model.score_fasta(path_or_str)
        else:
            self.path = None
            assert load_seqs
            self._seqs = segment_fasta(path_or_str.splitlines())
            self.scores = seeker_model.score_fasta_segments(self._seqs)

    def phage_or_bacteria(self, phage_threshold=PHAGE_THRESHOLD, eval_str="{name}\t{kingdom}\t{score}"):
        """
        For each sequence in Fasta, yields evaluation whether it is phage or bacteria.
        :param phage_threshold: Threshold to determine phage.
        :param eval_str: String template to use for output.
        :return: Iterator for each sequence entry.
        """
        for name, scores in self.scores.items():
            mean_score = sum(scores) / len(scores)
            if mean_score < phage_threshold:
                kingdom = "Bacteria"
            else:
                kingdom = "Phage"

            yield eval_str.format(name=name, kingdom=kingdom, score=round(mean_score, 2))

    def meta2fasta(self, out_fasta_path="seeker_phage_contigs.fa", threshold=0.8, filter_func=lambda x: x[0:15] in x[-100:-1]):
        """
        Saves contigs that are predicted as phages to a fasta file.

        The default Seeker threshold is set to 0.8 to detect phages with high confidence. This parameter should be set
        based on the user's specific goals.

        :param out_fasta_path: path for the output Fasta file.
        :param threshold: seeker threshold to use for phage prediction.
        :param filter_func: custom function the user can use to filter specific contigs. The default is circularity.
        """
        orgname = (list(self.scores.keys()))
        scores = [(list(self.scores.items()))[i][1].tolist() for i in range(len(self.scores.items()))]
        msc = [sum(scores[i]) / len(scores[i]) for i in range(len(scores))]
        predv = [i for i, j in enumerate(msc) if j > threshold]

        with open(out_fasta_path, 'w') as faout:
            for i in range(len(predv)):
                fseq = "".join(self._seqs[orgname[predv[i]].replace('\n', '')])
                if filter_func(fseq):
                    faout.write('>'+orgname[predv[i]].replace('\n', '') + ', av_score: ' + str(msc[predv[i]]) +  '\n')
                    faout.write(fseq + '\n')

    def save2bed(self, out_path,
                 phage_threshold=0.5, skip=50, edge=15000, windsz=10, interval=SEGMENT_LENGTH, plenMN=0):
        """
        Saves prophage coordinates to bed file format. Prophage parameters should be set according to users needs.
        :param out_path: Output path for BED.
        :param phage_threshold: Seeker threshold for phage selection
        :param skip: segments to skip to concatenate high confidence regions
        :param edge: sequence edges to add to high confidence region
        :param windsz: Window size
        :param interval: intervals of segments
        :param plenMN: minimal length of detected prophage to consider.
        """
        with open(out_path, 'w') as bedout:
            print('# sequence\tstart\tend', file=bedout)

            for sequence_name, scores in self.scores.items():
                starts, ends = get_prophage_coords(scores, ''.join(list(self._seqs[sequence_name])),
                                                   phage_threshold=phage_threshold,
                                                   skip=skip,
                                                   edge=edge,
                                                   windsz=windsz,
                                                   interval=interval,
                                                   plenMN=plenMN)
                for start, end in zip(starts, ends):
                    print(sequence_name, start, end, sep="\t", file=bedout)

    def save2fasta(self, out_path,
                   phage_threshold=0.5, skip=50, edge=15000, windsz=10, interval=SEGMENT_LENGTH, plenMN=0):
        """
        Saves prophage locations to fasta file format. Prophage parameters should be set according to users needs.

        :param out_path: Output path for BED.
        :param phage_threshold: Seeker threshold for phage selection
        :param skip: segments to skip to concatenate high confidence regions
        :param edge: sequence edges to add to high confidence region
        :param windsz: Window size
        :param interval: intervals of segments
        :param plenMN: minimal length of detected prophage to consider.
        """
        assert self._seqs is not None, "load_seqs must be set to true on initialization in order to create phage Fasta"
        with open(out_path, 'w') as faout:
            for sequence_name, scores in self.scores.items():
                starts, ends = get_prophage_coords(scores, ''.join(list(self._seqs[sequence_name])),
                                                   phage_threshold=phage_threshold,
                                                   skip=skip,
                                                   edge=edge,
                                                   windsz=windsz,
                                                   interval=interval,
                                                   plenMN=plenMN)
                for start, end in zip(starts, ends):
                    sequence = "".join(self._seqs[sequence_name])[start:end]
                    print(
                        ">{}:{}-{}\n{}".format(sequence_name, str(start), str(end), sequence),
                        file=faout
                    )
