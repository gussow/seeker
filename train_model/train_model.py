#!/usr/bin/env python3
###############################################################################
"""
Summary: Train an LSTM model for sequence classification.
Input: Bacterial Fasta, Phage Fasta, in segments of 1K.
Output: LSTM model.
"""
###############################################################################


# Imports ---------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import re

import numpy as np

from Bio import SeqIO

from keras.models import Sequential
from keras.layers import Dense, LSTM


# Configuration ---------------------------------------------------------------
NUC_ORDER = {y: x for x, y in enumerate(["A", "T", "C", "G"])}
NUC_COUNT = len(NUC_ORDER)

# Functions -------------------------------------------------------------------
def seq2matrix(sequence, nuc_order, fragment_length):
    """Convert sequence to binary matrix."""
    # Set sequence length to a maximum of our set fragment length
    assert len(sequence) <= fragment_length
    sequence_len = min(len(sequence), fragment_length)
    sequence = sequence[:sequence_len]

    # Create a dataframe to represent the sequence
    ret = np.zeros((4, fragment_length))

    # Set each base in the matrix
    for idx, base in enumerate(sequence):
        ret[nuc_order[base], idx] = 1

    return ret


def read_fasta(fasta_path, nuc_order, fragment_length):
    """Read a Fasta into array of arrays, for LSTM input."""
    # Read in file
    fasta = SeqIO.parse(fasta_path, "fasta")

    # Iterate fasta and convert to matrices
    matrices = []
    for idx, entry in enumerate(fasta, start=1):
        entry.seq = entry.seq.upper()

        if re.match('^[ACGT]+$', str(entry.seq)) is None:
            continue

        matrix = seq2matrix(str(entry.seq), nuc_order, fragment_length)

        matrices.append(matrix)
    
    ret = np.array(matrices)
        
    return ret

def build_model():
    """Build the LSTM model for sequence classification."""
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(
       LSTM(5, input_shape=(NUC_COUNT, 1000))
    )

    # Add Dense NN layer
    model.add(
        Dense(1, activation='tanh')
    )

    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
    )

    return model


# Main ------------------------------------------------------------------------
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--bacteria",
                    type=argparse.FileType('r'),
                    required=True)
PARSER.add_argument("--phage",
                    type=argparse.FileType('r'),
                    required=True)
PARSER.add_argument("--out",
                    type=str,
                    required=True)

args = PARSER.parse_args()

# Read phage sequences
X_phage = read_fasta(
    args.phage, NUC_ORDER, 1000
)

# Read bacteria sequences
X_bac = read_fasta(
    args.bacteria, NUC_ORDER, 1000
)

# Build, train model
model = build_model()
X_train = np.concatenate((X_phage, X_bac))
y_train = np.concatenate((np.ones(len(X_phage)), np.zeros(len(X_bac))))

model.fit(
    X_train, y_train, epochs=100, batch_size=27
)

# Save model
model.save(args.out)
