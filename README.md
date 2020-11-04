![Seeker](seeker.png)
Seeker is a python library for discriminating between bacterial and phage genomes.
Seeker is based on an LSTM deep-learning models and does not rely on a reference genome,
genomic alignment or any direct genome comparison. 

## Overview 
This file describes a python package that implements Seeker, an alignment-free discrimination between Bacterial vs. phages DNA sequences, based on a deep learning framework [1]. 
This package can call classifiers that were trained with (a) either Python Keras LSTM with embedding layer, or (b) Matlab trained LSTM with a sequence imput layer, which was converted to a Keras model. A web portal is also available (http://seeker.pythonanywhere.com/).

If you have any trouble installing or using Seeker, please let us know by opening an issue on GitHub or emailing us 
(either ayal.gussow@gmail.com or noamaus@gmail.com).

<em>Note</em>: Seeker relies on tensorflow, which is not yet supported in python 3.8. Therefore, to use
Seeker you need to use Python 3.6 or 3.7. Creating different Python environments is easy using conda 
(https://docs.conda.io/en/latest/).
 

## Citation
[1]Noam Auslander*, Ayal B. Gussow1*#, Sean Benler, Yuri I. Wolf, Eugene V. Koonin# [Seeker: Alignment-free identification of bacteriophage genomes by deep learning](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkaa856/5921300) Nucleic Acid Research, October 2020.
(*) These authors contributed equally, (#) Corresponding authors


## Installation
Seeker requires python3, and has been tested with python3.6 and python3.7. 
Seeker can be installed using pip. From a terminal, run:

`pip install seeker` 

This will install Seeker and all of its dependencies.

## Installation using Conda
Conda provides an easy method to install Seeker. First, install conda or miniconda
(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run the following commands to install seeker:
```
conda create --name seeker python=3.7 pip
conda activate seeker
pip install seeker
```

<em>Note</em>: If you rely on conda, any time you want to use Seeker's libraries or commands you have to first run:
```
conda activate seeker
```

## Usage
The Seeker library consists of binaries that can be run from the command line and a python library that
can be incorporated into Python scripts.

### Binaries
Seeker includes a binary that predicts whether an entire sequence is bacterial or phage.

To predict whether sequences are bacterial or phage, run the following from the terminal:
 
`predict-metagenome input.fa`

This will output a prediction for each sequence in `input.fa` along with Seeker's score. Scores are between 0 and 1.
Higher scores correspond to phage predictions while lower scores correspond to bacterial predictions. Sequences with 
scores above 0.5 are predicted phages, while sequences with scores below 0.5 
are predicted bacteria.  

### Python Library
The primary class in the Python library is SeekerFasta. SeekerFasta can load a Fasta file and score its entries using 
Seeker. SeekerFasta has the following parameters:

1. path_or_str. Either a path to a Fasta or a Fasta string. 
2. LSTM_type. Which LSTM implementation to use. Options are "python", "matlab", "prophage" (not recommended). Default is Matlab.
3. seeker_model. If you've already loaded a model into a SeekerModel object and prefer to use that model, you can
provide it as a parameter here. Default is None, in which case the model will be loaded from file.  
1. load_seqs. Whether to preload all Fasta sequences to memory. Default is True. 
5. is_fasta_str. Set to True if you provided a Fasta string instead of a path to a Fasta file. Default is False. 

Once a Fasta is loaded, there are several functions that can be used to generate Seeker predictions.
For example, to predict whether the entries of a Fasta are phage or bacteria:
```
from seeker import SeekerFasta
seeker_fasta = SeekerFasta("input.fa")
predictions = seeker_fasta.phage_or_bacteria()  # This returns a list of phage/bacteria predictions for the Fasta
print("\n".join(predictions))   # print predictions

# To filter the Fasta file for predicted phage sequences, the following will
# create a new fasta and save it to "seeker_phage_contigs.fa" with all sequences with 
# a Seeker score of 0.5 and above (threshold can be adjusted per user goals)
seeker_fasta.meta2fasta(out_fasta_path="seeker_phage_contigs.fa", threshold=0.5)
```

Alternatively, to predict prophages:
```
seeker_fasta = SeekerFasta("input.fa", LSTM_type="prophage")
seeker_fasta.save2bed("output.bed")  # Save prophage coordinates to BED file
seeker_fasta.save2fasta("output.fa")  # Save prophage sequences to Fasta file 
```
<em>NOTE</em>: Seeker was not trained to predict prophages. The prophage model is the output of the first training step, that has been described in [1]. This model has not been tested thoroughly for prophage prediction, and its performance is affected by the prophage prediction parameters which depend on the organism and the user's goals. Due to this, the use of this model for prophage detection is not recommended, unless it is done as an initial filtering step. 
 
## LSTM Models
The LSTM models can be found in the `models` directory. 
1. model.h5. Metagenome LSTM model, trained in python using Keras.
1. MatModel0.h5. Metagenome LSTM model, trained in matlab.
1. MatModePRO.h5. Prophage LSTM model, trained in matlab.

## Datasets 
Training, validation and test datasets are available from:
ftp://ftp.ncbi.nih.gov/pub/wolf/_suppl/Seeker/

## Contact
If you run into any issues or have any questions, feel free to open an issue on Github or email us 
at either ayal.gussow@gmail.com or noamaus@gmail.com.
