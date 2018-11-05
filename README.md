# Speech recognition project at UCT Prague
This project is an essential part of my master's thesis at [University of Chemistry and Technology, Prague](https://www.vscht.cz/?jazyk=en). As it is still work in progress, it is not very well refined. The core concept in this repository is developing an [ASR](https://en.wikipedia.org/wiki/Speech_recognition) pipeline which is able to transform Czech natural speech to textual transcription in real time. The transcription is then to be used for controlling a robot using natural Czech language utilizing [keyword spotting](https://en.wikipedia.org/wiki/Keyword_spotting) mechanisms. The complete pipeline involves (subject to change):
* transfroming the raw audio into [Mel Frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCC) features
* feeding the features into a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) with [Bidirectional long short-term memory](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) (BLSTM) cell architecture
* using <a href=https://en.wikipedia.org/wiki/Connectionist_temporal_classification>Connectionist temporal classification</a> (CTC) or simple duplicate removal (in Czech language we don't usually have words with duplicate graphemes back to back)
* (using a Language model for correcting syntactic and semantic mistakes)
* finding keywords (using simple [LuT](https://en.wikipedia.org/wiki/Lookup_table) or another RNN) which are relevant to controlling the robot
* inferring intent from the keywords
* sending inferred commands to the robot

## Functionality so far
* PDTSCLoader() class for processing audiofiles and transcripts from the Prague DaTabase of Spoken Czech 1.0 (see [1] in References)
* MFCC() class for transforming list of raw audio arrays into usable MFCC features (with deltas and delta-deltas included if specified)

## Requirements
<a href="https://www.python.org/downloads/">Python 3.6</a> environment with modules specified in __requirements.txt__ file which is located in the root of this repository. 

If you have <a href="https://pip.pypa.io/en/stable/installing/">pip</a>, you can install the required modules in bulk using the following command:
```
pip install -r requirements.txt
```

## References
1. Hajič, Jan; Pajas, Petr; Ircing, Pavel; et al., 2017, Prague DaTabase of Spoken Czech 1.0, LINDAT/CLARIN digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-2375.
