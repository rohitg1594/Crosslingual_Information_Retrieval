# Cross-lingual Information Retrieval
Authors :  Rohit Gupta, Fabian Schmidt, Leon Slr

This is our submission to the final project for Information Retrieval Project Spring 2018. This project provides an
easy interface to perform experiments for cross-lingual information retrieval on the Eurparl dataset and to learn
linear mappings from one language's word embedding space to another.

## Setup
1. Unzip this archive.
2. Make setup.sh executable by running chmod +x setup.sh.
3. Run bash setup.sh. This will create the data directory structure, download Europarl data for 5 language pairs and
   also download fasttext embeddings for the same.(NOTE: This might take several hours depending on your internet
   connection).
4. Check if the following directories exist in the data directory: dictionaries, embs, experiments and mapping.
5. It is recommended to setup a virtual environment for this project. You can install the required libraries using
   either pip or conda:
       pip install -r requirements.txt
       conda install --yes --file requirements.txt

## Experiments
All experiments can be conducted using a Command Line Interface(CLI). The following files support this functionality:
words.py, sentences.py, word_experiment.py and sentence_experiment.py. To see the options available for each file, you
can use the help option : python [FILE-NAME] -h.

### Words

1. The file words.py can be used to perform individual mapping experiments, the main options are src_lang, tgt_lang and
   data_dir.
2. The file word_experiment.py can be used to perform all the word mapping experiments described in the paper in one go.
   You can provide an experiment name which will be integrated into the final results file name saved under the data/experiments
   directory.


### Unsupervised Sentences

1. The file sentences.py can be used to perform individual unsupervised sentences experiments, the main options are src_lang,
   tgt_lang, data_dir and method.
2. The file sentence_experiment.py can be used to perform all the unsupervised sentences experiments described in the paper in one go.
   The default configuration for max_sents and eval_sents was also the one used in the paper. You can provide an experiment
   name which will be integrated into the final results file name saved under the data/experiments directory.