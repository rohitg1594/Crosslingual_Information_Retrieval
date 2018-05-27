
###### IMPORT LIBRARIES ######
import argparse
import datetime
from keras.models import Model
from keras import models
from keras.layers import Embedding, LSTM, Input, Dense, Flatten, MaxPooling1D, Concatenate, BatchNormalization, LeakyReLU, Bidirectional
from keras.optimizers import Nadam, SGD
import pickle
import numpy as np
import os
import sys
import pandas as pd
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from utils import read_sents, generate_train_random, generate_train_unsupervised, gen_file_paths, compute_tf_idf
import faiss
from collections import defaultdict


###### PARSE CMD LINE ARGUMENTS ######

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory
parser.add_argument("--data_dir", default="./data", help="directory path of data")

# Word embeddings
parser.add_argument("--src_lang", default="en", help="Source lang")
parser.add_argument("--tgt_lang", default="de", help="Target lang")
parser.add_argument("--emb_dim",  default="300", type=int, help="dimension of embeddings")
parser.add_argument("--max_vocab", default=200000, type=int, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--pad_len", default=53, type=int,  help="Length of sentence padding for neural network; 95 pct of English sentences comprise 53 tokens")
parser.add_argument("--norm", default=1, type=int,  help="Normalize embeddings")

## Data
parser.add_argument("--max_sent", default=1.9e6+10000, type=int, help="Maximum number of aligned translation pairs")
parser.add_argument("--test_sent", default=10000, type=int, help="Number of aligned translation pairs for testing")
parser.add_argument("--create_test", default=0, type=int, help="Merely create test data for evaluation")
parser.add_argument("--save_test", default=0, type=int, help="Save test sentences")
parser.add_argument("--ratio", default=1, type=int, help="Generates training data that 50:50 comprises true translation and random sentences pairs. Ratio denotes scale of data, e.g. 1 doubles original translation pairs, 2 copies translation pairs 1 and generates 2 random instances per pair")
parser.add_argument("--fooling", default=0, type=int, help="Whether to create tough fooling instances or not")
parser.add_argument("--faisssize", default=80000, type=int, help="Bucket size to scan for near translations for creating tough instances")
parser.add_argument("--weight", default=0.4,type=float, help="Percentage of random instances when creating fooling instances")

# Network Architecture
parser.add_argument("--lstm_units", default=32, type=int, help="LSTM units")
parser.add_argument("--dropout", default=0.3, type=float, help="LSTM dropout")
parser.add_argument("--recurrent_dropout", default=0.5, type=float, help="LSTM recurrent dropout")
parser.add_argument("--dense_units", default=1024, type=int, help="LSTM units")

# Network Compilation
parser.add_argument("--optimizer", default="Nadam", help="Choose between Nadam or SGD for network optimizer")

# Network Fitting
parser.add_argument("--epochs", default=30, type=int, help="Choose maximum number of epochs for training")
parser.add_argument("--batch_size", default=512,type=int, help="Choose batch size for model fitting")
parser.add_argument("--validation_split", default=0.1,type=float, help="Choose percentage of training data to use for validation")
parser.add_argument("--callbacks", default=1, type=int, help="Set callbacks on / off for trial runs")
parser.add_argument("--earlystopping", default=1,type=int, help="Include Keras.Callbacks.EarlyStopping")
parser.add_argument("--earlystopping_patience", default=2,type=int,help="Patience of Keras.Callbacks.EarlyStopping")

args = parser.parse_args()

###### GENERATE PATHS ######

# Source
src_embs_file, src_map_file, src_sent_file = gen_file_paths(args.data_dir, args.src_lang, args.tgt_lang, source = True)

# Target
tgt_embs_file, tgt_map_file, tgt_sent_file = gen_file_paths(args.data_dir, args.src_lang, args.tgt_lang, source = False)

###### COMMENCE TRAINING ######
print()

if args.tgt_lang == "de": print("Europarl.de-en.lang was renamed to Europarl.en-de.lang to align with remaining training data")

if args.create_test != 1:
    print("Starting network training for " + args.src_lang + "-" + args.tgt_lang + " with " + str(int(args.max_sent)) + " translation pairs of which " + str(int(args.test_sent)) + " are reserved for testing")
    print("Time initiated: ",datetime.datetime.now().hour,":", datetime.datetime.now().minute, sep = "")
else:
    print("Only create evaluation data for " + args.src_lang + "-" + args.tgt_lang + " with " + str(int(args.max_sent)) + " translation pairs of which " + str(int(args.test_sent)) + " are reserved for testing")
    print("If this behaviour is unintended set --create_test 0 at execution")
    print("Time initiated: ", datetime.datetime.now().hour,":", datetime.datetime.now().minute, sep="")

print()

###### PRINT FILEPATHS FOR SANITY CHECK ######

assert os.path.exists(src_embs_file)
assert os.path.exists(src_embs_file)
assert os.path.exists(src_map_file)
assert os.path.exists(tgt_embs_file)
assert os.path.exists(tgt_sent_file)

print(args.src_lang, " filepaths")
print("----------------------------------------------")
print("Embedding: ", src_embs_file," - path exists: ", os.path.exists(src_embs_file))
print("Mapping: ", src_map_file," - path exists: ", os.path.exists(src_embs_file))
print("Sentences: ", src_sent_file," - path exists: ", os.path.exists(src_map_file))
print()
print(args.tgt_lang, " filepaths")
print("----------------------------------------------")
print("Embedding: ", tgt_embs_file," - path exists: ", os.path.exists(tgt_embs_file))
#print(tgt_map_file)
print("Sentences: ", tgt_sent_file," - path exists: ", os.path.exists(tgt_sent_file))
print()


######## READ EMBEDDINGS & SENTENCES ########
print("Reading " + args.src_lang + " data")
print("----------------------------------------------")
src_embs, src_word2vec, src_word2id, src_id2word, src_original_sents, src_encoded_sents, src_encoded_sents_unsup, src_tf_idf = read_sents(src_sent_file, src_embs_file, src_map_file, \
                                                                                                       maxvocab=args.max_vocab, max_sent=args.max_sent, padlen=args.pad_len,\
                                                                                                        random_state=42, comp_tfidf = args.fooling, project=True, evaluate = True)

print()
print("Reading " + args.tgt_lang + " data")
print("----------------------------------------------")
tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word, tgt_original_sents, tgt_encoded_sents, tgt_encoded_sents_unsup, tgt_tf_idf = read_sents(tgt_sent_file, tgt_embs_file, tgt_map_file, \
                                                                                                       maxvocab=args.max_vocab, max_sent=args.max_sent, padlen=args.pad_len,\
                                                                                                        random_state=42, comp_tfidf = args.fooling,project=False, evaluate = True)
print()

######## PRINT FIRST SENTENCE FOR SANITY CHECK ########

print("First sentence")
print("------------------------------------------")
print(args.src_lang,": ", src_original_sents[0])
print(args.tgt_lang,": ", tgt_original_sents[0])
print()

######## GENERATE TESTING & TRAINING DATA ########
print("Generating testing and training dataframes...")
data = pd.DataFrame(data = {"src" : pd.Series(src_encoded_sents), "tgt" : pd.Series(tgt_encoded_sents)})
test_size = args.test_sent
train_df = data.loc[:(data.shape[0]-test_size-1), :]
test_df = data.loc[(data.shape[0]-test_size):, :]
train_df["label"] = 1

## Save test data for later evaluation
if args.save_test == 1:
    if args.max_sent == (1.9e6 + 10000):
        print("Saving test data")
        test_data_path = os.path.join(args.data_dir , "test_data/test_10000_{}-{}".format(args.src_lang, args.tgt_lang))
        test_df.to_pickle(test_data_path)
        src_encoded_sents_unsup_path = os.path.join(args.data_dir, "test_data/src_enc_sents_unsup_{}-{}".format(args.src_lang, args.tgt_lang))
        tgt_encoded_sents_unsup_path = os.path.join(args.data_dir, "test_data/tgt_enc_sents_unsup_{}-{}".format(args.src_lang, args.tgt_lang))
        with open(src_encoded_sents_unsup_path, 'wb') as file_pi:
            pickle.dump(src_encoded_sents_unsup, file_pi)
        with open(tgt_encoded_sents_unsup_path, 'wb') as file_pi:
            pickle.dump(tgt_encoded_sents_unsup, file_pi)
        print("Test data saved")
    else:
        print("Test data not saved since max_sent is smaller than 1,910,000")

if args.create_test == 1:
    print("Exiting program as only test data was created")
    sys.exit(0)

print("Testing and training dataframes generated")

print()
if args.fooling:
    print("Generating fooling instances...")
    src_input, tgt_input, label = generate_train_unsupervised(train_df, src_tf_idf, tgt_tf_idf, weight=args.weight, ratio=args.ratio, faiss_size=args.faisssize)
else:
    print("Generating random instances...")
    src_input, tgt_input, label = generate_train_random(train_df, ratio=args.ratio)
print("Instances generated")
print()

######## NEURAL NETWORK ########

# Create Network
def supervised_model(pad_len = args.pad_len, max_vocab = args.max_vocab, emb_dim = args.emb_dim,\
                     lstm_units = args.lstm_units, dropout = args.dropout, recurrent_dropout = args.recurrent_dropout,\
                     dense_units = args.dense_units,\
                     left_emb = src_embs, right_emb = tgt_embs):
    """
    Creates a Keras model to predict whether sentence pair is a translation of one another
    
    :param left_emb: multi-lingual embedding of left sentences
    :param right_emb: multi-lingual embedding of right sentences 
    :param pad_len: maximum sentence length
    :param max_vocab: words included in embedding matrices
    :param emb_dim: dimensionality of embedding matrices
    :param lstm_units: hidden units in LSTM, defaults to 32
    :param dropout: LSTM dropout, default of 0.3
    :param recurrent_dropout: LSTM recurrent dropout, default of 0.5
    :param dense_units: hidden units in dense layers of classification block, defaults to 1,024
    :return: Keras model
    """


    ## Encode left sentence
    left_sent = Input(shape=(pad_len,))
    x = Embedding(max_vocab, emb_dim, input_length = pad_len, weights = [left_emb], trainable = False)(left_sent)
    x = Bidirectional(LSTM(lstm_units, dropout = dropout, recurrent_dropout = recurrent_dropout,return_sequences = True))(x)
    x = MaxPooling1D(pad_len)(x)
    x = Flatten()(x)
    left_enc = x

    ## Encode right sentence
    right_sent = Input(shape=(pad_len,))
    x = Embedding(max_vocab, emb_dim, input_length = pad_len,  weights = [right_emb], trainable = False)(right_sent)
    x = Bidirectional(LSTM(lstm_units, dropout = dropout, recurrent_dropout = recurrent_dropout, return_sequences = True))(x)
    x = MaxPooling1D(pad_len)(x)
    x = Flatten()(x)
    right_enc = x

    ## Classify
    x = Concatenate()([left_enc, right_enc])
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model([left_sent, right_sent], prediction)
    model.summary()

    return model

print("Generate model...")
model = supervised_model()

# Compile Network
## Select optimizer
if args.optimizer == "Nadam": optimizer = Nadam()
else: optimizer = SGD()
## Compile
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# Fit Network
## Including fooling in naming
if args.fooling == 1:
    fooling = "fooling"
else:
    fooling = "no-fooling"
## Implement callbacks
#### CSVLogger
date = str(datetime.datetime.now().year) + "-" + str(datetime.datetime.now().month) + "-" + str(datetime.datetime.now().day)
csv_name = date + "_{}-{}_{}_ratio-{}_{}".format(args.src_lang, args.tgt_lang, args.optimizer, args.ratio, fooling)
csv_path = os.path.join("training_logs/csv/", csv_name)
logger = CSVLogger(filename=csv_path)
#### Model Checkpoint
cp_path = "./models/monitoring/" + date + "_{}-{}_{}_ratio-{}_{}_cp".format(args.src_lang, args.tgt_lang, args.optimizer, args.ratio, fooling)
checkpoint = ModelCheckpoint(filepath = cp_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose = 1)

callbacks_list = [logger, checkpoint]

if args.earlystopping:
    callbacks_list.append(EarlyStopping(patience = args.earlystopping_patience, monitor = "val_loss"))

if not args.callbacks:
    callbacks_list = []

## Fit

history = model.fit([src_input, tgt_input], label, epochs = args.epochs, batch_size = args.batch_size, validation_split = args.validation_split, callbacks = callbacks_list)

## save model
model_path = "./models/{}-{}".format(args.src_lang, args.tgt_lang)

model_name = "model_{}-{}_{}_ratio-{}_{}".format(args.src_lang, args.tgt_lang, args.optimizer, args.ratio, fooling)

print("Saving model...")
model.save(filepath=os.path.join(model_path, model_name))
print("Model saved")

## save pickle
pickle_name = date + "_trainingHist_{}-{}_{}_ratio-{}_{}".format(args.src_lang, args.tgt_lang, args.optimizer, args.ratio, fooling)
pickle_path = os.path.join("training_logs/trainingHist/", pickle_name)
with open(pickle_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


print("Model trained and saved successfully, exiting Python environment")
sys.exit(0)