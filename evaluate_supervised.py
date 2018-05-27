from keras.models import load_model
#from utils import tough_baseline
from utils import read_sents, gen_file_paths, load_embs
import faiss
import datetime
import numpy as np
import argparse
import os
import pickle
import pandas as pd
import sys
from collections import Counter
from numpy.linalg import svd as svd
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description="Cross Lingual Sentence Retrieval", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory
parser.add_argument("--data_dir", default="./data", help="directory path of data")

# Word embeddings
parser.add_argument("--src_lang", default="en", help="Source lang")
parser.add_argument("--tgt_lang", default="de", help="Target lang")
parser.add_argument("--emb_dim",  default="300", type=int, help="dimension of embeddings")
parser.add_argument("--max_vocab", default=200000, type=int, help="Maximum vocabulary size loaded from embeddings")
parser.add_argument("--norm", default=1, type=int,  help="Normalize embeddings")
parser.add_argument("--max_sent", default=1.9e6+10000, type=int, help="Maximum number of aligned translation pairs")
parser.add_argument("--test_sent", default=10000, type=int, help="Number of aligned translation pairs for testing")
parser.add_argument("--pad_len", default=53, type=int,  help="Length of sentence padding for neural network; 95 pct of English sentences comprise 53 tokens")
parser.add_argument("--fooling", default=0, type=int, help="Whether to read a fooling model or not")
parser.add_argument("--ratio", default=1, type=int, help="What ratio the model trains on")
parser.add_argument("--optimizer", default="Nadam", help="Choose between Nadam or SGD for network optimizer")

# Scoring
parser.add_argument("--threshold", default=100, type=int, help="Determines no. of sentences to evaluate on")
parser.add_argument("--filtersize", default=500, type=int,  help="Indicates size of prefiltering, only the k top candidates w.r.t. tough_baseline are evaluated on")


args = parser.parse_args()

###### INITIATE EVALUATION ######
print()
print("Starting evaluation for " + args.src_lang + "-" + args.tgt_lang + " with " + str(int(args.test_sent)) + " translation pairs")
print("Threshold: ", args.threshold)
print("Filtersize: ", args.filtersize)
print("Time initiated: ", datetime.datetime.now().hour, ":", datetime.datetime.now().minute, sep="")
print()

######## READ EMBEDDINGS & SENTENCES ########
# Source
src_embs_file, src_map_file, src_sent_file = gen_file_paths(args.data_dir, args.src_lang, args.tgt_lang, source = True)

# Target
tgt_embs_file, tgt_map_file, tgt_sent_file = gen_file_paths(args.data_dir, args.src_lang, args.tgt_lang, source = False)

###### PRINT FILEPATHS FOR SANITY CHECK ######
print(args.src_lang, " filepaths")
print("----------------------------------------------")
print("Embedding: ", src_embs_file," - path exists: ", os.path.exists(src_embs_file))
print("Mapping: ", src_map_file," - path exists: ", os.path.exists(src_embs_file))
print()
print(args.tgt_lang, " filepaths")
print("----------------------------------------------")
print("Embedding: ", tgt_embs_file," - path exists: ", os.path.exists(tgt_embs_file))
print()

assert os.path.isdir(args.data_dir)
assert os.path.exists(src_embs_file)
assert os.path.exists(tgt_embs_file)


######## READ EMBEDDINGS & SENTENCES ########
print("Reading " + args.src_lang + " data")
print("----------------------------------------------")

print("Reading embedding...")
src_embs, src_word2vec, src_word2id, src_id2word = load_embs(src_embs_file, args.max_vocab)
print("Embedding read")
print("Projecting embedding to multi-lingual space...")
with open(src_map_file, "rb") as f:
    src_mapping = pickle.load(f)

def project(embs, word2vec, mapping):
    embs = np.matmul(embs, mapping)
    ## update word2vec to proper embedding space
    for i, word in enumerate(word2vec):
        word2vec[word] = embs[i, :]
    return embs, word2vec

src_embs, src_word2vec = project(src_embs, src_word2vec, src_mapping)
print("Embedding projected to multi-lingual space")

print()
print("Reading " + args.tgt_lang + " data")
print("----------------------------------------------")
print("Reading embedding...")
tgt_embs, tgt_word2vec, tgt_word2id, tgt_id2word = load_embs(tgt_embs_file, args.max_vocab)
print("Embedding read")
print()

print("Reading test sentences from pickle...")
## read test dataframe
test_df = pd.read_pickle("./data/test_data/test_10000_{}-{}".format(args.src_lang, args.tgt_lang))
## read encoded test sentences for unsupervised pre-filtering
with open("./data/test_data/src_enc_sents_unsup_{}-{}".format(args.src_lang, args.tgt_lang), "rb") as f:
    src_encoded_sents_unsup = pickle.load(f)
with open("./data/test_data/tgt_enc_sents_unsup_{}-{}".format(args.src_lang, args.tgt_lang), "rb") as f:
    tgt_encoded_sents_unsup = pickle.load(f)
print("Test sentences pickle read")
print()

### Slice encoded sentences
test_size = int(args.max_sent - args.test_sent)
tgt_encoded_sents_test = tgt_encoded_sents_unsup[test_size:]
src_encoded_sents_test = src_encoded_sents_unsup[test_size:]

######## IMPORT MODEL ########
print("Loading model...")

if args.fooling == 1:
    fooling = "fooling"
else:
    fooling = "no-fooling"
model_folder = "./models/{}-{}".format(args.src_lang, args.tgt_lang)
model_name = "model_{}-{}_{}_ratio-{}_{}".format(args.src_lang, args.tgt_lang, args.optimizer, args.ratio, fooling)
model_path = os.path.join(model_folder, model_name)
model_path = "./models/monitoring/2018-5-24_en-es_Nadam_ratio-3_fooling_cp"
print(model_path)
model = load_model(model_path)
model.summary()
print()

def tough_baseline(corpus, word2vec, id2word, emb_dim=300, a=10 ** -3, source=True, mapper=np.ones(300), norm=True):
    """
    Compute a simple unsupervised aggregation of word embeddings as described in:
       https://openreview.net/pdf?id=SyK00v5xx
    :param corpus:
    :param word2vec:
    :param id2word:
    :param vec_dim:
    :param a:
    :param source: bool: if set to true apply mapper to word embeddings
    :param mapper:
    :return:
    """

    def calc_word_probs(corpus):
        """
        Returns the estimated word probabilites of the corpus
        :param corpus: list of sentences in the corpus, words in the form of ids
        :return: c : dic of the form => word :  probability
        """
        c = Counter()
        corpus = [word for sent in corpus for word in sent]
        total_words = len(corpus)
        c = Counter(corpus)

        c = {word: count / total_words for word, count in c.items()}

        return dict(c)

    word_probs = calc_word_probs(corpus)

    N = len(corpus)
    corpus_vec = np.zeros((N, emb_dim))

    # Create tf-idf weights
    for sent_idx, sentence in enumerate(corpus):
        vec = np.zeros(emb_dim)

        # For every unique word in sentence
        for word_idx, word_id in enumerate(sentence):
            try:
                if source:
                    vec += a / (a + word_probs.get(word_id, a / 10)) * word2vec[id2word[word_id]] @ mapper
                else:
                    vec += a / (a + word_probs.get(word_id, a / 10)) * word2vec[id2word[word_id]]
            except KeyError:
                continue

        corpus_vec[sent_idx] = vec

    x = corpus_vec.T
    U, d, v_t = svd(x)
    u = U[:, 0]

    corpus_vec = corpus_vec - corpus_vec @ np.outer(u, u.T)

    if norm:
        corpus_vec = normalize(corpus_vec, axis=1, norm='l2')

    return corpus_vec



def supervised_scores(model, src_encoded_sents_unsup, tgt_encoded_sents_unsup, src_word2vec, tgt_word2vec, src_id2word, tgt_id2word, threshold=args.threshold, filtersize=args.filtersize, topk=[1, 5, 10, 100]):
    """

    Supervised scores computes MAP for

    :param model: trained neural network for input language pair
    :param src_encoded_sents_unsup:
    :param tgt_encoded_sents_unsup:
    :param src_word2vec:
    :param tgt_word2vec:
    :param src_id2word:
    :param tgt_id2word:
    :param threshold: determines no. of sentences to evaluate on
    :param filtersize: indicates size of prefiltering, only the k top candidates w.r.t. tough_baseline are evaluated on
    :param topk:
    :return:
    """

    assert (filtersize <= len(test_df))
    assert (threshold <= len(test_df))


    tgt_tough = tough_baseline(tgt_encoded_sents_unsup, tgt_word2vec, tgt_id2word, source=False)
    src_tough = tough_baseline(src_encoded_sents_unsup, src_word2vec, src_id2word, source=False)
    index = faiss.IndexFlatL2(300)
    index.add(tgt_tough.astype(np.float32))
    D, I = index.search((src_tough).astype(np.float32), filtersize)

    print("Starting Predictions...")
    total_scores = []
    for i in range(threshold):
        if (i % 50) == 0: print(i,"==>", sep = " ")
        if (i % 1000) == 0: print()
        indices = I[i]
        tgt_test = np.array(test_df['tgt'].tolist())[indices]
        src_test = np.tile(np.array(test_df['src'])[i], (filtersize, 1))
        predictions = model.predict([src_test, tgt_test])
        scorelist = []
        for ix, score in zip(indices, predictions):
            scorelist.append((ix, score))
        sorted_scores = sorted(scorelist, key=lambda x: x[1], reverse=True)
        sorted_indices = []
        for ix, score in sorted_scores:
            sorted_indices.append(ix)

        try:
            current_index = sorted_indices.index(i)
        except:
            current_index = filtersize
        total_scores.append(current_index)
    index_counts = pd.Series(total_scores).value_counts()
    ic_indices = index_counts.index.tolist()
    ic_counts = index_counts.values.tolist()
    ic_indices_all = list(range(filtersize))
    ic_counts_all = [ic_counts[ic_indices.index(i)] if i in ic_indices else 0 for i in ic_indices_all]
    ic_counts_all_sorted = [n for _, n in sorted(zip(ic_indices_all, ic_counts_all))]
    cusum = np.cumsum(ic_counts_all_sorted) / threshold
    print("Evaluating on", threshold, "out of", index.ntotal, "sentences...")
    topks = []
    for k in topk:
        print("Top", k, "precision:", np.round(cusum[k - 1], 4))
        topks.append(cusum[k - 1])

    return total_scores, topks


print("Computing scores ...")
total_scores, topks = supervised_scores(model, src_encoded_sents_test, tgt_encoded_sents_test, src_word2vec, tgt_word2vec, src_id2word, tgt_id2word, threshold=args.threshold)

### WRITE RESULTS
total_scores_path = "./results/{}-{}/results_total-scores_{}-{}_{}_ratio-{}_{}_threshold-{}_filtersize-{}".format(args.src_lang, args.tgt_lang,args.src_lang, args.tgt_lang,args.optimizer,args.ratio,fooling, args.threshold, args.filtersize)
with open(total_scores_path, 'wb') as file_pi:
    pickle.dump(total_scores, file_pi)

topks_path = "./results/{}-{}/results_topks_{}-{}_{}_ratio-{}_{}_threshold-{}_filtersize-{}".format(args.src_lang, args.tgt_lang, args.src_lang, args.tgt_lang, args.optimizer,args.ratio,fooling, args.threshold, args.filtersize)
with open(topks_path, 'wb') as file_pi:
    pickle.dump(topks, file_pi)
