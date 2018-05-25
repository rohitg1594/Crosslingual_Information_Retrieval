import os
import numpy as np
from sklearn.preprocessing import normalize
import pickle
import argparse
import re
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from collections import Counter, defaultdict
import faiss
np.set_printoptions(edgeitems=5)


def tokenize(text):
    sent = text_to_word_sequence(text, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    return sent


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


def load_embs_bin(path):
    """Read word embeddings from pickle file at path.
    :param: path : str :file path of word embedding in pickle format

    """
    with open(path, 'rb') as f:
        embs, word2vec, word2id, id2word = pickle.load(f)

    return embs, word2vec, word2id, id2word


def load_embs(path, max_vocab, norm=True):
    """Read word embeddings from vec file at path.
    :param: path : str :file path of word embedding in txt format
    :param: max_vocab : int : max number of word embeddings loaded from disk
    :param: norm : bool : whether to normalize matrices
    """
    word2vec = {}
    word2id = {}
    vectors = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):

            if len(word2vec) >= max_vocab:
                break
            if i == 0:
                vocab_size, emb_size = line.split()
            else:
                word, vec = line.split(' ', 1)
                vec = np.fromstring(vec.strip(), sep=' ')
                assert len(vec) == int(emb_size)
                if word in word2vec:
                    print("Same word encountered twice\n")
                    continue
                else:
                    word2vec[word] = vec
                    word2id[word] = int(len(word2id))
                    vectors.append(vec[None])

    id2word = {v:k for k,v in word2id.items()}
    embs = np.concatenate(vectors, 0)

    if norm:
        embs = normalize(embs, axis=1, norm='l2')

    return embs, word2vec, word2id, id2word


def load_dictionary(path, max_vocab, src_word2id, tgt_word2id):
    """Loads training dict from file at path."""

    assert os.path.isfile(path)
    dico_list = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == max_vocab:
                break
            split = line.split()
            if split[0] in src_word2id and split[1] in tgt_word2id:
                id1 = int(src_word2id[split[0]])
                id2 = int(tgt_word2id[split[1]])
                dico_list.append([id1, id2])

    dico = np.array(dico_list)

    return dico


def get_parallel_data(src_embs, tgt_embs, dico):
    """Return parallel X and Y matrices corresponding to source
       and target embedding matrices respectively."""

    X = src_embs[dico[:, 0].astype(np.int16)]
    Y = tgt_embs[dico[:, 1].astype(np.int16)]

    return X, Y


def sigmoid(x):
    """Sigmoid function"""

    return 1/(1 + np.exp(-x))

def str2bool(v):
    """
    thanks : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_tf_idf(corpus, word2vec, id2word, vec_dim=300, mapper=np.ones(300), source=True, norm=True):
    'Computes the tf-idf weight for the corpus'
    from collections import Counter, defaultdict
    N = len(corpus)
    idfmap = defaultdict(int)
    tf = defaultdict(int)
    corpus_vec = np.zeros((N, vec_dim))

    # Create idfmap
    for sent_i, sentence in enumerate(corpus):
        word_indices = np.unique(sentence)
        for word_i in word_indices:
            idfmap[word_i] += 1

    # Create tf-idf weights
    for sent_i, sentence in enumerate(corpus):
        vec = np.zeros(vec_dim)
        index, row_count = np.unique(sentence, return_counts=True)
        index = index.astype(np.int32)
        try:
            f_max = np.max(row_count)
        except ValueError:
            corpus_vec[sent_i] = np.zeros(vec_dim)
            continue

        # For every unique word in sentence
        for i, f in zip(index, row_count):
            idf = np.log(N/idfmap[i])
            tf = (1 + np.log(f))/(1 + np.log(f_max))
            try:
                if source:
                    vec += tf*idf*word2vec[id2word[i]]@mapper
                else:
                    vec += tf*idf*word2vec[id2word[i]]
            except KeyError:
                continue

        vec = vec[None]
        if norm:
            vec = normalize(vec, axis=1, norm='l2')
        corpus_vec[sent_i] = vec

    return corpus_vec

def read_sents(sentpath, embpath, mappath, limit=None, max_sent=20000, padlen=30, test=False, maxvocab=200000, random_state=42, comp_tfidf = True, project=True, evaluate = False):
    """

    :param sentpath: path to europarl sentence textfile
    :param embpath: path to monolingual embedding textfile
    :param mappath: path to mono- to multi-lingual embedding map
    :param limit: number of sentences to randomly select from sequentially read sentences
    :param max_sent: maximum number of sentences read sequentially
    :param padlen: unified length of words considered in sentence embedding, longer sentences truncated, shorter ones padded with zeros'
    :param test: indicates whether test set read (last 100.000 sentences)
    :param maxvocab:
    :param random_state: maximum vocabulary size considred when reading from embedding file
    :param comp_tfidf: boolean, compute tf-idf scores
    :param project: boolean, whether to project

    :return:
    """

    print("Reading embeddings...")
    assert os.path.isfile(sentpath)
    assert os.path.isfile(embpath)
    embs, word2vec, word2id, id2word = load_embs(embpath, maxvocab)
    print("Embeddings read")

    if project:
        print("Projecting embedding to multi-lingual space...")
        assert os.path.isfile(embpath)
        with open(mappath, "rb") as f:
            mapping = pickle.load(f)
        embs = np.matmul(embs, mapping)
        ## update word2vec to proper embedding space
        for i, word in enumerate(word2vec):
            word2vec[word] = embs[i, :]
        print("Embedding projected to multi-lingual space")

    # shuffle logic to randomly select sentences according to limit

    if limit is not None:
        np.random.seed(random_state)
        retrieve_indices = np.random.choice(range(max_sent), size=limit, replace=False)
        retrieve_indices = sorted(retrieve_indices)

    N = 1965734
    test_threshold = N - 100000

    original_sents = []
    encoded_sents = []
    encoded_sents_unsup = []

    print("Generating sentence representations...")

    if not test:
        if limit is not None:
            with open(sentpath, 'r') as f:
                for i, line in enumerate(f):

                    if i >= max_sent:
                        break
                    else:
                        original_sent = re.sub(r"[\n]", "", line).lower()
                        original_sents.append(original_sent)

                        encoded_sent = [word2id[word] for word in tokenize(original_sent) if (word in word2vec)]
                        encoded_sents_unsup.append(encoded_sent)
                        encoded_sent = [i for sub in pad_sequences([encoded_sent], maxlen=padlen) for i in sub]
                        encoded_sents.append(encoded_sent)

                print("Sampling sentence representations...")

                original_sents = [original_sents[i] for i in retrieve_indices]

        else:
            with open(sentpath, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_sent:
                        break
                    else:
                        original_sent = re.sub(r"[\n]", "", line).lower()
                        original_sents.append(original_sent)

                        encoded_sent = [word2id[word] for word in tokenize(original_sent) if (word in word2vec)]
                        encoded_sents_unsup.append(encoded_sent)
                        encoded_sent = [i for sub in pad_sequences([encoded_sent], maxlen=padlen) for i in sub]
                        encoded_sents.append(encoded_sent)


    else:
        with open(sentpath, 'r') as f:
            for i, line in enumerate(f):
                if i >= test_threshold:
                    original_sent = re.sub(r"[\n]", "", line).lower()
                    original_sents.append(original_sent)

                    encoded_sent = [word2id[word] for word in tokenize(original_sent) if (word in word2vec)]
                    encoded_sents_unsup.append(encoded_sent)
                    encoded_sent = [i for sub in pad_sequences([encoded_sent], maxlen=padlen) for i in sub]
                    encoded_sents.append(encoded_sent)

    if comp_tfidf:
        print("Generating TF-IDF unsupervised representations...")
        tf_idf_sents = compute_tf_idf(encoded_sents_unsup, word2vec, id2word, source=False)
    else:
        ## create dummy
        tf_idf_sents = None

    print("Done")
    if evaluate:
        return embs, word2vec, word2id, id2word, original_sents, encoded_sents, encoded_sents_unsup, tf_idf_sents
    else:
        return embs, original_sents, encoded_sents, encoded_sents_unsup, tf_idf_sents

def get_sample(arr, n, n_iter=None, sample_size=10, fast=True):
    """Get random sample from arr.

    Parameters
    ----------
    arr: np.array, array to sample from.
    n_iter: int, current iteration number.
    sample_size: int, sample size
    fast: bool, use sampling optimized for fast consecutive samples from the same array.

    Returns
    -------
    sample: np.array, sample from arr of length n_iter.

    credits to: https://medium.freecodecamp.org/how-to-get-embarrassingly-fast-random-subset-sampling-with-python-da9b27d494d9
    """
    if fast:
        np.random.seed(42)
        # find the index we last sampled from
        start_idx = (n_iter * sample_size) % n
        if start_idx + sample_size >= n:
            # shuffle array if we have reached the end and repeat again
            np.random.shuffle(arr)

        return arr[start_idx:start_idx + sample_size]
    else:
        np.random.seed(42)
        return np.random.choice(arr, sample_size, replace=False)


def collect_samples(arr, sample_size, n, n_samples, fast=False):
    """
    Collect several samples from arr.

    Parameters
    ----------
    arr: np.array, array to sample from.
    sample_size: int, sample size.
    n_samples: int, number of samples to take.
    fast: bool, use sampling optimized for fast consecutive samples from the same array.

    Returns
    -------
    samples: np.ndarray, sample matrix of shape (n_samples, sample_size)

    credits to: https://medium.freecodecamp.org/how-to-get-embarrassingly-fast-random-subset-sampling-with-python-da9b27d494d9
    """
    samples = np.zeros((n_samples + 1, sample_size), np.int32)

    for sample_n in range(0, n_samples):
        sample = get_sample(arr,
                            n,
                            n_iter=sample_n,
                            sample_size=sample_size,
                            fast=fast)
        samples[sample_n] = sample

    return samples


def generate_train_random(train_base, ratio=1):
    """
    Generates training data that 50:50 comprises true translation and random sentences pairs.
    True pairs are labelled 1, whereas random pairs are labelled 0.

    :param train_base: pd.DataFrame comprising true translation pairs
    :param ratio: integer, denotes scale of data, e.g. 1 doubles original translation pairs, 2 copies translation pairs 1 and generates 2 random instances per pair
    :return: pd.DataFrame, training data with true translation and random sentence pairs
    """

    num_neg_train_instances = train_base.shape[0]
    all_indices = np.array(train_base.index)
    train_data = pd.DataFrame(columns=['src', 'tgt'])

    two_indices_array = collect_samples(np.array(range(0, num_neg_train_instances)), 2, int(num_neg_train_instances / 10), num_neg_train_instances * ratio, fast=True)[:-1]
    src_indices = []
    tgt_indices = []
    for entry in two_indices_array:
        src_indices.append(entry[0])
        tgt_indices.append(entry[1])

    train_data = pd.DataFrame(data={"src": np.array(train_base.loc[src_indices]["src"]), \
                                    "tgt": np.array(train_base.loc[tgt_indices]["tgt"])})

    train_data = train_data.sample(frac=1, random_state=42)  # shuffle data
    train_data['label'] = 0

    framelist = [train_data]
    for i in range(ratio):
        framelist.append(train_base)

    train_data = pd.concat(framelist, ignore_index=True)
    train_data = train_data.sample(frac=1, random_state=42)  # shuffle data

    return np.array(train_data["src"].tolist()), np.array(train_data["tgt"].tolist()), np.array(train_data["label"].tolist())


def generate_train_unsupervised(train_base, src_tf_idf, tgt_tf_idf, weight=0.4, ratio=1, faiss_size=70000):
    num_instances = train_base.shape[0]
    sep_index = int(weight * num_instances)
    src_encoded_sents = np.array(train_base['src'].tolist())
    tgt_encoded_sents = np.array(train_base['tgt'].tolist())
    src_encoded_sents_random = src_encoded_sents[0:sep_index]
    tgt_encoded_sents_random = tgt_encoded_sents[0:sep_index]
    src_encoded_sents_hard = src_encoded_sents[sep_index:]
    tgt_encoded_sents_hard = tgt_encoded_sents[sep_index:]

    src_tfidf = src_tf_idf[:train_base.shape[0]][sep_index:]
    tgt_tfidf = tgt_tf_idf[:train_base.shape[0]][sep_index:]

    two_indices_array = collect_samples(np.array(range(0, sep_index)), 2, int(sep_index / 10), ratio * sep_index, fast=True)[:-1]
    src_indices = []
    tgt_indices = []
    for entry in two_indices_array:
        src_indices.append(entry[0])
        tgt_indices.append(entry[1])

    train_data = pd.DataFrame(data={"src": np.array(train_base.loc[src_indices]["src"]), \
                                    "tgt": np.array(train_base.loc[tgt_indices]["tgt"])})
    num_folds_hard = int((num_instances - sep_index) / faiss_size)
    topk = 20
    for i in range(num_folds_hard + 1):

        index = faiss.IndexFlatL2(300)
        if i != (num_folds_hard):
            index.add(tgt_tfidf[(i * faiss_size):((i + 1) * faiss_size)].astype(np.float32))
            D, I = index.search((src_tfidf[(i * faiss_size):((i + 1) * faiss_size)]).astype(np.float32), topk)
        else:
            index.add(tgt_tfidf[(i * faiss_size):].astype(np.float32))
            D, I = index.search((src_tfidf[(i * faiss_size):]).astype(np.float32), topk)

        match_indices = pd.DataFrame(I).iloc[:, 0:(ratio + 1)]
        match_indices['index'] = list(range(index.ntotal))
        intermediate = match_indices.iloc[:, 0:(ratio + 1)].apply(lambda x: x - match_indices['index'])
        for j in range(ratio):
            replacement = intermediate.loc[intermediate[j] == 0, j + 1]
            intermediate.loc[intermediate[j] == 0, j + 1] = 0
            intermediate.loc[intermediate[j] == 0, j] = replacement

        intermediate = intermediate.apply(lambda x: x + match_indices['index'])
        intermediate.drop(labels=[ratio], axis=1, inplace=True)
        if i != (num_folds_hard):
            src_current_base = np.array(src_encoded_sents_hard[(i * faiss_size):((i + 1) * faiss_size)])
            tgt_current_base = np.array(tgt_encoded_sents_hard[(i * faiss_size):((i + 1) * faiss_size)])
        else:
            src_current_base = np.array(src_encoded_sents_hard[(i * faiss_size):])
            tgt_current_base = np.array(tgt_encoded_sents_hard[(i * faiss_size):])

        src_current_indices = []
        tgt_current_indices = []

        for j in range(ratio):
            src_current_indices += match_indices['index'].tolist()
            tgt_current_indices += intermediate[j].tolist()

        src_current_base = list(np.array(src_current_base)[src_current_indices])
        tgt_current_base = list(np.array(tgt_current_base)[tgt_current_indices])

        current_frame = pd.concat([pd.DataFrame([[i] for i in src_current_base]),
                                   pd.DataFrame([[i] for i in tgt_current_base])], axis=1)
        current_frame.columns = ['src', 'tgt']

        train_data = pd.concat([train_data, current_frame], ignore_index=True)

        print('Processed fold', i + 1, 'out of', num_folds_hard + 1)

    train_data = train_data.sample(frac=1, random_state=42)  # shuffle data
    train_data['label'] = 0

    framelist = [train_data]
    for i in range(ratio):
        framelist.append(train_base)

    train_data = pd.concat(framelist, ignore_index=True)
    train_data = train_data.sample(frac=1, random_state=42)  # shuffle data

    return np.array(train_data["src"].tolist()), np.array(train_data["tgt"].tolist()), np.array(train_data["label"].tolist())


def gen_file_paths(data_dir, src_lang, tgt_lang, source=True):
    """

    :param data_dir: Subdirectory containing data in respective subdirectories /embs/; /europarl/
    :param src_lang: source language [en]
    :param tgt_lang: target language [de, es, fr, it]
    :param source: boolean, whether to read paths for src_lang or tgt_lang
    :return: mono-lingual embeddings, embedding mapping, and sentences by lang
    """

    if source:
    # Source
        src_embs_file = os.path.join(data_dir, "embs", "wiki." + src_lang + ".vec")
        src_map_file = os.path.join(data_dir, "embs", "{}-{}-200000-supervised.pickle".format(src_lang, tgt_lang))
        src_sent_file = os.path.join(data_dir, "europarl", "Europarl." + "{}-{}.{}".format(src_lang, tgt_lang, src_lang))
        return src_embs_file, src_map_file, src_sent_file
    else:
        # Target
        tgt_embs_file = os.path.join(data_dir, "embs", "wiki." + tgt_lang + ".vec")
        tgt_map_file = os.path.join(data_dir, "embs", "{}-{}-200000-supervised.pickle".format(tgt_lang, src_lang))
        tgt_sent_file = os.path.join(data_dir, "europarl", "Europarl." + "{}-{}.{}".format(src_lang, tgt_lang, tgt_lang))
        return tgt_embs_file, tgt_map_file, tgt_sent_file