from os.path import isfile
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

def load_w2v_embeddings(word_index, max_vocab_size):
    # get word2vec embeddings
    # TODO how to get embeddings if not locally?
    emb_len = 300
    fn = '/Users/charlesj/tools/nlp/word2vec/GoogleNews-vectors-negative300.bin.gz'
    if isfile(fn):
        w2v = KeyedVectors.load_word2vec_format(fn, binary=True)
    else:
        raise FileNotFoundError("No pretrained embeddings. Use default initialization.")
    num_words = min(max_vocab_size, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, emb_len))
    for word, i in word_index.items():
        if i >= num_words - 1:
            continue
        if word in w2v.wv:
            embedding_vector = w2v.wv[word]
            embedding_matrix[i] = embedding_vector
        else:
            # words not found in embedding index will be random
            embedding_matrix[i] = np.random.uniform(-0.001, 0.001, (1, emb_len))

    print('Number of words in embedding matrix: ', num_words)
    return [embedding_matrix]