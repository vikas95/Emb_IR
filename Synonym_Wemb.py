import heapq
import math
import ast
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
#becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
# f = open('glove.840B.300d.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in glove_emb:
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)


def synonym(term1, WordEmb):
    max_dot_val=0
    synonym_word=""
    min_dot_val=0
    antonym_word=""

    for curr_key in WordEmb.keys():
        if curr_key!=term1:
            val=np.dot(WordEmb[term1],WordEmb[curr_key])
            if val>max_dot_val:
               max_dot_val=val
               synonym_word=str(curr_key)
            if val<min_dot_val:
               min_dot_val=val
               antonym_word=str(curr_key)

    print(max_dot_val, min_dot_val)
    return (synonym_word, antonym_word)

#word_vectors = KeyedVectors.load_word2vec_format('ss_qz_04.dim50vecs.txt', binary=False)

#word_vectors.most_similar("blood")

syn1, ant1 = synonym("cell",embeddings_index)

print(syn1, " ", ant1)