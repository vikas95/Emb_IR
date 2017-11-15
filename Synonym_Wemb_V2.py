import heapq
import math
import ast
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))


"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
#becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
f = open('glove.840B.300d.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
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

    if term1 not in WordEmb.keys():  ###### Discuss this with Becky, we have to do something about this.
       synonym_word=term1
       antonym_word=term1
    else:

        for curr_key in WordEmb.keys():
            if curr_key!=term1 or lmtzr.lemmatize(curr_key)!= term1:   ## we dont want synonyms or close words with same parent word, hence lemmatization
                if len(WordEmb[curr_key])==len(WordEmb[term1]):
                    val=np.dot(WordEmb[term1],WordEmb[curr_key])
                    if val>max_dot_val:
                       max_dot_val=val
                       synonym_word=str(curr_key)
                    if val<min_dot_val:
                       min_dot_val=val
                       antonym_word=str(curr_key)
    return (synonym_word, antonym_word)


def Question_synonym(ques1, WordEmb):
    New_ques1=""
    for term1 in ques1:
        syn_term1,oppterm1=synonym(term1,WordEmb)
        New_ques1=New_ques1+" "+str(term1)+" "+str(syn_term1)

    return New_ques1



#word_vectors = KeyedVectors.load_word2vec_format('ss_qz_04.dim50vecs.txt', binary=False)

#word_vectors.most_similar("blood")

#syn1, ant1 = synonym("cell",embeddings_index)


Question_file = open('training_set.tsv', 'r')
 #[]

Question_set=[]



counter=0
file1=open("structured_kerasInput_train_bestIR_08j5.tsv","r")
Final_scores=[]
All_terms=[]
All_Ques_terms=[]
Q_terms_list=[]



for line1 in Question_file:
    counter+=1

    #print(counter)
    Question = ""
    Option_A = ""  # []  ####### These will contain justification text also and later on, becky features will be added.
    Option_B = ""  # []
    Option_C = ""  # []
    Option_D = ""
    Cand_score = []
    line1 = line1.strip()
    cols = line1.split("\t")

    A_index = cols[10].index("(A)")
    B_index = cols[10].index("(B)")
    C_index = cols[10].index("(C)")
    D_index = cols[10].index("(D)")

    Question = (cols[10][:A_index - 1])
    Option_A = (cols[10][A_index + 4:B_index - 1])
    Option_B = (cols[10][B_index + 4:C_index - 1])
    Option_C = (cols[10][C_index + 4:D_index - 1])
    Option_D = (cols[10][D_index + 4:])

    Question = tokenizer.tokenize(Question.lower())
    Question=[lmtzr.lemmatize(w1) for w1 in Question]
    Question = [w for w in Question if not w in stop_words]

    Option_A = tokenizer.tokenize(Option_A.lower())
    Option_A = [lmtzr.lemmatize(w1) for w1 in Option_A]
    Option_A = [w for w in Option_A if not w in stop_words]

    Option_B = tokenizer.tokenize(Option_B.lower())
    Option_B = [lmtzr.lemmatize(w1) for w1 in Option_B]
    Option_B = [w for w in Option_B if not w in stop_words]

    Option_C = tokenizer.tokenize(Option_C.lower())
    Option_C = [lmtzr.lemmatize(w1) for w1 in Option_C]
    Option_C = [w for w in Option_C if not w in stop_words]

    Option_D = tokenizer.tokenize(Option_D.lower())
    Option_D = [lmtzr.lemmatize(w1) for w1 in Option_D]
    Option_D = [w for w in Option_D if not w in stop_words]

    All_Ques_terms = Question + Option_A + Option_B + Option_C + Option_D
    All_Ques_terms = list(set(All_Ques_terms))
    All_terms+=All_Ques_terms
    All_Ques_terms = []

    Ques1 = Question + Option_A  ###### Question + Candidate answer 1
    Question_set.append(Ques1)

    Ques2 = Question + Option_B
    Question_set.append(Ques2)

    Ques3 = Question + Option_C
    Question_set.append(Ques3)

    Ques4 = Question + Option_D
    Question_set.append(Ques4)


print ("number of questions are: ",len(Question_set))
print (len(All_terms))
All_terms=list(set(All_terms))
print ("all unique terms",len(All_terms))

Synonym_terms={}
for ind1, term1 in enumerate(All_terms):
    if ind1%20==0:
       print (ind1)
    #print (term1)
    syn_term1, opp_term1=synonym(term1,embeddings_index)
    Synonym_terms.update({term1:syn_term1})

print (len(Synonym_terms))

syn_file=open("synonyms_ques.txt","w")
syn_file.write(str(Synonym_terms))

#New_ques_file=open("Synonym_added_questions.txt","w")
