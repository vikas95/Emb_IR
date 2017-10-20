import numpy as np

import ast
"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
#f = open('glove.6B.100d.txt','r', encoding='utf-8')
# f = open('glove.840B.300d.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in becky_emb:
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
term_absent=0
for term1 in All_words:
    if term1 in embeddings_index.keys():
       pass
    else:
       term_absent+=1

print("number of terms not present in W2V is:  ", term_absent)

"""

mat1=np.matrix([[4,5],[6,7]])
mat2=np.matrix([[1,5],[2,7],[3,6]])
mat2=mat2.transpose()
c=np.matmul(mat1,mat2)
d=np.amax(c,axis=1)
print(c)
print(d)

"""






import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')


file2=open("IDF.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)



def Word2Vec_score(Question, Corpus, IDF, Doc_Length_all, Doc_Avg_Len):

    Doc_Score=[0]

    Score = 0
    Ques_score=[]
    Document_score=[]
    SCORES=[]


    for doc_ind, Document1 in enumerate(Corpus):
        if doc_ind==0:  #### length
           pass
        else:
            if doc_ind%2==0:
               print(doc_ind)

            Doc_dict=ast.literal_eval(Document1)
            Doc_Length = Doc_Length_all[doc_ind-1]



            for ques1 in Question:
                Ques_terms = tokenizer.tokenize(ques1.lower())
                Ques_terms=list(set(Ques_terms))





Question_file = open('training_set.tsv', 'r')
Question = ""  # []
Correct_ans = []  # []
Option_A = ""  # []  ####### These will contain justification text also and later on, becky features will be added.
Option_B = ""  # []
Option_C = ""  # []
Option_D = ""  # []

All_questions = []
for line1 in Question_file:
    Cand_score = []
    line1 = line1.strip()
    cols = line1.split("\t")
    Correct_ans.append(cols[3])
    A_index = cols[10].index("(A)")
    B_index = cols[10].index("(B)")
    C_index = cols[10].index("(C)")
    D_index = cols[10].index("(D)")

    Question = (cols[10][:A_index - 1])
    Option_A = (cols[10][A_index + 4:B_index - 1])
    Option_B = (cols[10][B_index + 4:C_index - 1])
    Option_C = (cols[10][C_index + 4:D_index - 1])
    Option_D = (cols[10][D_index + 4:])

    Ques1 = Question + " " + Option_A  ###### Question + Candidate answer 1

    Ques2 = Question + " " + Option_B

    Ques3 = Question + " " + Option_C

    Ques4 = Question + " " + Option_D

    All_questions += [Ques1, Ques2, Ques3, Ques4]  ###### All_questions will be having 10000 questions.

Score_matrix = Word2Vec_score(All_questions, file1, IDF, Doc_Length, Doc_Avg_Len)



