

import heapq
import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

"""
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
"""
becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
# f = open('glove.840B.300d.txt','r', encoding='utf-8')

#f = open('ss_qz_04.dim50vecs.txt')
for line in becky_emb:
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
"""
term_absent=0
for term1 in All_words:
    if term1 in embeddings_index.keys():
       pass
    else:
       term_absent+=1

print("number of terms not present in W2V is:  ", term_absent)


mat1=np.matrix([[4,5],[6,7]])
mat2=np.matrix([[1,5],[2,7],[3,6]])
mat2=mat2.transpose()
c=np.matmul(mat1,mat2)
d=np.amax(c,axis=1)
e=np.amin(c,axis=1)

idf=numpy.matrix([[2],[3]])
g=np.multiply(idf,c)  ####### elementwise matrix multiplication.


"""









file2=open("IDF.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)



def Word2Vec_score(Question, IDF_Mat, Corpus, All_terms):

    Doc_Score=[0]

    Top_docs=5  ## top 20 scores for each question

    max_score=0
    min_score=0
    #Ques_score=[]
    Document_score=[[0] for i in range(len(Question))]
    #SCORES=[]


    for doc_ind, Document1 in enumerate(Corpus):
        if doc_ind==0:  #### length
           pass
        else:
            if doc_ind%5000==0:

               print(doc_ind)

            Doc_dict=ast.literal_eval(Document1)
            Doc_set=set(Doc_dict.keys())
            Common_term_sum=0
            for terms1 in All_terms:
                Common_term_sum+=len(list(set(Doc_dict.keys()) & terms1))

            if Common_term_sum<1000: ### (not relevant to any of the question in the batch)
               #print("Yep, there are cases like these....")
               pass
            else:
                print ("what is wrong:::::::: ")
                Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
                for key in Doc_dict:
                    if key in embeddings_index.keys():
                       Doc_Matrix=np.append(Doc_Matrix, np.array([embeddings_index[key]]), axis=0)

                if Doc_Matrix.size==0:
                   pass
                else:

                    Doc_Matrix=Doc_Matrix.transpose()
                    #print(Doc_Matrix.shape)


                    for qind, ques1 in enumerate(Question):
                        if len(list(set(Doc_dict.keys()).intersection(All_terms[int(qind/4)])))<4:  ## atleast one of the 4 option has 3 common terms
                           pass
                        else:
                            Score=np.matmul(ques1,Doc_Matrix)
                            max_score=np.amax(Score,axis=1)
                            max_score=np.multiply(IDF_Mat[qind],max_score)
                            max_score=(sum(max_score)).item(0)
                            min_score=np.amin(Score,axis=1)
                            min_score = np.multiply(IDF_Mat[qind], min_score)
                            min_score=(sum(min_score)).item(0)
                            total_score=max_score+min_score
                            if doc_ind>Top_docs:
                               Document_score[qind].append(total_score)  ## here Top_docs+1, but will be reduced in below step
                               Document_score[qind]=heapq.nlargest(Top_docs,Document_score[qind])  ### this will keep only Top_docs
                               pass
                            else:
                               Document_score[qind].append(total_score)


    return Document_score



def Ques_Emb(ques1, IDF):
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)
    return Ques_Matrix, IDF_Mat


Question_file = open('training_set.tsv', 'r')
 #[]
Correct_ans = []#[]
All_questions = []
IDF_Mat=[]


counter=0
file1=open("TF_Doc_len.txt","r")
Final_scores=[]
All_terms=[]
All_Ques_terms=[]
start=1
End=start+100
for line1 in Question_file:
    counter+=1
    print(counter)
    if counter<start:
       pass
    else:
        Question = ""
        Option_A = ""  # []  ####### These will contain justification text also and later on, becky features will be added.
        Option_B = ""  # []
        Option_C = ""  # []
        Option_D = ""
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

        Question = tokenizer.tokenize(Question.lower())
        Question=[lmtzr.lemmatize(w1) for w1 in Question]

        Option_A = tokenizer.tokenize(Option_A.lower())
        Option_A = [lmtzr.lemmatize(w1) for w1 in Option_A]

        Option_B = tokenizer.tokenize(Option_B.lower())
        Option_B = [lmtzr.lemmatize(w1) for w1 in Option_B]

        Option_C = tokenizer.tokenize(Option_C.lower())
        Option_C = [lmtzr.lemmatize(w1) for w1 in Option_C]

        Option_D = tokenizer.tokenize(Option_D.lower())
        Option_D = [lmtzr.lemmatize(w1) for w1 in Option_D]

        All_Ques_terms=Question+Option_A+Option_B+Option_C+Option_D
        All_Ques_terms=(set(All_Ques_terms))
        All_terms.append(All_Ques_terms)
        All_Ques_terms=[]


        Ques1 = Question + Option_A  ###### Question + Candidate answer 1
        Q1_matrix, IDF_Mat1=Ques_Emb(Ques1, IDF)

        Ques2 = Question + Option_B
        Q2_matrix, IDF_Mat2 = Ques_Emb(Ques2, IDF)

        Ques3 = Question + Option_C
        Q3_matrix, IDF_Mat3 = Ques_Emb(Ques3, IDF)

        Ques4 = Question + Option_D
        Q4_matrix, IDF_Mat4 = Ques_Emb(Ques4, IDF)



        All_questions += [Q1_matrix, Q2_matrix, Q3_matrix, Q4_matrix]
        IDF_Mat += [IDF_Mat1, IDF_Mat2, IDF_Mat3, IDF_Mat4]
        if counter%End==0:  ## taking 25 questions at a time

           Score_matrix = Word2Vec_score(All_questions, IDF_Mat,  file1, All_terms)

           print(Score_matrix)
           out_file=open("Ques"+str(start)+"_"+str(End)+".txt","w")
           out_file.write(str(Score_matrix))


           All_questions=[]
           IDF_Mat = []
           break
        # All_questions += [Ques1, Ques2, Ques3, Ques4]  ###### "All_questions" will be having 10000 questions.

