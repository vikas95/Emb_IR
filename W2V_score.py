###### This version
import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization

tokenizer = RegexpTokenizer(r'\w+')

################## TF values
file1 = open("TF_Doc_len.txt", "r")
Doc_Length = []
All_Documents = []
counter = 1

for line1 in file1:
    if counter == 1:
        Doc_Length = ast.literal_eval(line1)
    else:
        break
    counter += 1

Doc_Avg_Len = sum(Doc_Length) / float(len(Doc_Length))
print("Avg doc length is : ", Doc_Avg_Len)
Doc_Length = np.asarray(Doc_Length)

################## IDF values
file2 = open("IDF.txt", "r")
for line2 in file2:
    IDF = ast.literal_eval(line2)
print("IDF of when is : ", IDF["when"])

###################

Corpus1 = open("quizlet_corpus.qz.txt", "r")
Corpus2 = open("studystack_corpus.st.txt", "r")
Corpus = [Corpus1, Corpus2]


## Scoring function which calculates the score of TF-IDF
def BM25_score(Question, Corpus, IDF, Doc_Length_all, Doc_Avg_Len):
    dummy_TF = 0
    Doc_Score = [0]
    K = 1.2
    b = 0.75
    Score = 0
    Ques_score = []
    Document_score = []
    SCORES = []

    for doc_ind, Document1 in enumerate(Corpus):
        if doc_ind == 0:  #### length
            pass
        else:
            if doc_ind % 2 == 0:
                print(doc_ind)

            Doc_dict = ast.literal_eval(Document1)
            Doc_Length = Doc_Length_all[doc_ind - 1]

            for ques1 in Question:
                Ques_terms = tokenizer.tokenize(ques1.lower())
                Ques_terms = list(set(Ques_terms))
                for term1 in Ques_terms:
                    if term1 in Doc_dict.keys():
                        dummy_TF = Doc_dict[term1]
                    else:
                        dummy_TF = 0

                    Score = Score + ((IDF[str(term1)]) * (
                    dummy_TF * (K + 1) / float(dummy_TF + K * (1 - b + (b * (Doc_Length / float(Doc_Avg_Len)))))))

                Ques_score.append(Score)
                Score = 0

            Document_score.append(Ques_score)  ##### each element of Document_score is 10000 in length
            Ques_score = []

    # Doc_Score=sorted(Doc_Score)
    # Doc_Score=Doc_Score[0:5]    ####### Returning top 5 scores
    # Doc_Score=sum(Doc_Score)
    # Doc_Score=max(Doc_Score)
    # Doc_Score=Doc_Score[0]
    return Document_score


###################

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

Score_matrix = BM25_score(All_questions, file1, IDF, Doc_Length, Doc_Avg_Len)


