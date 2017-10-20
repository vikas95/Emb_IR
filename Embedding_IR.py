import math
import ast
import numpy as np
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')


################## TF values
file1=open("TF_Doc_len.txt","r")
Doc_Length=[]
All_Documents=[]
counter=1
for line1 in file1:
    if counter==1:
        Doc_Length=ast.literal_eval(line1)
    else:
        All_Documents.append(ast.literal_eval(line1))

    counter+=1

Doc_Avg_Len=sum(Doc_Length)/float(len(Doc_Length))
print("Avg doc length is : ",Doc_Avg_Len)
Doc_Length=np.asarray(Doc_Length)
################## IDF values
file2=open("IDF.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)
print("IDF of when is : ", IDF["when"])
###################
## Scoring function which calculates the score of TF-IDF
def BM25_score(Question, Corpus, IDF, Doc_Avg_Len, Doc_Length):
    dummy_TF=0
    Doc_Score=[0]
    K=1.2
    b=0.75
    Ques_terms=tokenizer.tokenize(Question.lower())

    for doc_ind, Document1 in enumerate(Corpus):
        Score = 0
        for term1 in Ques_terms:
            if term1 in Document1.keys():
               dummy_TF=int(Document1[str(term1)])
            else:
               dummy_TF=0

            #Score=Score+ ( (IDF[str(term1)]) * (dummy_TF*(K+1)/float(dummy_TF + K*(1-b+ (b*(Doc_Length[doc_ind]/float(Doc_Avg_Len)))))  )  )

        Doc_Score.append(Score)

    #Doc_Score=sorted(Doc_Score)
    #Doc_Score=Doc_Score[0:5]    ####### Returning top 5 scores
    #Doc_Score=sum(Doc_Score)
    #Doc_Score=max(Doc_Score)
    Doc_Score=Doc_Score[0]
    return Doc_Score
###################

Question_file = open('training_set.tsv', 'r')
Question = "" #[]
Correct_ans = []#[]
Option_A ="" # []  ####### These will contain justification text also and later on, becky features will be added.
Option_B = "" #[]
Option_C = "" #[]
Option_D = "" #[]

Final_Answers=[]

Scores_A=0#[]
Scores_B=0#[]
Scores_C=0#[]
Scores_D=0#[]
Quest_num=0
for line1 in Question_file:
    Quest_num+=1
    print("We are on Quest num: ", Quest_num)
    Cand_score = []
    line1 = line1.strip()
    cols = line1.split("\t")
    Correct_ans.append(cols[3])
    A_index = cols[10].index("(A)")
    B_index = cols[10].index("(B)")
    C_index = cols[10].index("(C)")
    D_index = cols[10].index("(D)")
    """
    Question.append(cols[10][:A_index - 1])
    Option_A.append(cols[10][A_index + 4:B_index - 1])
    Option_B.append(cols[10][B_index + 4:C_index - 1])
    Option_C.append(cols[10][C_index + 4:D_index - 1])
    Option_D.append(cols[10][D_index + 4:])
    """
    Question=(cols[10][:A_index - 1])
    Option_A=(cols[10][A_index + 4:B_index - 1])
    Option_B=(cols[10][B_index + 4:C_index - 1])
    Option_C=(cols[10][C_index + 4:D_index - 1])
    Option_D=(cols[10][D_index + 4:])

    Ques1=Question + " "+ Option_A  ###### Question + Candidate answer 1
    Scores_A=BM25_score(Ques1,All_Documents,IDF, Doc_Avg_Len, Doc_Length)   ##### This is sum of top 5 max scores

    Ques2= Question + " " + Option_B
    Scores_B = BM25_score(Ques2, All_Documents, IDF, Doc_Avg_Len, Doc_Length)  ##### This is sum of top 5 max scores

    Ques3 = Question + " " + Option_C
    Scores_C = BM25_score(Ques3, All_Documents, IDF, Doc_Avg_Len, Doc_Length)   ##### This is sum of top 5 max scores

    Ques4 = Question + " " + Option_D
    Scores_D = BM25_score(Ques4, All_Documents, IDF, Doc_Avg_Len, Doc_Length)   ##### This is sum of top 5 max scores

    Cand_score=[Scores_A, Scores_B, Scores_C, Scores_D]
    Cand_score=np.asarray(Cand_score)
    Final_Answers.append(np.argmax(Cand_score))



