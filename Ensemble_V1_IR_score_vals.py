import ast
import numpy as np
from statistics import mean
from collections import Counter
import math
import nltk

file1=open("Becky_files_W2V_score_10_Final.txt","r")
scores=[]

for line in file1:
    scores=ast.literal_eval(line)

# print (scores[205])
for sind, sc1 in enumerate(scores):
    curr_score=np.asarray(sc1)
    non_z=np.count_nonzero(curr_score)
    if non_z == 0:
       #print (sind)
       scores[sind]=sum(sc1)
       #print (scores[sind])
    else:
       ranked_score=0
       for i1,s1 in enumerate(sc1):
           ranked_score+=(s1/float(i1+1))

       #scores[sind]=sum(sc1)/float(non_z)
       scores[sind]=ranked_score
# scores=[sum(score1) for score1 in scores]

file2=open("structured_kerasInput_train_bestIR_08j5.tsv","r")
IR_score=[]
Ques_score=[]
Final_ques_score=[]
Correct_ans_2=[]    ### for each candidate level
Ques_ans=[]
Just_score=0
Predicted_ans=[]

W2Vec_score=[]

for ind2,line2 in enumerate(file2):

    line2 = line2.strip()
    cols = line2.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
    Feature_col = cols[4].split()

    Feature_col = cols[6].split(";;")
    # print (len(Feature_col))
    Justification_threshold = 2
    if len(Feature_col) >= Justification_threshold:
        for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
            ##["AggregatedJustification"]["text"]
            dict1 = ast.literal_eval(Feature_col[ind1])
            Just_score += (int(dict1["AggregatedJustification"]["score"])/float(ind1+1))








    #Just_score += float(Feature_col[2])
    Ques_score.append(Just_score)
    Just_score = 0

    W2Vec_score.append(scores[ind2])


    if cols[1]==cols[2]:
       Ques_ans.append(1)
    else:
       Ques_ans.append(0)

    if ind2%4==3:
       if len(Ques_score)!=4:
          print(ind2)
          Ques_score = []
          Ques_ans = []
       else:

           m = max(W2Vec_score)
           if m!=0:
              W2Vec_score = [math.pow(s1/float(m),2) for s1 in W2Vec_score]  #### normalizing value
           else:
              print ("well, its here ", ind2)
           # print (W2Vec_score)
           qm=max(Ques_score)
           Ques_score=[qs1/float(qm) for qs1 in Ques_score] #### normalizing value

           #Ques_score=[sum(x) for x in zip(Ques_score, W2Vec_score)]
           #Ques_score = zip(Ques_score, W2Vec_score)
           for ind_q, qs in enumerate(Ques_score):
               Ques_score[ind_q]=qs+(0.2*W2Vec_score[ind_q])  ##

           Ques_score=np.asarray(Ques_score)
           Final_ques_score.append(np.argmax(Ques_score))
           Ques_score=[]
           W2Vec_score=[]


           Ques_ans=np.asarray(Ques_ans)
           Correct_ans_2.append(np.argmax(Ques_ans))
           Ques_ans=[]


accuracy=0
for pind, pred1 in enumerate(Final_ques_score):

    if pred1==Correct_ans_2[pind]:
       accuracy+=1

print ("Accuracy is: ", accuracy/pind)


"""
Input_len=[]
for ques_s in Final_ques_score:
    Input_len.append(len(ques_s))

print(len(Final_ques_score))
print(len(Correct_ans_2))

train_data=Final_ques_score[0:1500]
train_label=Correct_ans_2[0:1500]
test_data=Final_ques_score[1500:]
test_label=Correct_ans_2[1500:]
from sklearn import svm
clf = svm.SVC()
print("training")
clf.fit(train_data,train_label)
predictions=clf.predict(test_data)
print("predicted")
accuracy=0
for pind, pred1 in enumerate(predictions):
    if pred1==test_label[pind]:
       accuracy+=1

print ("Accuracy is: ", accuracy/pind)


"""
