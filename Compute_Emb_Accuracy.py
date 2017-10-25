import ast
import numpy as np
from statistics import mean
file1=open("Becky_files_W2V_score_5.txt","r")
scores=[]

for line in file1:
    scores=ast.literal_eval(line)

print(len(scores))


ind_score=[]
All_score=[]
Predicted_ans=[]
for ind1, s1 in enumerate(scores):

    ind_score.append(sum(s1))
    All_score.append(sum(s1))
    if ind1%4==3:
       ind_score=np.asarray(ind_score)

       Predicted_ans.append(np.argmax(ind_score))
       ind_score=[]

print((All_score))
print (Predicted_ans)

Question_file = open('training_set.tsv', 'r')
Correct_ans = []#[]
counter=0
for line1 in Question_file:
    counter += 1
    if counter>2500:
       break
    if counter<1:
       pass
    else:
        line1 = line1.strip()
        cols = line1.split("\t")
        Correct_ans.append(cols[3])


print(Correct_ans)

Accuracy=0
if len(Correct_ans)==len(Predicted_ans):
   for Pind, Pred1 in enumerate(Predicted_ans):

       if Pred1==int(Correct_ans[Pind]):
          Accuracy+=1

print("Accuracy for all ques is: ",str(Accuracy/float(len(Predicted_ans))))

