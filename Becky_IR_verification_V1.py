import ast
import numpy as np
import nltk

file2=open("structured_kerasInput_train_bestIR_08j5.tsv","r")
IR_score=[]
Ques_score=[]
Correct_ans_2=[]    ### for each candidate level
Ques_ans=[]
Just_score=0
Predicted_ans=[]
for ind2,line2 in enumerate(file2):


    line2=line2.strip()
    cols=line2.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
    Feature_col=cols[4].split()


    Just_score+=float(Feature_col[0])
    Ques_score.append(Just_score)
    Just_score=0


    ## Adding answer tag
    if cols[1]==cols[2]:
       Ques_ans.append(1)
    else:
       Ques_ans.append(0)

    if ind2%4==3:
       Ques_score=np.asarray(Ques_score)
       Predicted_ans.append(np.argmax(Ques_score))
       Ques_score=[]

       Ques_ans=np.asarray(Ques_ans)
       Correct_ans_2.append(np.argmax(Ques_ans))
       Ques_ans=[]



print("len of IR score is: ",len(Predicted_ans))
print("len of Correct ans is : ", len(Correct_ans_2))
accuracy=0
counter1=0
for ind3, Pred1 in enumerate(Predicted_ans):
    if ind3<0:
       pass
    else:
       counter1+=1
       if Pred1==Correct_ans_2[ind3]:
          accuracy+=1


print("P@1 is : ", accuracy/counter1)
print(counter1)
