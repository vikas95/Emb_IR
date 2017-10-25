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
    Feature_col=cols[6].split(";;")
    # print (len(Feature_col))
    Justification_threshold=1
    if len(Feature_col)>=Justification_threshold:
        for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
              ##["AggregatedJustification"]["text"]
            dict1 = ast.literal_eval(Feature_col[ind1])
            Just_score+=dict1["AggregatedJustification"]["score"]
        Ques_score.append(Just_score)
        Just_score=0

    if ind2==3524:
       print (Ques_score)
    ## Adding answer tag
    #print (ind2)
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
    if ind3<2000:
       pass
    else:
        if Pred1==Correct_ans_2[ind3]:
           accuracy+=1


print("P@1 is : ", accuracy/float(500))
print(len(Predicted_ans))
