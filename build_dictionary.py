import math
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')



###################

Question_file = open('training_set.tsv', 'r')
Question = "" #[]
Correct_ans = []#[]
Option_A ="" # []  ####### These will contain justification text also and later on, becky features will be added.
Option_B = "" #[]
Option_C = "" #[]
Option_D = "" #[]

Final_IDF=[]
counter=0
All_query_words=[]
for line1 in Question_file:
    counter+=1

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

    Ques_cand_ans = tokenizer.tokenize((Question + " " + Option_A + " " + Option_B + " " + Option_C + " " + Option_D).lower() ) ###### Question + Candidate answer 1
    All_query_words+=Ques_cand_ans

All_query_words=list(set(All_query_words))

IDF={}
for each_word in All_query_words:
    IDF[str(each_word)]=0



def get_weights(file1, IDF):
    Doc_len=[]
    Corpus=[]
    All_words=[]
    for line in file1:    ####### each line is a doc
        line=line.lower()
        #words=line.split()
        words=tokenizer.tokenize(line)
        #print(words)
        if words[0]=="section" or words[0]=="page":
           #print("something")
           continue

        else:
           Document={}  ########## dictionary - having terms as key and TF as values of the key.
           Doc_len.append(len(words))
           unique_words=list(set(words))
           for w1 in unique_words:
               if w1 in IDF.keys():
                  IDF[str(w1)]+=1


           All_words += unique_words
           for term1 in unique_words:
               Document[str(term1)]=words.count(term1)

           Corpus.append(Document)
    All_words=list(set(All_words))
    return Doc_len, Corpus, All_words, IDF

file1=open("quizlet_corpus.qz.txt","r")
file2=open("studystack_corpus.st.txt","r")

doc_len1, corp1, AW1, IDF1 = get_weights(file1, IDF) ##### Here IDF  is a dict with all value=0 corresponding to all keys
print (len(AW1))
doc_len2, corp2, AW2, IDF2 = get_weights(file2, IDF1)  ### IDF1 is IDF values based on corpus 1
print ("len of AW2 is : ",len(AW2))
Doc_lengths=doc_len1+doc_len2
Avg_Doc_len=sum(Doc_lengths)/float(len(Doc_lengths))
############# TF is calculated in the above part and has been stored in the variable below
All_Documents = corp1+corp2  ############## List of dictionaries having TF for each word
Total_doc=len(All_Documents)  ## Total number of documents in the corpus
print("Total number of documents are : ",Total_doc)

Avg_len_file=open("Avg_len.txt","w")
Avg_len_file.write(str(Avg_Doc_len))
TF_file=open("TF_Doc_len.txt","w")
TF_file.write(str(Doc_lengths)+"\n")
for terms_TF in All_Documents:
    TF_file.write(str(terms_TF)+"\n")
#############

for each_word in All_query_words:
    doc_count=IDF2[str(each_word)]
    IDF2[str(each_word)]=math.log10((Total_doc-doc_count+0.5)/float(doc_count+0.5))

IDF_file=open("IDF.txt","w")
IDF_file.write(str(IDF2))
#################### Calculating IDF
All_words=AW1+AW2
Vocab=list(set(All_words))    ######### Vocab for calculating IDF of each term.
print("Length of Vocab is: " , len(Vocab))
Vocab_file=open("Vocab.txt","w")
Vocab_file.write(str(Vocab))

