import ast


def Vocab_from_IDF(IDF_file):
    for line in IDF_file:
        IDF_dict = ast.literal_eval(line)
    Vocab = []
    for key, value in IDF_dict.items():
        Vocab.append(key)

    return Vocab



Test_IDF_vocab = Vocab_from_IDF(open("IDF_test.txt","r"))
Train_IDF_vocab = Vocab_from_IDF(open("IDF.txt","r"))

Test_IDF_doc_vocab = Vocab_from_IDF(open("IDF_test_doc.txt","r"))
Train_IDF_doc_vocab = Vocab_from_IDF(open("IDF_doc.txt","r"))

Final_Vocab = list(set(Test_IDF_vocab+Test_IDF_doc_vocab+Train_IDF_vocab+Train_IDF_doc_vocab))

Final_vocab_file = open("PMI_individual_term.txt","w")

for word in Final_Vocab:
    Final_vocab_file.write(word+"\n")