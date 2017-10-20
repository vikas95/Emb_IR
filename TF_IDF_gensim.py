import gensim
from gensim import corpora
import math
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
class BM25 :
    def __init__(self, fn_docs) :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self) :
        raw_data = []

        raw_data.append(tokenizer.tokenize(self.fn_docs)) ##### regex tokenizer
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in (self.fn_docs) :
            doc = tokenizer.tokenize(line) ##### regex tokenizer
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75) :
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items










###############################

def get_weights(file1):

    Corpus=[]

    for line in file1:    ####### each line is a doc
        line=line.lower()
        #words=line.split()
        words=tokenizer.tokenize(line)
        #print(words)
        if words[0]=="section" or words[0]=="page":
           #print("something")
           continue

        else:
           Corpus.append(line)
    return Corpus

file1=open("quizlet_corpus.qz.txt","r")
file2=open("studystack_corpus.st.txt","r")
Corpus=get_weights(file2)
###############################






if __name__ == '__main__' :
    #mycorpus.txt is as following:
    '''
    Human machine interface for lab abc computer applications
    A survey of user opinion of computer system response time
    The EPS user interface management system
    System and human system engineering testing of EPS
    Relation of user perceived response time to error measurement
    The generation of random binary unordered trees
    The intersection graph of paths in trees
    Graph IV Widths of trees and well quasi ordering
    Graph minors A survey
    '''
    for ind,line in enumerate(Corpus):

        #fn_docs = 'mycorpus.txt'
        bm25 = BM25(line)
        Query = 'The intersection graph of paths in trees survey Graph'
        Query = Query.split()
        scores = bm25.BM25Score(Query)
        tfidf = bm25.TFIDF()
        #print (bm25.Items())
        print(ind,scores)