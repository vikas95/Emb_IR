file1=open("studystack_corpus.st.txt","r")
file2=open("dummy_studystack.txt","w")
for line in file1:
    words=line.split()
    if words[0]=="<PAGE>":
       pass
    elif words[0]=="<SECTION>":
         words[0]="#"
         words[1]=words[1].split("__")[1]
         line=" ".join(words)
         file2.write(line+"\n")
    else:
        file2.write(line)