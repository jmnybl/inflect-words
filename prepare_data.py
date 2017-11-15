import sys

ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conllu(f):
    sent=[]
    comment=[]
    for line in f:
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comment,sent
            comment=[]
            sent=[]
        elif line.startswith("#"):
            comment.append(line)
        else: #normal line
            sent.append(line.split("\t"))
    else:
        if sent:
            yield comment, sent

ignore_categories="Derivation,Abbr".split(",")
def create_data(args):

    f_inp=open(args.output+".input","wt",encoding="utf-8")
    f_out=open(args.output+".output","wt",encoding="utf-8")

    counter=0
    for comm, sent in read_conllu(open(args.file,"rt",encoding="utf-8")):
        
        for token in sent:
            if token[UPOS]!="NOUN" and token[UPOS]!="PROPN": # TODO
                continue
            lemma=" ".join(c for c in token[LEMMA].replace(" ","_")) # TODO
            wordform=" ".join(c for c in token[FORM].replace(" ","_")) # TODO
            tags=[]
            if args.extra_tag!="":
                tags.append(args.extra_tag)
            tags.append(token[UPOS])
            
            accepted=True
            for t in token[FEAT].split("|"):
                if t.split("=",1)[0] in args.categories.split(","):
                    tags.append(t)
                elif t.split("=",1)[0] in ignore_categories:
                    continue
                else: # remove for example words with possessive suffixes and clitics
                    accepted=False
                    break
            if accepted:
                tags=" ".join(tags)
                print(lemma,tags,file=f_inp)
                print(wordform,file=f_out)
                counter+=1
    print("Done, files have",counter,"examples.",file=sys.stderr)



if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-f', '--file', type=str, help='Input file name')
    g.add_argument('-o', '--output', type=str, help='Output file name, will create file extentions .input and .output')
    g.add_argument('--categories', type=str, default="Case,Number", help='Which tag categories are used? Comma separated list, default=Case,Number')
    g.add_argument('--extra_tag', type=str, default="", help='extra tag, for example mark autoencoding training examples')
    
    args = parser.parse_args()

    create_data(args)
