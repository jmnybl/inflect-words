# create all inflections for a name list given as json file

import sys
import json
import subprocess

tags="Case=Nom,Case=Gen,Case=Par,Case=Tra,Case=Ess,Case=Ine,Case=Ela,Case=Ill,Case=Ade,Case=Abl,Case=All".split(",")
def inflect_names(args):

    data=json.loads(open(args.file).read())
    tmp_file=open("names.to_be_inflected","wt",encoding="utf-8")
    words=[]
    for name in data:
        n=" ".join(c for c in name)
        pos="PROPN"
        number="Number=Sing"
        for tag in tags:
            print(n,pos,tag,number,file=tmp_file)
            words.append((name,tag))
    tmp_file.close()
          
    # inflect
    beam_size=10
    subprocess.call(["python", "predict.py", "-model", args.model1, "-src", "names.to_be_inflected", "-output", "names.inflected.beam1", "-gpu", "0", "-beam_size", str(beam_size), "-n_best", str(beam_size), "-verbose"])
    if args.model2!="":
        subprocess.call(["python", "predict.py", "-model", args.model2, "-src", "names.to_be_inflected", "-output", "names.inflected.beam2", "-gpu", "0", "-beam_size", str(beam_size), "-n_best", str(beam_size), "-verbose"])
    
    inflections=[]
    files=["names.inflected.beam1"]
    if args.model2!="":
        files.append("names.inflected.beam2")
    
    for i,inf_file in enumerate(files):
        counter=0
        for option1 in open(inf_file,"rt",encoding="utf-8"):
            if counter==0 and i==0:
                inflections.append({})
            inf,score=option1.strip().split("\t")
            if inf not in inflections[-1]:
                inflections[-1][inf]=[float(score)]
            else:
                inflections[-1][inf].append(float(score))
            counter+=1
            if counter==10:
                counter=0
    f=open(args.outfile,"w")
    for inflected,(name,tag) in zip(inflections,words):
        inf=sorted(inflected.items(),key=lambda x:sum(x[1])/2 if len(x[1])>1 else sum(x[1])*2,reverse=True)[0][0]
        print(name,tag,inf,sep="\t",file=f)
    f.close()





if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-f', '--file', type=str, help='json file name')
    g.add_argument('-m', '--model1', type=str, help='model name')
    g.add_argument('--model2', type=str, default="", help='model name')
    g.add_argument('-o', '--outfile', type=str, help='output file name')
    
    args = parser.parse_args()

    inflect_names(args)
