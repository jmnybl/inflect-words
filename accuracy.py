import sys
from itertools import zip_longest



tags="NOUN,PROPN,Case=Nom,Case=Gen,Case=Par,Case=Tra,Case=Ess,Case=Ine,Case=Ela,Case=Ill,Case=Ade,Case=Abl,Case=All,Number=Sing,Number=Plur".split(",")
def accuracy(args):

    correct=0
    total=0
    labels={}
    if args.original_input:
        original=open(args.original_input,"rt",encoding="utf-8")
    else:
        original=[]
    for (g,s),inp in zip_longest(zip(open(args.gold,"rt",encoding="utf-8"),open(args.system,"rt",encoding="utf-8")),original):
        if g.strip()==s.strip():
            correct+=1    
        total+=1
        
        # tag based evaluation
        if not inp:
            continue
        for tag in tags:
            if tag not in inp:
                continue
            if tag not in labels:
                labels[tag]=0
                labels[tag+"_total"]=0
            labels[tag+"_total"]+=1
            if g.strip()==s.strip():
                labels[tag]+=1

    print("\n"+"*"*50+"\n")
    print("Correct:",correct)
    print("Total:",total)
    print("Accuracy:",correct/total*100)

    print("\n"+"*"*50+"\n")

    for tag in tags:
        if tag not in labels:
            continue
        print("Tag:",tag)
        print("Correct:",labels[tag])
        print("Total:",labels[tag+"_total"])
        print("Accuracy:",labels[tag]/labels[tag+"_total"]*100)
        print("-"*50)
    print("\n"+"*"*50+"\n")


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-g', '--gold', type=str, help='Gold standard inflections')
    g.add_argument('-s', '--system', type=str, help='System output inflections')
    g.add_argument('--original_input', type=str, help='Original input file, in case we want to count accuracy for each tag')
    
    args = parser.parse_args()

    accuracy(args)
