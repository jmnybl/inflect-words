# copied from OpenNMT/translate.py

from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch
import sys

sys.path.insert(0,os.getcwd()+"/OpenNMT-py")
import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()
if opt.batch_size != 1:
    print("WARNING: -batch_size isn't supported currently, "
          "we set it to 1 for now!")
    opt.batch_size = 1


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)
    
def read_word_statistics(fname, min_freq=25):
    words={}
    for line in open(fname, "rt", encoding="utf-8"):
        line=line.strip()
        count,word=line.split()
        if int(count)<min_freq:
            return words
        words[word]=int(count)
    return words
    
def score_beam(predictions, scores, word_statistics):
    rescored=[]
    for p,sc in zip(predictions, scores):
        count=word_statistics.get(p,0)
        rescored.append((p,sc,count))
#    if [p for p,s in rescored+tmp][0] != predictions[0]:
#        print(src_words,"---",predictions[0],"---",[p for p,s in rescored+tmp][0])
#    print(predictions)
#    print([p for p,s in rescored+tmp])
#    print()
    return [p for p,s,c in sorted(rescored,key=lambda x:x[2], reverse=True)], [s for p,s,c in sorted(rescored,key=lambda x:x[2], reverse=True)], [c for p,s,c in sorted(rescored,key=lambda x:x[2], reverse=True)]


def main():

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.Translator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    withAttention=len(opt.save_attention) > 0
    if withAttention:
        if opt.n_best>1:
            print("-save_attention works only with -n_best 1 right now. Attention weights are not saved!")
            withAttention=False
        else:
            print("saving attention weights to",opt.save_attention)
            attn_file=open(opt.save_attention,"wt")
    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator.fields, None)

    word_statistics=read_word_statistics("/home/jmnybl/finnish_vocab")

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    counter = count(1)
    for batch in test_data:
        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = translator.translate(batch, data)
        pred_score_total += sum(score[0] for score in pred_scores)
        pred_words_total += sum(len(x[0]) for x in pred_batch)
        if opt.tgt:
            gold_score_total += sum(gold_scores)
            gold_words_total += sum(len(x) for x in batch.tgt[1:])

        # z_batch: an iterator over the predictions, their scores,
        # the gold sentence, its score, and the source sentence for each
        # sentence in the batch. It has to be zip_longest instead of
        # plain-old zip because the gold_batch has length 0 if the target
        # is not included.
        z_batch = zip_longest(
                pred_batch, gold_batch,
                pred_scores, gold_scores, attn,
                (sent.squeeze(1) for sent in src.split(1, dim=1)))

        for pred_sents, gold_sent, pred_score, gold_score, attention, src_sent in z_batch:
            n_best_preds = ["".join(pred)+"\t"+str(score) for pred,score in zip(pred_sents[:opt.n_best],pred_score[:opt.n_best])]
            
            #n_best_preds, scores, counts = score_beam(n_best_preds, pred_score[:opt.n_best], word_statistics)
            
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()
            
            sent_number = next(counter)
            
            if withAttention:
                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                words = get_src_words(
                    src_sent, translator.fields["src"].vocab.itos)
                    
                print(" ||| ".join([str(sent_number),best_pred,str(best_score),words,str(len(words.split(" ")))+" "+str(len(best_pred.split(" ")))]),file=attn_file)
                for y in attention:
                    for i,v in enumerate(y.cpu().numpy()):
                        print(" ".join(str(x) for x in v),file=attn_file)

            if opt.verbose:
                
                words = get_src_words(
                    src_sent, translator.fields["src"].vocab.itos)

                os.write(1, bytes('\nSENT %d: %s\n' %
                                  (sent_number, words), 'UTF-8'))

                best_pred = n_best_preds[0]
                best_score = pred_score[0]
                os.write(1, bytes('PRED %d: %s\n' %
                                  (sent_number, best_pred), 'UTF-8'))
                print("PRED SCORE: %.4f" % best_score)

                if opt.tgt:
                    tgt_sent = ' '.join(gold_sent)
                    os.write(1, bytes('GOLD %d: %s\n' %
                             (sent_number, tgt_sent), 'UTF-8'))
                    print("GOLD SCORE: %.4f" % gold_score)

                if len(n_best_preds) > 1:
                    print('\nBEST HYP:')
                    for score, sent in zip(pred_score, n_best_preds):
                        os.write(1, bytes("[%.4f] %s\n" % (score, sent),
                                 'UTF-8'))

    report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        report_score('GOLD', gold_score_total, gold_words_total)

    if opt.dump_beam:
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))
                  
    if withAttention:              
        attn_file.close()


if __name__ == "__main__":
    main()
