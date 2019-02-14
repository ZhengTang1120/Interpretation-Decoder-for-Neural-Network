# from language import *
import argparse
import random
import pickle
import json

from model_dy import *
from bio_utils import *

from nltk.translate.bleu_score import corpus_bleu

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    parser.add_argument('outdir')
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()
    input_lang, pl1, char, rule_lang, raw_train = prepare_data(args.datadir,input_lang, pl1, char, rule_lang, raw_train)
    input_lang, pl1, char, rule_lang, raw_train = prepare_data("pubmed2",input_lang, pl1, char, rule_lang, raw_train, "valids2.json")
    input2_lang, pl2, char2, rule_lang2, raw_test = prepare_data(args.dev_datadir, valids="valids.json")
    embeds = load_embeddings("embeddings_november_2016.txt", input_lang)
    model = LSTMLM(input_lang.n_words, char.n_words, 50, 50, 100, 200, pl1.n_words, 5, len(input_lang.labels), 
        100, 200, rule_lang.n_words, 200, 2, embeds)
    trainning_set = list()
    test = list()
    i = j = 0
    for datapoint in raw_train:
        if datapoint[3][0] != -1:
            i += len(datapoint[3])
            trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
                datapoint[1],#entity
                datapoint[2],#entity position
                datapoint[3],#trigger position
                [input_lang.label2id[l] for l in datapoint[4]],#trigger label
                [pl1.word2index[p] for p in datapoint[5]]+[0],#positions
                [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
                [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]]))
        else:
            j += 1
            trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], [0], 
                [pl1.word2index[p] for p in datapoint[5]]+[0],
                [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
                [rule_lang.word2index["EOS"]]))
    print(i,j)
    i = j = 0
    for datapoint in raw_test:
        if datapoint[3][0] != -1:
            i += len(datapoint[3])
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],#entity
                datapoint[2],#entity position
                datapoint[3],#trigger position
                [input_lang.label2id[l] for l in datapoint[4]],#trigger label
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],#positions
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]]))
        else:
            j += 1
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], [0], 
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [rule_lang.word2index["EOS"]]))
    print(i,j)
    for i in range(100):
        random.shuffle(trainning_set)
        model.train(trainning_set)
        if (i % 10) == 0 :
            predict = 0.0
            label_correct = 0.0
            trigger_correct = 0.0
            both_correct = 0.0
            tp = 0
            s = 0
            r = 197
            references = []
            candidates = []
            for datapoint in test:
                triggers = datapoint[3]  
                rules = datapoint[-1]             
                pred_triggers, score, contexts, hidden, pred_rules = model.get_pred(datapoint)
                if len(pred_triggers) != 0 or triggers[0] != -1:
                    if len(pred_triggers) != 0:
                        s += len(pred_triggers)
                    # if triggers[0] != -1:
                    #     r += len(triggers)
                    for k, t in enumerate(pred_triggers):
                        if t in triggers:
                            tp += 1
                            j = triggers.index(t)
                            references.append([rules[j]])
                            candidates.append(pred_rules[k])
            precision = tp/s if s!= 0 else 0
            recall = tp/r
            f1 = 2*(precision*recall)/(recall+precision) if recall+precision != 0 else 0
            bleu = corpus_bleu(references, candidates)
            print (tp, r, s)
            print ("Recall: %.4f Precision: %.4f F1: %.4f BLEU: %.4f"%(recall, precision, f1, bleu))
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
            model.save("%s/%d"%(args.outdir,i/10))

