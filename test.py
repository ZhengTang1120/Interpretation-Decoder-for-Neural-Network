import argparse
import random
import pickle
import json
import os
import glob

from model_dy import *
from bio_utils import *

from nltk.translate.bleu_score import corpus_bleu

from collections import defaultdict

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    args = parser.parse_args()
    input_lang, pl1, char, raw_train = prepare_data(args.datadir)
    rule_lang = Lang("rule")
    input2_lang, pl2, char2, raw_test = prepare_data(args.dev_datadir)
    # input_lang, pl1, char, rule_lang, raw_train = parse_json_data(input_lang, pl1, char, raw_train)
    model = LSTMLM.load(args.model)
    i = j = 0
    # for fname in glob.glob(os.path.join(args.dev_datadir, '*.a1')):
    #     root = os.path.splitext(fname)[0]
    #     output = "preds/"+root.split("/")[-1]+ '.a2'
    #     test = []
    #     raw_test, tcount = prepare_data(fname, False, "valids.json")
    #     for datapoint in raw_test:
    #         if datapoint[4]:
    #             i += 1
    #             test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
    #                 datapoint[1],
    #                 datapoint[2],
    #                 datapoint[3], 
    #                 input_lang.label2id[datapoint[4]], 
    #                 [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
    #                 [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
    #                 datapoint[-3]))
    #         else:
    #             j += 1
    #             test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
    #                 datapoint[1],
    #                 datapoint[2],
    #                 datapoint[3], 0, 
    #                 [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
    #                 [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
    #                 datapoint[-3]))
    #     predict = 0.0
    #     label_correct = 0.0
    #     trigger_correct = 0.0
    #     both_correct = 0.0
    #     references = []
    #     candidates = []
    #     phosphos = dict()
    #     events = defaultdict(list)
    test = []
    for datapoint in raw_test:
        if datapoint[4]:
            try:
                i += 1
                test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                    datapoint[1],
                    datapoint[2],
                    datapoint[3], 
                    input_lang.label2id[datapoint[4]], 
                    [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],
                    [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],[]))
            except:
                print (datapoint)
        else:
            # try:
            j += 1
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], 0, 
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],[]))
            # except:
            #     print (datapoint)

    print(i,j)
    s = 0
    r = 0
    tp = 0
    for datapoint in test:
        sentence = datapoint[0]
        eid = datapoint[1]
        entity = datapoint[2]
        triggers = datapoint[3]
        pos = datapoint[-3]
        chars = datapoint[-2]
        pred_triggers, score = (model.get_pred(sentence, pos,chars, entity))
        if len(pred_triggers) != 0 or triggers[0] != -1:
            if len(pred_triggers) != 0:
                s += len(pred_triggers)
            if triggers[0] != -1:
                r += len(triggers)
            for t in pred_triggers:
                if t in triggers:
                    tp += 1
    precision = tp/s if s!= 0 else 0
    print ("Recall: %f Precision: %f"%(tp/r, precision))
        # for i, datapoint in enumerate(test):
        #     sentence = datapoint[0]
        #     eid = datapoint[1]
        #     entity = datapoint[2]
        #     pos = datapoint[-3]
        #     chars = datapoint[-2]
        #     attention, pred_label, score, rule = (model.get_pred(sentence, pos,chars, entity))
        #     pred_trigger = attention.index(max(attention)) if attention.index(max(attention)) != len(attention)-1 else -1
        #     if pred_label != 0:
        #         predict += 1.0
        #         if pred_trigger == datapoint[3]:
        #             trigger_correct += 1.0
        #         if pred_label == datapoint[4]:
        #             label_correct += 1.0
        #         if pred_trigger == datapoint[3] and pred_label == datapoint[4]:
        #             both_correct += 1.0
        #         starts = raw_test[i][-2]
        #         ends = raw_test[i][-1]
        #         text = raw_test[i][0]
        #         if str(starts[pred_trigger])+" "+str(ends[pred_trigger])+"\t"+text[pred_trigger] not in phosphos:
        #             tcount += 1
        #             phosphos[str(starts[pred_trigger])+" "+str(ends[pred_trigger])+"\t"+text[pred_trigger]] = "T"+str(tcount)
        #         if eid not in events[phosphos[str(starts[pred_trigger])+" "+str(ends[pred_trigger])+"\t"+text[pred_trigger]]]:
        #             events[phosphos[str(starts[pred_trigger])+" "+str(ends[pred_trigger])+"\t"+text[pred_trigger]]].append(eid)
        # with  open(output, "w") as f:
        #     for k in phosphos:
        #         f.write(phosphos[k]+"\tPhosphorylation "+k+"\n")
        #     ecount = 1
        #     for tid in events:
        #         for e in events[tid]:
        #             f.write("E"+str(ecount)+"\tPhosphorylation:"+tid+" Theme:"+e+"\n")
        #             ecount += 1
