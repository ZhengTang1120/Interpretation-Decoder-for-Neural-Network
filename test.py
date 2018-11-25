import argparse
import random
import pickle
import json

from model_dy import *
from bio_utils import *

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    args = parser.parse_args()
    input_lang, pl1, char, raw_train = prepare_data(args.datadir)
    input_lang, pl1, char, rule_lang, raw_train = parse_json_data(input_lang, pl1, char, raw_train)
    input2_lang, pl2, char2, raw_test = prepare_data(args.dev_datadir)
    # raw_test = prepare_test_data(args.dev_datadir)
    model = LSTMLM.load(args.model)
    test = []
    i = j = 0
    i = j = 0
    for datapoint in raw_test:
        if datapoint[4]:
            try:
                i += 1
                test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                    datapoint[1],
                    datapoint[2],
                    datapoint[3], 
                    input_lang.label2id[datapoint[4]], 
                    [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-2]]+[0],
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

    predict = 0.0
    label_correct = 0.0
    trigger_correct = 0.0
    both_correct = 0.0
    for datapoint in test:
        sentence = datapoint[0]
        eid = datapoint[1]
        entity = datapoint[2]
        pos = datapoint[-3]
        chars = datapoint[-2]
        attention, pred_label, score, rule = (model.get_pred(sentence, pos,chars, entity))
        pred_trigger = attention.index(max(attention)) if attention.index(max(attention)) != len(attention)-1 else -1
        if pred_label != 0:
            predict += 1.0
            if pred_trigger == datapoint[3]:
                trigger_correct += 1.0
            if pred_label == datapoint[4]:
                label_correct += 1.0
            if pred_trigger == datapoint[3] and pred_label == datapoint[4]:
                both_correct += 1.0
            # with open("attention%d"%(i/10), "a") as f:
            #     f.write(' '.join([input_lang.index2word[sentence[i1]]+" %.4f"%attention[i1] for i1 in range(0, len(sentence))]))
            #     t = input_lang.index2word[sentence[datapoint[3]]] if datapoint[3]!=-1 else "None"
            #     f.write("\t"+' '.join([input_lang.index2word[sentence[e]] for e in entity]))
            #     f.write("\ttrigger: %s pred_trigger: %s\n"%(t, input_lang.index2word[sentence[pred_trigger]]))
        with open("rules", "a") as f:
            f.write(' '.join([rule_lang.index2word[p] for p in rule])+"\n")
    with open("result", "w") as f:
        f.write("predict: %d, trigger correct: %d, label correct: %d, both correct: %d\n"
            %(predict, trigger_correct, label_correct, both_correct))
        precision = label_correct/predict if predict !=0 else 0
        recall = label_correct/197.0
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall) != 0 else 0
        f.write("precision: %.4f, recall: %.4f, f1: %.4f"
            %(precision, recall, f1))
