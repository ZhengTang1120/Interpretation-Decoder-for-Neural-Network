# from language import *
import argparse
import random
import pickle
import json

from model_dy import *
from bio_utils import *

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('dev_datadir')
    args = parser.parse_args()

    input_lang, pl1, char, raw_train = prepare_data(args.datadir)
    rule_lang = Lang("rule")
    # input_lang, pl1, char, rule_lang, raw_train = parse_json_data(input_lang, pl1, char, raw_train)
    input2_lang, pl2, char2, raw_test = prepare_data(args.dev_datadir)
    embeds = load_embeddings("embeddings_november_2016.txt", input_lang)
    model = LSTMLM(input_lang.n_words, char.n_words, 50, 50, 100, 200, pl1.n_words, 5, len(input_lang.labels), 
        100, 200, rule_lang.n_words, 200, 2, embeds)
    trainning_set = dict()
    test = dict()
    i = j = 0
    for key in raw_train:
        (words, starts, ends) = raw_train[key][0]
        trainning_set[key] = ([input_lang.word2index[w] for w in words]+[1], 
            [[char.word2index[c] for c in w] for w in words+["EOS"]], dict())
        for entity in raw_train[key][1]:
            trainning_set[key][2][entity] = [None ,list()]
            for (entity, e_pos, trigger_pos, tlbl, pos, rule) in raw_train[key][1][entity]:
                if (trigger_pos) != -1:
                    i += 1
                else:
                    j += 1
                if trainning_set[key][2][entity][0] == None:
                    trainning_set[key][2][entity][0] = (entity, e_pos, [pl1.word2index[p] for p in pos]+[0])
                trainning_set[key][2][entity][1].append((trigger_pos, input_lang.label2id[tlbl], 
                    [rule_lang.word2index[p] for p in rule +["EOS"]]))
        # if datapoint[4]:# and datapoint[6]:
        #     i += 1
        #     trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
        #         datapoint[1],
        #         datapoint[2],
        #         datapoint[3], 
        #         input_lang.label2id[datapoint[4]], 
        #         [pl1.word2index[p] for p in datapoint[5]]+[0],
        #         [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
        #         [rule_lang.word2index[p] for p in datapoint[6]+["EOS"]]))
        # elif datapoint[4]:
        #     continue
        # else:
        #     j += 1
        #     trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
        #         datapoint[1],
        #         datapoint[2],
        #         datapoint[3], 0, 
        #         [pl1.word2index[p] for p in datapoint[5]]+[0],
        #         [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
        #         [rule_lang.word2index["EOS"]]))
    print(i,j)
    i = j = 0
    for key in raw_test:
        (words, starts, ends) = raw_test[key][0]
        test[key] = ([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in words]+[1], 
            [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in words+["EOS"]], dict())
        for entity in raw_test[key][1]:
            test[key][2][entity] = [None ,list()]
            for (entity, e_pos, trigger_pos, tlbl, pos, rule) in raw_test[key][1][entity]:
                if (trigger_pos) != -1:
                    i += 1
                else:
                    j += 1
                if test[key][2][entity][0] == None:
                    test[key][2][entity][0] = (entity, e_pos, 
                        [pl1.word2index[p] if p in pl1.word2index else 2 for p in pos]+[0])
                test[key][2][entity][1].append((trigger_pos, input_lang.label2id[tlbl], 
                    [rule_lang.word2index[p] for p in rule +["EOS"]]))
    print(i,j)
    sids = list(trainning_set.keys())
    for i in range(100):
        random.shuffle(sids)
        model.train(sids, trainning_set)
        if (i % 10) == 0 :
            predict = 0.0
            label_correct = 0.0
            trigger_correct = 0.0
            both_correct = 0.0
            tp = 0
            s = 0
            r = 0
            for sid in test:
                sentence = test[sid][0]
                chars = test[sid][1]
                for eid in test[sid][2]:
                    eid, entity, pos = test[sid][2][eid][0]
                    triggers = []
                    for trigger, label, rule in test[sid][2][eid][1]:
                        triggers.append(trigger)
                    pred_triggers, score = model.get_pred(sentence, pos,chars, entity)
                    if len(pred_triggers) != 0 or triggers[0] != -1:
                        if len(pred_triggers) != 0:
                            s += len(pred_triggers)
                        if triggers[0] != -1:
                            r += len(triggers)
                        for t in pred_triggers:
                            if t in triggers:
                                tp += 1
            precision = tp/s if s!= 0 else 0
            print (tp, r, s)
            print ("Recall: %f Precision: %f"%(tp/r, precision))
            model.save("model_embeds%d"%(i/10))

