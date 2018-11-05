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
    input2_lang, pl2, char2, raw_test = prepare_data(args.dev_datadir)
    model = LSTMLM(input_lang.n_words, char.n_words, 50, 50, 200, 200, len(input_lang.labels), 2)
    trainning_set = []
    test = []
    i = j = 0
    for datapoint in raw_train:
        if datapoint[2]:
            try:
                i += 1
                trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]],
                    [input_lang.word2index[w] for w in word_tokenize(datapoint[1])],
                    datapoint[0].index(word_tokenize(datapoint[2])[0]), 
                    input_lang.label2id[datapoint[3]], [pl1.word2index[p] for p in datapoint[-1]],
                    [[char.word2index[c] for c in w] for w in datapoint[0]]))
            except:
                pass
        else:
            j += 1
            trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]],
                [input_lang.word2index[w] for w in word_tokenize(datapoint[1])],
                0, 0, [pl1.word2index[p] for p in datapoint[-1]],
                [[char.word2index[c] for c in w] for w in datapoint[0]]))
    print(i,j)
    i = j = 0
    for datapoint in raw_test:
        if datapoint[2]:
            try:
                i += 1
                test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]],
                    [input_lang.word2index[w] if w in input_lang.word2index else 2 for w in word_tokenize(datapoint[1])],
                    datapoint[0].index(word_tokenize(datapoint[2])[0]), 
                    input_lang.label2id[datapoint[3]], 
                    [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-1]],
                    [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]]))
            except:
                pass
        else:
            j += 1
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]],
                [input_lang.word2index[w] if w in input_lang.word2index else 2 for w in word_tokenize(datapoint[1])],
                0, 0, 
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[-1]],
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]]))
    print(i,j)
    for i in range(100):
        random.shuffle(trainning_set)
        model.train(trainning_set)
        if (i % 10) == 0 :
            i = 0
            j = 0
            k = 0
            l = 0
            n = 0
            m = 0
            x = 0
            random.shuffle(test)
            for datapoint in test:
                sentence = datapoint[0]
                entity = datapoint[1]
                pos = datapoint[-2]
                chars = datapoint[-1]
                pred_trigger, pred_label, score = (model.get_pred(sentence, pos,chars, entity))
                # print (pred_trigger, datapoint[2], pred_label, datapoint[3], score)
                if pred_trigger == datapoint[2]:
                    i += 1
                if pred_label == datapoint[3]:
                    j += 1
                if pred_trigger == datapoint[2] and pred_label == datapoint[3]:
                    k += 1
                if pred_label != 0:
                    x += 1
                    if pred_trigger == datapoint[2]:
                        l += 1
                    if pred_label == datapoint[3]:
                        m += 1
                    if pred_trigger == datapoint[2] and pred_label == datapoint[3]:
                        n += 1
            print (len(test), i, j, k, x, l ,m, n)

