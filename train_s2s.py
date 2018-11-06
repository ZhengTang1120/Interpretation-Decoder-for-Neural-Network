from language import *
import argparse
import random
from model_mt import *
import pickle
import json
# from utils import *

if __name__ == '__main__':

    # try:
    #     with open('corpra_r.pickle', 'rb') as f:
    #         input_lang, output_lang, pairs = pickle.load(f)
    # except FileNotFoundError:
    #     input_lang, output_lang, pairs = prepareData('eng', 'rule', False)
    #     # with open('corpra_r.pickle', 'wb') as f:
    #     #     pickle.dump((input_lang, output_lang, pairs), f)

    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)

    model = LSTMLM(input_lang.n_words, 200, 200, 200, 200, output_lang.n_words, 2, 50)

    trainning_set = list()
    for pair in pairs:
        trainning_set.append(([input_lang.word2index[w] for w in pair[0].split()], [output_lang.word2index[w] for w in pair[1].split()]))
    test =  random.choice(pairs)
    print ("ENG: %s"%test[0])
    print ("FRA: %s"%test[1])
    for i in range(100):
        random.shuffle(trainning_set)
        model.train(trainning_set)
        if (i % 10) == 0 :
            sentence = [input_lang.word2index[w] for w in test[0].split()]
            output = (model.translate(sentence))
            trans = (' '.join([output_lang.index2word[i] for i in output]))
            print ("Translation%d: %s"%(i/10,trans))
            model.save("model%d"%(i/10))

