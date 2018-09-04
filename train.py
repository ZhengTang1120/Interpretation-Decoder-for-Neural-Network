from language import *
import argparse
import random
from model_mt import *

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--rules', default=None)
    # parser.add_argument('--data', default=None)
    # parser.add_argument('--outdir', default='out')
    # parser.add_argument('--w_embed_dim',         type=int,   default=100)
    # parser.add_argument('--lstm_hidden_size',    type=int,   default=125)
    # parser.add_argument('--lstm_num_layers',     type=int,   default=3)
    # parser.add_argument('--p_hidden_size',  type=int,   default=100)
    # parser.add_argument('--epochs',              type=int,   default=30)
    # args = parser.parse_args()

    # rules = get_rules("rules")
    # trainning_set = get_trainning_data("causalOut_oct7_replicate", rules)
    # words, patterns, rels = make_vocabularies(trainning_set)
    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)

    model = LSTMLM(input_lang.n_words, 200, 100, 100, 200, output_lang.n_words, 2, 20)

    trainning_set = list()
    for pair in pairs:
        trainning_set.append(([input_lang.word2index[w] for w in pair[0].split()], [output_lang.word2index[w] for w in pair[1].split()]))
    for i in range(100):
        model.train(trainning_set[:100])
        if (i % 10) == 0 :
            sentence = [input_lang.word2index[w] for w in pairs[0][0].split()]
            print (sentence)
            output = (model.translate(sentence))
            print (output)
            print (' '.join([output_lang.index2word[i] for i in output]))
            print (pairs[0][1])
