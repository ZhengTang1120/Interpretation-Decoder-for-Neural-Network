import argparse
import pickle
from model_mt import *
import random


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--sentence', default=None)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()


    model = LSTMLM.load(args.model)
    sentence = args.sentence

    try:
        with open('corpra.pickle', 'rb') as f:
            output_lang, input_lang, pairs = pickle.load(f)
    except FileNotFoundError:
        output_lang, input_lang, pairs = prepareData('eng', 'fra', False)
        with open('corpra.pickle', 'wb') as f:
            pickle.dump((output_lang, input_lang, pairs), f)

    if args.random:
        test =  random.choice(pairs)
        print ("source: %s"%test[1])
        print ("target: %s"%test[0])
        sentence = [input_lang.word2index[w] for w in test[1].split()]
    elif sentence:
        sentence = [input_lang.word2index[w] for w in sentence.split()]
    else:
        exit()
    output = (model.translate(sentence))
    trans = (' '.join([output_lang.index2word[i] for i in output]))
    print ("result: %s"%(trans))
