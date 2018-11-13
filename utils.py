import os
import glob
import yaml
import json

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1        

def get_trainning_data(folder, rules):
    res = []
    for fname in glob.glob(os.path.join(folder, '*.detail')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        with open(root+'.detail') as f:
            for line in f:
                if 'List(' in line:
                    sentence = line.split('=>')[1].strip()
                    label = line.split('(')[1].split(',')[0].strip()
                if 'controller (' in line:
                    entity1 = line.split('=>')[1].strip()
                if 'controlled (' in line:
                    entity2 = line.split('=>')[1].strip()
                if 'Rule =>' in line:
                    rule = line.split('=>')[1].strip()
        node = (sentence.lower(), entity1, entity2, label, rules[rule])
        res.append(node)
    return res

def get_rules(folder):
    rules = dict()
    for fname in glob.glob(os.path.join(folder, '*.yml')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        with open(root+'.yml') as stream:
            docs = yaml.load_all(stream)
            for doc in docs:
                for rule in doc:
                    rules[rule['name']] = rule['pattern']
    return rules

def make_vocabularies(trainning_set):
    words = set()
    patterns = set()
    rels = set()
    for sentence, entity1, entity2, label, rule in trainning_set:
        for w in sentence.split():
            words.add(w)
        for p in rule.split()[1:-1]:
            patterns.add(p)
        rels.add(label)
    special = ['*unk*']
    special2 = ['*start*', '*end*']
    words = special + list(words)
    patterns = special2 + list(patterns)
    rels = list(rels)
    return (words, patterns, rels)


