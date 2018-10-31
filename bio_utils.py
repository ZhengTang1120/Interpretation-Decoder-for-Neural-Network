from nltk import sent_tokenize, word_tokenize

import os
import glob
import argparse

simple_events = ["Gene_expression", "Transcription", "Protein_catabolism", "Localization", "Binding", "Protein_modification",
 "Phosphorylation", "Ubiquitination", "Acetylation", "Deacetylation"]

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 2  # Count SOS and EOS
        self.labels = ["NoRel"]
        self.label2id = {"NoRel":0}

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_token_spans(text):
    """
    returns (words, start_offsets, end_offsets)
    for each sentence in the provided text
    """
    offset = 0
    for s in sent_tokenize(text):
        offset = text.find(s, offset)
        yield sentence_tokens(s, offset)

def sentence_tokens(sentence, offset):
    """this is meant to be used by get_token_spans() only"""
    pos = 0
    starts = []
    ends = []
    words = word_tokenize(sentence)
    for w in words:
        pos = sentence.find(w, pos)
        starts.append(pos + offset)
        pos += len(w)
        ends.append(pos + offset)
    return words, starts, ends

def get_trigger(s, e, phosphorylations):
    for tlbl, trigger, entity in phosphorylations:
        if trigger[2] >= s and trigger[1] <= e:
            yield tlbl, trigger, entity

def get_entity(s, e, d):
    for e in d:
        if d[e][1] > s and d[e][0] < e and d[e][0] == "Protein":
            return d[e]

def refine_words(words, e):
    if e not in words:
        new_words = []
        for word in words:
            if e in word:
                if word[:word.find(e)]:
                    new_words.append(word[:word.find(e)])
                if word[word.find(e):word.find(e)+len(e)]:
                    new_words.append(word[word.find(e):word.find(e)+len(e)])
                if word[word.find(e)+len(e):]:
                    new_words.append(word[word.find(e)+len(e):])
            else:
                new_words.append(word)
        return new_words
    return words

def prepare_data(dirname):
    maxl = 0
    input_lang = Lang("input")
    pos_lang = Lang("position")
    train = []
    for fname in glob.glob(os.path.join(dirname, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        txt = root + '.txt'
        a1 = root + '.a1'
        a2 = root + '.a2'
        d = dict()
        with open(a1) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    d[id] = (label, int(start), int(end), text)
        phosphorylations = []
        with open(a2) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    d[id] = (label, int(start), int(end), text)
                if line.startswith('E') and "Phosphorylation" in line:
                    [id, data] = line.split('\t')
                    temp = data.split(' ')
                    [tlbl, trigger] = temp[0].split(':')
                    [elbl, entity] = temp[1].split(':')
                    phosphorylations.append((tlbl, d[trigger], d[entity]))
                    if tlbl not in input_lang.labels:
                        input_lang.label2id[tlbl] = len(input_lang.labels)
                        input_lang.labels.append(tlbl)
        with open(txt) as f:
            text = f.read()
            for words, starts, ends in get_token_spans(text):
                if len(words) > maxl:
                    maxl = len(words)
                s = starts[0]
                e = ends[-1]
                x = list(get_trigger(s, e, phosphorylations))
                y = get_entity(s, e, d)
                if x:
                    for i in x:
                        tlbl, trigger, entity = i
                        for w in word_tokenize(entity[-1] + " " + trigger[-1]):
                            words = refine_words(words, w)
                    for i in x:
                        tlbl, trigger, entity = i
                        try:
                            ent = word_tokenize(entity[-1])[0]
                            e_pos = words.index(ent)
                        except:
                            e_pos = 0
                        st_pos = e_pos-20 if e_pos-20 > 0 else 0
                        ed_pos = e_pos+21 if e_pos+21 < len(words) else len(words)
                        words = words[st_pos:ed_pos]
                        pos = [i-e_pos for i in range(0, len(words))]
                        pos_lang.addSentence(pos)
                        train.append((words, entity[-1], trigger[-1], tlbl, pos))
                elif y:
                    for w in word_tokenize(y[-1]):
                        words = refine_words(words, w)
                    try:
                            ent = word_tokenize(y[-1])[0]
                            e_pos = words.index(ent)
                    except:
                        e_pos = 0
                    st_pos = e_pos-20 if e_pos-20 > 0 else 0
                    ed_pos = e_pos+21 if e_pos+21 < len(words) else len(words)
                    words = words[st_pos:ed_pos]
                    pos = [i-e_pos for i in range(0, len(words))]
                    pos_lang.addSentence(pos)
                    train.append((words, y[-1], None, None, pos))
                input_lang.addSentence(words)
    return input_lang, pos_lang, train

# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('datadir')
#     args = parser.parse_args()

#     input_lang, train = prepare_data(args.datadir)
#     for t in train:
#         print (t)