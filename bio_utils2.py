from nltk import sent_tokenize
import re
import numpy as np
np.random.seed(1)
import math
import json

import os
import glob
import argparse

simple_events = ["Gene_expression", "Transcription", "Protein_catabolism", "Localization", "Binding", "Protein_modification",
 "Phosphorylation", "Ubiquitination", "Acetylation", "Deacetylation"]

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"EOS":1,"UNK":2,"THEME":3}
        self.index2word = {1: "EOS", 2:"UNK", 3:"THEME"}
        self.n_words = 4  # Count SOS and EOS
        self.labels = ["NoRel"]
        self.label2id = {"NoRel":0}

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def load_embeddings(file, lang):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            vector = np.array([float(i) for i in line_split[1:]])
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
    for i in range(5, lang.n_words):
        base = math.sqrt(6/embedding_size)
        word = lang.index2word[i]
        try:
            vector = emb_dict[word]
        except KeyError:
            vector = np.random.uniform(-base,base,embedding_size)
        if np.any(emb_matrix):
            emb_matrix = np.vstack((emb_matrix, vector))
        else:
            emb_matrix = np.random.uniform(-base,base,(5, embedding_size))
            emb_matrix[4] = vector
    return emb_matrix

def sanitizeWord(w):
    if w.startswith("$T"):
        return w
    if w == ("xTHEMEx"):
        return "THEME"
    w = w.lower()
    if is_number(w):
        return "xnumx"
    w = re.sub("[^a-z_]+","",w)
    if w:
        return w

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def word_tokenize(text):
    return re.findall(r"[\w|#\w|@\w]+|[^\w\s,]",text)

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

def get_trigger(s, e, entities, phosphorylations):
    for tlbl, trigger, entity in phosphorylations:
        if entities[trigger][2] >= s and entities[trigger][1] <= e:
            yield tlbl, trigger, entity

def check_entity(e, x):
    for p in x:
        if e == p[-1]:
            return False
    return True

def get_entity(s, e, entities, x):
    for entity in entities:
        if entities[entity][2] > s and entities[entity][1] < e and entities[entity][0] == "Protein" and check_entity(entity,x):
            yield entity

def token_span(entity, starts):
    res = []
    offset = entity[1]
    for w in word_tokenize(entity[-1]):
        if entity[-1][offset-entity[1]] == " ":
            offset += 1
        try:
            res.append(starts.index(offset))
        except:
            for i, s in enumerate(starts):
                if i<len(starts)-1 and s<offset and starts[i+1]>offset:
                    res.append(i)
                    break
        offset += len(w)
    return res

def get_id(proteins, i, starts, entities):
    for p in proteins:
        if token_span(entities[p], starts) and i == token_span(entities[p], starts)[0]:
            return p
    return None

def replace_protein(words, entities, starts, proteins):
    res = []
    ps = []
    for p in proteins:
        ps += token_span(entities[p], starts)
    for i, w in enumerate(words):
        if i not in ps:
            res.append(w)
        p = get_id(proteins, i, starts, entities)
        if p:
            res.append("$"+p)
    return res

def prepare_data(dirname):
    maxl = 0
    input_lang = Lang("input")
    pos_lang = Lang("position")
    char_lang = Lang("char")
    train = []
    for fname in glob.glob(os.path.join(dirname, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        txt = root + '.txt'
        a1 = root + '.a1'
        a2 = root + '.a2'
        entities = dict()
        with open(a1) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    entities[id] = (label, int(start), int(end), text)
        phosphorylations = []
        with open(a2) as f:
            for line in f:
                line = line.strip()
                if line.startswith('T'):
                    [id, data, text] = line.split('\t') 
                    [label, start, end] = data.split(' ')
                    entities[id] = (label, int(start), int(end), text)
                if line.startswith('E') and "Phosphorylation" in line:
                    [id, data] = line.split('\t')
                    temp = data.split(' ')
                    [tlbl, trigger] = temp[0].split(':')
                    [elbl, entity] = temp[1].split(':')
                    if ((tlbl, trigger, entity)) not in phosphorylations:
                        phosphorylations.append((tlbl, trigger, entity))
                    if tlbl not in input_lang.labels:
                        input_lang.label2id[tlbl] = len(input_lang.labels)
                        input_lang.labels.append(tlbl)
        with open(txt) as f:
            text = f.read()
            for words, starts, ends in get_token_spans(text):
                if len(words) > maxl:
                    maxl = len(words)
                s = int(starts[0])
                e = int(ends[-1])
                x = list(get_trigger(s, e, entities, phosphorylations))
                y = list(get_entity(s, e, entities, x))
                # proteins = [t[2] for t in x]
                # triggers = [t[1] for t in x]
                # new_words = replace_protein(words, entities, starts, triggers+proteins)
                # temp = []
                # for i,w in enumerate(new_words):
                #     sw = sanitizeWord(w)
                #     if sw:
                #         temp.append(sw)
                # new_words = temp
                for res in x:
                    try:
                        tlbl, trigger, entity = res
                        new_words = replace_protein(words, entities, starts, [trigger, entity])
                        temp = []
                        for i,w in enumerate(new_words):
                            sw = sanitizeWord(w)
                            if sw:
                                temp.append(sw)
                        new_words = temp
                        trigger_pos = (new_words.index("$"+trigger))
                        new_words[trigger_pos] = sanitizeWord(entities[trigger][-1])
                        e_pos = new_words.index("$"+entity)
                        st_pos = e_pos-10 if e_pos-10 > 0 else 0
                        ed_pos = e_pos+11 if e_pos+11 < len(new_words) else len(new_words)
                        if trigger_pos < ed_pos and trigger_pos > st_pos:
                            trigger_pos = trigger_pos - st_pos
                        else:
                            trigger_pos = -1
                        pos = [i-e_pos for i in range(st_pos, e_pos)]+[0]+[i-e_pos for i in range(e_pos+1, ed_pos)]
                        pos_lang.addSentence(pos)
                        res = new_words[st_pos: ed_pos]#["OTHER" if w.startswith("$T") and w != "$"+entity else w for w in new_words[st_pos: ed_pos]]
                        res[e_pos-st_pos] = "THEME"
                        input_lang.addSentence(res)
                        train.append((res, entity, e_pos-st_pos, trigger_pos, tlbl, pos, []))
                    except:
                        continue
                        # print (words)
                        # print (new_words)
                        # print ([(p,entities[p]) for p in triggers+proteins])
                for entity in y:
                    new_words = replace_protein(words, entities, starts, [entity])
                    temp = []
                    for i,w in enumerate(new_words):
                        sw = sanitizeWord(w)
                        if sw:
                            temp.append(sw)
                    new_words = temp
                    try:
                        e_pos = new_words.index("$"+entity)
                        st_pos = e_pos-10 if e_pos-10 > 0 else 0
                        ed_pos = e_pos+11 if e_pos+11 < len(new_words) else len(new_words)
                        pos = [i-e_pos for i in range(st_pos, e_pos)]+[0]+[i-e_pos for i in range(e_pos+1, ed_pos)]
                        pos_lang.addSentence(pos)
                        res = new_words[st_pos: ed_pos]#["OTHER" if w.startswith("$T") and w != "$"+entity else w for w in new_words[st_pos: ed_pos]]
                        res[e_pos-st_pos] = "THEME"
                        input_lang.addSentence(res)
                        train.append((res,entity, e_pos-st_pos, -1, None, pos, []))
                    except:
                        continue
                        # print (words)
                        # print (new_words)
                        # print ([(p,entities[p]) for p in triggers+proteins])
    for i, w in input_lang.index2word.items():
        char_lang.addSentence(w)
    return input_lang, pos_lang, char_lang, train

def parse_json_data(input_lang, pos_lang, char_lang, train):
    with open("rules/raw.json") as f:
        rules = json.load(f)
    rule_lang = Lang("rule")
    with open("events.json") as f:
        j = json.load(f)
        for event in j:
            sentence = event["sentence"]
            rule = rules[event["rule"]]
            rule = word_tokenize(rule)
            entity = event["entity"]
            sentence = sentence[:entity[1][0]]+"xTHEMEx"+sentence[entity[1][1]:]
            trigger = event["trigger"]
            words = []
            for w in word_tokenize(sentence):
                w = sanitizeWord(w)
                if w:
                    words.append(w)
            e_pos = words.index("THEME")
            st_pos = e_pos-10 if e_pos-10 > 0 else 0
            ed_pos = e_pos+11 if e_pos+11 < len(words) else len(words)
            words = words[st_pos: ed_pos]
            pos = [i-e_pos for i in range(st_pos, e_pos)]+[0]+[i-e_pos for i in range(e_pos+1, ed_pos)]
            strigger = sanitizeWord(word_tokenize(trigger[0])[0])
            if not strigger:
                for w in word_tokenize(trigger[0]):
                    if sanitizeWord(w):
                        strigger = w
                        break
            trigger_pos = None
            try:
                trigger_pos = words.index(strigger)
            except:
                continue
            if trigger_pos:
                input_lang.addSentence(words)
                pos_lang.addSentence(pos)
                rule_lang.addSentence(rule)
                train.append((words, "T?", e_pos-st_pos, trigger_pos, "Phosphorylation", pos, rule))
    for i, w in input_lang.index2word.items():
        char_lang.addSentence(w)
    return input_lang, pos_lang, char_lang, rule_lang, train

# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('datadir')
#     args = parser.parse_args()

#     input_lang, pos_lang, char_lang, train = prepare_data(args.datadir)
#     input_lang, pos_lang, char_lang, rule_lang, train = parse_json_data(input_lang, pos_lang, char_lang, train)
#     # print (load_embeddings("embeddings_november_2016.txt", input_lang))
#     offset_dict = dict()
#     for t in train:
#         if t[3] != -1:
#             print (t[-1])
    #         print (t[-1])
    #         print (len(t[0]), len(t[-1]))
    #         print (t[3])
    #         print (t[0][t[3]], t[4])
    #     if t[3] != -1:
    #         print (t[2], t[3], t[-1][t[3]])
    #         try:
    #             offset_dict[t[-1][t[3]]] += 1
    #         except KeyError:
    #             offset_dict[t[-1][t[3]]] = 1
    # with open("histogram.tsv", "w") as f:
    #     for i in range(min(offset_dict.keys()), max(offset_dict.keys())+1):
    #         l = offset_dict[i] if i in offset_dict else 0
    #         f.write("%d\t%d\n"%(i,l)) 









