import numpy as np
import dynet_config
dynet_config.set(
    mem=2048,
    random_seed=1,
    # autobatch=True
)
import dynet as dy
import math

import pickle

class TNLM:

    def __init__(self, vocab_size, char_size, char_embedding_dim, char_hidden_size,
        word_embedding_dim, hidden_dim, pos_size, pos_embeddings_size, label_size,
        pattern_hidden_dim, pattern_embeddings_dim, rule_size, max_rule_length, 
        lstm_num_layers, pretrained):
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_dim = hidden_dim
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        self.label_size = label_size
        self.char_hidden_size = char_hidden_size
        self.pos_size = pos_size
        self.pos_embeddings_size = pos_embeddings_size
        self.pretrained = pretrained
        self.pattern_hidden_dim = pattern_hidden_dim
        self.pattern_embeddings_dim = pattern_embeddings_dim
        self.rule_size = rule_size
        self.max_rule_length = max_rule_length
        self.lstm_num_layers = lstm_num_layers
        if np.any(self.pretrained):
            self.word_embeddings = self.model.lookup_parameters_from_numpy(self.pretrained)
        else:
            self.word_embeddings = self.model.add_lookup_parameters((self.vocab_size, self.word_embedding_dim))
        self.pos_embeddings = self.model.add_lookup_parameters((self.pos_size, self.pos_embeddings_size))
        self.char_embeddings = self.model.add_lookup_parameters((self.char_size, self.char_embedding_dim))
        
        self.character_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.char_embedding_dim,
            self.char_hidden_size,
            self.model,
            dy.VanillaLSTMBuilder,
        )

        self.weight_eq = self.model.add_parameters((2 * self.hidden_dim, 
            self.word_embedding_dim + self.char_hidden_size + self.pos_embeddings_size))
        self.weight_ek = self.model.add_parameters((2 * self.hidden_dim, 
            self.word_embedding_dim + self.char_hidden_size + self.pos_embeddings_size))
        self.weight_ev = self.model.add_parameters((2 * self.hidden_dim, 
            self.word_embedding_dim + self.char_hidden_size + self.pos_embeddings_size))
        self.eff = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim))
        self.eff_bias = self.model.add_parameters((self.hidden_dim))

        self.pattern_embeddings = self.model.add_lookup_parameters((self.rule_size, self.pattern_embeddings_dim))

        self.weight_dq = self.model.add_parameters((2 * self.hidden_dim, self.pattern_embeddings_dim))
        self.weight_dk = self.model.add_parameters((2 * self.hidden_dim, self.pattern_embeddings_dim))
        self.weight_dv = self.model.add_parameters((2 * self.hidden_dim, self.pattern_embeddings_dim))

        self.dff = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim))
        self.dff_bias = self.model.add_parameters((self.hidden_dim))

        self.weight_q = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))
        self.weight_k = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))
        self.weight_v = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))
        
        self.pt = self.model.add_parameters((self.rule_size, self.pattern_hidden_dim))
        self.pt_bias = self.model.add_parameters((self.rule_size))

        self.lb = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim))
        self.lb_bias = self.model.add_parameters((self.hidden_dim))
        self.lb2 = self.model.add_parameters((1, self.hidden_dim))
        self.lb2_bias = self.model.add_parameters((1))

        self.self_attention_weight = self.model.add_parameters((1, self.hidden_dim))

    def save(self, name):
        params = (
            self.vocab_size, self.char_size, self.char_embedding_dim, self.char_hidden_size, 
            self.word_embedding_dim, self.hidden_dim, self.pos_size, self.pos_embeddings_size,
            self.label_size, self.pattern_hidden_dim, self.pattern_embeddings_dim, 
            self.rule_size, self.max_rule_length, self.lstm_num_layers, self.pretrained
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = LSTMLM(*params)
            parser.model.populate(f'{name}.model')
            return parser

    def char_encode(self, word):
        c_seq = [self.char_embeddings[c] for c in word]
        return self.character_lstm.transduce(c_seq)[-1]

    def encode_sentence(self, sentence, pos, chars):
        embeds_sent = [dy.concatenate([self.word_embeddings[sentence[i]], self.char_encode(chars[i]), self.pos_embeddings[pos[i]]]) 
         for i in range(len(sentence))]
        embeds = dy.concatenate_cols(embeds_sent)
        scores = []
        for embed in embeds_sent:
            q = self.weight_eq.expr() * embed
            k = self.weight_ek.expr() * embed
            scores.append(q*dy.transpose(k)/math.sqrt(self.hidden_dim * 2))
        scores = dy.softmax(dy.concatenate_cols(scores))
        V = self.weight_ev.expr() * embeds
        features = dy.cmult(scores, V)
        features = self.eff.expr() * features + self.eff_bias.expr()
        return features

    def self_attend(self, H_e):
        H_e = dy.concatenate_cols(H_e)
        S = self.self_attention_weight.expr() * H_e
        S = dy.transpose(S)
        A = dy.softmax(S)
        context_vector = H_e * A
        return A, context_vector

    def decoder(features, pres):
        encode = dy.concatenate_cols(features)
        decoded = [self.pattern_embeddings[p] for p in pres]
        decoded2 = dy.concatenate_cols(pres)
        scores = []
        for embed in decoded:
            q = self.weight_eq.expr() * embed
            k = self.weight_ek.expr() * embed
            scores.append(q*dy.transpose(k)/math.sqrt(self.hidden_dim * 2))
        scores = dy.softmax(dy.concatenate_cols(scores))
        V = self.weight_dv.expr() * decoded2
        Q2 = self.dff.expr() * (scores * V) + self.dff_bias.expr()
        Q2 = self.weight_q.expr() * Q2
        K2 = self.weight_k.expr() * encode
        V2 = self.weight_v.expr() * encode
        output = dy.softmax(Q*dy.transpose(K)/math.sqrt(self.hidden_dim * 2)) * V
        output = self.pt.expr() * output + self.pt_bias.expr()
        return dy.softmax(output)

    def train(self, trainning_set):
        for sentence, eid, entity, trigger, label, pos, chars, rule in trainning_set:
            features = self.encode_sentence(sentence, pos, chars)
            loss = []            

            entity_embeds = features[entity]

            attention, context = self.self_attend(features)
            ty = dy.vecInput(len(sentence))
            ty.set([0 if i!=trigger else 1 for i in range(len(sentence))])
            loss.append(dy.binary_log_loss(dy.reshape(attention,(len(sentence),)), ty))
            h_t = dy.concatenate([context, entity_embeds])
            hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
            out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
            label = dy.scalarInput(label)
            loss.append(dy.binary_log_loss(out_vector, label))

            pres = [0]
            for pattern in rule:
                probs = self.decoder(features, pres)
                loss.append(-dy.log(dy.pick(probs, pattern)))
                pres.append(pattern)

            loss = dy.esum(loss)
            loss.backward()
            self.trainer.update()
            dy.renew_cg()

    def get_pred(self, sentence, pos, chars, entity):
        features = self.encode_sentence(sentence, pos, chars)
        entity_embeds = features[entity]
        attention, context = self.self_attend(features)
        attention = attention.vec_value()
        # pred_trigger = attention.index(max(attention))
        h_t = dy.concatenate([context, entity_embeds])
        hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
        out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
        res = 1 if out_vector.npvalue() > 0.0005 else 0
        rule = [0]
        while rule[-1] != 0:
            probs = self.decoder(features, rule)
            rule.append(probs.index(max(probs)))

        return attention, res, out_vector.npvalue(), rule[1:]