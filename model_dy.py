import numpy as np
import dynet_config
dynet_config.set(
    mem=6144,
    random_seed=1978,
    # autobatch=True
)
import dynet as dy

import pickle

class LSTMLM:

    def __init__(self, vocab_size, char_size, char_embedding_dim, char_hidden_size,
        word_embedding_dim, hidden_dim, label_size, lstm_num_layers):
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_dim = hidden_dim
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        self.label_size = label_size
        self.lstm_num_layers = lstm_num_layers
        self.char_hidden_size = char_hidden_size

        self.word_embeddings = self.model.add_lookup_parameters((self.vocab_size, self.word_embedding_dim))
        self.char_embeddings = self.model.add_lookup_parameters((self.char_size, self.char_embedding_dim))
        
        self.character_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.char_embedding_dim,
            self.char_hidden_size,
            self.model,
            dy.VanillaLSTMBuilder,
        )
        self.encoder_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.word_embedding_dim + char_hidden_size,
            self.hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )

        self.attention_weight = self.model.add_parameters((self.hidden_dim + self.word_embedding_dim, self.hidden_dim))

        self.lb = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim + self.word_embedding_dim))
        self.lb_bias = self.model.add_parameters((self.hidden_dim))

        self.lb2 = self.model.add_parameters((1, self.hidden_dim))
        self.lb2_bias = self.model.add_parameters((1))

    def save(self, name):
        params = (
            self.vocab_size, self.char_size, self.char_embedding_dim, self.char_hidden_size, 
            self.word_embedding_dim, self.hidden_dim, 
            self.pattern_hidden_dim, self.pattern_embeddings_dim,
            self.rule_size, self.lstm_num_layers, self.max_rule_length
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
        embeds_sent = [dy.concatenate([self.word_embeddings[sentence[i]], self.char_encode(chars[i])])#self.word_embeddings[sentence[i]] #dy.concatenate([self.word_embeddings[sentence[i]], self.pos_embeddings[pos[i]]])
         for i in range(len(sentence))]
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def attend(self, H_e, h_t):
        H_e =dy.concatenate_cols(H_e)
        S = dy.transpose(h_t) * self.attention_weight * H_e
        S = dy.transpose(S)
        A = dy.softmax(S)
        context_vector = H_e * A
        return A, context_vector/H_e.npvalue().shape[-1]

    def train(self, trainning_set):
        for sentence, entity, trigger, label, pos, chars in trainning_set:
            features = self.encode_sentence(sentence, pos, chars)
            loss = []            

            entity_embeds = dy.average([self.word_embeddings[word] for word in entity])

            h_t = dy.concatenate([features[-1], entity_embeds])
            attention, context = self.attend(features, h_t)
            loss.append(-dy.log(dy.pick(attention, trigger)))
            hidden = dy.tanh(self.lb * dy.concatenate([context, h_t]) + self.lb_bias)
            out_vector = dy.reshape(dy.logistic(self.lb2 * hidden + self.lb2_bias), (1,))
            # probs = dy.softmax(out_vector)
            label = dy.scalarInput(label)
            loss.append(dy.binary_log_loss(out_vector, label))
            
            loss = dy.esum(loss)
            loss.backward()
            self.trainer.update()
            dy.renew_cg()

    def get_pred(self, sentence, pos, chars, entity):
        features = self.encode_sentence(sentence, pos, chars)
        entity_embeds = dy.average([self.word_embeddings[word] for word in entity])
        h_t = dy.concatenate([features[-1], entity_embeds])
        attention, context = self.attend(features, h_t)
        attention = attention.vec_value()
        hidden = dy.tanh(self.lb * dy.concatenate([context, h_t]) + self.lb_bias)
        out_vector = dy.reshape(dy.logistic(self.lb2 * hidden + self.lb2_bias), (1,))
        res = 1 if out_vector.npvalue() > 0.05 else 0
        # probs = dy.softmax(out_vector).vec_value()
        return attention.index(max(attention)), res, out_vector.npvalue()