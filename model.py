import numpy as np
import dynet_config
dynet_config.set(
    mem=2048,
    random_seed=1,
    # autobatch=True
)
import dynet as dy

import pickle

class LSTMLM:

    def __init__(self, vocab_size, char_size, char_embedding_dim, char_hidden_size,
        word_embedding_dim, hidden_dim, label_size, lstm_num_layers, pattern_hidden_dim, 
        pattern_embeddings_dim, rule_size, max_rule_length):
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
        self.rule_size = rule_size
        self.max_rule_length = max_rule_length
        self.pattern_hidden_dim = pattern_hidden_dim
        self.pattern_embeddings_dim = pattern_embeddings_dim

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
            self.word_embedding_dim,# + char_hidden_size,
            self.hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )

        self.attention_weight = self.model.add_parameters((1, self.hidden_dim))

        self.lb = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim))
        self.lb_bias = self.model.add_parameters((self.hidden_dim))

        self.lb2 = self.model.add_parameters((1, self.hidden_dim))
        self.lb2_bias = self.model.add_parameters((1))

        self.pattern_embeddings = self.model.add_lookup_parameters((self.rule_size, self.pattern_embeddings_dim))
        self.decoder_lstm = dy.LSTMBuilder(
            self.lstm_num_layers,
            self.hidden_dim + self.pattern_embeddings_dim,
            self.pattern_hidden_dim,
            self.model,
        )
        self.pt = self.model.add_parameters((self.rule_size, self.pattern_hidden_dim + self.hidden_dim))
        self.pt_bias = self.model.add_parameters((self.rule_size))

    def save(self, name):
        params = (
            self.vocab_size, self.char_size, self.char_embedding_dim, self.char_hidden_size, 
            self.word_embedding_dim, self.hidden_dim, self.label_size, self.lstm_num_layers,
            self.pattern_hidden_dim, self.pattern_embeddings_dim, self.rule_size, self.max_rule_length
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
        embeds_sent = [self.word_embeddings[sentence[i]] #dy.concatenate([self.word_embeddings[sentence[i]], self.char_encode(chars[i])]) #dy.concatenate([self.word_embeddings[sentence[i]], self.pos_embeddings[pos[i]]])
         for i in range(len(sentence))]
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def attend(self, H_e):
        H_e =dy.concatenate_cols(H_e)
        S = self.attention_weight * H_e
        S = dy.transpose(S)
        A = dy.softmax(S)
        context_vector = H_e * A
        return A, context_vector

    def train(self, trainning_set):
        for sentence, eid, entity, trigger, label, pos, chars in trainning_set:
            features = self.encode_sentence(sentence, pos, chars)
            loss = []            

            entity_embeds = dy.average([features[word] for word in entity])

            attention, context = self.attend(features)
            # loss.append(-dy.log(dy.pick(attention, trigger)))
            h_t = dy.concatenate([context, entity_embeds])
            hidden = dy.tanh(self.lb * h_t + self.lb_bias)
            out_vector = dy.reshape(dy.logistic(self.lb2 * hidden + self.lb2_bias), (1,))
            # probs = dy.softmax(out_vector)
            label = dy.scalarInput(label)
            loss.append(dy.binary_log_loss(out_vector, label))
            
            # Get decoding losses
            last_output_embeddings = self.pattern_embeddings[0]
            s = self.decoder_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden_dim), last_output_embeddings]))

            rule.append(1)
            for pattern in rule:
                h_t = s.output()
                context = self.attend(features, h_t)
                out_vector = self.pt.expr() * dy.concatenate([context, h_t]) + self.pt_bias.expr()
                probs = dy.softmax(out_vector)
                loss.append(-dy.log(dy.pick(probs, pattern)))
                last_output_embeddings = self.pattern_embeddings[pattern]
                s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            loss = dy.esum(loss)

            loss = dy.esum(loss)
            loss.backward()
            self.trainer.update()
            dy.renew_cg()

    def decode(self, features):
        last_output_embeddings = self.pattern_embeddings[0]
        s = self.decoder_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden_dim), last_output_embeddings]))
        out = []
        for i in range(self.max_rule_length):
            h_t = s.output()
            context = self.attend(features, h_t)
            out_vector = self.pt.expr() * dy.concatenate([context, h_t]) + self.pt_bias.expr()
            probs = dy.softmax(out_vector).vec_value()
            last_output = probs.index(max(probs))
            last_output_embeddings = self.pattern_embeddings[last_output]
            s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            if last_output != 1:
                out.append(last_output)
            else:
                break
        return out

    def get_pred(self, sentence, pos, chars, entity):
        features = self.encode_sentence(sentence, pos, chars)
        entity_embeds = dy.average([features[word] for word in entity])
        attention, context = self.attend(features)
        attention = attention.vec_value()
        h_t = dy.concatenate([context, entity_embeds])
        hidden = dy.tanh(self.lb * h_t + self.lb_bias)
        out_vector = dy.reshape(dy.logistic(self.lb2 * hidden + self.lb2_bias), (1,))
        res = 1 if out_vector.npvalue() > 0.05 else 0
        # probs = dy.softmax(out_vector).vec_value()
        return attention, res, out_vector.npvalue(), self.decode(features)