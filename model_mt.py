import numpy as np
import dynet_config
dynet_config.set(
    mem=6144,
    random_seed=1978,
    autobatch=True
)
import dynet as dy

import pickle

class LSTMLM:

    def __init__(self, vocab_size, word_embedding_dim, hidden_dim, pattern_hidden_dim, pattern_embeddings_dim,
     rule_size, lstm_num_layers, max_rule_length):
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        self.rule_size = rule_size
        self.lstm_num_layers = lstm_num_layers
        self.max_rule_length = max_rule_length
        self.pattern_hidden_dim = pattern_hidden_dim
        self.pattern_embeddings_dim = pattern_embeddings_dim

        self.word_embeddings = self.model.add_lookup_parameters((self.vocab_size, self.word_embedding_dim))
        self.encoder_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.word_embedding_dim,
            self.hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )
        self.attention_weight = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))
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
            self.vocab_size, self.word_embedding_dim, self.hidden_dim, 
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

    def encode_sentence(self, sentence):
        embeds_sent = [self.word_embeddings[word] for word in sentence]
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def attend(self, H_e, h_t):
        context_vector = dy.vecInput(self.hidden_dim)
        for h_e in H_e:
            s = dy.transpose(h_t) * self.attention_weight * h_e
            a = dy.softmax(s)
            context_vector += h_e * a
        return context_vector / len(H_e)

    def train(self, trainning_set):
        for sentence, rule in trainning_set:
            features = self.encode_sentence(sentence)
            loss = []
            # Get decoding losses
            last_output_embeddings = self.pattern_embeddings[0]
            s = self.decoder_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden_dim), last_output_embeddings]))

            rule.append(1)
            for pattern in rule:
                h_t = s.output()
                context = self.attend(features, h_t)
                out_vector = self.pt * dy.concatenate([context, h_t]) + self.pt_bias
                probs = dy.softmax(out_vector)
                loss.append(-dy.log(dy.pick(probs, pattern)))
                last_output_embeddings = self.pattern_embeddings[pattern]
                s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            loss = dy.esum(loss)
            loss.backward()
            self.trainer.update()
            dy.renew_cg()

    def get_pred(self, features):
        probs = dy.softmax(self.lb * features[-1] + self.lb_bias)
        return probs.index(max(probs))

    def decode(self, features):
        last_output_embeddings = self.pattern_embeddings[0]
        s = self.decoder_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden_dim), last_output_embeddings]))
        out = []
        for i in range(self.max_rule_length):
            h_t = s.output()
            context = self.attend(features, h_t)
            out_vector = self.pt * dy.concatenate([context, h_t]) + self.pt_bias
            probs = dy.softmax(out_vector).vec_value()
            last_output = probs.index(max(probs))
            last_output_embeddings = self.pattern_embeddings[last_output]
            s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            if last_output != 1:
                out.append(last_output)
            else:
                break
        return out

    def translate(self, sentence):
        features = self.encode_sentence(sentence)
        return self.decode(features)

     