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
        self.lstm_num_layers = lstm_num_layers
        self.char_hidden_size = char_hidden_size
        self.pos_size = pos_size
        self.pos_embeddings_size = pos_embeddings_size
        self.pretrained = pretrained
        self.pattern_hidden_dim = pattern_hidden_dim
        self.pattern_embeddings_dim = pattern_embeddings_dim
        self.rule_size = rule_size
        self.max_rule_length = max_rule_length
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
        self.encoder_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.word_embedding_dim + self.char_hidden_size + self.pos_embeddings_size,
            self.hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )

        self.self_attention_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))

        self.query_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))
        self.key_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))
        self.value_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))

        self.attention_weight = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))

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
            self.word_embedding_dim, self.hidden_dim, self.pos_size, self.pos_embeddings_size,
            self.label_size, self.pattern_hidden_dim, self.pattern_embeddings_dim, 
            self.rule_size, self.max_rule_length,
            self.lstm_num_layers, self.pretrained
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
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def self_attend(self, H_e):
        H = dy.concatenate_cols(H_e)
        keys = self.key_weight.expr() * H
        queries = self.query_weight.expr() * H
        values = self.value_weight.expr() * H
        context_vectors = []
        for q in dy.transpose(queries):
            S = dy.transpose(dy.transpose(q) * keys)
            A = dy.softmax(S)
            context_vectors.append(values * A)
            # S = dy.transpose(h_e) * self.self_attention_weight.expr() * H
            # S = dy.transpose(S)
            # A = dy.softmax(S)
            # context_vectors.append(H * A)
        return context_vectors

    def entity_attend(self, H_e, h_e):
        H = dy.concatenate_cols(H_e)
        keys = self.key_weight.expr() * H
        query = self.query_weight.expr() * h_e
        values = self.value_weight.expr() * H
        context_vectors = []
        S = dy.transpose(query) * keys
        A = dy.softmax(S)
        context_vectors = dy.cmult(values, A)
        return dy.transpose(context_vectors)

    def attend(self, H_e, h_t):
        H_e =dy.concatenate_cols(H_e)
        S = dy.transpose(h_t) * self.attention_weight.expr() * H_e
        S = dy.transpose(S)
        A = dy.softmax(S)
        context_vector = H_e * A
        return context_vector

    def train(self, sids, trainning_set):
        for sid in sids:
            sentence = trainning_set[sid][0]
            chars = trainning_set[sid][1]
            for eid in trainning_set[sid][2]:
                eid, entity, pos = trainning_set[sid][2][eid][0]
                triggers = list()
                for trigger, label, rule in trainning_set[sid][2][eid][1]:
                    triggers.append(trigger)
                features = self.encode_sentence(sentence, pos, chars)
                loss = []            

                # context1 = self.self_attend(features)
                entity_vec = features[entity]
                context = self.entity_attend(features, entity_vec)

                for i, c in enumerate(context):
                    if i != entity:
                        h_t = dy.concatenate([c, entity_vec])
                        hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
                        out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
                        out = dy.scalarInput(1) if i in triggers else dy.scalarInput(0)
                        loss.append(dy.binary_log_loss(out_vector, out))
                # ty = dy.vecInput(len(sentence))
                # ty.set([0 if i!=trigger else 1 for i in range(len(sentence))])
                # loss.append(dy.binary_log_loss(dy.reshape(attention,(len(sentence),)), ty))
                # h_t = dy.concatenate([context, entity_embeds])
                # hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
                # out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
                # label = dy.scalarInput(label)
                # loss.append(dy.binary_log_loss(out_vector, label))

                # # Get decoding losses
                # last_output_embeddings = self.pattern_embeddings[0]
                # s = self.decoder_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden_dim), last_output_embeddings]))
                # for pattern in rule:
                #     h_t = s.output()
                #     context = self.attend(features, h_t)
                #     out_vector = self.pt.expr() * dy.concatenate([context, h_t]) + self.pt_bias.expr()
                #     probs = dy.softmax(out_vector)
                #     loss.append(-dy.log(dy.pick(probs, pattern)))
                #     last_output_embeddings = self.pattern_embeddings[pattern]
                #     s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            
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
            if last_output != 0:
                out.append(last_output)
            else:
                return out
        return out

    def get_pred(self, sentence, pos, chars, entity):
        features = self.encode_sentence(sentence, pos, chars)
        entity_embeds = features[entity]
        # context = self.self_attend(features)
        # entity_vec = context[entity]
        entity_vec = features[entity]
        context = self.entity_attend(features, entity_vec)
        res = []
        for i, c in enumerate(context):
            if i != entity:
                h_t = dy.concatenate([c, entity_vec])
                hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
                out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
                res.append(i) if out_vector.npvalue() > 0.0005 else 0
        # # pred_trigger = attention.index(max(attention))
        # h_t = dy.concatenate([context, entity_embeds])
        # hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
        # out_vector = dy.reshape(dy.logistic(self.lb2.expr() * hidden + self.lb2_bias.expr()), (1,))
        # res = 1 if out_vector.npvalue() > 0.0005 else 0
        # rule = self.decode(features)
        # # probs = dy.softmax(out_vector).vec_value()
        return res, out_vector.npvalue()#, rule