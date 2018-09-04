import numpy as np
import dynet_config
dynet_config.set(
    mem=6144,
    random_seed=1978
)
import dynet as dy

class LSTMLM:

    def __init__(self, vocab_size, word_embedding_dim, hidden_dim, pattern_hidden_dim, pattern_embeddings_dim,
     rule_size, label_size, lstm_num_layers, max_rule_length):
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.model = dy.Model()
        self.rule_size = rule_size
        self.label_size = label_size
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
        self.lb = self.model.add_parameters((self.label_size, self.hidden_dim + 2 * self.word_embedding_dim))
        self.lb_bias = self.model.add_parameters((self.label_size))
        self.attention_weight = self.model.add_parameters((self.hidden_dim, self.pattern_hidden_dim))
        self.pattern_embeddings = self.model.add_lookup_parameters((self.rule_size, self.pattern_embeddings_dim))
        self.decoder_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.hidden_dim + 2 * self.word_embedding_dim + self.pattern_embeddings_dim,
            self.pattern_hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )
        self.pt = self.model.add_parameters((self.rule_size, self.pattern_hidden_dim + self.hidden_dim))
        self.pt_bias = self.model.add_parameters((self.rule_size))

    def encode_sentence(self, sentence, entity1, entity2):
        embeds_sent = [self.word_embeddings[word] for word in sentence]
        embed_e1 = self.word_embeddings[entity1]
        embed_e2 = self.word_embeddings[entity2]
        features = [dy.concatenate((f, embed_e1, embed_e2)) for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def attend(H_e, h_t):
        context_vector = None
        for h_e in H_e:
            s = h_e * self.attention_weight * h_t
            a = dy.softmax(s)
            context_vector += a * h_e
        return context_vector

    def train(self, trainning_set):
        trainer = dy.SimpleSGDTrainer(self.model)
        for sentence, entity1, entity2, label, rule in trainning_set:
            features = self.encode_sentence(sentence, entity1, entity2)
            loss = []
            # Get classifying losses
            pred = probs = dy.softmax(self.lb.expr() * features + self.lb_bias.expr())
            loss.append(-dy.log(dy.pick(probs, label)))
            # Get decoding losses
            last_output_embeddings = self.pattern_embeddings[0]
            s = self.decoder_lstm.initial_state().add_input(dy.concatenate((features[-1], last_output_embeddings)))

            for pattern in rule:
                h_t = s.output()
                context = attend(features, h_t)
                out_vector = self.pt * dy.concatenate((context, h_t)) + self.pt_bias
                probs = dy.softmax(out_vector)
                loss.append(-dy.log(dy.pick(probs, pattern)))
                last_output_embeddings = self.pattern_embeddings[pattern]
                s.add_input(dy.concatenate((context, last_output_embeddings)))
            loss = dy.esum(loss)
            loss.backward()
            trainer.update()

    def get_pred(self, features):
        probs = dy.softmax(self.lb.expr() * features + self.lb_bias.expr())
        return probs.index(max(probs))

    def decode(self, features):
        s = self.decoder_lstm.initial_state().add_input(features)
        out = []
        last_output = 0
        last_output_embeddings = pattern_embeddings[0]
        s = s.add_input(last_output_embeddings)
        for i in range(self.max_rule_length):
            h_t = s.output()
            context = attend(features, h_t)
            out_vector = self.pt * dy.concatenate((context, h_t)) + self.pt_bias
            probs = dy.softmax(out_vector)
            last_output = probs.index(max(probs))
            last_output_embeddings = pattern_embeddings[last_output]
            s.add_input(dy.concatenate((context, last_output_embeddings)))
            if last_output != 0:
                out.append(last_output)
            else:
                break
        return out
     