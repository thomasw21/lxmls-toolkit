from __future__ import division
import sys
import numpy as np
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb


class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    """ Implements Structured Perceptron"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs=10, learning_rate=1.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in range(self.num_epochs):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            self.params_per_epoch.append(self.parameters.copy())
            acc = 1.0 - num_mistakes_total / num_labels_total
            print("Epoch: %i Accuracy: %f" % (epoch, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def perceptron_update(self, sequence):

        # ----------
        # Solution to Exercise 3

        predicted_sequence, score = self.viterbi_decode(sequence)

        # update initial features
        true_initial_features = self.feature_mapper.get_initial_features(sequence, sequence.y[0])
        predicted_initial_features = self.feature_mapper.get_initial_features(sequence, predicted_sequence.y[0])
        self.parameters[true_initial_features] += self.learning_rate
        self.parameters[predicted_initial_features] -= self.learning_rate

        # update transmission features
        for pos in range(len(sequence) -1):
            true_transition_features = self.feature_mapper.get_transition_features(sequence, pos + 1, sequence.y[pos+1], sequence.y[pos])
            predicted_transition_features = self.feature_mapper.get_transition_features(sequence, pos + 1, predicted_sequence.y[pos+1], predicted_sequence.y[pos])
            self.parameters[true_transition_features] += self.learning_rate
            self.parameters[predicted_transition_features] -= self.learning_rate

        # update emission features
        for pos in range(len(sequence)):
            true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, sequence.y[pos])
            predicted_emission_features = self.feature_mapper.get_emission_features(sequence, pos, predicted_sequence.y[pos])
            self.parameters[true_emission_features] += self.learning_rate
            self.parameters[predicted_emission_features] -= self.learning_rate

        # update initial features
        true_final_features = self.feature_mapper.get_final_features(sequence, sequence.y[-1])
        predicted_final_features = self.feature_mapper.get_final_features(sequence, predicted_sequence.y[-1])
        self.parameters[true_final_features] += self.learning_rate
        self.parameters[predicted_final_features] -= self.learning_rate

        number_of_labels = len(sequence.y)
        num_mistakes = (predicted_sequence.y != sequence.y).sum()
        return number_of_labels, num_mistakes

        # End of Solution to Exercise 3
        # ----------

    def save_model(self, dir):
        fn = open(dir + "parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        fn = open(dir + "parameters.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
