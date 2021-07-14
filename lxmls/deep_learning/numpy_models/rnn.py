import numpy as np
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self.log_forward(input)[0])
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m] -= learning_rate * gradients[m]

    def log_forward(self, input):

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        hidden_size = W_h.shape[0]
        nr_steps = input.shape[0]

        # Embedding layer
        z_e = W_e[input, :]

        # Recurrent layer
        h = np.zeros((nr_steps + 1, hidden_size))
        for t in range(nr_steps):

            # Linear
            z_t = W_x.dot(z_e[t, :]) + W_h.dot(h[t, :])

            # Non-linear
            h[t+1, :] = 1.0 / (1 + np.exp(-z_t))

        # Output layer
        y = h[1:, :].dot(W_y.T)

        # Softmax
        log_p_y = y - logsumexp(y, axis=1, keepdims=True)

        return log_p_y, y, h, z_e, input

    def backpropagation(self, input, output):

        '''
        Compute gradientes, with the back-propagation method
        inputs:
            x: vector with the (embedding) indicies of the words of a
                sentence
            outputs: vector with the indicies of the tags for each word of
                        the sentence outputs:
            gradient_parameters: vector with parameters gradientes
        '''

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        nr_steps = input.shape[0]

        K, J = W_y.shape
        I, E = W_e.shape

        log_p_y, y, h, z_e, x = self.log_forward(input)
        p_y = np.exp(log_p_y)

        # Initialize gradients with zero entrances
        gradient_W_e = np.zeros(W_e.shape)
        # gradient_W_x = np.zeros(W_x.shape)
        # gradient_W_h = np.zeros(W_h.shape)
        # gradient_W_y = np.zeros(W_y.shape)

        # ----------
        # Solution to Exercise 1

        # print(output.shape)
        one_hot_y = np.zeros((nr_steps, K))
        one_hot_y[np.arange(nr_steps), output] = 1
        e_y = np.einsum("kj, mk -> mj", W_y, one_hot_y - p_y)
        e_r = np.zeros(J)
        e_h = np.zeros((nr_steps, J))
        e_e = np.zeros((nr_steps, E))
        for m in range(nr_steps -1, -1, -1):
            e_h[m] = np.einsum("j,j,j-> j",(e_r + e_y[m]), h[m+1], (1-h[m+1]))
            e_r = np.einsum("ij,i->j", W_h, e_h[m])
            e_e[m] = np.einsum("ij,i->j", W_x, e_h[m])

        # gradient_W_e =
        gradient_W_h = - np.einsum("ni, nj->ij", e_h, h[:-1]) / nr_steps
        gradient_W_x = - np.einsum("nj, ne->je", e_h, z_e) / nr_steps

        gradient_W_e[x] = - e_e / nr_steps
        gradient_W_y = - np.einsum("nk, nj -> kj", one_hot_y - p_y, h[1:]) / nr_steps



        # raise NotImplementedError("Implement Exercise 1")

        # End of Solution to Exercise 1
        # ----------

        # Normalize over sentence length
        gradient_parameters = [
            gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
        ]

        return gradient_parameters

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        nr_steps = input.shape[0]
        log_probability = self.log_forward(input)[0]
        return -log_probability[range(nr_steps), output].mean()
