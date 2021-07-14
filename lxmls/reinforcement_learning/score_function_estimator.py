import torch

import numpy as np
import random
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.t_policy = torch.autograd.Variable(torch.FloatTensor(
            [[1/3 for x in range(3)] for y in range(3)]),
            requires_grad=True
        )

    def forward(self):
        policy = torch.nn.functional.log_softmax(self.t_policy)
        return policy


def gt(rewardlist, gamma=0.1):
    '''
    function to compute the total discounted return from time-step t
    >>> gt([10, 2, 3], gamma=0.1)
    10.23
    '''
    summe = 0
    for i, value in enumerate(rewardlist):
        summe += (gamma ** i) * value
    return summe


def train():

    valuelist = []
    rewards = np.array([10., 2., 3.])/10
    model = Model()
    optim = torch.optim.SGD([model.t_policy], lr=0.01)
    alpha = 1.
    gamma = 0.99

    for i in range(10001):
        poli = torch.nn.functional.softmax(model.t_policy).data.numpy()
        state_action_list = []
        start_state = random.randint(0, 2)
        next_state = start_state
        rewardlist = []

        for k in range(40):
            rewardlist.append(rewards[next_state]) # action, ie the reward from the state after
            action = np.random.choice(np.arange(0, 3), p=poli[next_state])
            state_action_list.append((next_state, action))
            next_state = action

        rew = gt(rewardlist[:], gamma)
        grad_list = []
        eye = np.eye(3)
        for j, (state, action) in enumerate(state_action_list):

            # ----------
            # Solution to Exercise 6.3

            # grad_j softmax(a)_i = ( delta_{i,j} - softmax(a)_j ) * softmax(a)_i
            rew = ( rew - rewardlist[j] ) / gamma

            # Why does this not depend on action?
            grad = np.zeros((3,3))
            grad[state] = rew * (eye[action] - poli[state])

            # model.t_policy[action] += alpha * grad
            grad_list.append(grad)

            # raise NotImplementedError("Exercise 6.3")

        model.t_policy.grad = torch.from_numpy(np.sum(np.stack(grad_list), axis=0)).to(model.t_policy.dtype)
        optim.step()

        # code needed at this identation level!

        # End of solution to Exercise 6.3
        # ----------

        value = (gt(rewardlist, 1))
        valuelist.append(value)

        if i % 100 == 0:
            print(poli)
            print(rewardlist)
            plt.plot(valuelist)
            plt.show()
