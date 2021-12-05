import random as rand
import numpy as np


class QLearner:

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0):
        self.num_states = num_states
        self.num_actions = num_actions

        self.alpha = alpha
        self.gamma = gamma
        self.Rar = rar
        self.Radr = radr
        self.dyna = dyna

        # current state
        self.s = 0

        # current action
        self.a = 0

        # Q table: (state, action)
        self.Q = np.zeros(shape=(num_states, num_actions))

        # rewards: (state, reward)
        self.R = np.zeros(shape=(num_states, num_actions))

        # transitions: {(state, action): {next state: count}}
        self.T = {}

    def querySetState(self, s):
        # if taking a random action:
        if rand.uniform(0.0, 1.0) < self.Rar:
            action = rand.randint(0, self.num_actions - 1)

        else:
            action = self.Q[s, :].argmax()

        self.s = s
        self.a = action

        return action

    def query(self, s_prime, r):
        # Update Q table:
        self.Q[self.s, self.a] = ((1 - self.alpha) * self.Q[self.s, self.a]) + \
                                 (self.alpha * (r + (self.gamma * self.Q[s_prime, self.Q[s_prime, :].argmax()])))

        # Dyna rewards/transitions
        if self.dyna > 0:
            self.dynaQ(s_prime, r)

        # pick next action
        a_prime = self.querySetState(s_prime)

        # decay random action rate
        self.Rar *= self.Radr

        return a_prime

    def dynaQ(self, s_prime, r):
        # update rewards table
        self.R[self.s, self.a] = ((1 - self.alpha) * self.R[self.s, self.a]) + (self.alpha * r)

        # Query if transition has occurred, update/init count
        if (self.s, self.a) in self.T:
            if s_prime in self.T[(self.s, self.a)]:
                self.T[(self.s, self.a)][s_prime] += 1
            else:
                self.T[(self.s, self.a)][s_prime] = 1
        else:
            self.T[(self.s, self.a)] = {s_prime: 1}

        for i in range(self.dyna):
            s = rand.randint(0, self.num_states - 1)
            a = rand.randint(0, self.num_actions - 1)

            if (s, a) in self.T:
                # argmax of t[(action, state)] dict
                s_prime = max(self.T[(s, a)], key=lambda x: self.T[(s, a)][x])

                self.Q[s, a] = ((1 - self.alpha) * self.Q[s, a]) + \
                               (self.alpha * (self.R[s, a] + self.gamma * self.Q[s_prime, self.Q[s_prime, :].argmax()]))
