from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random

class DQNAgent:

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None, modelFile=None):

        if len(activations) != len(n_neurons) + 1:
            raise ValueError("n_neurons and activations do not match, "
                             f"expected a n_neurons list of length {len(activations) - 1}")

        if replay_start_size is not None and replay_start_size > mem_size:
            raise ValueError("replay_start_size must be <= mem_size")

        if mem_size <= 0:
            raise ValueError("mem_size must be > 0")

        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        else: 
            self.epsilon = 0
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size

        if modelFile is not None:
            self.model = load_model(modelFile)
        else:
            self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model


    def add_to_memory(self, current_state, next_state, reward, done):
      
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):

        return random.random()


    def predict_value(self, state):

        return self.model.predict(state, verbose=0)[0]


    def act(self, state):

        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def best_state(self, states):

        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state


    def train(self, batch_size=32, epochs=3):

        if batch_size > self.mem_size:
            print('WARNING: batch size is bigger than mem_size. The agent will not be trained.')

        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:

                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay


    def save_model(self, name):
        
        self.model.save(name)