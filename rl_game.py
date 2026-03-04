from time import sleep
import os

import numpy as np

learning_rate = 0.1

class Action:
    def __init__(self, wait_time):
        self.wait_time = wait_time
        self.q_value = 0.0
        self.N = 0
        self.correct_guesses = 0

    def update(self, reward):
        self.N += 1
        
        self.correct_guesses += 1 if reward > 0 else 0

        # Apply exponential decay to the reward based on wait time
        print(f"\n\nUpdating action with wait_time: {self.wait_time}, current q_value: {self.q_value}, N: {self.N}, reward: {reward}")
        penalty = 1 - np.exp(-self.wait_time)
        reward = reward + (reward * (-1) * penalty)
        self.q_value += learning_rate * ((reward - self.q_value) / self.N)
        print(f"Updated q_value: {self.q_value:.2f} after applying penalty: {penalty:.2f}\n\n")


    def __repr__(self):
        return f"Action(wait_time: {self.wait_time:.2f}, q_value: {self.q_value:.2f}, N: {self.N}, correct_guesses: {self.correct_guesses})"

actions = [Action(wait_time) for wait_time in [0.05, 0.1, 0.2, 0.5, 1.0]]

for i in range(100):

    epsilon = np.max([0.75 * (1 - i / 100), 0.1])  # Decaying epsilon
    if np.random.rand() < epsilon:
        action = np.random.choice(actions)
        state = "exploration"
    else:
        action = max(actions, key=lambda a: a.q_value)
        state = "exploitation"
    iteration_wait_time = action.wait_time
    random_number = np.random.randint(0, 100)

    os.system('clear')
    print(f'Random number: {random_number}')
    sleep(iteration_wait_time)
    os.system('clear')

    print(f"Running in {state} mode, epsilon: {epsilon:.3f}, waited for {iteration_wait_time:.2f} seconds, Q-value: {action.q_value:.2f}, N: {action.N}")
    guess = input("Guess the number (0-99): ")
    
    if guess.isdigit() and int(guess) == random_number:
        reward = 1.0
        print("Correct guess!")
    else:
        reward = -1.0
        print(f"Wrong guess! The correct number was {random_number}.")
    
    action.update(reward)
    for action in actions:
        print(action)
    input('Press Enter to continue...')
    os.system('clear')
