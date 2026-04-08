import numpy as np


def weather_simulator(start_weather, num_days, transition_matrix, weather_states):
    '''Write a simulator that can randomly generate sequences of “weathers” from this state transition function'''
    state_to_idx = {state: idx for idx, state in enumerate(weather_states)}
    current_idx = state_to_idx[start_weather]
    weather_sequence = [start_weather]

    for i in range(num_days - 1):
        current_idx = np.random.choice(
            len(weather_states), p=transition_matrix[current_idx]
        )
        weather_sequence.append(weather_states[current_idx])

    return weather_sequence


def steady_state_distribution(transition_matrix, weather_states):
    '''Calculate the steady state distribution of the Markov chain defined by the transition matrix'''
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.isclose(eigenvalues, 1).nonzero()[0][0]
    steady_state_eig = np.real(eigenvectors[:, idx])
    steady_state_eig = steady_state_eig / np.sum(steady_state_eig)

    for state, probability in zip(weather_states, steady_state_eig):
        print(f"{state}: {probability:.4f}")


# Example:
transition_matrix = np.array([[0.8, 0.2, 0.0], 
                              [0.4, 0.4, 0.2],
                              [0.2, 0.6, 0.2]])
weather_states = ["Sunny", "Cloudy", "Rainy"]

# Simulate weather for 10 days starting from 'Sunny'
start_weather = "Sunny"
num_days = 10
simulated_weather = weather_simulator(
    start_weather, num_days, transition_matrix, weather_states
)
print(simulated_weather)

# Calculate steady state distribution
steady_state_distribution(transition_matrix, weather_states)
