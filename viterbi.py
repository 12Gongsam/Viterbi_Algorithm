import numpy as np


class Viterbi:
    def __init__(self, p: float):
        # p : measurement noise
        self.A = np.array([
            [0.975, 0.025],
            [0.025, 0.975]
        ])
        self.B = np.array([
            [1 - p, p],
            [p, 1 - p]
        ])
        self.pi = np.array([1, 0]) # z_1 = 0 

    def generate_hidden_state(self, sequence_length: int, seed: int) -> np.ndarray:
        
        np.random.seed(seed)
        states = [np.random.choice([0, 1], p=self.pi)] # initial state
        for _ in range(sequence_length - 1):
            next_state = np.random.choice([0, 1], p=self.A[states[-1]])
            states.append(next_state)
        return np.array(states)
    
    def generate_observation(self, hidden_states: np.ndarray, seed: int) -> np.ndarray:
        
        np.random.seed(seed)
        observations = []
        for state in hidden_states:
            observation = np.random.choice([0, 1], p=self.B[state])
            observations.append(observation)
        return np.array(observations)
    
    def algorithm(self, observations):
        T = len(observations)
        N = self.A.shape[0]
        # row : state id, col : time
        deltas = np.zeros((N, T))
        # backtracking table
        backpointer = np.zeros((N, T), dtype=int)  

        # Initialization (t=0)
        deltas[0][0] = self.pi[0] * self.B[0][observations[0]]
        deltas[1][0] = self.pi[1] * self.B[1][observations[0]]

        # Forward Pass
        for time, observation in enumerate(observations):
            if time == 0:
                continue
            for z in [0, 1]:
                # Record the maximum value of all paths coming from the previous point and the status of that path.
                max_val_0 = deltas[0][time - 1] * self.A[0][z]
                max_val_1 = deltas[1][time - 1] * self.A[1][z]
                if max_val_0 > max_val_1:
                    deltas[z][time] = max_val_0 * self.B[z][observation]
                    backpointer[z][time] = 0
                else:
                    deltas[z][time] = max_val_1 * self.B[z][observation]
                    backpointer[z][time] = 1

        # Backward Pass
        # Finding the optimal state at the very last point
        best_last_state = np.argmax(deltas[:, -1])
        best_path = [best_last_state]

        # Backtracking to restore optimal path
        for time in range(T - 1, 0, -1):
            best_next_state = backpointer[best_path[-1]][time]
            best_path.append(best_next_state)

        # Since we traced the optimal path in reverse, we need to invert the results.
        best_path.reverse()

        return np.array(best_path), deltas

def simulate_viterbi(p: float, sequence_length: int, num_realizations: int):
    viterbi = Viterbi(p)
    total_errors = 0

    for seed in range(num_realizations):
        true_hidden_states = viterbi.generate_hidden_state(sequence_length, seed)
        observations = viterbi.generate_observation(true_hidden_states, seed)
        estimated_states, _ = viterbi.algorithm(observations)
        # compute error
        errors = np.sum(true_hidden_states != estimated_states)
        total_errors += errors

    # compute average error probability
    avg_error_probability = total_errors / (sequence_length * num_realizations)
    return avg_error_probability

if __name__=="__main__":
    sequence_length = 100
    num_realization = 1000
    p_values = [0.01, 0.05, 0.1]

    for p in p_values:
        avg_error = simulate_viterbi(p, sequence_length, num_realization)
        print(f"p = {p}, Average Error Probability: {avg_error:.4f}")