import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def objective_function(features, labels, selected_features):
    if np.count_nonzero(selected_features) == 0:
        return 0
    clf = GaussianNB()
    X_selected = features[:, selected_features == 1]
    return cross_val_score(clf, X_selected, labels, cv=5).mean()

def bat_algorithm(features, labels, num_bats=20, max_gen=50, loudness=0.5, pulse_rate=0.5):
    num_features = features.shape[1]
    positions = np.random.randint(0, 2, (num_bats, num_features))
    velocities = np.zeros((num_bats, num_features))
    freq_min, freq_max = 0, 2
    fitness = np.array([objective_function(features, labels, pos) for pos in positions])
    best_idx = np.argmax(fitness)
    best_solution = positions[best_idx].copy()
    best_score = fitness[best_idx]

    convergence_curve = [best_score]

    for t in range(max_gen):
        for i in range(num_bats):
            freq = freq_min + (freq_max - freq_min) * np.random.rand()
            velocities[i] += (positions[i] - best_solution) * freq
            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            new_position = np.array([1 if np.random.rand() < s else 0 for s in sigmoid])

            if np.random.rand() > pulse_rate:
                new_position = best_solution.copy()
                rand_idx = np.random.randint(num_features)
                new_position[rand_idx] = 1 - new_position[rand_idx]

            new_score = objective_function(features, labels, new_position)

            if (new_score > fitness[i]) and (np.random.rand() < loudness):
                positions[i] = new_position
                fitness[i] = new_score

            if new_score > best_score:
                best_solution = new_position.copy()
                best_score = new_score

        convergence_curve.append(best_score)

    return best_solution, convergence_curve
