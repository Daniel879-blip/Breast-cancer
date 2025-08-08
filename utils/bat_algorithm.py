import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def objective_function(features, labels, position):
    """
    features: numpy array (n_samples, n_features)
    labels: numpy array (n_samples,)
    position: boolean-like list/array indicating selected features (True/False or 1/0)
    """
    if len(position) == 0:
        return 0.0
    # convert boolean mask to column indices
    mask = np.array(position, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    X_sel = features[:, mask]
    # use GaussianNB as the internal eval classifier
    clf = GaussianNB()
    try:
        scores = cross_val_score(clf, X_sel, labels, cv=5, error_score='raise')
        return float(np.mean(scores))
    except Exception:
        # if CV fails for any reason return 0
        return 0.0

def bat_algorithm(features, labels, num_bats=20, max_gen=50, loudness=0.5, pulse_rate=0.5, progress_callback=None):
    """
    Binary-styled BAT algorithm for feature selection.
    - features: numpy (n_samples, n_features)
    - labels: numpy (n_samples,)
    - num_bats: population
    - max_gen: generations
    - loudness, pulse_rate: algorithm params
    - progress_callback(gen_index, best_score, curve_so_far, max_gen) -> optional function to animate progress
    Returns: selected_feature_indices (list), convergence_curve (list)
    """
    n_features = features.shape[1]
    # initialize positions randomly (0/1)
    positions = (np.random.rand(num_bats, n_features) > 0.5).astype(int)
    velocities = np.zeros((num_bats, n_features))
    freq_min, freq_max = 0.0, 2.0

    fitness = np.array([objective_function(features, labels, positions[i]) for i in range(num_bats)])
    best_idx = np.argmax(fitness)
    best_position = positions[best_idx].copy()
    best_score = float(fitness[best_idx])

    convergence = [best_score]

    for gen in range(max_gen):
        for i in range(num_bats):
            freq = freq_min + (freq_max - freq_min) * np.random.rand()
            # velocity update (binary idea: xor with best and scaled freq)
            velocities[i] = velocities[i] + freq * (positions[i] ^ best_position)
            # sigmoidal transform to get probabilities
            prob = 1.0 / (1.0 + np.exp(-velocities[i]))
            new_pos = (np.random.rand(n_features) < prob).astype(int)

            # local search (pulse)
            if np.random.rand() > pulse_rate:
                # flip a small number of bits of the best
                rand_mask = (np.random.rand(n_features) < 0.05).astype(int)
                new_pos = best_position.copy()
                new_pos[rand_mask==1] = 1 - new_pos[rand_mask==1]

            new_score = objective_function(features, labels, new_pos)

            # acceptance
            if (new_score > fitness[i]) and (np.random.rand() < loudness):
                positions[i] = new_pos
                fitness[i] = new_score

            # update global best
            if new_score > best_score:
                best_position = new_pos.copy()
                best_score = float(new_score)

        convergence.append(best_score)
        # call progress callback if provided
        if progress_callback is not None:
            try:
                progress_callback(gen, best_score, convergence.copy(), max_gen)
            except Exception:
                # ignore callback errors to not break optimization
                pass

    # compute selected indices
    selected_indices = [int(i) for i, v in enumerate(best_position) if v == 1]
    return selected_indices, convergence
