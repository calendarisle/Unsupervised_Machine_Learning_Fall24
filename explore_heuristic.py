"""Heuristic that stays away from previous samples."""

import numpy as np
from sklearn.metrics import pairwise_distances

def explore_heuristic(num_samples, bounds, previous_samples, epsilon=0.5):
    """Sample away from previous samples.

    Parameters:
        num_samples: int. The number of new samples to create.
        bounds: numpy array. Dx2 array of lower and upper bounds for
            each feature.
        previous_samples: numpy array, None. The samples that we've already taken.
            If none, this is the first iteration.
        epsilon: float. The minimum distance away that the new samples must exceed.

    Return: numpy array, numpy array {KMeans object, None}.
        Nxdimensionality array of data.

    """

    # Repeatedly sample uniformly and remove samples
    # that are within some distance of the original samples.
    max_iter = 75
    iteration = 0
    sample_list = []
    while len(sample_list) != num_samples:
        features = []
        for bound_idx in range(bounds.shape[0]):
            features.append(
                np.random.uniform(bounds[bound_idx, 0], bounds[bound_idx, 1], num_samples)
            )
        # Turn this into one array.
        new_samps = np.stack(features)
        new_samps = new_samps.transpose()

        if previous_samples is None:
            return new_samps

        # Determine if the new samples are far enough away from
        # the previous samples.
        dists = pairwise_distances(new_samps, previous_samples)
        total_exceeded = (dists > epsilon).astype(int).sum(axis=1)

        good_sample_map = (total_exceeded == previous_samples.shape[0])
        good_samples = new_samps[good_sample_map]

        if good_samples.shape[0] != 0:

            # Also ensure that the samples aren't too close to each other.
            good_sample_dists = pairwise_distances(good_samples, good_samples)
            good_total_exceeded = (good_sample_dists > epsilon).astype(int).sum(axis=1)

            final_sample_map = (good_total_exceeded == (good_samples.shape[0] - 1))
            final_samples = good_samples[final_sample_map]

            # Add the appropriate amount of samples
            add_number = num_samples - len(sample_list)


            if final_samples.shape[0] != 0:
                sample_list.extend(np.split(final_samples[:add_number], final_samples[:add_number].shape[0]))

        if iteration >= max_iter:
            print("Exceeded max iterations!")
            if len(sample_list)==0:
                return None
            return np.concatenate((sample_list))
        iteration += 1
    
    return np.concatenate((sample_list))

# classifier = train_log_regressor(subsample_data_array, subsample_k_means.labels_)
# # Create the predictions
# classif_labels = classifier.predict(subsample_data_array)
# create_prediction_plot(
#     subsample_data_array, classif_labels, centers=classifier.coef_
# )

# create_prediction_plot(
#     location_data_array, all_locations_k_means.labels_, centers=all_locations_k_means.cluster_centers_
# )

# create_prediction_plot(
#     subsample_data_array, subsample_k_means.labels_, centers=subsample_k_means.cluster_centers_
# )

def explore_stable_samples(num_samples, bounds, old_samples, epsilon, old_kmeans, prev_kmeans=None):
    """
    Generate samples starting uniformly at random, then discard any samples whose predicted labels
    change between two prediction steps (e.g., before and after updating KMeans or comparing two models).
    Replace discarded points using the explore_heuristic approach.
    
    Parameters:
        num_samples: int, number of samples to generate.
        bounds: numpy array (D, 2), lower and upper bounds for each feature.
        old_samples: numpy array or None, previously chosen samples (used by explore_heuristic).
        epsilon: float, minimum distance used in explore_heuristic.
        old_kmeans: A trained KMeans or classifier object for predicting labels.
        prev_kmeans: A previous KMeans or classifier state from another iteration (optional).
                     If not provided, we might simulate a second state or skip the comparison step.
                     
    Returns:
        A numpy array of shape (num_samples, D) with the new stable samples.
    """
    # 1. Generate initial random samples
    features = []
    for bound_idx in range(bounds.shape[0]):
        features.append(
            np.random.uniform(bounds[bound_idx, 0], bounds[bound_idx, 1], num_samples)
        )
    new_samps = np.stack(features).T  # (num_samples, D)

    # 2. Predict labels from one state (e.g., old_kmeans)
    labels_state_1 = old_kmeans.predict(new_samps)

    # If we have a previous state (prev_kmeans), predict from that as well
    if prev_kmeans is not None:
        labels_state_2 = prev_kmeans.predict(new_samps)
    else:
        # If no prev_kmeans provided, one approach is to slightly alter old_kmeans or simulate a scenario.
        # For demonstration, we'll just duplicate the same predictions (no changes).
        # In practice, you'd have two different prediction steps to compare.
        labels_state_2 = labels_state_1.copy()

    # 3. Identify stable samples: those that have not changed labels
    stable_mask = (labels_state_1 == labels_state_2)
    stable_samps = new_samps[stable_mask]

    # 4. For points that changed labels, replace them using explore_heuristic
    num_unstable = num_samples - stable_samps.shape[0]
    if num_unstable > 0:
        # Use explore_heuristic to generate replacements
        # This ensures points are epsilon away from old_samples and each other
        replacement_samps = explore_heuristic(num_unstable, bounds, old_samples, epsilon=epsilon)

        # If explore_heuristic fails to find replacements (returns None), handle gracefully
        if replacement_samps is None:
            # As a fallback, just return the stable samples we have
            return stable_samps
        # Combine stable samples and the newly explored stable samples
        final_samps = np.concatenate((stable_samps, replacement_samps), axis=0)
    else:
        final_samps = stable_samps

    return final_samps