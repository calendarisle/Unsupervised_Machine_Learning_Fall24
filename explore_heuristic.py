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

