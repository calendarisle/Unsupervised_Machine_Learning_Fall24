""" Util File for functions."""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
import sklearn 
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import pairwise_distances
import warnings
from sklearn.exceptions import ConvergenceWarning

import explore_heuristic

def determine_bounds(input_array):
    bounds = []
    for dim_idx in range(input_array.shape[1]):
        bounds.append(np.array([
            np.min(input_array[:, dim_idx]), np.max(input_array[:, dim_idx])
        ]))
    all_bounds = np.stack(bounds)
    return all_bounds

def ground_truth_kmeans(full_data, num_clusters): #Potential fix
    temp_kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    temp_kmeans.fit(full_data)
    return temp_kmeans

def random_subsample(full_data, num_samples):
    """Return a random subset of N samples from all the data."""
    perm = np.random.permutation(full_data.shape[0])
    less_data = copy.deepcopy(full_data[perm][:num_samples])
    return less_data

# Train a multiclass logistic regression model
def train_log_regressor(train_data, labels):    
    classif = LogisticRegression(multi_class='multinomial', max_iter=1000)
    classif.fit(train_data, labels)
    return classif

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

def generate_new_cluster_labels(original_data, num_new_samples, bounds,
                                method="random", retraining="online",
                                old_kmeans=None, old_samples=None, epsilon=0.5,
                                num_clusters=10, prev_kmeans=None):
    
    """Sample some data by re-training k-means with the new data.
    
    Parameters:
        original_data: numpy array. The base data that is hidden.
        num_new_samples: int. The number of new samples to create.
        bounds: numpy array. Dx2 array of lower and upper bounds for
            each feature.
        method: string. How we'll create the new data.
            "random": Randomly sample points from the space.
            "distance": Sample points perpetually further away from other points.
        retraining: string. How we'll create the new clusters.
            "online": Start from a previously learned set of centers.
            "restart": Start from scratch with the same random init.
        old_kmeans: sklearn KMeans object or None. Another KMeans object.
            Used to extract the old cluster centres for "online"
            or to predict with "query".
        old_samples: numpy array or None. Previously created centres
            to use when the sample method is "distance".
        epsilon: float. If "distance" is the method, the min distance to exceed.
        num_clusters: int.
            
    Return: numpy array, numpy array {KMeans object, None}.
        Nx2 array of data, N-dimensional vector of labels.
        May return a KMeans object if "online".
    
    """
    if method == "random":
        # Sample uniformly within the bounds.
        features = []
        for bound_idx in range(bounds.shape[0]):
            features.append(
                np.random.uniform(bounds[bound_idx, 0], bounds[bound_idx, 1], num_new_samples)
            )
        # Turn this into one array.
        new_samps = np.stack(features)
        new_samps = new_samps.transpose()
    elif method == "distance":
        new_samps = explore_heuristic(
            num_new_samples, bounds, old_samples, epsilon=epsilon
        )
        if new_samps is None: 
            return None, None, None
    elif method == "stable_random":
        new_samps = explore_stable_samples(
            num_new_samples, bounds, old_samples, epsilon, old_kmeans, prev_kmeans
        )
        if new_samps is None:
            return None, None, None
    else:
        raise NotImplementedError("No other methods.")
    
    # Add the new samples to the original array and retrain kmeans.
    all_samples = np.concatenate((original_data, new_samps))
    
    # Initialize k-means according to which retraining paradigm
    # we will be working with.
    if retraining == "online":
        assert old_kmeans is not None
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, init=old_kmeans.cluster_centers_, n_init=1)
    elif retraining == "restart":
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    elif retraining == "query":
        kmeans = copy.deepcopy(old_kmeans)
    

    if retraining == "query":
        new_samp_labels = kmeans.predict(new_samps)
    else:
        kmeans.fit(all_samples)

        # Extract the new labels and return.
        new_samp_labels = kmeans.labels_[-new_samps.shape[0]:]
    
    if retraining == "online":
        return new_samps, new_samp_labels, kmeans
    else:
        return new_samps, new_samp_labels

def attack_experiment(original_data, original_kmeans, num_rounds, samps_per_round,
                      method, retraining, num_repeats=100, epsilon=15.0, num_clusters=10,
                      add_gaussian_noise=False):
    """Run a k-means attack experiment.

    Parameters:


    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        bounds = determine_bounds(original_data)
        acc_list = []
        for repeat in range(num_repeats):
            new_data = []
            new_labels = []
            # Copy the kmeans object as we'll be overwriting
            # this for the online learning.
            new_kmeans = copy.deepcopy(original_kmeans)
            prev_kmeans = copy.deepcopy(new_kmeans)
            for iteration in range(num_rounds):
                if add_gaussian_noise:
                    noise = np.random.normal(loc=0.0, scale=0.1, size=original_data.shape)
                    data_for_round = original_data + noise
                else: 
                    data_for_round = original_data
                if method == "random":
                    # Only online returns a kmeans object:
                    if retraining == "online":
                        re_new_samps, re_new_labels, new_kmeans = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method=method, retraining=retraining, old_kmeans=new_kmeans,
                            num_clusters=num_clusters
                        )
                    else:
                        re_new_samps, re_new_labels = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method=method, retraining=retraining, old_kmeans=new_kmeans,
                            num_clusters=num_clusters
                    )
                elif method == "distance":
                    if len(new_data) == 0:
                        old_samps = None
                    else:
                        old_samps = np.concatenate(new_data)
                    # Only online returns a kmeans object:
                    if retraining == "online":
                        re_new_samps, re_new_labels, new_kmeans = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method=method, retraining=retraining, old_kmeans=new_kmeans,
                            old_samples=old_samps, num_clusters=num_clusters, epsilon=epsilon
                        )
                    else:
                        re_new_samps, re_new_labels = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method=method, retraining=retraining, old_kmeans=new_kmeans,
                            old_samples=old_samps, num_clusters=num_clusters, epsilon=epsilon
                    )
                elif method == "stable_random":
                # If stable_random needs old_samps similarly to distance:
                    if len(new_data) == 0:
                        old_samps = None
                    else:
                        old_samps = np.concatenate(new_data)
                    
                    if retraining == "online":
                        re_new_samps, re_new_labels, new_kmeans = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method="stable_random",
                            retraining="online",
                            old_kmeans=new_kmeans,
                            old_samples=old_samps,
                            epsilon=epsilon,   # if needed by stable_random
                            num_clusters=num_clusters,
                            prev_kmeans=prev_kmeans
                            
                        )
                    else:
                        re_new_samps, re_new_labels = generate_new_cluster_labels(
                            data_for_round, samps_per_round, bounds,
                            method="stable_random",
                            retraining="query", # or whatever is specified
                            old_kmeans=new_kmeans,
                            old_samples=old_samps,
                            epsilon=epsilon,
                            num_clusters=num_clusters
                            # Add prev_kmeans or other params if needed
                        )

                    # Note that "distance" might fill the space. In this case, we'll
                    # add the final batch of data nd then break the loop.
                    if re_new_samps is None:
                        print("Space filled, did not retrieve samples")
                        break
                    if re_new_samps.shape[0] < samps_per_round:
                        if re_new_samps.shape[0] != 0:
                            new_data.append(re_new_samps)
                            new_labels.append(re_new_labels)
                        print("Space filled, only retrieved {} samples".format(re_new_samps.shape[0]))
                        break
                else:
                    raise ValueError("Not a valid method!")
                if re_new_samps is None: 
                    break
                else:     
                    new_data.append(re_new_samps)
                    new_labels.append(re_new_labels)
                    prev_kmeans = copy.deepcopy(new_kmeans)
            
            new_data = np.concatenate(new_data)
            new_labels = np.concatenate(new_labels)
            classifier = train_log_regressor(new_data, new_labels)

            acc = sklearn.metrics.accuracy_score(
                original_kmeans.labels_, classifier.predict(data_for_round)
            )
            acc_list.append(acc)

        print("Mean accuracy {}".format(np.mean(acc_list)))
        print("Std accuracy {}".format(np.std(acc_list)))
        return acc_list


def acc_plot(title, all_acc_lists, names, xticks=None):
    """ Create a plot for an ablation experiment.

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    for acc_lists, name in zip(all_acc_lists, names):
        mean_array = np.array([np.mean(acc_list) for acc_list in acc_lists])
        std_array = np.array([np.std(acc_list) for acc_list in acc_lists])
        plt.plot(
            np.arange(len(mean_array)), mean_array, label=name
        )
        plt.fill_between(
            np.arange(len(mean_array)), mean_array - std_array, mean_array + std_array,
            alpha=0.2
        )
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=12)
    ax.set_xlabel("Sampling", fontsize=14) 

    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()