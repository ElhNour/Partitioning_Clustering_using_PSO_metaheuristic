import numpy
from kmeans import KMeans
from sklearn.metrics.cluster import completeness_score
from itertools import permutations
 

def normalize(x: numpy.ndarray):
    """Scale to 0-1
    """
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def standardize(x: numpy.ndarray):
    """Scale to zero mean unit variance
    """
    return (x - x.mean(axis=0)) / numpy.std(x, axis=0)

def get_clusters_from_centroid(data, gbest_centroids):
    pso_kmeans = KMeans(n_cluster = gbest_centroids.shape[0], init_pp=False, seed=2018)
    pso_kmeans.centroid = gbest_centroids.copy()
    return pso_kmeans.predict(data)

def completeness_score(real_labels, predected_labels):
    return completeness_score(real_labels, predected_labels)

def correct_count(real_labels, predected_labels, permutation):
    correct = 0
    for i in range(len(real_labels)):
        if predected_labels[i] == permutation[real_labels[i]]:
            correct +=1
    return correct

def stupid_precision_ratio(real_labels, predected_labels, nb_clusters):
    permut = permutations(range(nb_clusters))
    total_number_observations = len(real_labels)
    max_ = 0
    config = {}
    for per in permut:
        correct = correct_count(real_labels, predected_labels, per)/ total_number_observations
        if max_ < correct:
            max_ = correct
            config = per
    return max_, config