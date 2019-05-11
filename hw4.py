import os
import sys
import random
import operator
import math

# dataset_dir = os.path.join(os.getcwd(), sys.argv[1])
# K = sys.argv[2]
# iterations = sys.argv[3]

dataset_dir = "/Users/chenhang91/TEMP/HW4Group/em_data.txt"
K = 3
iterations = 1000

# initialize params

# read lines as a list https://qiita.com/visualskyrim/items/1922429a07ca5f974467
data_list = [float(line.rstrip('\n')) for line in open(dataset_dir)]
data_length = len(data_list)

# use K-means to initialize mu in theta
def initialize_mu_K_means():
    old_centroids = []
    new_centroids = []
    # initialize centroids
    for i in range(K):
        random_point = data_list[random.randint(0, data_length-1)]
        while random_point in new_centroids:
            random_point = data_list[random.randint(0, data_length-1)]
        new_centroids.append(random_point)
    # check for convergence
    while old_centroids != new_centroids:
        # assign each data point to its closest centroid
        old_centroids = new_centroids
        cluster_dict = dict([(key, []) for key in old_centroids])
        for data_point in data_list:
            distance_dict = {}
            for centroid in cluster_dict:
                # https://www.geeksforgeeks.org/abs-in-python/
                distance_dict[centroid] = abs(data_point - centroid)
            best_centroid = min(distance_dict.items(), key=operator.itemgetter(1))[0]
            cluster_dict[best_centroid].append(data_point)
        # recalculate centroids(mean value, mu) for each cluster
        new_centroids = []
        for cluster in cluster_dict:
            new_centroids.append(sum(cluster_dict[cluster])/len(cluster_dict[cluster]))
    return new_centroids, cluster_dict

means, clusters = initialize_mu_K_means()

# based on the clusters calculated from K-means, initialize stand_deviation and variance for each model
def initialize_std_and_variance():
    variances = []
    for mean in means:
        variance_temp = 0
        for data_point in clusters[mean]:
            variance_temp += (data_point - mean) ** 2
        variances.append(1/len(clusters[mean]) * variance_temp)
    # https://stackoverflow.com/questions/26894227/sum-of-squares-in-a-list-in-one-line
    stds = list(map(lambda variance:math.sqrt(variance), variances))
    return stds, variances
    
stds, variances = initialize_std_and_variance()

# initialize alphas
# A list of random numbs sum to 1
# https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
alphas = [random.random() for i in range(K)]
temp_sum = sum(alphas)
alphas = [alpha/temp_sum for alpha in alphas]

# create models_list and feed in their initial values
models_list =[]
for i in range(K):
    models_list.append({'mu': means[i], 'std': stds[i], 'variance': variances[i], 'alpha':alphas[i]})

# begin EM-algorithm

# construct data_points_dict
# https://stackoverflow.com/questions/671403/memory-efficiency-one-large-dictionary-or-a-dictionary-of-smaller-dictionaries
data_points_dict_list = []
for data_point in data_list:
    # the list stored in {data_point: []} is used to store k omegles for this data point, and its index could indicate K, like index 0 means omegle1 for model 1 for this data_point, index 1 means omegle2 for model 2
    data_points_dict_list.append({data_point: [0] * K})

# helper function for calculating PDF of Gaussian
def gaussian_pdf(mean, std, variance, data_point):
    return 1/(std * math.sqrt(2 * math.pi)) * math.exp(-(data_point - mean)**2 / (2 *variance))

# begin learning
for i in range(iterations):
    # E step
    for data_dict in data_points_dict_list:
        # https://stackoverflow.com/questions/3545331/how-can-i-get-dictionary-key-as-variable-directly-in-python-not-by-searching-fr
        data_point = list(data_dict.keys())[0]
        omegle_numerator_each_model = []
        for model in models_list:
            omegle_numerator_each_model.append(gaussian_pdf(model['mu'], model['std'], model['variance'], data_point) * model['alpha'])
        for model_iter in range(K):
            data_dict[data_point][model_iter] = omegle_numerator_each_model[model_iter]/sum(omegle_numerator_each_model)
    # M step
    for model_iter in range(K):
        # calculate nk, mu_k_temp_sum, variance_k_temp_sum
        nk = 0
        mu_k_temp = 0
        variance_k_temp_sum = 0
        mu_k_temp_sum = 0
        for data_dict in data_points_dict_list:
            omegle_k = list(data_dict.values())[0][model_iter]
            data_point_val = list(data_dict.keys())[0]
            nk += omegle_k
            mu_k_temp_sum += omegle_k * data_point_val
            variance_k_temp_sum += omegle_k * ((data_point_val - models_list[model_iter]['mu']) ** 2)
        # re-calculate alpha
        models_list[model_iter]['alpha'] = nk/data_length
        # re-calculate mu
        models_list[model_iter]['mu'] = (1/nk) * mu_k_temp_sum
        # re-calculate variance
        models_list[model_iter]['variance'] = (1/nk) * variance_k_temp_sum

print(models_list)







# data_points_dict = {}
# for data_iter in data_list:
#     data_points_dict[data_iter] = {}
#     data_points_dict[data_iter][data_list[data_iter]] = []


# for i in range(iterations):