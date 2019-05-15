import os
import sys
import random
import operator
import math

dataset_dir = os.path.join(os.getcwd(), sys.argv[1])
K = int(sys.argv[2])
iterations = int(sys.argv[3])
# By default we use k-means to initialize mu for each model and we don't assign variance as 1
variance_one = False
km_algo = True

try:
  km_algo = sys.argv[4].lower() == 't'
  variance_one = sys.argv[5].lower() == 't'
except:
    pass

# initialize params

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
    # To answer the question "Is the result sensitive to the initial values?", we provide two ways to assgin the means(theta) here
    # If km_algo is set to False, meaning we don't use k-means to assign the means for theta, then we just return the randomly selected centroids
    if not km_algo:
        print("Randomly assign thetas.")
        cluster_dict = dict([(key, []) for key in new_centroids])
        # randomly assign K clusters to calculate std and variance later
        random.shuffle(data_list)
        clusters_list = [data_list[i::K] for i in range(K)]
        clusters_list_index = 0
        for random_centroid in cluster_dict:
            cluster_dict[random_centroid] = clusters_list[clusters_list_index]
            clusters_list_index += 1
    else:
        print("Using K-means to assign thetas.")
        # check for convergence
        while old_centroids != new_centroids:
            # assign each data point to its closest centroid
            old_centroids = new_centroids
            cluster_dict = dict([(key, []) for key in old_centroids])
            for data_point in data_list:
                distance_dict = {}
                for centroid in cluster_dict:
                    distance_dict[centroid] = abs(data_point - centroid)
                best_centroid = min(distance_dict.items(), key=operator.itemgetter(1))[0]
                cluster_dict[best_centroid].append(data_point)
            # recalculate centroids(mean value, mu) for each cluster
            new_centroids = []
            for cluster in cluster_dict:
                new_centroids.append(sum(cluster_dict[cluster])/len(cluster_dict[cluster]))
    return new_centroids, cluster_dict

means, clusters = initialize_mu_K_means()

if variance_one:
    print("Variance are known as 1 for all models.")

# based on the clusters calculated from K-means, initialize stand_deviation and variance for each model
def initialize_std_and_variance():
    variances = []        
    for mean in means:
        if variance_one:
            variances.append(1)
        else:
            variance_temp = 0
            for data_point in clusters[mean]:
                variance_temp += (data_point - mean) ** 2
            variances.append(1/len(clusters[mean]) * variance_temp)
    stds = list(map(lambda variance:math.sqrt(variance), variances))
    return stds, variances
    
stds, variances = initialize_std_and_variance()

# initialize alphas
# A list of random numbs sum to 1
alphas = [random.random() for i in range(K)]
temp_sum = sum(alphas)
alphas = [alpha/temp_sum for alpha in alphas]

# create models_list and feed in their initial values
models_list =[]
for i in range(K):
    models_list.append({'mu': means[i], 'std': stds[i], 'variance': variances[i], 'alpha':alphas[i]})

# helper function to print the params for the model
def print_model_params(models_list):
    for i in range(K):
        print(f"\nModel {i + 1}")
        print("mu:", models_list[i]['mu'])
        print("variance:", models_list[i]['variance'])
        print("alpha:", models_list[i]['alpha'])

print("=============================================")
print("Before EM-algorithm")
print_model_params(models_list)

# begin EM-algorithm

# construct data_points_dict
data_points_dict_list = []
for data_point in data_list:
    # the list stored in {data_point: []} is used to store k omegles for this data point, and its index could indicate K, like index 0 means omegle1 for model 1 for this data_point, index 1 means omegle2 for model 2
    data_points_dict_list.append({data_point:{'omegle_ks': [0] * K, 'z_ks': [0] * K,}})

# helper function for calculating PDF of Gaussian
def gaussian_pdf(mean, std, variance, data_point):
    try:
        return 1/(std * math.sqrt(2 * math.pi)) * math.exp(-(data_point - mean)**2 / (2 * variance))
    except:
        print("Warning: wik calculated as 0 leading to variance 0 for this model.")
        return 0

# begin learning
for i in range(iterations):
    # E step
    for data_dict in data_points_dict_list:
        data_point = list(data_dict.keys())[0]
        omegle_numerator_each_model = []
        z_numerator_each_model = []
        for model in models_list:
            gaussian_likelihood = gaussian_pdf(model['mu'], model['std'], model['variance'], data_point)
            omegle_numerator_each_model.append(gaussian_likelihood * model['alpha'])
            z_numerator_each_model.append(gaussian_likelihood)
        for model_iter in range(K):
            data_dict[data_point]['omegle_ks'][model_iter] = omegle_numerator_each_model[model_iter]/sum(omegle_numerator_each_model)
            data_dict[data_point]['z_ks'][model_iter] = z_numerator_each_model[model_iter]/sum(z_numerator_each_model)
    # M step
    for model_iter in range(K):
        # calculate nk, mu_k_temp_sum, variance_k_temp_sum
        nk = 0
        mu_k_temp = 0
        variance_k_temp_sum = 0
        mu_k_temp_sum = 0
        for data_dict in data_points_dict_list:
            omegle_k = list(data_dict.values())[0]['omegle_ks'][model_iter]
            data_point_val = list(data_dict.keys())[0]
            nk += omegle_k
            mu_k_temp_sum += omegle_k * data_point_val
            variance_k_temp_sum += omegle_k * ((data_point_val - models_list[model_iter]['mu']) ** 2)
        # re-calculate alpha
        models_list[model_iter]['alpha'] = nk/data_length
        # re-calculate mu
        models_list[model_iter]['mu'] = (1/nk) * mu_k_temp_sum
        # re-calculate variance (if not known as 1)
        if not variance_one:
            models_list[model_iter]['variance'] = (1/nk) * variance_k_temp_sum

print("=============================================")
print("After EM-algorithm")
print_model_params(models_list)

# calculate the log likelihood of the whole data
log_likelihood_whole_data = 0
for data_dict in data_points_dict_list:
    data_point = list(data_dict.keys())[0]
    numerator_sum_in_exp = 0
    for model_iter in range(K):
        zik = data_dict[data_point]['z_ks'][model_iter]
        model = models_list[model_iter]
        numerator_sum_in_exp += -1 * zik * (data_point - model['mu']) ** 2
        likelihood_each_data = 1/(model['std'] * math.sqrt(2 * math.pi)) * math.exp(numerator_sum_in_exp / (2 * model['variance']))
    log_likelihood_whole_data += math.log(likelihood_each_data)


print("=============================================")
print(f"Log likelihood of these data generated from this mixture of models is: {log_likelihood_whole_data}")