
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
from data import *
from plotting_signatures import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
import pickle
import variational_autoencoder_for_transition
import time_point_transition_vae
import time_point_transition_vae_nn_for_first_tp
from math import sqrt

if __name__ == '__main__':
	#model = "VAE_one_for_transition"
	model = "transition_time_point_VAE_only_mut_types"
	#model = "transition_tp_VAE_nn_for_first_tp"
	# model = "transition_tp_VAE_nn_for_first_tp_w_cancer_type"
	data_set = "vae/training_transition_data.only_mut_types.csv"
	annot_path = "vae/training_annotation.csv"
	# data_set = 'vae/training_transition_data_w_cancer_type.csv'
	latent_dim = 10

	if len(sys.argv) > 1:
		latent_dim = int(sys.argv[1])

	# Training parameters
	param_scale = 0.01
	batch_size = 200
	num_epochs = 200
	step_size = 0.0005
	kl_weight_decay_time = 1

	model_prefix = "vae/" + model + "/" + data_set[4:-4] + "/" + model + "_tp."
	param_postfix = "latent_dim_" + str(latent_dim) + ".kl_weight_decay_time_" + str(kl_weight_decay_time) + ".pdf"

	N, train_samples, test_samples, training_annot, test_annot = load_signature_data(data_set, annot_path)

	all_samples = np.concatenate((train_samples, test_samples), axis=0)

	if model == "VAE_one_for_transition_w_cancer_type" or model == "VAE_one_for_transition" or model == "VAE_one_for_transition_only_mut_types":
		if model == "VAE_one_for_transition_only_mut_types":
			data_dim = tp_dim = 96
			exposures_included = False
		else:
			tp_dim = data_dim = 126
			exposures_included = True

		train_first_tp = all_samples[:,:tp_dim]
		train_second_tp = all_samples[:,tp_dim:(tp_dim*2)]

		gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
		rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]

		with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params = pickle.load(handle)

		gen_params, rec_params = optimized_params

		sample_second_tp, z_first_tp_means = variational_autoencoder_for_transition.vae_predict(optimized_params, train_first_tp)
		prediction_prob =  variational_autoencoder_for_transition.p_data_given_latents(gen_params, train_second_tp, z_first_tp_means)

	if model == "transition_time_point_VAE_w_cancer_type" or model == "transition_time_point_VAE" or model == "transition_time_point_VAE_only_mut_types":
		if model == "transition_time_point_VAE_only_mut_types":
			data_dim = 192
			tp_dim = 96
			exposures_included = False
		else:
			data_dim = 252  # How many pixels in each image (28x28).
			tp_dim = 126
			exposures_included = True

		if data_set == 'vae/training_transition_data_w_cancer_type.csv':
			data_dim = 253

		train_first_tp = all_samples[:,:tp_dim]
		train_second_tp = all_samples[:,tp_dim:(tp_dim*2)]

		gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
		rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]
		rec_layer_first_tp_sizes = [data_dim/2, 200, 300, latent_dim * 2]

		with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params = pickle.load(handle)

		gen_layer_optimized, rec_layer_optimized = optimized_params

		with open(model_prefix + "optimized_params_first_tp" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params_first_tp = pickle.load(handle)
			
		sample_second_tp, z_first_tp_means = time_point_transition_vae.two_step_vae_predict(optimized_params_first_tp, gen_layer_optimized, train_first_tp, tp_dim) 
		prediction_prob = time_point_transition_vae.p_second_tp_given_latents(gen_layer_optimized, train_second_tp, z_first_tp_means, tp_dim)

	if model == "transition_tp_VAE_nn_for_first_tp_w_cancer_type" or model == "transition_tp_VAE_nn_for_first_tp" or model == "transition_tp_VAE_nn_for_first_tp_only_mut_types":
		if model == "transition_tp_VAE_nn_for_first_tp_only_mut_types":
			data_dim = 192
			tp_dim = 96
			exposures_included = False
		else:
			data_dim = 252  # How many pixels in each image (28x28).
			tp_dim = 126
			exposures_included = True

		if data_set == 'vae/training_transition_data_w_cancer_type.csv':
			data_dim = 253

		train_first_tp = all_samples[:,:tp_dim]
		train_second_tp = all_samples[:,tp_dim:(tp_dim*2)]

		gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
		rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]
		rec_layer_first_tp_sizes = [data_dim/2, 200, 300, latent_dim * 2]

		with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params = pickle.load(handle)

		gen_layer_optimized, rec_layer_optimized = optimized_params
		z_means, z_stds = time_point_transition_vae_nn_for_first_tp.nn_predict_gaussian(rec_layer_optimized, train_samples)
		z_labels = np.concatenate((z_means, z_stds), axis=1)

		with open(model_prefix + "optimized_params_first_tp" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params_first_tp = pickle.load(handle)

		sample_second_tp, z_first_tp_means = time_point_transition_vae_nn_for_first_tp.two_step_vae_plus_nn_predict(optimized_params_first_tp, \
																						gen_layer_optimized, train_first_tp, latent_dim, tp_dim)
		prediction_prob = time_point_transition_vae_nn_for_first_tp.p_second_tp_given_latents(gen_layer_optimized, train_second_tp, z_first_tp_means, tp_dim)


	print("N samples: " + str(sample_second_tp.shape[0]))
	
	dist_second_list = []
	dist_first_list = []
	for i in range(sample_second_tp.shape[0]):
		dist_second = sqrt(sum((sample_second_tp[i,]- train_second_tp[i,]) ** 2))
		dist_first = sqrt(sum((sample_second_tp[i,]- train_first_tp[i,]) ** 2))
		dist_second_list.append(dist_second)
		dist_first_list.append(dist_first)

	count_less = 0
	count_greater = 0
	for i in range(len(dist_second_list)):
		if (dist_second_list[i] - dist_first_list[i] < -0.001):
			count_less +=1

		if (dist_second_list[i] - dist_first_list[i] > 0.001):
			count_greater +=1

	print("% samples with prediction closer to input (bad) : " + str(count_less/len(dist_second_list)))
	print("% samples with prediction closer to target (good) : " + str(count_greater/len(dist_second_list)))






	# pca = PCA(n_components=20)
	# pca_transformed = pca.fit_transform(z_first_tp_means[:1000,])

	cancer_types = get_cancer_types()

	print("Making TSNE plot...")
	#plot_tsne(z_first_tp_means[:,], training_annot[:,0], cancer_types, "vae/" + model + "/" + data_set[4:-4] + "/TSNE." + param_postfix)
	#plot_tsne(z_first_tp_means[:5000,], training_annot[:5000,0], cancer_types, "vae/" + model + "/" + data_set[4:-4] + "/TSNE.reduced." + param_postfix)
	print("vae/" + model + "/" + data_set[4:-4] + "/TSNE." + param_postfix)

	np.savetxt("vae/" + model + "/" + data_set[4:-4] + "/predictions_" + param_postfix[:-4] + ".csv", sample_second_tp, delimiter=",")
	np.savetxt("vae/" + model + "/" + data_set[4:-4] + "/prediction_prob_" + param_postfix[:-4] + ".csv", prediction_prob, delimiter=",")
	
	# print("Making example plots...")
	# example_dir = "vae/" + model + "/" + data_set[4:-4] + "/" + "examples" + param_postfix[:-4] + "/" 
	# if not os.path.exists(example_dir):
	# 	os.makedirs(example_dir)

	# for i in range(10):
	# 	plot_mut_types(sample_second_tp[i][:96], example_dir + "example" + str(i) + ".mut_types.predicted.pdf")
	# 	if exposures_included:
	# 		plot_exposures(sample_second_tp[i][96:tp_dim], example_dir + "example" + str(i) + ".exposures.predicted.pdf")

	# 	plot_mut_types(train_samples[i,tp_dim:(tp_dim + 96)], example_dir + "example" + str(i) + ".mut_types.pdf")
	# 	if exposures_included:
	# 		plot_exposures(train_samples[i,(tp_dim + 96):(tp_dim*2)], example_dir + "example" + str(i) + ".exposures.pdf")



