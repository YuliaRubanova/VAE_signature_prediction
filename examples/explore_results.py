
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

	N, train_samples, test_images, training_annot, test_annot = load_signature_data(data_set, annot_path)

	if model == "VAE_one_for_transition_w_cancer_type" or model == "VAE_one_for_transition" or model == "VAE_one_for_transition_only_mut_types":
		if model == "VAE_one_for_transition_only_mut_types":
			data_dim = tp_dim = 96
			exposures_included = False
		else:
			tp_dim = data_dim = 126
			exposures_included = True

		train_first_tp = train_samples[:,:tp_dim]
		train_second_tp = train_samples[:,tp_dim:(tp_dim*2)]

		gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
		rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]

		with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params = pickle.load(handle)

		sample_second_tp, z_first_tp_means = variational_autoencoder_for_transition.vae_predict(optimized_params, train_first_tp)


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

		train_first_tp = train_samples[:,:tp_dim]
		train_second_tp = train_samples[:,tp_dim:(tp_dim*2)]

		gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
		rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]
		rec_layer_first_tp_sizes = [data_dim/2, 200, 300, latent_dim * 2]

		with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params = pickle.load(handle)

		gen_layer_optimized, rec_layer_optimized = optimized_params

		with open(model_prefix + "optimized_params_first_tp" + param_postfix + '.pickle', 'rb') as handle:
			optimized_params_first_tp = pickle.load(handle)
			
		sample_second_tp, z_first_tp_means = time_point_transition_vae.two_step_vae_predict(optimized_params_first_tp, gen_layer_optimized, train_first_tp) 

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

		train_first_tp = train_samples[:,:tp_dim]
		train_second_tp = train_samples[:,tp_dim:(tp_dim*2)]

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

		sample_second_tp, z_first_tp_means = time_point_transition_vae_nn_for_first_tp.two_step_vae_plus_nn_predict(optimized_params_first_tp, gen_layer_optimized, train_first_tp, latent_dim)


	# pca = PCA(n_components=20)
	# pca_transformed = pca.fit_transform(z_first_tp_means[:1000,])

	cancer_types = get_cancer_types()

	print("Making TSNE plot...")
	plot_tsne(z_first_tp_means[:,], training_annot[:,0], cancer_types, "vae/" + model + "/" + data_set[4:-4] + "/TSNE." + param_postfix)
	plot_tsne(z_first_tp_means[:5000,], training_annot[:5000,0], cancer_types, "vae/" + model + "/" + data_set[4:-4] + "/TSNE.reduced." + param_postfix)
	print("vae/" + model + "/" + data_set[4:-4] + "/TSNE." + param_postfix)

	print("Making example plots...")
	example_dir = "vae/" + model + "/" + data_set[4:-4] + "/" + "examples" + param_postfix[:-4] + "/" 
	if not os.path.exists(example_dir):
		os.makedirs(example_dir)

	for i in range(10):
		plot_mut_types(sample_second_tp[i][:96], example_dir + "example" + str(i) + ".mut_types.predicted.pdf")
		if exposures_included:
			plot_exposures(sample_second_tp[i][96:tp_dim], example_dir + "example" + str(i) + ".exposures.predicted.pdf")

		plot_mut_types(train_samples[i,tp_dim:(tp_dim + 96)], example_dir + "example" + str(i) + ".mut_types.pdf")
		if exposures_included:
			plot_exposures(train_samples[i,(tp_dim + 96):(tp_dim*2)], example_dir + "example" + str(i) + ".exposures.pdf")



