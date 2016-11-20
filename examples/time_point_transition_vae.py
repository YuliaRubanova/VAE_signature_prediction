# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.optimizers import adam
from data import *
from plotting_signatures import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os 
import pickle

model = "transition_time_point_VAE"

def diag_gaussian_log_density(x, mu, log_std):
	return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
	# Params of a diagonal Gaussian.
	D = np.shape(params)[-1] / 2
	mean, log_std = params[:, :D], params[:, D:]
	return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
	return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, unnormalized_logprobs):
	# unnormalized_logprobs are in R
	# Targets must be -1 or 1
	label_probabilities = -np.logaddexp(0, -unnormalized_logprobs*targets)
	return np.sum(label_probabilities, axis=-1)   # Sum across pixels.


def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
	"""Build a (weights, biases) tuples for all layers."""
	return [(scale * rs.randn(m, n),   # weight matrix
			 scale * rs.randn(n))      # bias vector
			for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
	mbmean = np.mean(activations, axis=0, keepdims=True)
	return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
	"""Params is a list of (weights, bias) tuples.
	   inputs is an (N x D) matrix.
	   Applies batch normalization to every layer but the last."""
	for W, b in params[:-1]:
		outputs = batch_normalize(np.dot(inputs, W) + b)  # linear transformation
		inputs = relu(outputs)                            # nonlinear transformation
	outW, outb = params[-1]
	outputs = np.dot(inputs, outW) + outb
	return outputs

def nn_predict_gaussian(params, inputs):
	# Returns means and diagonal variances
	return unpack_gaussian_params(neural_net_predict(params, inputs))

def generate_from_prior(gen_params, num_samples, noise_dim, rs):
	latents = rs.randn(num_samples, noise_dim)
	return neural_net_predict(gen_params, latents)

def p_data_given_latents(gen_params, labels, latents):
	pred_means, pred_log_stds = nn_predict_gaussian(gen_params, latents)
	return diag_gaussian_log_density(labels, pred_means, pred_log_stds)

def p_first_tp_given_latents(gen_params, labels, latents):
	pred_means, pred_log_stds = nn_predict_gaussian(gen_params, latents)
	pred_means_first_tp = pred_means[:,:tp_dim]
	pred_log_stds_first_tp = pred_log_stds[:,:tp_dim]
	return diag_gaussian_log_density(labels, pred_means_first_tp, pred_log_stds_first_tp)

def p_second_tp_given_latents(gen_params, labels, latents, tp_dim):
	pred_means, pred_log_stds = nn_predict_gaussian(gen_params, latents)
	pred_means_second_tp = pred_means[:,tp_dim:]
	pred_log_stds_second_tp = pred_log_stds[:,tp_dim:]
	return diag_gaussian_log_density(labels, pred_means_second_tp, pred_log_stds_second_tp)

def vae_lower_bound(gen_params, rec_params, data, rs, kl_weight, likelihood_given_latents = p_data_given_latents):
	# We use a simple Monte Carlo estimate of the KL
	# divergence from the prior.
	q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
	latents = sample_diag_gaussian(q_means, q_log_stds, rs)
	q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
	p_latents = diag_gaussian_log_density(latents, 0, 1)
	likelihood = likelihood_given_latents(gen_params, data, latents)
	kl_divergence = - np.mean(p_latents - q_latents)
	return np.mean(likelihood + (p_latents - q_latents) * kl_weight), np.mean(likelihood), kl_divergence, q_means, q_log_stds

def vae_lower_bound_first_tp(gen_params, rec_params, data, rs, kl_weight):
	return vae_lower_bound(gen_params, rec_params, data, rs, kl_weight, p_first_tp_given_latents)

def two_step_vae_predict(rec_params, gen_params, data, tp_dim):
	z_first_tp_means, z_first_tp_stds = nn_predict_gaussian(rec_params, data)

	# latents_first_tp = []
	# for i in range(100):
	# 	latents_first_tp.append(sample_diag_gaussian(z_first_tp_means, z_first_tp_stds, seed))

	#latents_first_tp = np.mean(np.array(latents_first_tp), axis=0)
	
	latents_first_tp = z_first_tp_means
	second_tp_means, second_tp_stds = nn_predict_gaussian(gen_params, latents_first_tp)

	#sample_second_tp = sample_diag_gaussian(second_tp_means, second_tp_stds, seed)
	sample_second_tp = second_tp_means[:,tp_dim:(tp_dim*2)]

	return(sample_second_tp, z_first_tp_means)

if __name__ == '__main__':
	# Model hyper-parameters
	latent_dim = 10

	if len(sys.argv) > 1:
		latent_dim = int(sys.argv[1])

	data_dim = 252  # How many pixels in each image (28x28).
	tp_dim = 126
	exposures_included = True
	
	#data_set = 'vae/training_transition_data_w_cancer_type.csv'
	data_set = 'vae/training_transition_data.only_mut_types.csv'
	annot_path = "vae/training_annotation.csv"

	if data_set == 'vae/training_transition_data_w_cancer_type.csv':
		model += "_w_cancer_type"
		data_dim += 1

	if data_set == 'vae/training_transition_data.only_mut_types.csv':
		model += "_only_mut_types"
		data_dim = 192
		tp_dim = 96
		exposures_included = False

	gen_layer_sizes = [latent_dim, 300, 200, data_dim * 2]
	rec_layer_sizes = [data_dim, 200, 300, latent_dim * 2]
	rec_layer_first_tp_sizes = [data_dim/2, 200, 300, latent_dim * 2]

	# Training parameters
	param_scale = 0.01
	batch_size = 200
	num_epochs = 1000
	step_size = 0.0005
	kl_weight_decay_time = 1

	print(data_dim)
	print(model)

	if not os.path.exists("vae/" + model + "/"):
		os.makedirs("vae/" + model + "/")

	if len(sys.argv) > 3:
		data_set = sys.argv[3]

	model_prefix = "vae/" + model + "/" + data_set[4:-4] + "/"
	if not os.path.exists(model_prefix):
		os.makedirs(model_prefix)

	model_prefix = model_prefix + model + "_tp."
	param_postfix = "latent_dim_" + str(latent_dim) + ".kl_weight_decay_time_" + str(kl_weight_decay_time) + ".pdf"

	log_file = model_prefix + param_postfix[:-4] + ".log.txt"

	if os.path.exists(log_file):
		os.remove(log_file)

	print("Loading training data...")
	N, train_samples, test_images, training_annot, test_annot = load_signature_data(data_set, annot_path)

	print(train_samples.shape)

	train_first_tp = train_samples[:,:tp_dim]
	print(train_first_tp.shape)

	# ==============================================================
	# Train first step: VAE on both time steps
	init_gen_params = init_net_params(param_scale, gen_layer_sizes)
	init_rec_params = init_net_params(param_scale, rec_layer_sizes)
	combined_init_params = (init_gen_params, init_rec_params)

	num_batches = int(np.ceil(len(train_samples) / batch_size))
	def batch_indices(iter):
		idx = iter % num_batches
		return slice(idx * batch_size, (idx+1) * batch_size)

	# Define training objective
	seed = npr.RandomState(0)
	def lower_bound_for_batch(combined_params, iter):
		data_idx = batch_indices(iter)
		gen_params, rec_params = combined_params
		kl_weight = min(1, iter / kl_weight_decay_time)

		lower_bound, likelihood, kl_divergence, q_means, q_log_stds = vae_lower_bound(gen_params, rec_params, train_samples[data_idx], seed, kl_weight)
		return lower_bound / data_dim, likelihood/data_dim, kl_divergence/data_dim, q_means, q_log_stds

	def objective(combined_params, iter):
		lower_bound, _, _, _, _ = lower_bound_for_batch(combined_params, iter)
		return - lower_bound

	# Get gradients of objective using autograd.
	objective_grad = grad(objective)

	elbo_list = []
	likelihood_list = []
	kl_divergence_list = []
	sample_from_prior = []

	print("     Epoch     |     Lower bound     |     Likelihood     |     KL divergence     ")
	def print_perf(combined_params, iter, grad):
		if iter % num_batches == 0:
			# plt.cla()

			gen_params, rec_params = combined_params
			lower_bound, likelihood, kl_divergence, q_means, q_log_stds = lower_bound_for_batch(combined_params, iter)
			bound = np.mean(lower_bound)
			print("{:15}|{:20} |{:20}| {:20}".format(iter//num_batches, bound, likelihood, kl_divergence))


			with open(log_file, "a") as log:
				log.write("{:15}|{:20} |{:20}| {:20}".format(iter//num_batches, bound, likelihood, kl_divergence))

			fake_data = generate_from_prior(gen_params, 20, latent_dim, seed)
			#print(fake_data[0])
			sample_from_prior.append(fake_data)

			#save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)

			elbo_list.append(lower_bound)
			likelihood_list.append(likelihood)
			kl_divergence_list.append(kl_divergence)

			# plt.draw()
			# plt.pause(1.0/60.0)

	# The optimizers provided can optimize lists, tuples, or dicts of parameters.
	optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
							num_iters=num_epochs * num_batches, callback=print_perf)


	with open(model_prefix + "optimized_params" + param_postfix + '.pickle', 'wb') as handle:
		pickle.dump(optimized_params, handle)

	plot_elbo(elbo_list, likelihood_list, kl_divergence_list, model_prefix + param_postfix)

	plot_mut_types(sample_from_prior[-1][0][:96], model_prefix + ".mutation_types.tp_1." + param_postfix)
	if (exposures_included):
		plot_exposures(sample_from_prior[-1][0][96:tp_dim], model_prefix + "exposures.tp_1." + param_postfix)

	plot_mut_types(sample_from_prior[-1][0][tp_dim:(tp_dim + 96)], model_prefix + ".mutation_types.tp_2." + param_postfix)
	if (exposures_included):
		plot_exposures(sample_from_prior[-1][0][(tp_dim + 96):(tp_dim*2)], model_prefix + "exposures.tp_2." + param_postfix)

	# ==============================================================
	# Train second step: VAE for recognition net on the first time step
	gen_layer_optimized, rec_layer_optimized = optimized_params
	z_labels = nn_predict_gaussian(rec_layer_optimized, train_samples)

	init_rec_params_first_tp = init_net_params(param_scale, rec_layer_first_tp_sizes)

	def lower_bound_for_batch(rec_params_first_tp, iter):
		data_idx = batch_indices(iter)
		kl_weight = min(1, iter / kl_weight_decay_time)

		lower_bound, likelihood, kl_divergence, q_means, q_log_stds = \
				vae_lower_bound_first_tp(gen_layer_optimized, rec_params_first_tp, train_first_tp[data_idx], seed, kl_weight)
		return lower_bound / data_dim, likelihood/data_dim, kl_divergence/data_dim, q_means, q_log_stds

	def objective_first_tp(rec_params_first_tp, iter):
		lower_bound, _, _, _, _ = lower_bound_for_batch(rec_params_first_tp, iter)
		return - lower_bound

	# Get gradients of objective using autograd.
	objective_first_tp_grad = grad(objective_first_tp)

	elbo_list = []
	likelihood_list = []
	kl_divergence_list = []
	sample_from_prior = []

    # Set up figure.
	# fig = plt.figure(figsize=(12,8), facecolor='white')
	# ax = fig.add_subplot(111, frameon=False)
	# plt.show(block=False)

	print("     Epoch     |     Lower bound     |     Likelihood     |     KL divergence     ")
	def print_perf_first_tp_net(rec_params_first_tp, iter, grad):
		if iter % num_batches == 0:
			# plt.cla()
			lower_bound, likelihood, kl_divergence, q_means, q_log_stds = lower_bound_for_batch(rec_params_first_tp, iter)
			bound = np.mean(lower_bound)
			print("{:15}|{:20} |{:20}| {:20}".format(iter//num_batches, bound, likelihood, kl_divergence))


			with open(log_file, "a") as log:
				log.write("{:15}|{:20} |{:20}| {:20}".format(iter//num_batches, bound, likelihood, kl_divergence))

			fake_data = generate_from_prior(gen_layer_optimized, 20, latent_dim, seed)
			#print(fake_data[0])
			sample_from_prior.append(fake_data)

			#save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)

			elbo_list.append(lower_bound)
			likelihood_list.append(likelihood)
			kl_divergence_list.append(kl_divergence)

			# plt.draw()
			# plt.pause(1.0/60.0)

	# The optimizers provided can optimize lists, tuples, or dicts of parameters.
	optimized_params_first_tp = adam(objective_first_tp_grad, init_rec_params_first_tp, step_size=step_size,
							num_iters=num_epochs * num_batches, callback=print_perf_first_tp_net)

	with open(model_prefix + "optimized_params_first_tp" + param_postfix + '.pickle', 'wb') as handle:
		pickle.dump(optimized_params_first_tp, handle)

	sample_second_tp, z_first_tp_means = two_step_vae_predict(optimized_params_first_tp, gen_layer_optimized, train_first_tp, tp_dim) 


	plot_elbo(elbo_list, likelihood_list, kl_divergence_list, model_prefix + "_vae_first_tp." + param_postfix)

	plot_mut_types(sample_second_tp[0,:96], model_prefix + ".mutation_types.predicted_tp_2." + param_postfix)
	if (exposures_included):
		plot_exposures(sample_second_tp[0,96:tp_dim], model_prefix + "exposures.predicted_tp_2." + param_postfix)

	plot_mut_types(train_samples[0,tp_dim:(tp_dim + 96)], "vae/" + model + "/reproduced_samples.mut_types.pdf")
	if (exposures_included):
		plot_exposures(train_samples[0,(tp_dim + 96):(tp_dim*2)], "vae/" + model + "/reproduced_samples.exposures.pdf")
