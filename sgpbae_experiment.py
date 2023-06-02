import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import absl.app

from torch.utils.data import DataLoader, TensorDataset
from utils.data import import_rotated_mnist, generate_init_inducing_points, plot_mnist
from utils.util import set_seed, ensure_dir, inf_loop
from utils.logger.logger import setup_logging
from nn.nets.mnist_nets import Encoder, Decoder
from priors.fixed_priors import PriorGaussian
from models.sgpbae import SGPBAE
from distributions import ConditionalMeanNormal
from samplers import AdaptiveSGHMC
from gp.models.bsgp import BSGP
from gp.kernels import MNISTKernel
from gp.likelihoods import Gaussian

import warnings
warnings.filterwarnings("ignore")

FLAGS = absl.app.flags.FLAGS

f = absl.app.flags

f.DEFINE_integer("seed", 1, "The random seed for reproducibility")
f.DEFINE_bool("train", True, "Whether or not to be in the training phase.")
f.DEFINE_string("out_dir", "./exp/mnist", "The path to the directory containing the experimental results")
f.DEFINE_string("data_path", "data/mnist/", "Path where rotated MNIST data is stored.")
f.DEFINE_string("dataset", "3", "")
f.DEFINE_integer("n_units", 500, "The number of hidden units")
f.DEFINE_integer("n_layers", 1, "The number of hidden layers")
f.DEFINE_integer("latent_size", 16, "The size of latent space")
f.DEFINE_float("lengthscale", 1., "Lengthscale of kernel for GP")
f.DEFINE_float("variance", 0.001, "Variance of kernel for GP")
f.DEFINE_float("likelihood_var", 0.00001, "Variance of Gaussian likelihood for GP")
f.DEFINE_bool("optim_gp", True, "Whether to optimize GP hyper-parameters")
f.DEFINE_integer("nr_inducing_points", 2, "Number of object vectors per angle.")
f.DEFINE_string("inducing_prior", "uniform", "Type of the prior on inducing inputs")
f.DEFINE_bool("PCA", True, "Use PCA embeddings for initialization of object vectors.")
f.DEFINE_integer("M", 8, "Dimension of GPLVM vectors.")
f.DEFINE_integer("batch_size", 512, "The mini-batch size")
f.DEFINE_integer("K", 30, "The number of MCMC steps for each update of the decoder and GP")
f.DEFINE_integer("J", 50, "The number of optimization steps for each update of the latent code")
f.DEFINE_float("decoder_var", 1., "Prior variance for the decoder")
f.DEFINE_float("lr", 0.005, "The learning rate for the sampler")
f.DEFINE_float("mdecay", 0.05, "The momentum for the sampler")
f.DEFINE_integer("n_burnin_iters", 500, "The number of burn-in iterations of the sampler")
f.DEFINE_integer("n_samples", 50, "The number of prior samples for each forward pass")
f.DEFINE_integer("collect_every", 10, "The thinning interval")

FLAGS(sys.argv)
set_seed(FLAGS.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup
BASE_DIR = FLAGS.out_dir
FIGS_DIR = os.path.join(BASE_DIR, "figures")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ensure_dir(FIGS_DIR)
ensure_dir(SAMPLES_DIR)
ensure_dir(LOG_DIR)

logger = setup_logging(LOG_DIR)

logger.info("====="*20)
for k, v in FLAGS.flag_values_dict().items():
    logger.info(">> {}: {}".format(k, v))

ending = FLAGS.dataset + ".p"

# Load data
train_data, eval_data, test_data = import_rotated_mnist(
    FLAGS.data_path, ending, FLAGS.batch_size)

inducing_points_init = generate_init_inducing_points(
    FLAGS.data_path + "train_data" + ending,
    n=FLAGS.nr_inducing_points, remove_test_angle=None,
    PCA=FLAGS.PCA, M=FLAGS.M)
inducing_points_init = torch.Tensor(inducing_points_init).double().to(device)

object_vectors_init = pickle.load(
     open(FLAGS.data_path + 'pca_ov_init{}.p'.format(FLAGS.dataset), 'rb'))
object_vectors_init = torch.Tensor(object_vectors_init).double().to(device)

# Create dataloaders
train_dataset = TensorDataset(torch.Tensor(train_data['images']).permute(0, 3, 1, 2).to(device),
                              torch.Tensor(train_data['aux_data']).to(device))
train_dataloader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size, shuffle=True,
                              drop_last=True)   

eval_dataset = TensorDataset(torch.Tensor(eval_data['images']).permute(0, 3, 1, 2).to(device),
                              torch.Tensor(eval_data['aux_data']).to(device))
eval_dataloader = DataLoader(eval_dataset,
                             batch_size=len(eval_dataset), shuffle=False,
                             drop_last=True)

test_dataset = TensorDataset(torch.Tensor(test_data['images']).permute(0, 3, 1, 2).to(device),
                             torch.Tensor(test_data['aux_data']).to(device))
test_dataloader = DataLoader(test_dataset,
                             batch_size=len(test_dataset), shuffle=False,
                             drop_last=True)

n_data = len(train_dataset)

# Initialize the encoder and decoder networks
encoder_net = Encoder(in_channels=2, latent_dim=FLAGS.latent_size)
encoder_net = encoder_net.to(device)

decoder_net = Decoder(FLAGS.latent_size)
decoder_net = decoder_net.to(device)

# Initialize decoder prior
decoder_prior = PriorGaussian(FLAGS.decoder_var)

# Initialize the Bayesian Autoencoder
encoder = encoder_net
decoder = ConditionalMeanNormal(decoder_net, scale=1) 

model = SGPBAE(decoder, encoder, decoder_prior, FLAGS.optim_gp)

# Sample a batch of data
data = next(iter(train_dataloader))
Y = data[0] # BCHW
X = data[1]

# Initialize the latent variable
model.init_z(Y)

# Initialize the GP Prior
kernel = MNISTKernel(input_dim=10, ARD=False,
                     variance=torch.tensor([FLAGS.variance], dtype=torch.double),
                     lengthscales=torch.tensor([1], dtype=torch.double),
                     period=torch.tensor([2*np.pi], dtype=torch.double),
                     object_vectors=object_vectors_init)
likelihood = Gaussian(variance=FLAGS.likelihood_var)

n_inducing = inducing_points_init.shape[1]
gp = BSGP(X, model.Z.double(),
          kernel, likelihood, FLAGS.inducing_prior,
          inputs=X.shape[-1], outputs=FLAGS.latent_size, n_data=n_data,
          inducing_points_init=inducing_points_init, full_cov=False)
gp.kern.period.requires_grad = False
gp.kern.variance.requires_grad = False
gp.likelihood.variance.requires_grad = False

gp = gp.to(device)

model.init_gp(gp)

# Train model
if FLAGS.train:

    # Initialize the sampler for BAE
    bae_sampler = AdaptiveSGHMC(model.get_parameters(),
                                lr=FLAGS.lr, num_burn_in_steps=2000,
                                mdecay=FLAGS.mdecay, scale_grad=n_data)

    # Initialize the optimizer for the encoder
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

    n_burnin_iters = FLAGS.n_burnin_iters
    collect_every = FLAGS.collect_every
    n_samples = FLAGS.n_samples
    n_sampling_iters = n_burnin_iters + n_samples * collect_every

    # Start sampling
    iter = 0
    sample_idx = 0  

    for data in inf_loop(train_dataloader):
        if iter > n_sampling_iters:
            break

        Y = data[0] # BCHW
        X = data[1]

        model.init_z(Y)

        # Sample the latent variables and decoder parameters
        for k in range(FLAGS.K):
            log_prob, log_lik, log_prior, gp_log_lik = model.log_prob(
                Y, X, n_data)
            bae_sampler.zero_grad()
            bae_loss = -log_prob
            bae_loss.backward()
            bae_sampler.step()
        
        # Update the encoder
        for j in range(FLAGS.J):
            encoder_optimizer.zero_grad()
            z_loss = model.z_loss(Y)
            z_loss.backward()
            encoder_optimizer.step()

        # Collect sample
        if (iter > n_burnin_iters) and (iter % collect_every == 0):
            model.save_sample(SAMPLES_DIR, sample_idx)
            sample_idx += 1
            model.set_samples(SAMPLES_DIR, cache=True)

        if iter % 10 == 0:
            logger.info("Iter: {}/{}, log_joint: {:.5f}, log_lik: {:.5f}, log_prior: {:.5f}, gp_log_lik: {:.5f}".format(
                iter, n_sampling_iters, log_prob.detach(), log_lik.detach(), log_prior.detach(), gp_log_lik.detach()))  
            logger.info("Iter: {}/{}, z_loss: {:.5f}".format(
                iter, n_sampling_iters, z_loss.detach()))

            # Evaluate conditional generatation on test set
            Ys = []
            Y_preds = []
            for Y, X in test_dataloader:
                Y_pred, _ = model.conditional_generate(X)

                Ys.append(Y)
                Y_preds.append(Y_pred)

            Ys = torch.cat(Ys, dim=0)
            Y_preds = torch.cat(Y_preds, dim=0)

            MSE = float(F.mse_loss(Ys, Y_preds).detach())
            logger.info("Iter: {}/{}, Test Conditional Generation MSE: {:.5f}".format(
                iter, n_sampling_iters, MSE))

            plot_mnist(Ys.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    Y_preds.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    "Iter: {}, Test MSE: {:.5f}".format(iter, MSE))
            plt.savefig(os.path.join(FIGS_DIR, "cgen_iter_{}.png".format(iter)))

            plot_mnist(Ys.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    Y_preds.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    "Iter: {}, Test MSE: {:.5f}".format(iter, MSE))
            plt.savefig(os.path.join(FIGS_DIR, "cgen.png".format(iter)))

        iter += 1

# Test model
model.set_samples(SAMPLES_DIR, cache=True)
 
# Evaluate conditional generatation on test set
Ys = []
Y_preds = []
Y_vars = []
for Y, X in test_dataloader:
    Y_pred, Y_var = model.conditional_generate(X)
    Ys.append(Y)
    Y_preds.append(Y_pred)
    Y_vars.append(Y_var)

Ys = torch.cat(Ys, dim=0)
Y_preds = torch.cat(Y_preds, dim=0)
Y_vars = torch.cat(Y_vars, dim=0)

# Report per pixel MSE
MSE = float(F.mse_loss(Ys, Y_preds).detach())
logger.info("Test Conditional Generation MSE: {:.5f}".format(MSE))

plot_mnist(Ys.permute(0, 2, 3, 1).detach().cpu().numpy(),
           Y_preds.permute(0, 2, 3, 1).detach().cpu().numpy(),
           "Test MSE: {:.5f}".format(MSE))
plt.savefig(os.path.join(FIGS_DIR, "cgen.png"))


plot_mnist(Ys.permute(0, 2, 3, 1).detach().cpu().numpy(),
           Y_vars.permute(0, 2, 3, 1).detach().cpu().numpy(),
           "Uncertainty Estimate")
plt.savefig(os.path.join(FIGS_DIR, "cgen_uncertainty.png"))


Ys = Ys.detach().cpu().numpy()
Y_preds = Y_preds.detach().cpu().numpy()
Y_vars = Y_vars.detach().cpu().numpy()

pickle.dump((Ys, Y_preds, Y_vars), open(os.path.join(BASE_DIR, "cgen_images.p"), "wb"))
torch.save(model.encoder, os.path.join(BASE_DIR, "encoder.pt"))