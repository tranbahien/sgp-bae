import os
import sys
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import absl.app

from utils.util import set_seed, ensure_dir, inf_loop
from utils.logger.logger import setup_logging
from nn.nets import MLP
from priors.fixed_priors import PriorGaussian
from models.mask_sgpbae import MaskSGPBAE
from distributions import ConditionalMeanNormal
from samplers import AdaptiveSGHMC
from gp.kernels import RBF
from gp.models.bsdgp import BSDGP
from gp.likelihoods import Gaussian
from utils.dataset import TupleDataset
from utils.metric import mll, mae
from utils.data import load_jura


import warnings
warnings.filterwarnings("ignore")

FLAGS = absl.app.flags.FLAGS

f = absl.app.flags

f.DEFINE_integer("seed", 1, "The random seed for reproducibility")
f.DEFINE_string("out_dir", "./exp/jura", "The path to the directory containing the experimental results")
f.DEFINE_string("data_dir", "data/jura/", "Path where Jura data is stored.")
f.DEFINE_integer("K", 30, "The number of MCMC steps for each update of decoder and GP")
f.DEFINE_integer("J", 30, "The number of optimization steps for each update of latent code")
f.DEFINE_float("decoder_var", 1, "Prior variance for the decoder")
f.DEFINE_float("sigma", 0.5, "The scale of the decoder likelihood")
f.DEFINE_float("lr", 2e-3, "The learning rate for the sampler")
f.DEFINE_float("mdecay", 0.05, "The momentum for the sampler")
f.DEFINE_integer("batch_size", 100, "The mini-batch size")
f.DEFINE_integer("n_inducing", 128, "The number of inducing points")
f.DEFINE_integer("n_burnin_iters", 100, "The number of burn-in iterations of the sampler")
f.DEFINE_integer("n_samples", 50, "The number of prior samples for each forward pass")
f.DEFINE_integer("collect_every", 3, "The thinning interval")

FLAGS(sys.argv)

set_seed(FLAGS.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test data
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

train, test = load_jura(FLAGS.data_dir)

# Extract data into numpy arrays.
x = [[i, j] for (i, j) in train.index]
x = np.array(x)
y = np.array(train)

# Normalise observations.
y_mean, y_std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
y = (y - y_mean) / y_std

# Set up DataLoader.
x = torch.tensor(x).to(device)
y = torch.tensor(y).to(device).float()
dataset = TupleDataset(x, y, missing=True)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)

# Initialize the autoencoder
input_dim = y.shape[1]
latent_size = 2

encoder_net = MLP(input_dim *2, latent_size,
                  hidden_units=[20],
                  activation="relu")
decoder_net = MLP(latent_size, input_dim,
                  hidden_units=[5, 5],
                  activation="relu",
                  out_lambda=None)
encoder_net = encoder_net.to(device)
decoder_net = decoder_net.to(device)

encoder = encoder_net
decoder = ConditionalMeanNormal(decoder_net, FLAGS.sigma)

# Prior on the decoder
decoder_prior = PriorGaussian(FLAGS.decoder_var)

n_data = x.shape[0]

# Initialize the SGP-BAE model
model = MaskSGPBAE(decoder, encoder, decoder_prior, True)

# Sample a batch of data
data = next(iter(loader))
x_b = data[0]
y_b = data[1]

# Initialize latent variable models Z
model.init_z(y_b)

# Initialize Deep GP prior
likelihood = Gaussian(variance=1e-5)
kernels = []

n_inducing = FLAGS.n_inducing
n_layers = 3
for i in range(n_layers):
    output_dim = 196 if i >= 1 and x.shape[1] > 700 else x.shape[1]

    kernel = RBF(output_dim, ARD=True,
            variance=torch.tensor([1], dtype=torch.double),
            lengthscales=torch.tensor([0.1], dtype=torch.double))
    kernels.append(kernel)

gp = BSDGP(x.double(), model.Z.double(),
          kernels, likelihood, prior_type='uniform', output_dim=latent_size,
          n_data=n_data,  full_cov=False,
          n_inducing=n_inducing)

gp = gp.to(device)


model.init_gp(gp)

# Initialize the sampler
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

for data in inf_loop(loader):
    if iter > n_sampling_iters:
        break

    x_b, y_b, m_b, idx_b = data

    model.init_z(y_b)

    # Sample the latent variables and decoder parameters
    for k in range(FLAGS.K):
        log_prob, log_lik, log_prior, gp_log_lik = model.log_prob(y_b, x_b, m_b, n_data)
        bae_sampler.zero_grad()
        bae_loss = -log_prob
        bae_loss.backward()
        bae_sampler.step()
    
    # Update the encoder
    for j in range(FLAGS.J):
        encoder_optimizer.zero_grad()
        z_loss = model.z_loss(y_b)
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

        mean, var = model.predict(dataset.y)
        mean = mean.detach().cpu().numpy()
        mean = mean * y_std + y_mean

        pred = pd.DataFrame(mean, index=train.index,
                        columns=train.columns)

        mae_ = mae(pred, test).mean()

        if var is not None:
            var = var.detach().cpu().numpy()
            sigma = np.sqrt(var)
            sigma = sigma * y_std

            var = pd.DataFrame(sigma ** 2, index=train.index,
                            columns=train.columns)

            mll_ = mll(pred, var, test).mean()
            logger.info('MLL: {:.3f}'.format(mll_))

        logger.info('MAE: {:.3f}'.format(mae_))

    iter += 1
 