import os
import torch

from torch import nn
from torch.nn import functional as F
from utils import get_all_files


class SGPBAE(nn.Module):

    def __init__(self, decoder, encoder, decoder_prior, optim_gp=True):
        super(SGPBAE, self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.decoder_prior = decoder_prior
        self.optim_gp = optim_gp

        self.decoder_samples = None
        self.gp = None
        self.loaded_samples = False

        self.Z = None
        self.Y_tilde = None

    def init_gp(self, gp):
        self.gp = gp

    def init_z(self, Y):
        Z, Y_tilde = self.encode(Y)
        Z = Z.detach()
        self.Y_tilde = Y_tilde.detach()

        if self.Z is None:
            self.Z = torch.autograd.Variable(
                Z.data, requires_grad=True).to(Y.device)
        else:
            self.Z.data = Z.data

    def get_parameters(self):
        params = list(self.decoder.net.parameters()) + \
                 list([self.Z])

        if self.optim_gp:
            params += list(self.gp.parameters())
        
        return params

    def predict(self, Y, randomness=False, get_mean=True):
        X_pred = None
        if randomness:
            X_pred = []
            for i in range(len(self.decoder_samples)):
                decoder_params, _ = self.load_samples(i)
                self.decoder_params = decoder_params
                
                X_pred.append(self.decoder(self.encoder(Y)))

            X_pred = torch.stack(X_pred, dim=0)
            if not get_mean:
                return X_pred
            
            X_pred = torch.mean(X_pred, dim=0)
        else:
            Z = self.encoder(Y)
            X_pred = self.decoder(Z)
        
        return X_pred

    def conditional_generate(self, X):

        if self.decoder_samples is None:
            Z = self.gp.predict(X.double())[0]
            Y = self.decoder.net(Z.float())
            return Y, None
        else:
            Y = []
            for i in range(len(self.decoder_samples)):
                decoder_params, gp_params = self.load_samples(i)
                self.decoder_params = decoder_params
                self.gp_params = gp_params

                Z = self.gp.predict(X.double())[0]
                Y.append(self.decoder.net(Z.float()))
            
            Y = torch.stack(Y, dim=0)
            Y_var = torch.var(Y, dim=0)
            Y = torch.mean(Y, dim=0)
            return Y, Y_var

    def encode(self, Y):
        Y_tilde = torch.cat((Y, torch.randn_like(Y)), 1)
        Z = self.encoder(Y_tilde)

        return Z, Y_tilde


    def decode(self, Z, randomness=False, return_std=False):
        Y = None
        if randomness and (self.decoder_samples is not None):
            Y = []
            for i in range(len(self.decoder_samples)):
                decoder_params, _ = self.load_samples(i)
                self.decoder_params = decoder_params

                Y.append(self.decoder(Z))
            
            Y = torch.stack(Y, dim=0)
            x_std = torch.std(Y, dim=0)
            Y = torch.mean(Y, dim=0)
            
            if return_std:
                return Y, x_std
        else:
            Y = self.decoder.net(Z)

        return Y

    def z_loss(self, Y):
        Y_tilde = self.Y_tilde
        Z = self.Z
        loss = F.mse_loss(self.encoder(Y_tilde), Z)

        return loss

    def log_prob(self, Y, X, n_data):
        n_batch = Y.shape[0]
        Z = self.Z

        log_lik = self.decoder.log_prob(Y, context=Z)
        log_lik = (n_data / n_batch) * torch.sum(log_lik)

        log_prior = 0.
        log_prior += torch.sum(self.decoder_prior.log_prob(self.decoder.net))

        gp_log_lik = 0.
        gp_log_lik += self.gp.log_prob(X.double(), Z.double())

        log_prob = log_lik + log_prior +  gp_log_lik

        return log_prob, log_lik, log_prior, gp_log_lik

    def load_samples(self, idx):
        decoder_params = None
        gp_params = None

        if self.loaded_samples:
            decoder_params = self.decoder_samples[idx]
            gp_params = self.gp_samples[idx]
        else:
            decoder_params =  torch.load(self.decoder_samples[idx])
            gp_params =  torch.load(self.gp_samples[idx])

        return decoder_params, gp_params

    def set_samples(self, sample_dir, cache=False):
        decoder_files = get_all_files(os.path.join(sample_dir, "decoder*"))
        gp_files = get_all_files(os.path.join(sample_dir, "gp*"))

        if cache:
            self.decoder_samples = []
            self.gp_samples = []

            for i in range(len(decoder_files)):
                self.decoder_samples.append(
                    torch.load(decoder_files[i]))
                self.gp_samples.append(
                    torch.load(gp_files[i]))
        else:
            self.decoder_samples = decoder_files
            self.gp_samples = gp_files
        
        self.loaded_samples = cache
        
    def save_sample(self, sample_dir, idx):
        torch.save(self.decoder_params,
            os.path.join(sample_dir, "decoder_{:03d}.pt".format(idx)))
        torch.save(self.gp_params,
            os.path.join(sample_dir, "gp_{:03d}.pt".format(idx)))
    
    @property
    def params(self):
        return self.state_dict()

    @params.setter
    def params(self, params):
        self.load_state_dict(params)

    @property
    def decoder_params(self):
        return self.decoder.net.state_dict()

    @decoder_params.setter
    def decoder_params(self, params):
        self.decoder.net.load_state_dict(params)

    @property
    def gp_params(self):
        return self.gp.state_dict()

    @gp_params.setter
    def gp_params(self, params):
        self.gp.load_state_dict(params)
