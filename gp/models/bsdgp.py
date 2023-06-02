import torch
import torch.nn as nn
import copy
import numpy as np

from scipy.cluster.vq import kmeans2

from ..conditionals import conditional


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Strauss(nn.Module):
    
    def __init__(self, gamma=0.5, R=0.5):
        super(Strauss, self).__init__()
        self.gamma = gamma
        self.R = R

    def _euclid_dist(self, X):
        Xs = torch.sum(torch.square(X), dims=-1, keepdims=True)
        dist = -2 * X @ X.T
        dist += Xs + Xs.T

        return torch.sqrt(torch.maximum(dist, 1e-40))
    
    def _get_Sr(self, X):
        """
        Get the # elements in distance matrix dist that are < R
        """
        dist = self._euclid_dist(X)
        val = torch.where(dist <= self.R)
        Sr = val.shape[0] # number of points satisfying the constraint above
        dim = dist.shape[0]
        Sr = (Sr - dim)/2  # discounting diagonal and double counts
        return Sr

    def log_prob(self, X):
        return self._get_Sr(X) * np.log(self.gamma)


class GPLayer(nn.Module):

    def __init__(self, X, kernel, outputs, n_inducing, fixed_mean,
                 full_cov, prior_type="uniform", prior_lengthscale=1.,
                 prior_variance=0.05, inducing_points_init=None):
        super(GPLayer, self).__init__()
        self.kern = kernel
        self.inputs = kernel.input_dim
        self.outputs = outputs
        self.M = n_inducing
        self.fixed_mean = fixed_mean
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.prior_lengthscale = prior_lengthscale
        self.prior_variance = prior_variance

        if inducing_points_init is not None:
            self.M = inducing_points_init.shape[0]

        else:
            # TODO: n_inducing_point option
            if torch.is_tensor(X):
                X = X.cpu().numpy()

            inducing_points_init = torch.tensor(
                kmeans2(X, self.M, minit='points')[0], dtype=torch.float64)

        if prior_type == "strauss":
            self.pZ = Strauss(R=0.5)
        
        self.Z = nn.parameter.Parameter(
            inducing_points_init.double(),
            requires_grad=True)

        self.U = nn.parameter.Parameter(
            torch.zeros((self.M, self.outputs), dtype=torch.float64),
            requires_grad=True)

        self.Lm = None

        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)
            self.mean = V[:self.outputs, :].T

        self.mean = torch.from_numpy(self.mean).double().to(device)
        self.mean.requires_grad = False
        
    def conditional(self, X):
        mean, var, self.Lm = conditional(X, self.Z, self.kern, self.U,
                                         whiten=True, full_cov=self.full_cov,
                                         return_Lm=True)
        if self.fixed_mean:
            mean += X @ self.mean

        return mean, var

    def log_prior_Z(self):
        if self.prior_type == "uniform":
            return 0.
        
        if self.prior_type == "normal":
            return -torch.sum(torch.square(self.Z)) / 2.0
        
        if self.prior_type == "strauss":
            return self.pZ.log_prob(self.Z)

        #if self.Lm is not None: # determinantal;
        if self.prior_type == "determinantal":
            self.Lm = torch.cholesky(self.kern.K(self.Z) + torch.eye(self.M, dtype=torch.float64, device=self.Z.device) * 1e-7)
            log_prob = torch.sum(torch.log(torch.square(torch.diagonal(self.Lm))))
            return log_prob
        
        else:
            raise Exception("Invalid prior type")

    def log_prior_hyper(self):
        log_lengthscales = torch.log(self.kern.lengthscales.get())
        log_variance = torch.log(self.kern.variance.get())

        log_prob = 0.
        log_prob += -torch.sum(torch.square(log_lengthscales - np.log(self.prior_lengthscale))) / 2.
        log_prob += -torch.sum(torch.square(log_variance - np.log(self.prior_variance))) / 2.

        return log_prob

    def log_prior_U(self):
        return -torch.sum(torch.square(self.U)) / 2.0

    def log_prior(self):
        return self.log_prior_U() + self.log_prior_Z() + self.log_prior_hyper()


def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = torch.cholesky(var + torch.eye(mean.shape[0]), dtype=torch.float64)[None, :, :] * 1e-7
        rnd = chol @ torch.randn_like(torch.transpose(mean), dtype=torch.float64)[:, :, None]
        rnd = torch.transpose(torch.squeeze(rnd))
        return mean + rnd
    return mean + torch.randn_like(mean, dtype=torch.float64) * torch.sqrt(var)


class BSDGP(nn.Module):
 
    def __init__(self, X, Y, kernels, likelihood, prior_type, output_dim,
                 n_data=None, n_inducing=None, inducing_points_init=None,
                 full_cov=False, prior_lengthscale=2.,
                 prior_variance=0.05, prior_lik_var=0.05):
        super(BSDGP, self).__init__()
        self.kernels = kernels
        self.likelihood = likelihood
        self.full_cov = full_cov
        self.prior_type = prior_type
        # self.inputs = inputs
        # self.outputs = outputs
        self.prior_lengthscale = prior_lengthscale
        self.prior_variance = prior_variance
        self.prior_lik_var = prior_lik_var
        
        if n_data is None:
            self.N = X.shape[0]
        else:
            self.N = n_data


        self.rand = lambda x: get_rand(x, full_cov)
        self.output_dim = output_dim or Y.shape[1]

        n_layers = len(kernels)

        X_running = copy.deepcopy(X)

        self.layers = []
        for l in range(n_layers):
            outputs = self.kernels[l+1].input_dim if l+1 < n_layers else self.output_dim#Y.shape[1]
            self.layers.append(GPLayer(X_running, self.kernels[l], outputs, n_inducing,
                                       fixed_mean=(l+1 < n_layers),
                                       full_cov=full_cov if l+1<n_layers else False,
                                       prior_type=prior_type,
                                       inducing_points_init=inducing_points_init if l == 0 else None))
            X_running = torch.matmul(X_running, self.layers[-1].mean)
        self.layers = nn.ModuleList(self.layers)

    
    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for l, layer in enumerate(self.layers):
            mean, var = layer.conditional(Fs[-1])
            if l+1 < len(self.layers):
                F = self.rand([mean, var])
            else:
                F = get_rand([mean, var], False)
                
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def predict(self, X):
        f, fmeans, fvars = self.propagate(X)
        y_mean, y_var = self.likelihood.predict_mean_and_var(fmeans[-1], fvars[-1])
        
        return y_mean, y_var

    def log_prior(self):
        res = 0.
        for gp_layer in self.layers:
            res += gp_layer.log_prior()
        return res


    def log_likelihood(self, X, Y):
        f, fmeans, fvars = self.propagate(X)
        log_likelihood = torch.sum(self.likelihood.predict_density(fmeans[-1], fvars[-1], Y))

        return log_likelihood

    def log_prob(self, X, Y):
        log_likelihood = self.log_likelihood(X, Y)
        log_prior = self.log_prior()

        batch_size = X.shape[0]

        log_prob = (self.N / batch_size) * log_likelihood + log_prior

        return log_prob
