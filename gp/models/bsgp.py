import torch
import torch.nn as nn
import numpy as np

from scipy.cluster.vq import kmeans2

from ..conditionals import conditional, conditional2


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



class BSGP(nn.Module):
 
    def __init__(self, X, Y, kernel, likelihood, prior_type, inputs, outputs,
                 n_data=None, n_inducing=None, inducing_points_init=None, full_cov=False, prior_lengthscale=2.,
                 prior_variance=0.05, prior_lik_var=0.05):
        super(BSGP, self).__init__()
        self.kern = kernel
        self.likelihood = likelihood
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.inputs = inputs
        self.outputs = outputs
        self.prior_lengthscale = prior_lengthscale
        self.prior_variance = prior_variance
        self.prior_lik_var = prior_lik_var
        
        if n_data is None:
            self.N = X.shape[0]
        else:
            self.N = n_data
        self.M = n_inducing
        if inducing_points_init is not None:
            self.M = inducing_points_init.shape[0]
        else:
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

    def conditional(self, X):
        mean, var, self.Lm = conditional(X, self.Z, self.kern, self.U,
                                         whiten=True, full_cov=self.full_cov,
                                         return_Lm=True)
        return mean, var

    def predict(self, X):
        f_mean, f_var = self.conditional(X)
        y_mean, y_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        
        return y_mean, y_var

    def log_prior_hyper(self):
        log_lengthscales = torch.log(self.kern.lengthscales.get())
        log_variance = torch.log(self.kern.variance.get())
        log_lik_var = torch.log(self.likelihood.variance.get())

        log_prob = 0.
        log_prob += -torch.sum(torch.square(log_lengthscales - np.log(self.prior_lengthscale))) / 2.
        log_prob += -torch.sum(torch.square(log_variance - np.log(self.prior_variance))) / 2.
        log_prob += -torch.sum(torch.square(log_lik_var - np.log(self.prior_lik_var))) / 2.

        return log_prob

    def log_prior_U(self):
        return -torch.sum(torch.square(self.U)) / 2.0

    def log_prior(self):
        return self.log_prior_U() + self.log_prior_Z() + self.log_prior_hyper()

    def log_likelihood(self, X, Y):
        f_mean, f_var = self.conditional(X)
        log_likelihood = torch.sum(self.likelihood.predict_density(f_mean, f_var, Y))

        return log_likelihood

    def log_prob(self, X, Y):
        log_likelihood = self.log_likelihood(X, Y)
        log_prior = self.log_prior()

        batch_size = X.shape[0]

        log_prob = (self.N / batch_size) * log_likelihood + log_prior

        return log_prob



class BSGPTitsias(nn.Module):
 
    def __init__(self, X, Y, kernel, likelihood, prior_type, inputs, outputs,
                 n_data=None, n_inducing=None, inducing_points_init=None, full_cov=False, prior_lengthscale=2.,
                 prior_variance=0.05, prior_lik_var=0.05):
        super(BSGPTitsias, self).__init__()
        self.kern = kernel
        self.likelihood = likelihood
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.inputs = inputs
        self.outputs = outputs
        self.prior_lengthscale = prior_lengthscale
        self.prior_variance = prior_variance
        self.prior_lik_var = prior_lik_var
        
        if n_data is None:
            self.N = X.shape[0]
        else:
            self.N = n_data
        self.M = n_inducing
        if inducing_points_init is not None:
            self.M = inducing_points_init.shape[0]
        else:
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

    def conditional(self, X):
        mean, var, self.Lm = conditional(X, self.Z, self.kern, self.U,
                                         whiten=True, full_cov=self.full_cov,
                                         return_Lm=True)
        return mean, var
    
    def conditional2(self, X):
        mean, var, trace = conditional2(X, self.Z, self.kern, self.U,
                                         whiten=True, full_cov=self.full_cov,
                                         return_trace=True)
        return mean, var, trace

    def predict(self, X):
        f_mean, f_var = self.conditional(X)
        y_mean, y_var = self.likelihood.predict_mean_and_var(f_mean, f_var)
        
        return y_mean, y_var

    def log_prior_hyper(self):
        log_lengthscales = torch.log(self.kern.lengthscales.get())
        log_variance = torch.log(self.kern.variance.get())
        log_lik_var = torch.log(self.likelihood.variance.get())

        log_prob = 0.
        log_prob += -torch.sum(torch.square(log_lengthscales - np.log(self.prior_lengthscale))) / 2.
        log_prob += -torch.sum(torch.square(log_variance - np.log(self.prior_variance))) / 2.
        log_prob += -torch.sum(torch.square(log_lik_var - np.log(self.prior_lik_var))) / 2.

        return log_prob

    def log_prior_U(self):
        return -torch.sum(torch.square(self.U)) / 2.0

    def log_prior(self):
        return self.log_prior_U() + self.log_prior_Z() + self.log_prior_hyper()

    def log_likelihood(self, X, Y):
        f_mean, f_var, trace = self.conditional2(X)
        log_likelihood = torch.sum(self.likelihood.predict_density(f_mean, f_var, Y))
        trace = torch.sum(trace)

        return log_likelihood, trace

    def log_prob(self, X, Y):
        log_likelihood, trace = self.log_likelihood(X, Y)

        batch_size = X.shape[0]
        
        log_prob = (self.N / batch_size) * log_likelihood + trace

        return log_prob
