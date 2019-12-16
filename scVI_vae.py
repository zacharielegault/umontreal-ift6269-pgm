import torch
import torch.nn as nn
from torch import cuda
from torch.distributions import Normal, kl_divergence
import collections
from torch import optim
from torch import autograd
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class scVI_vae(nn.Module) :
    def __init__(
        self,
        n_genes: int,
        #n_hidden_layers : int = 1,
        n_hidden_neurons: int = 128,
        n_latent_z: int = 10,
        dropout_rate: float = 0.1,
        ):
        super().__init__()
        # dispersion parameter : cst for all cells
        self.log_theta = torch.nn.Parameter(torch.randn(n_genes), requires_grad=True)


        self.enc_z = nn.Sequential(
            nn.Linear(n_genes, n_hidden_neurons, bias=True),
            nn.BatchNorm1d(n_hidden_neurons, momentum=0.03, eps=0.001),  # 0.01 and 0.001 are the default values used in scVI tests
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )
        self.enc_mu_z = nn.Linear(n_hidden_neurons, n_latent_z)
        self.enc_sigma_z = nn.Linear(n_hidden_neurons, n_latent_z)

        self.enc_l =  nn.Sequential(
            nn.Linear(n_genes, n_hidden_neurons, bias=True),
            nn.BatchNorm1d(n_hidden_neurons, momentum=0.01, eps=0.001),  # 0.01 and 0.001 are the default values used in scVI tests
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )
        self.enc_mu_l = nn.Linear(n_hidden_neurons, 1)
        self.enc_sigma_l = nn.Linear(n_hidden_neurons, 1)

        self.dec = nn.Sequential(
            nn.Linear(n_latent_z, n_hidden_neurons, bias=True),
            #nn.BatchNorm1d(n_hidden_neurons , momentum=0.01, eps=0.001),  0.01 and 0.001 are the default values used in scVI
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
        )

        # mean gamma
        self.rho_dec = nn.Sequential(
            nn.Linear(n_hidden_neurons, n_genes),
            nn.Softmax(dim=-1)
        )

        # dropout
        self.h_dec = nn.Linear(n_hidden_neurons, n_genes)

    def forward(self, x: torch.Tensor):
        fx_z = self.enc_z(x)
        mu_z = self.enc_mu_z(fx_z)
        sigma_z = torch.exp(self.enc_sigma_z(fx_z)) + 1e-4
        # It’s better to model log sigma as it is more numerically stable to take exponent compared to computing log
        qz = Normal(mu_z, sigma_z.sqrt())
        # Reparametrization trick
        z = qz.rsample()

        fx_l = self.enc_l(x)
        mu_l = self.enc_mu_l(fx_l)
        sigma_l = torch.exp(self.enc_sigma_l(fx_l)) + 1e-4
        # It’s better to model log sigma as it is more numerically stable to take exponent compared to computing log
        ql = Normal(mu_l, sigma_l.sqrt())
        # Reparametrization trick
        # Apply exp to library because it follows a log-normal dist
        l = torch.exp(ql.rsample())

        fx = self.dec(z)
        rho = self.rho_dec(fx)
        mu = rho * l
        h = self.h_dec(fx)
        
        return qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h

    def loss(self, x, qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h):
        z_prior = Normal(torch.zeros_like(mu_z), torch.ones_like(sigma_z))
        kl_z = kl_divergence(qz, z_prior).sum(dim=1)

        l_prior =  Normal(torch.zeros_like(mu_l), torch.ones_like(sigma_l))
        kl_l = kl_divergence(ql, l_prior).sum(dim=1)
        theta = torch.exp(self.log_theta)
        #reconst_loss = self.zinb_ll(x, h, theta, mu)
        reconst_loss = self.log_zinb_positive(x, mu, theta, h, eps=1e-8)
        
        # Upper bound of the negative log-likelihood (torch optimizer is a minimizer)
        losses_minibatch = - reconst_loss + kl_z + kl_l
        
        return losses_minibatch.mean()

    def log_zinb_positive(self, x, mu, theta, pi, eps=1e-8):
        """Paper's implementation of the zinb ll
        Note: All inputs are torch Tensors
        log likelihood (scalar) of a minibatch according to a zinb model.
        Notes:
        We parametrize the bernoulli using the logits, hence the softplus functions appearing
        Variables:
        mu: mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
        theta: inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
        pi: logit of the dropout parameter (real support) (shape: minibatch x genes)
        eps: numerical stability constant
        """
        theta = theta.view(1, theta.size(0)).to(device) # reshape theta

        softplus_pi = F.softplus(-pi)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

        res = mul_case_zero + mul_case_non_zero

        return res.sum(-1)

    def zinb_ll(self, x, h, theta, mu):
        """Our implementation of the zinb ll : currently not working
        Compute the log likelihood of the reconstructed data under the ZINB model
        :param x: minibatch (shape minibatch x n_genes)
        :param h: expected dropout (shape minibatch x n_genes)
        :param theta: dispersion (shape n_genes)
        :param mu: output of the network (shape minibatch x n_genes)
        :return: log p(x| z, l) (shape minibatch)
        """
        theta = theta.view(1, theta.size(0)).to(device) # reshape theta
        drop_case = - torch.log(1 + torch.exp(-h)) + torch.lgamma(x + theta) - torch.lgamma(theta + 1e-8) - torch.lgamma(x + 1) + theta * (torch.log(theta + 1e-8) - torch.log(theta + mu + 1e-8)) + x * (torch.log(mu + 1e-8) - torch.log(theta + mu + 1e-8))
        no_drop_case = - h - torch.log(1 + torch.exp(-h)) + theta * (torch.log(theta + 1e-8) - torch.log(theta + mu + 1e-8))

        return (torch.mul((x <= 1e-8).type(torch.float32), drop_case) + torch.mul((x > 1e-8).type(torch.float32), no_drop_case)).sum(-1)

    def infer_z_posterior(self,  dataset: torch.Tensor) :
        dataset = torch.tensor(dataset).to(device)
        
        fx_z = self.enc_z(dataset)
        mu_z = self.enc_mu_z(fx_z)
        sigma_z = torch.exp(self.enc_sigma_z(fx_z)) + 1e-4
        qz = Normal(mu_z, sigma_z.sqrt())
        z = qz.sample()

        return z


def train_scvi(model, train_set, val_set, n_batches = 32, n_epochs = 300, lr = 0.001, save_path="./models"):
    """
    Trains the model
    :param model: The model to train
    :param dataset: The raw dataset (to split in train and test sets and mini-batches)
    :return:
    """
    model.to(device)

    val_set = torch.tensor(val_set).to(device)

    adam = optim.Adam(model.parameters(), lr=lr)

    losses_train = []
    losses_val = []

    # training
    for epoch in range(n_epochs):
        #np.random.shuffle(train_set)
        train_set_shuff = torch.tensor(train_set).to(device)

        model.train()
        for i in range(int(len(train_set)/n_batches) + 1):
            minibatch = train_set_shuff[i * n_batches:(i+1) * n_batches, :]
            qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h = model(minibatch)  # forward pass
            loss_train = model.loss(minibatch, qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h)  # compute ELBO
            autograd.backward(loss_train, retain_graph=True)  # backward pass
            adam.step()  # paramters update
            adam.zero_grad()  # put the gradients back to zero for the next mini-batch
               
        model.eval()
        with torch.set_grad_enabled(False) :
            for i in range(int(len(val_set)/n_batches)):
                minibatch = val_set[i * n_batches:(i+1) * n_batches, :]
                qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h = model(minibatch)
                loss_val = model.loss(minibatch, qz, mu_z, sigma_z, ql, mu_l, sigma_l, mu, h)
        
        losses_train.append(loss_train)
        losses_val.append(loss_val)

    return losses_train, losses_val
