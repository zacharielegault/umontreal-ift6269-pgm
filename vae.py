import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from collections import OrderedDict
from typing import Iterable


# Function from https://github.com/YosefLab/scVI/blob/b5e84724e173602aba3112d4ab904bc4cd61f4a7/scvi/models/log_likelihood.py#L203
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
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

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

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

    return res



class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
        activation: nn.Module = nn.ReLU(),
        use_bias: bool = True,
        use_batch_norm: bool = True,
        use_log_variational = True,
        dispersion: str = "gene",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.use_log_variational = use_log_variational
        self.dispersion = dispersion

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(input_dim), requires_grad=True)
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError("dispersion must be one of 'gene' or 'gene-cell' but input was {}".format(self.dispersion)
            )

        self.z_encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            activation=self.activation,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

        self.l_encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            n_layers=1,  # The original implementation only uses 1 hidden layer
            dropout=self.dropout,
            activation=self.activation,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

        self.decoder = Decoder(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,  # TODO: Check if dropout should be 0
            activation=self.activation,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

    def forward(self, x: torch.Tensor):
        qz_m, qz_v, z = self.z_encoder(x)
        ql_m, ql_v, library = self.l_encoder(x)
        px_scale, px_r, px_rate, px_dropout = self.decoder(z=z, library=library, dispersion=self.dispersion)
        if self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)
        return px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library

    def loss(self, x, qz_m, qz_v, ql_m, ql_v, prior_l_m, prior_l_v, px_r, px_rate, px_dropout):
        prior_z_m = torch.zeros_like(qz_m)
        prior_z_v = torch.ones_like(qz_v)
        kl_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(prior_z_m, prior_z_v)).sum(dim=1)
        kl_l = kl(Normal(ql_m, ql_v.sqrt()), Normal(prior_l_m, prior_l_v.sqrt())).sum(dim=1)

        reconst_loss = log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)

        return torch.mean(- reconst_loss + kl_z + kl_l)

    def inference(self, x, n_samples=1):
        qz_m, qz_v, z = self.z_encoder(x)
        ql_m, ql_v, library = self.l_encoder(x)

        if n_samples > 1:
            def resize(x, n_samples):
                return x.unsqueeze(0).expand((n_samples, x.size(0), x.size(1)))

            qz_m = resize(qz_m, n_samples)
            qz_v = resize(qz_v, n_samples)
            z = Normal(qz_m, qz_v.sqrt()).sample()

            ql_m = resize(ql_m, n_samples)
            ql_v = resize(ql_v, n_samples)
            library = Normal(ql_m, ql_v.sqrt()).sample()

        px_scale, px_r, px_rate, px_dropout = self.decoder(z=z, library=library, dispersion=self.dispersion)

        return px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library


def get_fc_layers(
    layers_dim: Iterable[int],
    dropout: float,
    activation: nn.Module = nn.ReLU(),
    use_bias: bool = True,
    use_batch_norm: bool = True,
):
    return nn.Sequential(
            OrderedDict([("Layer {}".format(i), nn.Sequential(
                nn.Linear(in_features, out_features, bias=use_bias),
                nn.BatchNorm1d(out_features, momentum=0.03, eps=0.001) if use_batch_norm else None,
                activation,
                nn.Dropout(p=dropout) if dropout > 0 else None)
            ) for i, (in_features, out_features) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))])
        )


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        activation: nn.Module = nn.ReLU(),
        use_bias: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.layers_dim = [input_dim] + (n_layers)*[hidden_dim] + [output_dim]

        self.fc_layers = get_fc_layers(
            layers_dim=self.layers_dim[:-1],
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
        )

        self.mean_encoder = nn.Linear(hidden_dim, output_dim)
        self.var_encoder = nn.Linear(hidden_dim, output_dim)

    def reparameterize(self, mean, var):
        return Normal(mean, var.sqrt()).rsample()  # torch.distributions.Normal expects mean and standard deviation

    def forward(self, x: torch.Tensor):
        q = x
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    # print(layer._get_name(), q.shape)
                    q = layer(q)

        q_mean = self.mean_encoder(q)
        q_var = torch.exp(self.var_encoder(q))  # Add 1e-4?
        latent = self.reparameterize(q_mean, q_var)
        return q_mean, q_var, latent


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
        activation: nn.Module = nn.ReLU(),
        use_bias: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.layers_dim = [input_dim] + (n_layers)*[hidden_dim] + [output_dim]

        self.fc_layers = get_fc_layers(
            layers_dim=self.layers_dim[:-1],
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
            use_batch_norm=use_batch_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # dispersion
        self.px_r_decoder = nn.Linear(hidden_dim, output_dim)

        # ZI dropout
        self.px_dropout_decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, library: torch.Tensor, dispersion: str):
        px = z
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    # print(layer._get_name(), px.shape)
                    px = layer(px)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
