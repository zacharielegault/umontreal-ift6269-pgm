import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal
from collections import OrderedDict
from typing import Iterable


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
        use_relu: bool = True,
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
        self.use_relu = use_relu
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.use_log_variational = use_log_variational
        self.dispersion = dispersion

        self.z_encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_relu=self.use_relu,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

        # TODO: make sure parameters are ok here, esp. `hidden_dim` and `n_layers`
        self.l_encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            n_layers=1,
            dropout=self.dropout,
            use_relu=self.use_relu,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

        self.decoder = Decoder(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,  # TODO: Check if dropout should be 0
            use_relu=self.use_relu,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
        )

    def forward(self):
        pass

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

        px_scale, px_r, px_rate, px_dropout = self.decoder(z=z, library=library, decoder_dispersion=self.decoder_dispersion)

        return (px_scale,
                px_r,
                px_rate,
                px_dropout,
                qz_m,
                qz_v,
                z,
                ql_m,
                ql_v,
                library,)


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
                nn.BatchNorm1d(out_features) if use_batch_norm else None,
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
        layers_dim = [input_dim] + (n_layers-1)*[hidden_dim] + [output_dim]

        self.fc_layers = get_fc_layers(
            layers_dim=layers_dim,
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
        q = None
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    q = layer(x)

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
        layers_dim = [input_dim] + (n_layers-1)*[hidden_dim] + [output_dim]

        self.fc_layers = get_fc_layers(
            layers_dim=layers_dim[:-1],
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

    def forward(self, z, library, decoder_dispersion):
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if decoder_dispersion else None
        return px_scale, px_r, px_rate, px_dropout
