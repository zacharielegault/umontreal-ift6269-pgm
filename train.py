import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch import optim
from torch import autograd

from vae import VAE
from clustering import plot_clustering

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_data: np.ndarray, valid_data: np.ndarray, batch_size: int, epochs: int, lr=0.001, verbose=1):
    model.to(DEVICE)

    X_train = torch.tensor(train_data).to(DEVICE)
    X_valid = torch.tensor(valid_data).to(DEVICE)

    adam = optim.Adam(model.parameters(), lr=lr)
    losses_train = []
    losses_val = []

    for epoch in range(epochs):
        if verbose == 1:
            print("Starting epoch", epoch+1)

        X_train = X_train[torch.randperm(X_train.size()[0])]  # Shuffle data at each epoch

        model.train()

        # Training loop
        for i in range(int(len(train_data)/batch_size) + 1):
            minibatch = X_train[i*batch_size:(i+1)*batch_size, :]
            px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library = model(minibatch)
            prior_l_m, prior_l_v = torch.zeros_like(qz_m), torch.ones_like(qz_v)  # TODO: Use correct prior
            loss_train = model.loss(minibatch, qz_m, qz_v, ql_m, ql_v, prior_l_m, prior_l_v, px_r, px_rate, px_dropout)

            if verbose == 1:
                print("Minibatch", i+1, "/", int(len(train_data)/batch_size) + 1, "loss", loss_train.item())

            autograd.backward(loss_train, retain_graph=True)
            adam.step()
            adam.zero_grad()

        # Validation step
        model.eval()
        with torch.set_grad_enabled(False):
            lv = []
            for i in range(int(len(valid_data)/batch_size) + 1):
                minibatch = X_valid[i*batch_size:(i+1)*batch_size, :]
                px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library = model(minibatch)
                prior_l_m, prior_l_v = torch.zeros_like(qz_m), torch.ones_like(qz_v)  # TODO: Use correct prior
                lv.append(model.loss(minibatch, qz_m, qz_v, ql_m, ql_v, prior_l_m, prior_l_v, px_r, px_rate, px_dropout))

        loss_val = np.mean(lv)
        if verbose == 1:
            print("Validation loss:", loss_val)

        losses_train.append(loss_train)
        losses_val.append(loss_val)

    return losses_train, losses_val


if __name__ == "__main__":
    x_train = np.load("./data/cortex_x_train.npy")
    y_train = np.load("./data/cortex_y_train.npy")
    x_test = np.load("./data/cortex_x_test.npy")
    y_test = np.load("./data/cortex_y_test.npy")

    model = VAE(
        input_dim=x_train.shape[1],
        hidden_dim=128,
        latent_dim=10,
        n_layers=1,
        dropout=0.2)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", pytorch_total_params)

    batch_size = 32
    epochs = 5
    losses_train, losses_val = train(model, train_data=x_train, valid_data=x_test, batch_size=batch_size, epochs=epochs)

    X_valid = torch.tensor(x_test).to(DEVICE)

    zz = np.zeros((x_test.shape[0], 10))
    model.eval()
    with torch.set_grad_enabled(False):
        lv = []
        for i in range(int(len(x_test) / batch_size) + 1):
            minibatch = X_valid[i * batch_size:(i + 1) * batch_size, :]
            qz_m, qz_v, z = model.z_encoder(minibatch)
            zz[i * batch_size:(i + 1) * batch_size, :] = z.numpy()

    z_tsne = TSNE(2).fit_transform(zz)
    plot_clustering(z_tsne, y_test[:,0])

    plt.plot(losses_train)
    plt.plot(losses_val)
    plt.show()
