import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# plots a grid of one example for each of the 10 digits
def test(net, device, test_loader, fig_path):
    net.eval()
    
    # get 1 example for each digit
    x_found = []
    x, y = next(iter(test_loader))
    digits = [i for i in range(10)]
    while len(digits) != 0:
        curr_digit = digits.pop(0)
        # keep iterating until we get a batch with the current digit
        while curr_digit not in y:
            x, y = next(iter(test_loader))
        # store the first x for the current digit
        x_curr_digit = x[y == curr_digit]
        x_found.append(x_curr_digit[0])

    # stack along batch dimension so shape is b x 1 x 28 x 28
    x = torch.stack(x_found, dim=0) 
    x = x.type(torch.FloatTensor).to(device)

    # forward
    _,_, x_ts, x_t_patches, x_orig_patches = net(x)

    # sum patches to create image
    x_hat = torch.sum(x_t_patches, dim=1)

    # plot
    fig, axes = plt.subplots(10 * 3, 5, figsize=(5, 20))
    for i in range(10):
        for j in range(4):
            axes[i * 3][j].imshow(x_t_patches[i][j].reshape(28, 28).detach().cpu().numpy(), cmap='gray')
            axes[i * 3 + 1][j].imshow(x_ts[i][j].reshape(28, 28).detach().cpu().numpy(), cmap='gray')
            axes[i * 3 + 2][j].imshow(x_orig_patches[i][j].reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        axes[i * 3][-1].imshow(x_hat[i].reshape(28, 28).detach().cpu().numpy(), cmap='gray')
        axes[i * 3 + 1][-1].imshow(np.zeros([28, 28]), cmap='gray')
        axes[i * 3 + 2][-1].imshow(x[i].reshape(28, 28).detach().cpu().numpy(), cmap='gray')
    plt.subplots_adjust(hspace=None)

    # save or show
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()
    