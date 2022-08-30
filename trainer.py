import torch
from tqdm import tqdm

def train(net, device, train_loader, train_board, optim, epoch, clip, loss_func):
    net.train()
    epoch_loss = 0
    for x, y in tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True):
        # forward
        x = x.type(torch.FloatTensor).to(device)
        _,_, x_ts, x_t_patches, x_orig_patches = net(x)
        x_hat = torch.sum(x_t_patches, dim=1)
        #x_hat = net(x)
        #x_hat = net(x)[:,-1]
        
        # loss
        batch_loss = loss_func(x_hat, x.view(-1, net.img_dim)).sum(1).mean()
        
        # backprop
        optim.zero_grad()
        batch_loss.backward()
        if clip is not None:
            # if we need to gradient clip
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optim.step()
        
        # store loss
        epoch_loss += batch_loss

    # tensorboard
    epoch_loss /= net.img_dim
    epoch_loss /= len(train_loader)
    train_board.add_scalar('Loss 1', epoch_loss, epoch)