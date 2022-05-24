import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


from cmd_line import parse_args
from trainer import train
from eval import test
from RNP import Net

def main():
    # Load args
    args = parse_args()

    # Load data
    transform = transforms.Compose([transforms.ToTensor()]) 
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=1)

    # paths
    cp_path = args.net_folder + "/" + args.session_name + "/"
    exp_path = args.exp_folder + "/" + args.session_name + "/"
    args_path = args.net_folder + "/" + args.session_name + "/args.txt"
    tb_path = args.tensorboard_folder + "/" + args.session_name + "/train"
    if not os.path.isdir(cp_path):
        os.system("mkdir -p " + cp_path)
    if not os.path.isdir(exp_path):
        os.system("mkdir -p " + exp_path)
    if args.print_folder == 1:
        print(cp_path)
        print(exp_path)
    cp_path += "checkpoint.pt"

    # boards
    train_board = SummaryWriter(tb_path, purge_step=True)
    #test_board = SummaryWriter(tb_path, purge_step=True)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()
    
    # net
    net = Net(args).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0

    # save args
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
        net.load_state_dict(checkpoint['model'])
        net.to(device)
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_e = checkpoint['epoch']
    else: 
        # init network
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        net.apply(init_weights)

    # train
    loss_func = nn.MSELoss(reduction='none')
    for epoch in tqdm(range(args.epochs), desc="Epoch", total=args.epochs, dynamic_ncols=True):
        # one train epoch
        train(net, device, train_loader, train_board, optim, epoch + initial_e, args.clip, loss_func)

        # test
        if (epoch + 1) % args.test_interval == 0:
            test(net, device, test_loader, exp_path + "grid_epoch_" + str(epoch + initial_e) + ".png")
        
        # adjust learning rate
        scheduler.step()

        # save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {'epoch': epoch + initial_e,
                          'model': net.state_dict(),
                          'optimizer': optim.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, cp_path)


if __name__ == "__main__":
    main()