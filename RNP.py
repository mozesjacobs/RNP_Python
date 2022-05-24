import torch
import torch.nn as nn
import torch.nn.functional as F

from Hyper import *

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        
        self.z_dim = args.z_dim
        self.a_dim = args.a_dim
        self.img_side_dim = args.img_dim
        self.img_dim = args.img_dim * args.img_dim
        self.encoder_hidden_dim = args.encoder_hidden_dim
        self.hypernet_hidden_dim = args.hypernet_hidden_dim
        self.decoder_hidden_dim = args.hypernet_hidden_dim
        self.z_hyper_dim = args.hypernet_hidden_dim
        self.theta = args.theta
        self.e_dim = args.e_dim
        self.channels = 1
        
        
        # img encoder
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.channels, 16, 4, 2, 1, bias=False), # 28x28 -> 14x14
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, 4, 2, 1, bias=False), # 14x14 -> 7x7
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),  # 7x7 -> 7x7
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, 3, 1, 1, bias=True) # 7x7 -> 7x7
        )
        self.fc_encoder = nn.Sequential(
            nn.Linear(16 * 7 * 7, self.e_dim, bias=False),
            nn.BatchNorm1d(self.e_dim),
            nn.ELU(),
            nn.Linear(self.e_dim, self.e_dim, bias=False),
            nn.BatchNorm1d(self.e_dim),
            nn.ELU(),
            nn.Linear(self.e_dim, self.e_dim, bias=False),
            nn.BatchNorm1d(self.e_dim),
            nn.ELU(),
            nn.Linear(self.e_dim, self.z_dim)
        )
        
        use_bias=True
        
        # z hypernet
        self.hypernet_z = Hypernet(self.z_dim, self.hypernet_hidden_dim, self.z_hyper_dim, num_layers=6, use_bias=use_bias)
        
        # x0 and a0 
        self.init_patch = HyperHead(self.z_hyper_dim, self.hypernet_hidden_dim, self.img_dim, use_bias=use_bias)
        self.init_policy = HyperHead(self.z_hyper_dim, self.hypernet_hidden_dim, self.a_dim, use_bias=use_bias)

        # patch encoder
        self.patch_encoder = HyperEncoder(self.img_dim, self.a_dim, self.decoder_hidden_dim, self.e_dim, self.z_hyper_dim, self.hypernet_hidden_dim, use_bias=use_bias)

        # rnns
        self.rnn_state = HyperRNN(self.e_dim, self.z_dim, self.z_hyper_dim, self.hypernet_hidden_dim, use_bias=use_bias)
        self.rnn_policy = HyperRNN(self.e_dim, self.z_dim, self.z_hyper_dim, self.hypernet_hidden_dim, use_bias=use_bias)

        # x decoder and a decoder
        self.decoder_img = HyperMLP(self.z_dim, self.decoder_hidden_dim, self.img_dim, self.z_hyper_dim, self.hypernet_hidden_dim, use_bias=use_bias)
        self.decoder_policy = HyperMLP(self.z_dim, self.decoder_hidden_dim, self.a_dim, self.z_hyper_dim, self.hypernet_hidden_dim, use_bias=use_bias)

        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        self.hc1, self.hc2 = None, None
        # encode img
        z = self.encode(x)

        # use biggest hypernet
        z_hyper = self.hypernet_z(z)
        
        # initial x0 and a0
        x0 = self.sig(self.init_patch(z_hyper))
        a0 = torch.sin(self.init_policy(z_hyper))

        # initialize patch encoder
        self.patch_encoder.init_net(z_hyper)
        self.decoder_policy.init_net(z_hyper)
        self.decoder_img.init_net(z_hyper)

        # initialize rnns
        self.rnn_state.init_net(z_hyper) # works and no bias in layernorm
        self.rnn_policy.init_net(z_hyper) # works and no bias in layernorm

        # generate each patch
        z_ts, x_ts, a_ts, x_t_patches, x_orig_patches = [], [], [], [], []
        x_prev, a_prev = x0, a0
        for t in range(self.theta):
            # encode
            e = self.patch_encoder(x_prev, a_prev)

            # do rnns
            z_t = self.rnn_state(e)
            h_policy = self.rnn_policy(e)

            # get first output of decoders
            x_t = self.sig(self.decoder_img(z_t))
            a_t = torch.sin(self.decoder_policy(h_policy))
            
            # place x_t on grid using a_t
            the_offset = 3.0
            the_vec = torch.tensor([the_offset, 0, 0, 0, the_offset, 0], device=x.device)
            a_t_view = (a_t + the_vec).view(-1, 2, 3)
            x_t_view = x_t.view(-1, 1, self.img_side_dim, self.img_side_dim)
            grid = F.affine_grid(a_t_view, x_t_view.size(), align_corners=False)
            x_t_patch = F.grid_sample(x_t_view, grid, align_corners=False)
            
            # zoom into the ground truth patch at location a_t
            with torch.no_grad():
                x_t_orig_patch = self.zoom_in(x, a_t.detach().clone(), the_offset).squeeze()
            
            # store values
            z_ts.append(z_t)
            x_ts.append(x_t)
            a_ts.append(a_t)
            x_t_patches.append(x_t_patch)
            x_orig_patches.append(x_t_orig_patch)
            
            x_prev = x_t
            a_prev = a_t

        self.hc1, self.hc2 = None, None
            
        # stack
        z_ts = torch.stack(z_ts, dim=1)  # b x T x z
        a_ts = torch.stack(a_ts, dim=1)  # b x T x 6
        x_ts = torch.stack(x_ts, dim=1)  # b x T x 784 
        x_t_patches = torch.stack(x_t_patches, dim=1).view(-1, self.theta, self.img_dim) # b x T x 784
        x_orig_patches = torch.stack(x_orig_patches, dim=1).view(-1, self.theta, self.img_dim) # b x T x 784

        return z_ts, a_ts, x_ts, x_t_patches, x_orig_patches
    
    def encode(self, x):
        x = x.view(-1, 1, self.img_side_dim, self.img_side_dim)
        result = self.conv_encoder(x)
        result = result.view(x.size(0), -1)
        return self.fc_encoder(result)

    def sample_patch(self, x, thetas, offset=None):
        ximg = x.view(-1, 1, self.img_side_dim, self.img_side_dim)
        thetas_view = thetas.view(-1, 2, 3)
        transformed_grid = F.affine_grid(thetas_view, ximg.size(), align_corners=False)
        return F.grid_sample(ximg, transformed_grid, align_corners=False)
    
    def zoom_in(self, x, thetas, offset=3.0):
        scale_dims = [0, 4]
        thetas_new = thetas
        if offset is not None:
            thetas_new = thetas + offset
        sc_ = thetas_new[:,scale_dims]
        inv_scale = 1 / sc_
        thetas_new[:, scale_dims] = inv_scale
        translation_dims = [2, 5]
        thetas_new[:, translation_dims] = sc_ * thetas_new[:, translation_dims]
        return self.sample_patch(x, thetas_new, offset)