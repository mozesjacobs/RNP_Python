import torch
import torch.nn as nn


class Hypernet(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, use_bias=True):
        super(Hypernet, self).__init__()
        
        self.layers = []
        self.use_bias = use_bias
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim, bias=self.use_bias))
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.ELU())
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=self.use_bias))
                self.layers.append(nn.LayerNorm(hidden_dim))
                self.layers.append(nn.ELU())
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.layers(z)

class HyperRNN(nn.Module):
    def __init__(self, e_dim, z_dim, in_dim, hidden_dim, use_bias=True):
        super(HyperRNN, self).__init__()
        
        self.e_dim = e_dim
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.wz_shape = e_dim * z_dim
        self.wh_shape = z_dim * z_dim
        self.out_dim = self.wz_shape + self.wh_shape + z_dim + z_dim
        
        self.hypernet = HyperHead(self.in_dim, self.hidden_dim, self.out_dim, use_bias=use_bias)
        self.elu = nn.ELU()
        
    def init_net(self, z):
        params = self.hypernet(z) # b x (mat x mat x z x z)
        self.Wz = params[:, 0:self.wz_shape].view(-1, self.z_dim, self.e_dim)
        self.Wh = params[:, self.wz_shape:self.wz_shape + self.wh_shape].view(-1, self.z_dim, self.z_dim)
        curr = self.wz_shape + self.wh_shape
        self.bias = params[:, curr:curr + self.z_dim]
        self.h = params[:, curr + self.z_dim:]

    def forward(self, e):
        # forward
        z_comp = torch.bmm(self.Wz, e.unsqueeze(2)).squeeze()
        h_comp = torch.bmm(self.Wh, self.h.unsqueeze(2)).squeeze()
        h = self.elu(z_comp + h_comp + self.bias)
        self.h = h
        return h


class HyperMLP(nn.Module):
    def __init__(self, z_dim, decoder_hidden_dim, out_dim, in_dim, hidden_dim, use_bias=True):
        super(HyperMLP, self).__init__()
        
        self.z_dim = z_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.w1_dim = self.z_dim * self.decoder_hidden_dim
        self.w2_dim = self.decoder_hidden_dim * self.decoder_hidden_dim
        self.w3_dim = self.decoder_hidden_dim * self.out_dim
        self.w_hypernet = HyperHead(self.in_dim, self.hidden_dim, self.w1_dim + self.w2_dim + self.w3_dim, use_bias=use_bias)
        self.b_hypernet = HyperHead(self.in_dim, self.hidden_dim, self.decoder_hidden_dim + self.decoder_hidden_dim + self.out_dim, use_bias=use_bias)
        self.elu = nn.ELU()

        
    def init_net(self, z):
        ws = self.w_hypernet(z) # batch x (decoder_hid_dim * 2 + img_dim)
        bs = self.b_hypernet(z) # batch x (decoder_hid_dim * 2 + img_dim)
        self.W1 = ws[:, 0:self.w1_dim].view(-1, self.decoder_hidden_dim, self.z_dim)
        self.W2 = ws[:, self.w1_dim:self.w1_dim + self.w2_dim].view(-1, self.decoder_hidden_dim, self.decoder_hidden_dim)
        self.W3 = ws[:, self.w1_dim + self.w2_dim:].view(-1, self.out_dim, self.decoder_hidden_dim)
        
        self.B1 = bs[:, 0:self.decoder_hidden_dim]
        self.B2 = bs[:, self.decoder_hidden_dim:self.decoder_hidden_dim + self.decoder_hidden_dim]
        self.B3 = bs[:, self.decoder_hidden_dim + self.decoder_hidden_dim:]
        
        
    def forward(self, z):
        z = self.elu(torch.bmm(self.W1, z.unsqueeze(2)).squeeze() + self.B1)
        z = self.elu(torch.bmm(self.W2, z.unsqueeze(2)).squeeze() + self.B2)
        return torch.bmm(self.W3, z.unsqueeze(2)).squeeze() + self.B3

class HyperEncoder(nn.Module):
    def __init__(self, img_dim, a_dim, decoder_hidden_dim, out_dim, in_dim, hidden_dim, use_bias=True):
        super(HyperEncoder, self).__init__()
        
        self.img_dim = img_dim
        self.a_dim = a_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.img_dim, self.decoder_hidden_dim)
        self.fc2 = nn.Linear(self.a_dim, self.decoder_hidden_dim)
        self.fc3 = nn.Linear(self.decoder_hidden_dim + self.decoder_hidden_dim, self.decoder_hidden_dim)
        self.fc4 = nn.Linear(self.decoder_hidden_dim, self.out_dim)

        self.hypernet = HyperHead(self.in_dim, self.hidden_dim, self.decoder_hidden_dim * 3 + self.out_dim)

        self.elu = nn.ELU()

        
    def init_net(self, z):
        weights = self.hypernet(z)
        self.w1 = weights[:, 0:self.decoder_hidden_dim]
        self.w2 = weights[:, self.decoder_hidden_dim:self.decoder_hidden_dim * 2]
        self.w3 = weights[:, self.decoder_hidden_dim * 2:self.decoder_hidden_dim * 3]
        self.w4 = weights[:, self.decoder_hidden_dim * 3:]
        
        
    def forward(self, x, a):
        x = self.fc1(x)
        x = self.w1 * x
        x = self.elu(x)

        a = self.fc2(a)
        a = self.w2 * a
        a = self.elu(a)

        x_a = torch.cat((x, a), dim=1)

        z = self.fc3(x_a)
        z = self.w3 * z
        z = self.elu(z)

        z = self.fc4(z)
        z = self.w4 * z

        return z


class HyperHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_bias=True):
        super(HyperHead, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        
        self.hypernet = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim, bias=self.use_bias),
            #nn.LayerNorm(self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, z):
        return self.hypernet(z)