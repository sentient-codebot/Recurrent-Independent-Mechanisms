import torch
import torch.nn as nn
import math
from RIM import RIMCell, SparseRIMCell, OmegaLoss, LayerNorm, Flatten, UnFlatten, Interpolate
import numpy as np


class MnistModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if args['sparse']:
            self.rim_model = SparseRIMCell(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
                args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout'],
                args['a'], args['b'], args['threshold']).to(self.device)
            self.eta_0 = torch.tensor(args['a']+args['b']-2, device=self.device)
            self.nu_0 = torch.tensor(args['b']-1, device=self.device)
            self.regularizer = OmegaLoss(1, self.eta_0, self.nu_0) # 1 for now
        else:
            self.rim_model = RIMCell(self.device, args['input_size'], args['hidden_size'], args['num_units'], args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
                args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout']).to(self.device)
            

        self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 10)
        self.Loss = nn.CrossEntropyLoss()


    def to_device(self, x):
        return torch.from_numpy(x).to(self.device) if type(x) is not torch.Tensor else x.to(self.device)

    def forward(self, x, y = None):
        x = x.float()
        
        # initialize hidden states
        hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
        cs = None
        if self.args['rnn_cell'] == 'LSTM':
            cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)

        x = x.reshape((x.shape[0],-1))
        xs = torch.split(x, self.args["input_size"], 1)

        # pass through RIMCell for all timesteps
        for x in xs[:-1]:
            hs, cs, nu = self.rim_model(x, hs, cs)
        preds = self.Linear(hs.contiguous().view(x.size(0), -1))

        if y is not None:
            # Compute Loss
            y = y.long()
            probs = nn.Softmax(dim = -1)(preds)
            entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1)) # = -entropy
            loss = self.Loss(preds, y) - entropy # what? should be + entropy
            if self.args['sparse']:
                eta = self.eta_0 + y.shape[0] # eta_0 + N
                loss = loss + self.regularizer(eta, nu)
            return probs, loss
        return preds


    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.hidden_size = args['hidden_size']
        self.lstm = nn.LSTMCell(args['input_size'], self.hidden_size)
        self.Linear = nn.Linear(self.hidden_size, 10)
        self.Loss = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

    def to_device(self, x):
        return x.to(self.device)

    def forward(self, x, y = None):
        x = x.float()
        hs = torch.randn(x.size(0), self.hidden_size).to(self.device)
        cs = torch.randn(x.size(0), self.hidden_size).to(self.device) 

        x = x.reshape((x.shape[0],-1))
        xs = torch.split(x, self.args["input_size"], 1)
        for x in xs:
            # x_ = torch.squeeze(x, dim = 1)
            hs, cs = self.lstm(x, (hs, cs))
        preds = self.Linear(hs)
        if y is not None:
            y = y.long()
            probs = nn.Softmax(dim = -1)(preds)
            entropy = torch.mean(torch.sum(probs*torch.log(probs), dim = 1))
            loss = self.Loss(preds, y) - entropy
            return probs, loss
        return preds

    
    def grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

class CopyingModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args['cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.rim_model = RIM(self.device, args['input_size'], args['hidden_size'], args['num_units'],args['k'], args['rnn_cell'], args['key_size_input'], args['value_size_input'] , args['query_size_input'],
            args['num_input_heads'], args['input_dropout'], args['key_size_comm'], args['value_size_comm'], args['query_size_comm'], args['num_input_heads'], args['comm_dropout']).to(self.device)

        self.Linear = nn.Linear(args['hidden_size'] * args['num_units'], 9)
        self.Loss = nn.CrossEntropyLoss()
        
    def to_device(self, x):
        return torch.from_numpy(x).to(self.device)

    def forward(self, x, y = None):
        x = x.float()
        hs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)
        cs = None
        if self.args['rnn_cell'] == 'LSTM':
            cs = torch.randn(x.size(0), self.args['num_units'], self.args['hidden_size']).to(self.device)

        xs = torch.split(x, 1, 1)
        preds_ = []
        loss = 0
        loss_last_10 = 0
        for i,k in enumerate(xs):
            hs, cs = self.rim_model(k, hs, cs)
            
            preds = self.Linear(hs.contiguous().view(x.size(0), -1))
            preds_.append(preds)
            if y is not None:
                loss+=self.Loss(preds, y[:,i].squeeze().long())
                if i >= len(xs) - 10:
                    loss_last_10+=self.Loss(preds, y[:,i].squeeze().long())
        preds_ = torch.stack(preds_, dim = 1)
        if y is not None:
            loss/=len(xs)
            loss_last_10/=10
            return preds_, loss, loss_last_10
        return preds_

def sparse_loss(beta, gamma):
    # NOTE: loss is defined for BATCH. so it should be the average across the whole batch
    # beta = batch x K
    # gamma = 1x1
    if beta.dim() > 2:
        raise IndexError('expect beta to be (BatchSize, K)')
    loss_sum = -gamma*torch.sum(beta/(2*gamma*beta-gamma-beta+1)*torch.log(beta/(2*gamma*beta-gamma-beta+1)), dim=1)
    loss = torch.mean(loss_sum)
    return loss

class BallModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.hidden_size * args.num_units # NOTE dimension of encoded input. not clearly mentioned in paper
        self.output_size = args.hidden_size * args.num_units
        self.core = args.core.upper()

        self.Encoder = self.make_encoder().to(self.args.device)
        
        self.Decoder = None
        self.make_decoder()

        if self.core == 'RIM':
            self.rim_model = RIMCell(
                                    device=self.args.device,
                                    input_size=self.input_size, 
                                    num_units=self.args.num_units,
                                    hidden_size=self.args.hidden_size,
                                    k=self.args.k,
                                    rnn_cell='GRU', # defalt GRU
                                    input_key_size=self.args.input_key_size,
                                    input_value_size=self.args.input_value_size,
                                    input_query_size = self.args.input_query_size,
                                    num_input_heads = self.args.num_input_heads,
                                    input_dropout = self.args.input_dropout,
                                    comm_key_size = self.args.comm_key_size,
                                    comm_value_size = self.args.comm_value_size, 
                                    comm_query_size = self.args.comm_query_size, 
                                    num_comm_heads = self.args.num_comm_heads, 
                                    comm_dropout = self.args.comm_dropout
            ).to(self.args.device)
        elif self.core == 'GRU':
            self.rim_model = nn.GRU(
                                    input_size=self.input_size,
                                    hidden_size=self.args.hidden_size * self.args.num_units,
                                    num_layers=1,
                                    batch_first=True,
            ).to(self.args.device)
        elif self.core == 'LSTM':
            raise ValueError('LSTM Baseline Not Implemented Yet. ')
        else:
            raise ValueError('Illegal RNN Core')

    def make_encoder(self):
        """Method to initialize the encoder"""
        print(self.input_size)
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            Flatten(),
            nn.Linear(2304, self.input_size),
            nn.ELU(),
            LayerNorm(),
        )
    
    def make_decoder(self):
        """Method to initialize the decoder"""
        self.Decoder = nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(self.output_size, 4096),
            nn.ReLU(),
            LayerNorm(),
            UnFlatten(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        ).to(self.args.device)

    def forward(self, x, h_prev):
        ctx = None
        encoded_input = self.Encoder(x)

        if self.core=='RIM':
            h_new, foo, bar, ctx = self.rim_model(encoded_input, h_prev)
        elif self.core=='GRU':
            h_shape = h_prev.shape # record the shape
            h_prev = h_prev.reshape((h_shape[0],-1)) # flatten
            _, h_new = self.rim_model(encoded_input.unsqueeze(1), 
                                        h_prev.unsqueeze(0))
            h_new = h_new.reshape(h_shape)
        elif self.core=='LSTM':
            raise ValueError('LSTM core not implemented yet!')
        
        # --- here just for test    ---
        # module_mask = torch.tensor([1,1,1,0,0,0]).reshape(1,-1,1).to(self.args.device)
        # h_new = h_new*module_mask
        # --- above for test        ---
        dec_out_ = self.Decoder(h_new.view(h_new.shape[0],-1))
        
        intm = ctx

        """ 
        [
            "input_mask",
            # "decoder_activation"
        ]
        """
        return dec_out_, h_new, intm

    def init_hidden(self, batch_size): 
        # assert False, "don't call this"
        return torch.zeros((batch_size, 
            self.args.num_units, 
            self.args.hidden_size), 
            requires_grad=False)

    def dec_actv_old(self, h, num_module, threshold=1e-5):
        """check the contribution of the (num_module)-th RIM by seeing how much they contribute to the activaiton of first relu"""
        h = h.detach()
        module_mask = torch.zeros((1, self.args.num_units, 1))
        module_mask[:, num_module, :] = 1
        module_mask = module_mask.to(self.args.device)
        masked_h = h*module_mask

        relu_layer_0 = self.Decoder[0:4](h.view(h.shape[0],-1))
        relu_layer_0_masked = self.Decoder[0:4](masked_h.view(masked_h.shape[0],-1))
        
        total_contri = torch.sum(relu_layer_0) + 1e-6
        module_contri = torch.sum(relu_layer_0_masked) / total_contri

        return module_contri

    def nan_hook(self, out):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            raise RuntimeError(f"Found NAN in {self.__class__.__name__}: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def clamp(input_tensor):
    return torch.clamp(input_tensor, min=-1e6, max=1e6)

def main():
    gamma = 0.1
    K = 6
    beta = torch.rand(10,6)
    sparse_l = sparse_loss(beta, gamma)
    print(f'sparse regularization loss is {sparse_l}')

if __name__ == "__main__":
    main()

