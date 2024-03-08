from types import SimpleNamespace
# from RODAN.basecall import load_model
from rnamodif.models.generic import GenericUnlimited
import torch
# from rnamodif.models.modules import ConvNet, RNNEncoder, MLP, Attention, Permute, BigConvNet, ResConvNet
from rnamodif.models.modules import SimpleCNN, Permute, RNNEncoder, MLP , Attention
import torch.nn as nn
# from RODAN.basecall import load_model
# from RODAN.model import Mish
from types import SimpleNamespace

#TODO add dropout to cnns?
class CNN_RNN(GenericUnlimited):
    def __init__(self, 
                 cnn_depth, 
                 initial_channels=8, 
                 rnn_hidden_size=32,  
                 rnn_depth=1,
                 rnn_dropout=0.2,
                 mlp_hidden_size=30,
                 dilation=1,
                 padding=0,
                 **kwargs): 
        super().__init__(**kwargs)
        self.save_hyperparameters() #For checkpoint loading
        self.architecture = torch.nn.Sequential(
            SimpleCNN(num_layers=cnn_depth, dilation=dilation, padding=padding),
            Permute(),
            RNNEncoder(
                input_size=initial_channels*(2**(cnn_depth-1)), 
                hidden_size=rnn_hidden_size, 
                num_layers=rnn_depth, 
                dropout=rnn_dropout
            ),
            #biridectional (2x) and pooling is mean+max+last (3x)
            MLP(input_size=2*3*rnn_hidden_size, hidden_size=mlp_hidden_size),
        )
 
        
