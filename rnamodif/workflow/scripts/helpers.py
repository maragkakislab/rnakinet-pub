# from rnamodif.models.model_uncut import RodanPretrainedUnlimited
# from rnamodif.models.model_mine import MyModel
from rnamodif.models.architectures import CNN_RNN

arch_map = {
    # 'rodan':RodanPretrainedUnlimited, 
    # 'cnn_gru':MyModel,
    'cnn_rnn':CNN_RNN,
}

print(arch_map)


