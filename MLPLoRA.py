import torch
import numpy as np

import utils

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))


class MLPLoRALayer(torch.nn.Module):
    
    def __init__(self, fixed_weight, rank):
        super(MLPLoRALayer, self).__init__()
        input_dim, output_dim = fixed_weight.shape
        if isinstance(fixed_weight, np.ndarray):
            self.fixed_weight = torch.from_numpy(fixed_weight.astype(np.float32)).cuda()
        else:
            self.fixed_weight = fixed_weight.cuda()
        self.rank = rank
        self.A_param = torch.nn.Parameter(torch.randn(input_dim, rank) / np.sqrt(input_dim))
        self.B_param = torch.nn.Parameter(torch.randn(rank, output_dim) / np.sqrt(output_dim))
        
    def forward(self, x):
        fixed_emb = torch.matmul(x, self.fixed_weight)
        trainable_emb = torch.matmul(torch.matmul(x, self.A_param), self.B_param)
        return fixed_emb + trainable_emb
    
    def set_param_data(self, mat_data):
        self.A_param.data = torch.from_numpy(mat_data[0]).cuda() / np.sqrt(self.rank) 
        self.B_param.data = torch.from_numpy(mat_data[1]).cuda() / np.sqrt(self.rank)
    
    def get_param_data(self):
        return self.A_param.data * np.sqrt(self.rank) , self.B_param.data * np.sqrt(self.rank)
    
    def get_fixed_weight(self):
        return self.fixed_weight
    
    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """ 
        return f'MLPLoRALayer: ({self.fixed_weight.shape[0]}, rank), (rank, {self.fixed_weight.shape[1]})'
    
    
def get_mlp_lora(fixed_weights, ranks, activation):
    """
    fixed_weights: list of numpy 2d arrays.
    ranks: list of integers.
    activation: string.
    """
    modules = [torch.nn.Flatten()]
    lora_layers = []
    for layer_idx, f_weight in enumerate(fixed_weights[:-1]):
        lora_layer = MLPLoRALayer(f_weight, ranks[layer_idx])
        lora_layers.append(lora_layer)
        modules.extend([
            lora_layer,
            get_activation(activation),
        ])
    last_lora_layer = MLPLoRALayer(fixed_weights[-1], ranks[-1])
    modules.append(last_lora_layer)
    lora_layers.append(last_lora_layer)
    return torch.nn.Sequential(*modules), lora_layers


def fully_connected_net(input_dim, output_dim, widths, activation, bias=True):
    modules = [torch.nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else input_dim
        modules.extend([
            torch.nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(torch.nn.Linear(widths[-1], output_dim, bias=False))
    return torch.nn.Sequential(*modules)
        

def train(train_data, test_data, model, loss_fn, optimizer, batch_size, num_epochs, measurement, verbose=False):
    """
    train_data: input-output pair (X, Y) where X is n*p_1*...*p_d tensor, Y is n*o tensor.
    test_data: input-output pair similar to train_data
    """
    train_X, train_Y = train_data
    num_train_samples = train_X.size(dim=0)
    num_batches = int(np.ceil(num_train_samples / batch_size))
    
    for epoch_idx in range(num_epochs):
        
        for batch_idx in range(num_batches):
            
            # Batching training data
            if batch_idx < num_batches - 1:
                batch_X = train_X[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_Y = train_Y[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                batch_X = train_X[batch_idx * batch_size:]
                batch_Y = train_Y[batch_idx * batch_size:]

            # Backpropagation step
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X.cuda()), batch_Y.cuda())
            loss.backward()
            optimizer.step()
            if verbose:
                print(f'Current loss value is {loss.detach().cpu().numpy():.5f}.', end='\r')
        
        if measurement is not None:
            measurement.measure(train_data, test_data, model, epoch_idx)
                

def train_block_sub_lora(train_data, test_data, model_t, all_ranks, activation, loss_fn, opt_constr, batch_size,
                         num_com_rounds, num_epoch_per_round, measurement):
    """
    model_t is a tuple (model, lora_layers).
    all_ranks is a list of list. all_ranks has length num_models, with each element has length num_layer.
    """
    full_model, full_layers = model_t
    num_lora_layers = len(full_layers)
    num_models = len(all_ranks)
    fixed_weights = [layer.get_fixed_weight() for layer in full_layers]
    
    # initial submodel partition. Initialization does not matter.
    sub_models = []
    for model_idx in range(num_models):
        sub_model_t = get_mlp_lora(fixed_weights, all_ranks[model_idx], activation)
        sub_models.append((sub_model_t[0].cuda(), sub_model_t[1]))
    
    # Start simulated distributed training
    for com_round_idx in range(num_com_rounds):
        sub_model_params = [[] for _ in range(num_lora_layers)]
        
        for sub_model_t in sub_models:
            train(train_data, test_data, sub_model_t[0], loss_fn, opt_constr(sub_model_t[0].parameters()), batch_size, 
                  num_epoch_per_round, None)
            for layer_idx, sub_model_layer in enumerate(sub_model_t[1]):
                sub_model_params[layer_idx].append(sub_model_layer.get_param_data())
        
        # Extract weights and reassign weights.
        full_layer_params = []
        if com_round_idx < num_com_rounds - 1:
            for layer_idx, layer_params in enumerate(sub_model_params):
                new_sub_matrices, mats = utils.reassign_sub_matrices(layer_params)
                full_layer_params.append(mats)
                for m_idx, sub_model_t in enumerate(sub_models):
                    sub_model_t[1][layer_idx].set_param_data(new_sub_matrices[m_idx])
        else:
            for layer_params in sub_model_params:
                mat1 = np.concatenate([a.cpu().numpy() for a, _ in layer_params], axis=1)
                mat2 = np.concatenate([b.cpu().numpy() for _, b in layer_params], axis=0)
                full_layer_params.append((mat1, mat2))
        
        # Update full model weights and perform measurements.
        for layer_idx, full_layer in enumerate(full_layers):
            full_layer.set_param_data(full_layer_params[layer_idx])
        measurement.measure(train_data, test_data, full_model, (com_round_idx + 1) * num_epoch_per_round)
        
        

def train_block_sub_lora_multi(train_data, all_train_data, test_data, model_t, all_ranks, activation, loss_fn, 
                               opt_constr, batch_size, num_com_rounds, num_epoch_per_round, measurement):
    """
    train_data is a list with length num_models
    model_t is a tuple (model, lora_layers).
    all_ranks is a list of list. all_ranks has length num_models, with each element has length num_layer.
    """
    full_model, full_layers = model_t
    num_lora_layers = len(full_layers)
    num_models = len(all_ranks)
    fixed_weights = [layer.get_fixed_weight() for layer in full_layers]
    
    # initial submodel partition. Initialization does not matter.
    sub_models = []
    for model_idx in range(num_models):
        sub_model_t = get_mlp_lora(fixed_weights, all_ranks[model_idx], activation)
        sub_models.append((sub_model_t[0].cuda(), sub_model_t[1]))
    
    # Start simulated distributed training
    for com_round_idx in range(num_com_rounds):
        sub_model_params = [[] for _ in range(num_lora_layers)]
        
        for model_idx, sub_model_t in enumerate(sub_models):
            train(train_data[model_idx], test_data, sub_model_t[0], loss_fn, opt_constr(sub_model_t[0].parameters()), 
                  batch_size, num_epoch_per_round, None)
            for layer_idx, sub_model_layer in enumerate(sub_model_t[1]):
                sub_model_params[layer_idx].append(sub_model_layer.get_param_data())
        
        # Extract weights and reassign weights.
        full_layer_params = []
        if com_round_idx < num_com_rounds - 1:
            for layer_idx, layer_params in enumerate(sub_model_params):
                new_sub_matrices, mats = utils.reassign_sub_matrices(layer_params)
                full_layer_params.append(mats)
                for m_idx, sub_model_t in enumerate(sub_models):
                    sub_model_t[1][layer_idx].set_param_data(new_sub_matrices[m_idx])
        else:
            for layer_params in sub_model_params:
                mat1 = np.concatenate([a.cpu().numpy() for a, _ in layer_params], axis=1)
                mat2 = np.concatenate([b.cpu().numpy() for _, b in layer_params], axis=0)
                full_layer_params.append((mat1, mat2))
        
        # Update full model weights and perform measurements.
        for layer_idx, full_layer in enumerate(full_layers):
            full_layer.set_param_data(full_layer_params[layer_idx])
        measurement.measure(all_train_data, test_data, full_model, (com_round_idx + 1) * num_epoch_per_round)
