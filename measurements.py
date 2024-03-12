import numpy as np

import torch

class Measurement:
    
    def __init__(self, verbose=False):
        self.on_train_data = []
        self.on_test_data = []
        self.verbose=verbose
        self.recorders = {
            'MSE': MSERecorder,
            'Cross Entropy': CERecorder,
            'Accuracy': AccuracyRecorder,
            'Binary Accuracy': BinaryAccuracyRecorder,
            'Binary Cross Entropy': BCELossRecorder
        }
    
    def measure(self, train_data, test_data, model, epoch_idx):
        if self.verbose:
            print(f'Epoch #{epoch_idx}')
            print(f'  Metrics on training data:')
        for m in self.on_train_data:
            m.record(train_data, model, epoch_idx, verbose=self.verbose)
        if self.verbose:
            print(f'  Metrics on testing data:')
        for m in self.on_test_data:
            m.record(test_data, model, epoch_idx, verbose=self.verbose)
    
    def add_train_recorder_raw(self, recorder):
        self.on_train_data.append(recorder)
    
    def add_test_recorder_raw(self, recorder):
        self.on_test_data.append(recorder)
    
    def add_train_recorder(self, rec_name, phys_batch_size, every=1, verbose=False):
        self.add_train_recorder_raw(self.get_recorder_constr(rec_name)(phys_batch_size, every, verbose))
    
    def add_test_recorder(self, rec_name, phys_batch_size, every=1, verbose=False):
        self.add_test_recorder_raw(self.get_recorder_constr(rec_name)(phys_batch_size, every, verbose))
        
    def get_recorder_constr(self, rec_name):
        if rec_name in self.available_recorders():
            return self.recorders[rec_name]
        else:
            raise Exception(f'Recorder name {rec_name} not available.')
    
    def get_train_recorder(self):
        return self.on_train_data
    
    def get_test_recorder(self):
        return self.on_test_data
    
    def available_recorders(self):
        return list(self.recorders.keys())
        
        
class Recorder:
    
    def __init__(self, physical_batch_size, every=1, verbose=False):
        self.batch_size = physical_batch_size
        self.verbose=verbose
        self.every = every
        self.records = []
    
    def record(self, data, model, epoch_idx, verbose):
        if epoch_idx % self.every == 0:
            self.records.append((epoch_idx, self.compute(data, model)))
            if verbose and self.verbose:
                print(f"    {self.get_name()}: {self.records[-1][1]}.")
        
    def batching(self, data):
        X, Y = data
        num_samples = X.size(dim=0)
        num_batches = int(np.ceil(num_samples / self.batch_size))
        data_batches = []
        for batch_idx in range(num_batches):
            if batch_idx < num_batches - 1:
                batch_X = X[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                batch_Y = Y[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            else:
                batch_X = X[batch_idx * self.batch_size:]
                batch_Y = Y[batch_idx * self.batch_size:]
            data_batches.append((batch_X, batch_Y))
        return data_batches
        
    def compute(self, data, model):
        raise Exception('Method "compute" not implemented.')

    def get_name(self):
        raise Exception('Method "compute" not implemented.')
    
    def get_records(self):
        return self.records
    

class MSERecorder(Recorder):
    
    def __init__(self, physical_batch_size, every=1, verbose=False):
        super(MSERecorder, self).__init__(physical_batch_size, every=every, verbose=verbose)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += 0.5 * self.loss_fn(model(X.cuda()), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "MSE"


class CERecorder(Recorder):

    def __init__(self, physical_batch_size, every=1, verbose=False):
        super(CERecorder, self).__init__(physical_batch_size, every=every, verbose=verbose)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += self.loss_fn(model(X.cuda()), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "Cross Entropy"


class AccuracyRecorder(Recorder):
    
    def __init__(self, physical_batch_size, every=1, verbose=False):
        super(AccuracyRecorder, self).__init__(physical_batch_size, every=every, verbose=verbose)
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += (model(X.cuda()).detach().cpu().numpy().argmax(1) == Y.numpy()).astype(np.float32).sum()
        error /= data[0].size(dim=0)
        return error
    
    def get_name(self):
        return "Accuracy"
    

class BinaryAccuracyRecorder(Recorder):
    
    def __init__(self, physical_batch_size, every=1, verbose=False):
        super(BinaryAccuracyRecorder, self).__init__(physical_batch_size, every=every, verbose=verbose)
        self.sig_layer = torch.nn.Sigmoid()
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            prob_output = self.sig_layer(model(X.cuda())).detach().cpu().numpy()
            error += ((prob_output > 0.5).astype(np.float32) == Y.numpy()).astype(np.float32).sum()
        error /= data[0].size(dim=0)
        return error
    
    def get_name(self):
        return "Binary Accuracy"
    

class BCELossRecorder(Recorder):
    
    def __init__(self, physical_batch_size, every=1, verbose=False):
        super(BCELossRecorder, self).__init__(physical_batch_size, every=every, verbose=verbose)
        self.sig_layer = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss(reduction='sum')
    
    def compute(self, data, model):
        data_batches = self.batching(data)
        error = 0.
        for X, Y in data_batches:
            error += self.loss_fn(self.sig_layer(model(X.cuda())), Y.cuda())
        error /= data[0].size(dim=0)
        return error.detach().cpu().item()
    
    def get_name(self):
        return "Binary Cross Entropy"
    