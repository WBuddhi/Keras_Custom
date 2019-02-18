'''
    Learner:
    
    Accomplishes the following:
    Compiling Model:
        defaults: 
            Optimizer = 'Adam', 
            loss = 'mean_squared_error', 
            metrics = ['accuracy']
            loss_weight = None
            sample_weight_mode = None
            weighted_metrics = None
            target_tensor = None
    Fitting:
        defaults:
            x = None,
            y = None,
            batch_size = 32,
            validation_split = 0,
            validation_data = None,
            shuffle = True
        
        fit: default fit function

        LR Finder: assists in determining best learning rate 
        to start training.
        Addopted from fastai library

        fit_one_Cycle: to be implemented
        
        *Adopted from fastai library

'''

#   Keras imports:

from keras import callbacks
from keras import backend
from keras import optimizers
from keras import Model
from keras import backend as K

#   Other Imports

import matplotlib.pyplot as plt 
import numpy as np
from collections import *

class Learner:

    def __init__(self, Model, optimizer = 'Adam', loss = 'mean_squared_error', 
            metrics = ['accuracy'], loss_weights=None, sample_weight_mode=None,
            weighted_metrics=None, target_tensor=None, 
            x=None, y=None, batch_size=32, validation_split=0, 
            validation_data=None, shuffle=True):
        
        self.model = Model
        
        #   Compile model and ready for training
        self.model.compile(optimizer = optimizer, loss = loss, 
                metrics = metrics)

        #   Fit function parameters
        self.x = x
        self.y = y
        self.bz = batch_size
        self.val_split = validation_split
        self.val_data = validation_data
        self.shuffle = shuffle
    
    def fit(self, epoch = 1, lr = 0.01, decay = None, step_sz = None, 
            val_step_sz = None, cb = None):
        '''
        
        Default keras fit function:
            defaults:
                epoch = 1,
                lr = 0.01,
                step_sz (steps_per_epoch) = None,
                cb (callbacks) = None

            if step_sz is not None and validation_steps is None Then 
                validation_steps = step_sz

        '''


        if decay is not None:
            if hasattr(self.model.optimizer, 'decay'):
                K.set_value(self.model.optimizer.decay, float(decay))
        if step_sz != None:
            batch_size = None
        else: batch_size = self.bz
        if step_sz != None and val_step_sz is None: val_step_sz = step_sz
        K.set_value(self.model.optimizer.lr, lr)
        return self.model.fit(x = self.x, y = self.y, epochs = epoch, 
                batch_size = batch_size, validation_split = self.val_split, 
                validation_data = self.val_data, shuffle = self.shuffle,
                steps_per_epoch = step_sz, validation_steps = val_step_sz, 
                callbacks = cb)
    
    def LR_Find(self, start_lr = 1e-7, end_lr = 10, it_num=100, 
            stop_div = True, decay = None):
        '''
        
        This function aids in finding the best learning rate to start training.
        Using the lr_find callback function. The fit function is run for 1 
        epoch varing the learning rate exponentially from start_lr to end_lr 
        for it_num iterations. If stop_div is True (default) then fitting is 
        stopped if current loss in batch is 4 x greater than minimum loss.

        Defaults:
            start_lr = 1e-7
            end_lr = 10
            it_num = 100
            stop_div = True
            wd = (To be added)

        *Adapted from fastai library.
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        https://arxiv.org/abs/1506.01186
       
       '''
        
        lr_finder = lr_find(start_lr, end_lr, it_num, stop_div)
        self.model.save_weights('tmp.h5')
        LR_fit = self.fit(cb = [lr_finder], decay = decay)
        self.model.load_weights('tmp.h5')

        #   plot lr vs loss graph

        plt.plot(lr_finder.logs['lr'], lr_finder.logs['loss'])
        plt.show()


class lr_find(callbacks.Callback):

    '''
    
    Callback that aids to pick best learning rate. Progressively increase
    learning rate from start_lr till end_lr and monitors loss till it stops
    improving.
    Model will be reset after finding best lr

    # Arguments
        start_lr: Initial learning rate
        end_lr: Final learning rate
        it_num: number of steps per epoch
        stop_div: Stop fitting if loss is not improving

    Adopted from fastai library.
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    https://arxiv.org/abs/1506.01186
    
    '''


    def __init__(self, start_lr, end_lr, it_num, stop_div):
        
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.it_num = float(it_num)
        self.stop_div = stop_div
        self.min_loss = float(0)
        self.logs = defaultdict(list)
        
        #   Learning rate calculations
        
        k = np.log(self.end_lr/self.start_lr)/self.it_num
        exp_function = lambda x: self.start_lr * np.exp(k * x) 
        self.lr_range = [exp_function(i) for i in range(100)]
               
       
    def on_train_begin(self, logs):
        #   Initialisation

        K.set_value(self.model.optimizer.lr,self.start_lr)
        self.current_lr = self.start_lr
        
    def on_batch_end(self, batch, logs):

        #   Update values
        #       Current loss
        #       min_loss: min_loss = current_loss when Batch No. = 0
        #       Logs: loss, lr
        
        if 'loss' in logs: 
            self.current_loss = logs['loss']
        else: 
            AttributeError('loss not in log')
        if (batch == 0): self.min_loss = self.current_loss

        if 'loss' in logs:
            self.logs['loss'].append(self.current_loss)
            self.logs['lr'].append(self.current_lr)    

        #   Monitor loss
        
        if self.min_loss > self.current_loss:
            self.min_loss = self.current_loss
        
        #   Kill training if current learning rate is 4 > than min loss 
        
        if (batch == len(self.lr_range)) or (
                self.current_loss > (self.min_loss * 4) and self.stop_div) or ( 
                np.isnan(self.current_loss)):
            self.model.stop_training = True
        else:
            #  Update learning rate
            self.current_lr = self.lr_range[batch]
            K.set_value(self.model.optimizer.lr, self.current_lr)

    def on_epoch_end(self, epoch, logs):

        #   End training, prevent validation
        self.model.stop_training = True
        
