import os
import librosa
import librosa.display
import numpy as np
import sys

class model1():
    def __init__(self, config):
        '''
        please initialize the network:
            1. framework
            2. hyperparameters(included in config)
            3. the initial value of parameters
        '''
        pass

    def train(self, dataflow):
        '''
        define what kind of dataflow you need: 
            1. what's the number of columns and rows
            2. what does the columns and rows mean
        '''
        pass

class model2():
    def __init__(self, config):
        '''
        please initialize the network:
            1. framework
            2. hyperparameters(included in config)
            3. the initial value of parameters
        '''
        pass

    def train(self, dataflow):
        '''
        define what kind of dataflow you need: 
            1. what's the number of columns and rows
            2. what does the columns and rows mean
        '''
        pass

class trainer():
    def __init__(self, config):
        self.config = config

    def trainNet1(self, dataflow):
        net1 = model1(self.config) #if any config needed for initialization
        '''
        input: dataflow
        output: model paras
        '''
        net1.train(dataflow)
        self.model1 = net1

    def trainNet2(self, dataflow):
        net2 = model2(self.config) # if any config needed for initialization
        '''
        input: dataflow
            though net1 get ppgs
            train net2
        output: model paras
        '''
        self.model2 = net2
        

    def startTrainer(self, Net1Dataflow, Net2Dataflow):
        '''
        dose tensorflow's training process need any config?
        add config if needed 
        '''
        self.trainNet1(Net1Dataflow)
        self.trainNet2(Net2Dataflow)


'''
utils
'''
def get_delta(features):
    delta_feat = np.zeros([features.shape[0],features.shape[1]])
    win_num = features.shape[1]
    delta_feat[:,0] = features[:,1] - features[:,0]
    delta_feat[:,1] = features[:,2] - features[:,1]
    for i in range(2,win_num-2):
        delta_feat[:,i] = features[:,i+1] - features[:,i-1] +2*(features[:,i+2] - features[:,i-2])
    delta_feat[:,win_num-2] = features[:,win_num-2] - features[:,win_num-1]
    delta_feat[:,win_num-1] = features[:,win_num-1] - features[:,win_num-2]
    delta_feat[:,2:win_num-2] /= np.sqrt(10)
    return delta_feat

def features_extraction(filename):
    y, sr = librosa.load(filename, sr=16000)
    feat = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    # feat = (feat - np.mean(feat))/np.std(feat)
    first_order_delta = get_delta(feat)
    second_order_delta = get_delta(first_order_delta)
    features = np.r_[feat, first_order_delta, second_order_delta]
    return features.T

def get_filename(dir):
    '''
    get all the helpful file in dir
    '''

'''
generate dataflow
'''

def dataflow_gen(dir, type):
    '''
    input:
    type: Net1Dataflow or Net2Dataflow or Conversion dataflow
    dir: file dir

        get filename
        feature extraction
        aggregation

    output:
    dataflow
    '''

def conversion(dataflow):
    '''
    dataflow: one file MFCC
        apply net1 to get ppgs
        apply net2 to get mel scale / linear scale spectrogram
        restruction 
    output: wav file
    '''
