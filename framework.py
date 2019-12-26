import os
import librosa
import librosa.display
import numpy as np
import sys
import glob
import random
import scipy
from scipy import signal


'''
hyperparameters define
'''
Net1TrainDatadir = "F://TIMIT/TIMIT/TIMIT/TRAIN/*/*/*.WAV"
Net1TestDatadir = "F://TIMIT/TIMIT/TIMIT/TEST/*/*/*.WAV"
Net2TrainDatadir = "D://study/programming/SLP/project/voice_conversion/datasets/LJSpeech-1.1/*.wav"
Net1Batchsize = 20

sr = 16000
frame_shift = 0.005
frame_length = 0.025
win_length = 400
hop_length = 80
n_fft = 512
preemphasis = 0.97
n_mfcc = 40
n_iter = 60 # Number of inversion iterations
n_mels = 80
duration = 2
max_db = 35
min_db = -55
phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

class model1():
    def __init__(self, config):
        '''
        `please initialize the network:
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
def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav

def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

def normalize(item, item_max, item_min):
    return (item - item_min)/(item_max - item_min)

def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):

    # Pre-emphasis
    y_preem = signal.lfilter([1, -preemphasis_coeff], [1], wav)
    # y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = librosa.amplitude_to_db(mag)
    mel_db = librosa.amplitude_to_db(mel)
    mfccs = np.array(librosa.feature.mfcc(y=y_preem, sr=sr, n_mfcc=n_mfcc, \
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels, ))
    # mfccs = np.array(np.dot(scipy.fftpack.dct() (n_mfcc, mel_db.shape[0]), mel_db))

    # Normalization (0 ~ 1)
    mag_db = normalize(mag_db, max_db, min_db)
    mel_db = normalize(mel_db, max_db, min_db)

    return mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)


def get_mfcc_and_phns(filename, trim=False, random_crop=True):
    # Load
    wav, _ = librosa.load(filename, mono=True, sr=sr, duration=duration)

    mfccs, _, _ = _get_mfcc_and_spec(wav, preemphasis, n_fft, win_length, hop_length)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    phn_file = filename.replace("WAV", "PHN")
    phn2idx, _ = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    with open(phn_file, 'r', encoding='UTF-8') as file:
        for line in file:
            start_point, _, phn = line.split()
            bnd = int(start_point) // hop_length
            phns[bnd:] = phn2idx[phn]
            bnd_list.append(bnd)

    # Trim
    if trim:
        start, end = bnd_list[1], bnd_list[-1]
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Random crop
    n_timesteps = (duration * sr) // hop_length + 1
    if random_crop:
        start = np.random.choice(range(np.maximum(1, len(mfccs) - n_timesteps)), 1)[0]
        end = start + n_timesteps
        mfccs = mfccs[start:end]
        phns = phns[start:end]
        assert (len(mfccs) == len(phns))

    # Padding or crop
    mfccs = librosa.util.fix_length(mfccs, n_timesteps, axis=0)
    phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

    return mfccs, phns

def get_filename(dir):
    '''
    get all the helpful file in dir
    '''
    wavfiles = glob.glob(dir)
    return wavfiles

def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    '''
    This is applied in `train2`, `test2` or `convert` phase.
    '''
    # Load
    wav, _ = librosa.load(wav_file, sr=sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)

    if random_crop:
        wav = wav_random_crop(wav, sr, duration)

    # Padding or crop
    length = sr * duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, preemphasis, n_fft, win_length, hop_length)
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
    dataflow = []
    data = {}
    # Net1Dataflow: x_mfcc, y_ppgs
    # Net2Dataflow: x_mcff, y_megDB, y_melDB

    if type == 'Net1Dataflow':
        filenames = get_filename(dir)
        for filename in filenames:
            data['x_mfcc'], data['y_ppgs'] = get_mfcc_and_phns(filename)
            dataflow.append(data)
    if type == 'Net2Dataflow':
        filenames = get_filename(dir)
        for filename in filenames:
            data['x_mfcc'], data['y_megDB'], data['y_melDB'] = get_mfccs_and_spectrogram(filename)
            dataflow.append(data)
    return dataflow

def conversion(dataflow):
    '''
    dataflow: one file MFCC
        apply net1 to get ppgs
        apply net2 to get mel scale / linear scale spectrogram
        restruction 
    output: wav file
    '''
    
if __name__ == "__main__":
    #train
    '''
    train1 
    '''
    config = []
    Net1Dataflow = dataflow_gen(Net1TrainDatadir, r'Net1Dataflow')
    
    # NetTrainer = trainer(config)
    # NetTrainer.trainNet1(Net1Dataflow)

    # Net2Dataflow = dataflow_gen(Net2TrainDatadir, r'Net2Dataflow')
    # NetTrainer.trainNet2(Net2Dataflow)

    '''
    evaluate train1
    '''

    
