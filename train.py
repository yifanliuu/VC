'''
hyperparameters define
'''
Net1TrainDatadir = "F://datasets/TIMIT/TIMIT/TIMIT/TRAIN/*/*/*.WAV"
Net1TestDatadir = "F://datasets/TIMIT/TIMIT/TEST/*/*/*.WAV"

Net2TrainDatadir = "../arctic/slt/wav/*.wav"
Net1Batchsize = 20

sr = 16000
frame_shift = 0.005
frame_length = 0.025
win_length = 400
hop_length = 80
n_fft = 512
preemphasis = 0.97
n_mfcc = 40
n_iter = 60  # Number of inversion iterations
n_mels = 80
duration = 2
max_db = 35
min_db = -55
phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
numbers = 10
cost_value = 0.01
acc_value = 0.9

train2_hidden_units = 512
train2_dropout_rate = 0
train2_num_banks = 8
train2_norm_type = 'ins'
train2_num_highway_blocks = 16
train2_steps_per_epoch = 100
train2_num_gpu = 4


train1_t = 1.0
train1_hidden_units = 128  # E
train1_dropout_rate = 0.2
train1_num_banks = 8
train1_num_highway_blocks = 4
train1_norm_type = 'ins'
train1_lr = 0.0003
train1_num_epochs = 10000
train1_steps_per_epoch = 100


def train():
    pass


if __name__ == "__main__":
    train()
