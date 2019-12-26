import os
import librosa
import librosa.display
import numpy as np
import sys
import glob
import random
import scipy
from scipy import signal
import tensorflow as tf

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
numbers =10
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
train1_hidden_units = 128 # E
train1_dropout_rate = 0.2
train1_num_banks = 8
train1_num_highway_blocks = 4
train1_norm_type = 'ins'
train1_lr = 0.0003
train1_num_epochs = 10000
train1_steps_per_epoch = 100


def gru(inputs, num_units=None, bidirection=False, seqlens=None, scope="gru", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            #inputs最后一维的维数
            num_units = inputs.get_shape().as_list[-1]
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            #双向RNN
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, 
                                                         sequence_length=seqlens,
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2) 
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs,  
                                           sequence_length=seqlens,
                                           dtype=tf.float32)
            return outputs

'''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
#宽度为k的Ck滤波器，一维卷积滤波器
def conv1d(inputs,filters=None,size=1,rate=1,padding="SAME",use_bias=False,activation_fn=None,
           scope="conv1d",reuse=None):
    with tf.variable_scope(scope):
        #转换为小写后
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            #填充函数，填充0
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        if filter is None :
            #input 最后一维的维数, 过滤器的个数
            filters = inputs.get_shape().as_list[-1]
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn, 
                "use_bias":use_bias, "reuse":reuse}
        #filters：过滤器的个数，kernel_size：卷积核的大小，卷积核的维度：filters*kernel_size
        #一维卷积
        outputs = tf.layers.conv1d(**params)
    return outputs




def normalize(inputs,type="bn",decay=.999,epsilon=1e-8,is_training=True, 
              reuse=None,activation_fn=None,scope="normalize"):
    if type=="bn":
        #inputs 的维度list
        inputs_shape = inputs.get_shape()
        #inputs 的维度
        inputs_rank = inputs_shape.ndims()
        if inputs_rank in [2, 3, 4]:
            if inputs_rank==2:
                #在axis轴处给input增加一个为1的维度。 
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank==3:
                inputs = tf.expand_dims(inputs, axis=1)
            #通过减少内部协变量加速神经网络的训练。
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,decay=decay,center=True,
                                               scale=True,updates_collections=None,
                                               is_training=is_training,scope=scope,
                                               zero_debias_moving_mean=True,fused=True,reuse=reuse)
            # restore original shape
            if inputs_rank==2:
                #删除[1,2]内的为1的维度
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif input_rank==3:
                #删除1内为1的维度
                outputs = tf.squeeze(outputs, axis=1)
        else:# fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,decay=decay,center=True,
                                               scale=True,updates_collections=None,is_training=is_training,
                                               scope=scope,reuse=reuse,fused=False)
    elif type in ("ln",  "ins"):
        reduction_axis = -1 if type=="ln" else 1
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            #最后一维存入params_shape
            params_shape = inputs_shape[-1:]
            #计算平均值和方差
            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            # beta = tf.Variable(tf.zeros(params_shape))
            beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer)
            # gamma = tf.Variable(tf.ones(params_shape))
            gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer)
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
    else:
        outputs = inputs
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs



'''
    Applies a series of conv1d separately.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is, 
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
    
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
'''
#size由1到k的一组卷积滤波器，输入的序列首先和k组一维卷积滤波器进行卷积，其中第k组包含宽度为k的Ck滤波器
def conv1d_banks(inputs, K=16, num_units=None, norm_type=None, is_training=True, scope="conv1d_banks", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = []
        for k in range(1, K+1):
            with tf.variable_scope("num_{}".format(k)):
                #对第k组一维卷积滤波器进行卷积
                output = conv1d(inputs, num_units, k)
                #tf.nn.relu：大于0的保持不变，小于0的数置为0
                output = normalize(output, type=norm_type, is_training=is_training, activation_fn=tf.nn.relu)
            outputs.append(output)
        #连接最后一维的张量
        outputs = tf.concat(outputs, -1)
    return outputs # (N, T, Hp.embed_size//2*K)


'''
    input：a 3d tensor of shape [N, T, hp.embed_size].
    is_training: A boolean.
    scope: Optional scope for `variable_scope`.  
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.
        
    Returns:
    A 3D tensor of shape [N, T, num_units/2].
'''
def prenet(inputs, num_units=None, dropout_rate=0., \
           is_training=True, scope="prenet", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        #tf.nn.relu ：将大于0的保持不变，小于零的数置为0
        #units：输出的维度大小，改变inputs的最后一维
        #tf.layer.dense：添加一层
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        #dropout：随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name="dropout2")
    return outputs # (N, T, num_units/2)


'''
    Highway networks, see https://arxiv.org/abs/1505.00387
    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3D tensor of shape [N, T, W].
'''
def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    if not num_units:
        #input 最后一维的维数
        num_units = inputs.get_shape()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        #tf.nn.sigmoid：计算x元素的sigmoid：y = 1/(1 + exp (-x))。
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C
    return outputs

#cbhg算法，修改了模型的泛化能力
def cbhg(input, num_banks, hidden_units, num_highway_blocks, norm_type='bn', is_training=True, scope="cbhg"):
    with tf.variable_scope(scope):
        #size由1到k的一组卷积滤波器，输入的序列首先和k组一维卷积滤波器进行卷积，其中第k组包含宽度为k的Ck滤波器
        out = conv1d_banks(input,K=num_banks,num_units=hidden_units,norm_type=norm_type,
                           is_training=is_training)  # (N, T, K * E / 2)
        
        #最大池化层：增加局部不变性，使结果更平滑
        out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (N, T, K * E / 2)
        
        #进一步将处理后的序列传入固定宽度的一维卷积层
        out = conv1d(out, hidden_units, 3, scope="conv1d_1")  # (N, T, E/2)
        out = normalize(out, type=norm_type, is_training=is_training, activation_fn=tf.nn.relu)
        out = conv1d(out, hidden_units, 3, scope="conv1d_2")  # (N, T, E/2)
        #将输出序列和原始序列相加
        out += input  # (N, T, E/2) # residual connections

        #输入进多层highway layers，高速公路网络可以让梯度更好地向前流动
        for i in range(num_highway_blocks):
           out = highwaynet(out, num_units=hidden_units,
                             scope='highwaynet_{}'.format(i))  # (N, T, E/2)
        #最后在顶部堆双向GRU RNN提取向前和向后的文本特征
        out = gru(out, hidden_units, True)  # (N, T, E)
    return out

def loss1(x_mfccs,logits,y_ppgs):
    istarget = tf.sign(tf.abs(tf.reduce_sum(x_mfccs, -1)))  # indicator: (N, T)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits / train1_t,
                                                              labels=y_ppgs)
    loss *= istarget
    loss = tf.reduce_mean(loss)
    return loss

def acc(x_mfccs,preds,y_ppgs):
    istarget = tf.sign(tf.abs(tf.reduce_sum(x_mfccs, -1)))  # indicator: (N, T)
    num_hits = tf.reduce_sum(tf.to_float(tf.equal(preds, y_ppgs)) * istarget)
    num_targets = tf.reduce_sum(istarget)
    acc = num_hits / num_targets
    return acc

def network1(x_mfcc, is_training):
    # Pre-net
    prenet_out = prenet(x_mfcc,num_units=[train1_hidden_units, train1_hidden_units // 2],
                 dropout_rate=train1_dropout_rate,is_training=is_training)  # (N, T, E/2)
    # CBHG
    out = cbhg(prenet_out, train1_num_banks, train1_hidden_units // 2,
               train1_num_highway_blocks, train1_norm_type, is_training)
    # Final linear projection
    logits = tf.layers.dense(out, len(phns))  # (N, T, V)
    ppgs = tf.nn.softmax(logits / train1_t, name='ppgs')  # (N, T, V)
    preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

    return ppgs, preds, logits


#x_mffcs：[batch_size, T, n_phns]:
def train1(inputs):    
    is_training = 1
    for k in range(numbers):
        x_mfccs, y_ppgs = inputs
        with tf.variable_scope('net1'):
            ppgs, preds, logits = network1(x_mfccs, is_training)
        cost = loss1(x_mfccs,logits,y_ppgs)
        acc = acc(x_mfccs,logits,y_ppgs)
        if cost<=cost_value and acc>=acc_value:
            print("train1 is over!")
            break
    outputs = ppgs 
    return outputs

def network2(ppgs, is_training,y_mel,y_spec):
    #pre-net
    prenet_out = prenet(ppgs,num_units=[train2_hidden_units, train2_hidden_units // 2],
                            dropout_rate=train2_dropout_rate,is_training=is_training)  # (N, T, E/2)
    # CBHG1: mel-scale
    pred_mel = cbhg(prenet_out, train2_num_banks, train2_hidden_units // 2,
                    train2_num_highway_blocks, train2_norm_type, is_training,scope="cbhg_mel")
    pred_mel = tf.layers.dense(pred_mel, y_mel.shape[-1], name='pred_mel')  # (N, T, n_mels)
    # CBHG2: linear-scale
    pred_spec = tf.layers.dense(pred_mel, train2_hidden_units // 2)  # (N, T, n_mels)
    pred_spec = cbhg(pred_spec, train2_num_banks, train2_hidden_units // 2,
                   train2_num_highway_blocks, train2_norm_type, is_training, scope="cbhg_linear")
    pred_spec = tf.layers.dense(pred_spec, y_spec.shape[-1], name='pred_spec')  # log magnitude: (N, T, 1+n_fft//2)

    return pred_spec,pred_mel

#计算损失函数,包括pred_spec和target的y_spec以及pred_mel和target的y_mel
def loss2(pred_spec,y_spec,pred_mel,y_mel):
    loss_spec = tf.reduce_mean(tf.squared_difference(pred_spec, y_spec))
    loss_mel = tf.reduce_mean(tf.squared_difference(pred_mel, y_mel))
    loss = loss_spec + loss_mel
    return loss

def train2(input,):
    is_training = 1
    for k in range(numbers):
        with tf.variable_scope('net2'):
            pred_spec, pred_mel = network(ppgs, is_training)
        cost = loss2(pred_spec, y_spec, pred_mel, y_mel)
        if cost<cost_value:
            print("train2 is ok!")
            break
    outputs = pred_spec,pred_mel
    return outputs

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

        is_training = 1
        for k in range(numbers):
            x_mfccs, y_ppgs = dataflow
            with tf.Session() as sess:
                sess.run(x_mfccs)
                sess.run(y_ppgs)
            
            outputs1 = network1(x_mfccs, is_training)
            with tf.Session() as sess:
                sess.run(outputs1)
            ppgs, preds, logits = outputs1
            with tf.Session() as sess:
                sess.run(ppgs)
                sess.run(preds)
                sess.run(logits)
            cost = loss1(x_mfccs,logits,y_ppgs)
            acc = acc(x_mfccs,logits,y_ppgs)
            with tf.Session() as sess:
                sess.run(cost)
                sess.run(acc)
            if cost<=cost_value and acc>=acc_value:
                print("train1 is over!")
                break
        outputs = ppgs
        with tf.Session() as sess:
            sess.run(outputs) 
        return outputs

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
        is_training = 1
        for k in range(numbers):
            pred_spec, pred_mel = network(ppgs, is_training)
            with tf.Session() as sess:
                sess.run(pred_spec)
                sess.run(pred_mel)
            cost = loss2(pred_spec,y_spec, pred_mel, y_mel)
            with tf.Session() as sess:
                sess.run(cost)
            if cost<cost_value:
                print("train2 is ok!")
                break
        outputs = pred_spec,pred_mel
        with tf.Session() as sess:
            sess.run(outputs)
        return outputs

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

def maxmin_normalize(item, item_max, item_min):
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
    mag_db = maxmin_normalize(mag_db, max_db, min_db)
    mel_db = maxmin_normalize(mel_db, max_db, min_db)

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

    
