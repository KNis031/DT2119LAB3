import tensorflow as tf
import numpy as np
from lab3_proto import *
from prondict import prondict
from lab2_proto import *
from lab2_tools import *
import h5py

def example_func():
    example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()
    #print(example.keys())

    #state_list = enumerate_states()
    #np.savez('state_list.npz', state_list=state_list)
    states = list(np.load('state_list.npz', allow_pickle=True)['state_list'])

    filename = 'tidigits/disc_4.1.1/tidigits/train/man/ae/z541a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc, mspecs = mfcc(samples)
    print(lmfcc)
    print(lmfcc.shape)
    
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict)

    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}

    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

    #maximum a posetri sequence of hidden states
    obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    viterbiLoglik, viterbiPath = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

    print(example['viterbiLoglik'])
    print(viterbiLoglik)

    print(example['viterbiPath'])
    print(viterbiPath)

    #get the indicies right 
    viterbiStateTrans = [stateTrans[int(i)] for i in viterbiPath] 
    print(viterbiStateTrans)

    #apply transcript to utterence 
    trans = frames2trans(viterbiStateTrans, 'z541a.lab')

    # the viterbisequence (with fixed labels) will be our targets for our network
    targets = [states.index(ph) for ph in viterbiStateTrans]
    data = []
    data.append({'filename': filename, 'lmfcc': lmfcc,
                    'mspec': mspecs, 'targets': targets})
    np.savez('trialdata.npz', trialdata=data)

def write_test_meta(test):
    no_of_frames = np.array([rec['lmfcc'].shape[0] for rec in test])
    name = 'test_no_of_frames.npz'
    print(np.sum(no_of_frames))
    np.savez(name, no_of_frames=no_of_frames)

def prepare_data(param, stacked):
    """
    read from npz, split into train val
    Acoustic Context (Dynamic Features)
    Feature Standardisation 
    """
    #4.4 read and split data
    train, val = split_data('traindata.npz')
    test = np.load('testdata.npz', allow_pickle=True)['testdata']

    #5.1 we neeeded the no of frames per utterence information
    write_test_meta(test)

    #4.5 stack acoustic context (add new key, value pair to every dicts)
    if stacked:
        train = stack_acoustic_context(train, param)
        val = stack_acoustic_context(val, param)
        test = stack_acoustic_context(test, param)

    #4.6 feature standardisation
    scaler = get_scalar(train, param, stacked)

    train_x = scaler.transform(data_reshape(train, param, stacked))
    val_x = scaler.transform(data_reshape(val, param, stacked))
    test_x = scaler.transform(data_reshape(test, param, stacked))

    train_y = targets_reshape(train)
    val_y = targets_reshape(val)
    test_y = targets_reshape(test)

    train_x = train_x.astype('float32')

    from keras.utils import np_utils
    stateList = list(np.load('state_list.npz', allow_pickle=True)['state_list'])
    output_dim = len(stateList)
    train_y = np_utils.to_categorical(train_y, output_dim) #one hot encoding

    data = [train_x, val_x, test_x]
    targets = [train_y, val_y, test_y]

    return data, targets   

def write_data():
    """
    write data to npz file
    """
    #4.3 feature extraction and save to file
    train_path =  'tidigits/disc_4.2.1/tidigits/test' #'tidigits/disc_4.1.1/tidigits/train'
    traindata = save_data(train_path)
    print(len(traindata[0]['lmfcc']))
    print(traindata[0]['filename'])
    #np.savez('traindata.npz', traindata=traindata)

def to_file(param, stacked):
    data, targets = prepare_data(param, stacked)
    print(np.shape(data))
    print(np.shape(targets))
    #data = [data, targets]
    name = param+'_and_targets.npz' if stacked else param+'and_targets_single.npz'
    if param == 'mspec' and stacked == True:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(targets[0].shape)
        print(targets[1].shape)
        print(targets[2].shape)
        f1 = h5py.File(param+"_and_targets.hdf5", "w")
        f1.create_dataset("train_data", data[0].shape, dtype='float32', data=data[0])
        f1.create_dataset("val_data", data[1].shape, dtype='float32', data=data[1])
        f1.create_dataset("test_data", data[2].shape, dtype='float32', data=data[2])
        f1.create_dataset("train_targets", targets[0].shape, dtype='float32', data=targets[0])
        f1.create_dataset("val_targets", targets[1].shape, dtype='float32', data=targets[1])
        f1.create_dataset("test_targets", targets[2].shape, dtype='float32', data=targets[2])
        f1.close()
    else:
        np.savez(name, data=data)



def main():
    #example_func()
    # trial = np.load('trialdata.npz', allow_pickle=True)['trialdata']
    # print(trial[0].keys())
    # targ = trial[0]['targets']
    # states = list(np.load('state_list.npz', allow_pickle=True)['state_list'])
    # vitpath = [states[ta] for ta in targ]
    
    #to_file('lmfcc', stacked=True)
    #to_file('mspec',stacked=True)
    #to_file('lmfcc', stacked=False)
    #to_file('mspec', stacked=False)



    #final_to_file()

    #param = 'lmfcc'
    #prepare_data(param, stacked=True)
    write_data()


if __name__ == '__main__':
    main()