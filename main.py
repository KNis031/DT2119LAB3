import tensorflow as tf
from lab3_proto import *

def main():
    example = np.load('lab3_example.npz', allow_pickle=True)['example'].item()
    print(example.keys())

    state_list = enumerate_states()
    print(state_list)

    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    
    print(example['lmfcc'])
    print(lmfcc)

    print(np.max(example['lmfcc']-lmfcc))

if __name__ == '__main__':
    main()
    #TODO fortsätt från 4.1 
    # kolla om keras finns