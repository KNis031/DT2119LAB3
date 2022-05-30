import numpy as np
from lab1_tools import *
from lab3_tools import *
from lab2_proto import *
from lab2_tools import *
from scipy import signal, fftpack, cluster
import os
from prondict import prondict
from tqdm import tqdm

def save_data(path):
    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    stateList = list(np.load('state_list.npz', allow_pickle=True)['state_list'])
    phones = sorted(phoneHMMs.keys())
    data = []
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                #...your code for feature extraction and forced alignment
                lmfcc, mspecs = mfcc(samples)

                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict)

                nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}

                utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

                stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

                obsloglik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
                viterbiLoglik, viterbiPath = viterbi(obsloglik, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

                viterbiStateTrans = [stateTrans[int(i)] for i in viterbiPath]

                targets = [stateList.index(ph) for ph in viterbiStateTrans]
                print(filename)
                print(len(lmfcc))
                data.append({'filename': filename, 'lmfcc': lmfcc,
                                'mspec': mspecs, 'targets': targets})
    return data

def split_data(path):
    """
    return:
    train: list of dicts containing 90% of data
    val: list of dicts containing 10% of data
    """
    train = []
    val = []
    data = np.load(path, allow_pickle=True)['traindata'] #8623 dicts
    men = []
    women = []
    for i in range(len(data)):
        split_path = data[i]['filename'].split('/')
        if split_path[4] == 'woman': #woman
            women.append(split_path[5])
        else:
            men.append(split_path[5])
        
    men = list(set(men))
    women = list(set(women))

    men_train = men[:int(len(men)*0.9)]
    women_train = women[:int(len(women)*0.9)]
    men_val = men[int(len(men)*0.9):]
    women_val = women[int(len(women)*0.9):]
    
    train_spkrs = men_train + women_train
    val_spkrs = men_val + women_val

    for i in range(len(data)):
        split_path = data[i]['filename'].split('/')
        if split_path[5] in train_spkrs:
            train.append(data[i])
        else:
            val.append(data[i])
    
    return train, val

def stack_acoustic_context_old(data):
    """
    takes a list of dicts
    returns a stacked version
    """
    for rec in data:
        N, feats = rec['lmfcc'].shape
        rec['lmfcc_stacked'] = np.zeros((N, feats, 7))
        for i in range(N):
            extended = []
            for k in range(-3, 4):
                index = abs(i+k)
                if index > (N-1):
                    index = (N-1) - (index%(N-1))
                extended.append(rec['lmfcc'][index])
            rec['lmfcc_stacked'][i] = np.array(extended).T

    return data

def stack_acoustic_context(data, param):
    """
    takes a list of dicts
    returns a stacked version
    """
    for rec in data:
        N, feats = rec[param].shape
        rec[param+'_stacked'] = np.zeros((N, feats, 7))
        for i in range(N):
            extended = []
            for k in range(-3, 4):
                index = abs(i+k)
                if index > (N-1):
                    index = (N-1) - (index%(N-1))
                extended.append(rec[param][index])
            rec[param+'_stacked'][i] = np.array(extended).T

    return data



def data_reshape(data, param, stacked):
    _, d = data[0][param].shape
    key = param+'_stacked' if stacked else param
    dim = d*7 if stacked else d

    data = [rec[key].reshape((rec[key].shape[0], dim)) for rec in data]
    data = np.array(np.concatenate(data, axis=0))

    return data

def targets_reshape(data):
    data = [rec['targets'] for rec in data]
    data = np.array(np.concatenate(data, axis=0))

    return data


def get_scalar(data, param, stacked):
    """
    inputs the data to standardize over
    returns the scalar used to standardize 
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data = data_reshape(data, param, stacked)
    scaler.fit(data)

    return scaler

def enumerate_states():
    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

    return stateList

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phone_list = []
    if addSilence:
        phone_list.append('sil')
    for word in wordList:
        [phone_list.append(w) for w in pronDict[word]]
        if addShortPause:
            phone_list.append('sp')
    if addSilence:
        phone_list.append('sil')
    
    return phone_list

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """

#### FROM LAB 1 ####

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff), mspecs #har lagt till mspecs här

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    sample_len = len(samples)
    
    N = int(np.floor((sample_len/(winlen-winshift))-1))
    
    windows = np.zeros((N,winlen))
    #print(windows.shape)
    for i in range(N):
        a = int(i * winshift)
        b = int(a + winlen)
        window = samples[a:b]
        windows[i,:] = window
      
    return windows
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame (91,400)
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    
    This is how the filter coefficients b and a are used:
    
    y[n] = b[0]*x[n] + b[1]*x[n−1] / a[0]
    """
    b = [1, -p]
    a = [1]
    
    return signal.lfilter(b, a, input)
    
def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N,M = input.shape
    ham = signal.hamming(M, sym=False)
    #plt.plot(ham) #looks like bell curve
    #plt.show()
    
    for i in range(N):
        input[i,:] *= ham

    return input

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    N,M = input.shape
    spec = np.zeros((N,nfft))
    
    for i in range(N):
        spec[i,:] = abs(fftpack.fft(input[i,:], nfft))**2
    
    return spec

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    N,nfft = input.shape    
    
    bank = trfbank(samplingrate,nfft)
    nmelfilters = np.shape(bank)[0]
    
    # Plotting the filterbank
    """
    for i in range(0,nmelfilters):
        melfilter = bank[i]
        freqs = np.linspace(0,samplingrate,nfft)
        plt.plot(freqs,melfilter) #plotting each filter of the bank in freq domain
    plt.show()
    """
    
    Melspec = np.zeros((N,))
    Melspec = np.matmul(input,bank.T) #[N x nfft]*[nfft x nmelfilter].T = [N x nfft]*[nmelfilter x nfft] = [N x nmelfilters]
    logMelspec = np.log(Melspec)
    
    return logMelspec

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    N, nmelfilters = input.shape
    
    cep_coeffs = np.zeros((N,nceps))
    for i in range(0,N):
        cep_coeffs[i] = fftpack.dct(input[i])[:nceps]
        
    return cep_coeffs