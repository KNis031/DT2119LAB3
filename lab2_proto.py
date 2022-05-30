import numpy as np
from prondict import prondict
#from IPython.lib.display import Audio
#import matplotlib.pyplot as plt
import lab2_tools

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    M, D = np.shape(hmm1['covars'])
    K = sum([hmm['covars'].shape[0] for hmm in [hmm1, hmm2]])

    means = np.concatenate((hmm1['means'], hmm2['means']), axis=0)
    covars = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    startprob = np.zeros(K+1)
    startprob[:M] = hmm1['startprob'][:M]
    startprob[M:] = hmm1['startprob'][M] * hmm2['startprob'] #Flagga

    transmat = np.zeros( ((K+1), (K+1)) )
    transmat[:M, :M] = hmm1['transmat'][:M, :M]
    transmat[M:, M:] = hmm2['transmat']
    transmat[:M, M:] = np.array([hmm1['transmat'][:-1, -1]]).T @ np.array([hmm2['startprob']])

    out_dict = {"startprob":startprob, "transmat":transmat, "means":means, "covars":covars}

    return out_dict

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N,M = np.shape(log_emlik)
    forward_prob = np.zeros((N,M))
    forward_prob[0,:] = log_startprob[:-1] + log_emlik[0,:]

    for n in range(1,N):
        for j in range(M):
            forward_prob[n,j] = lab2_tools.logsumexp(forward_prob[n-1,:] + log_transmat[:-1,j]) + log_emlik[n,j]

    return forward_prob

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N,M = log_emlik.shape

    #print(log_emlik.shape) #(71, 9)
    #print(log_startprob.shape) #(10,)
    #print(log_transmat.shape) #(10, 10)

    V = np.zeros((N,M))
    B = np.zeros((N,M))
    viterbi_path = np.zeros(N) #viterbi_path

    V[0,:] = log_startprob[:-1] + log_emlik[0,:]

    for n in range(1,N):
        for j in range(M):
            states = [V[n-1,i] + log_transmat[i,j] for i in range(M)]
            V[n,j] = np.max(states) + log_emlik[n,j]
            B[n,j] = np.argmax(states)

    if forceFinalState:
        viterbi_loglik = V[-1,M-1]
        viterbi_path[-1] = M-1
    else:
        viterbi_loglik = np.max(V[-1,:])
        viterbi_path[-1] = int(np.argmax(V[-1,:])) #forceFinalState=False, we always start the path backtrack at the best state at the last time step
        
    for n in range(N-2,-1,-1):
        viterbi_path[n] = B[n+1, int(viterbi_path[n+1])]

    return viterbi_loglik, viterbi_path


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N,M = log_emlik.shape
    backward_prob = np.zeros((N,M))
    for n in range(N-2,-1,-1):
        for i in range(M):
            backward_prob[n,i] = lab2_tools.logsumexp(log_transmat[i,:-1] + log_emlik[n+1,:] + backward_prob[n+1,:])

    return backward_prob

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N,M = log_alpha.shape
    log_gamma = np.zeros((N,M))
    for n in range(N):
        for i in range(M):
            log_gamma[n,i] = log_alpha[n,i] + log_beta[n,i] - lab2_tools.logsumexp(log_alpha[-1,:])
        #print(np.exp(lab2_tools.logsumexp(log_gamma[n,:]))) #all rows ≈ 1
    return log_gamma


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    N = no of time steps / frames (86)
    D = no of lmfcc features (13)
    M = no of states (15)

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N, D = X.shape
    _, M = log_gamma.shape
    means = np.zeros((M,D))
    covars = np.zeros((M,D))

    gamma = np.exp(log_gamma)

    for j in range(M):
        num = np.zeros(D)
        den = 0
        for n in range(N):
            num += gamma[n,j] * X[n,:]
            den += gamma[n,j]
        means[j,:] = num/den

        num = np.zeros(D)
        for n in range(N):
            num += gamma[n,j] * (X[n,:]-means[j,:]) * (X[n,:]-means[j,:])
        covars[j,:] = num/den

    covars[covars < varianceFloor] = varianceFloor

    return means, covars

def create_wordHMMs(phoneHMMs):
    isolated = {}
    wordHMMs = {}
    for digit in prondict.keys():
        isolated[digit] = ['sil'] + prondict[digit] + ['sil']
        wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
    return wordHMMs