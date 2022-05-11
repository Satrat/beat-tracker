# Coursework 1: Beat Tracker
# Music Informatics ECS7006P
# Sara Adkins 2022

import os
import librosa
import numpy as np
import pickle
from enum import Enum

#ballroom dance styles, can be used as input to beatTracker
class Style(Enum):
    CHACHA = 'ChaChaCha'
    JIVE = 'Jive'
    QUICK = 'Quickstep'
    RUMBA = 'Rumba'
    SAMBA = 'Samba'
    TANGO = 'Tango'
    VIENNESE = 'Viennese'
    WALTZ = 'Waltz'

#mean tempo for each style taken from Krebs 2013
tempo_dict = {}
tempo_dict[Style.CHACHA] = 60.0 / 125.0
tempo_dict[Style.JIVE] = 60.0 / 175.0
tempo_dict[Style.QUICK] = 60.0 / 205.0
tempo_dict[Style.RUMBA] = 60.0 / 100.0
tempo_dict[Style.SAMBA] = 60.0 / 105.0
tempo_dict[Style.TANGO] = 60.0 / 130.0
tempo_dict[Style.VIENNESE] = 60.0 / 180.0
tempo_dict[Style.WALTZ] = 60.0 / 82.0

#hyper-parameters
WINDOW_TIME = 0.023 #23ms, defined in Krebs 2013
HOP_TIME = 0.01 #10ms, defined in Krebs 2013
N_MELS = 82 #Bock 2012
F_MAX = 16000 #Bock 2012
LF_CUTOFF = 250 #Krebs 2013
MVG_AVG_TIME = 1.0 #Krebs 2013
LOG_SCALE = 1 #self-set
L_MIN = 1 #self-set
L_MAX = 501 #self-set
TAU0 = 0.5 #ellis07
STD_TAU = 0.9 #ellis07
WEIGHT = 500 #self-set

#onset detection function
#used pipeline defined in Krebs 2013 but left out the normalization
#details of algorithm adapted from Bock 2012, but used mel filterbank instead of const Q
def logFiltSpecFlux(x, window_len, hop_len, sr, n_mels=N_MELS, fmax=F_MAX, mel_cutoff=LF_CUTOFF, mel_low=True, mvg_avg_time=MVG_AVG_TIME):
    """
    Input:
      x: audio signal as a vector
      window_len: STFT window length in seconds
      hop_len: STFT hop size in seconds
      sr: sample rate of x
      n_mels: number of mel bins
      fmax: maximum mel frequency bin in Hz
      mel_cutoff: cutoff point between low and high frequency onsets, in Hz
      mel_low: whether to calculate low or high frequency onsets
      mv_avg_time: size of flux moving average window, in seconds
    Output:
      summed: vector containing estimated peaks in x, sampled every hop_len
      """
    mel = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=hop_len, win_length=window_len, n_mels=n_mels, fmax=fmax)
    mel_mag = np.abs(mel) #only care about magnitude
    mel_log = np.log(1 + LOG_SCALE * mel_mag) # scale magnitude logarithmically 
    
    # calculate which bin idx corresponds to the mel_cutoff frequency
    mel_bins = librosa.mel_frequencies(n_mels=n_mels, fmax=fmax, htk=False)
    lf_index = 0
    for i in range(len(mel_bins)):
        if mel_bins[i] > mel_cutoff:
            lf_index = i + 1
            break

    # calculate either the low frequency or high frequency onsets, depending on user input 
    if mel_low:
        mel_log = mel_log[0:lf_index]
    else:
        mel_log = mel_log[lf_index:len(mel_log)]
    
    # spectral flux calculation
    diff = np.maximum(mel_log[1:] - mel_log[:-1], 0)
    summed = np.sum(diff, axis=0)
    
    # calculate moving average of flux
    mv_avg_frames = int(mvg_avg_time * sr / hop_len)
    padding = int(mv_avg_frames / 2)
    summed_padded = np.concatenate((np.zeros(padding), summed, np.zeros(padding)))
    mvg_avg = np.zeros(len(summed))
    for i in range(len(summed)):
        mvg_avg[i] = np.sum(summed_padded[i:(i+(2*padding))]) / (padding * 2)

    summed = summed - mvg_avg # take the difference between each frame and moving average

    return np.maximum(summed, 0) # only care about positive changes

#weighted autocorrelation function defined in Ellis 07
def autocorrelate_weighted(onsets, hop_dur, tau_0=TAU0, std_tau=STD_TAU, l_min=L_MIN, l_max=L_MAX):
    """
    Input:
      onsets: vector of estimated onsets, ie output of ODF
      hop_dur: STFT hop size in seconds that was used to calculate onsets
      tau_0: tempo period prior, in sec
      std_tau: std deviation of tempo prior, in octaves
      l_min: minimum tempo lag to calculate, in BPM
      l_max: maximum tempo lag to calculate, in BPM
    Output:
      autocorrelation: vector of floats, autocorrelation metric for each lag l-min to l-max
      """
    autocorrelation = np.zeros(l_max - l_min)
    for l in range(l_min, l_max): # calculate autocorrelation for each tempo lag
        inner = np.inner(onsets[:-l], onsets[l:]) # autocorrelation
        weight = np.exp(-0.5 * pow(np.log2(l * hop_dur / tau_0) / std_tau, 2.0)) # log-based gaussian weighting
        autocorrelation[l - l_min] = weight * inner
    
    return autocorrelation

#find best tempo fit defined as non-zero lag with highest peak 
def tempo_from_auto(autocorrelation, hop_dur, l_min=L_MIN, l_max=L_MAX):
    """
    Input:
      autocorrelation: vector of floats, autocorrelation metric for each lag l-min to l-max
      hop_dur: hop size in seconds between successive tempo estimations
      l_min: minimum tempo lag calculated for autocorrelation, in BPM
      l_max: maximum tempo lag calculated for autocorrelatio, in BPM
    Output:
      tempo: tempo estimate in BPM from autocorrelation vector
      """
    max_idx = np.argmax(autocorrelation) #get lag index of max tempo
    tempo = 60.0 / (hop_dur * (max_idx + l_min)) #convert lag index to tempo in BPM
    return tempo

#dynamic programming algorithm defined in Ellis 07
def calculate_beats(onset, global_tempo, hop_dur=HOP_TIME, weight=WEIGHT):
    """
    Input:
      onset: vector of estimated onsets, ie output of ODF
      global_tempo: estimated tempo of audio, in BPM
      hop_dur: STFT hop length in seconds used to calculate onset
      weight: how much to weight onset strength vs global tempo adherence(higher means more weight to adherence)
    Output:
      final_beats: list of beat times in seconds
      """
    global_ioi_samp = round(60.0 / (global_tempo * HOP_TIME)) #convert the global tempo from BPM to onset samples
    max_back = round(-2 * global_ioi_samp) #we will only consider beats as far apart as 2x the global period
    min_back = -round(0.5 * global_ioi_samp) #we will only consider beats as close as 1/2 the global period

    #log-scale cost punishment for deviating from global tempo
    gaus_range = np.array(list(range(max_back,min_back)))
    gaus_cost = -1.0 * WEIGHT * np.abs(np.square(np.log(gaus_range * 1.0 / -global_ioi_samp)))

    costs = np.copy(onset) # set up to store best cost ending at each onset time
    beat_times = np.zeros(len(onset)) # set up to store previous beat time responding to best cost

    # loop through all possible beat times
    for i in range(-max_back, len(onset)):
        # calculated weighted penalty for each possible previous beat within range
        scores = gaus_cost + costs[(i+max_back):(i+min_back)]
        idx = np.argmax(scores) # get the beat index of lowest penalty
        max_score = scores[idx] # get lowest penalty value

        # combine best penalty with onset strength to get total cost of maximized sequence ending at i
        timerange = gaus_range + i
        costs[i] = max_score + onset[i]

        # store previous beat index that caused best cost at i
        beat_times[i] = timerange[idx]


    beats = np.zeros(len(onset))
    idx = int(np.argmax(costs)) # final beat is index with highest cost
    beats[0] = beat_times[idx] # get final beat's onset index
    N = 1

    # backtrack through previous indices to build beat sequence backwards
    while(beat_times[idx] > 0):
        idx = int(beat_times[idx])
        beats[N] = beat_times[idx]
        N = N + 1

    # convert beat sequence from onset samples to time in seconds, and put in ascending order
    final_beats = np.flip(beats[0:N]) * hop_dur
    return final_beats

def beatTracker(inputFile, danceStyle=None):
    """
    Usage: final_beats, _ = beatTracker(inputFile, danceStyle=Style.CHACHA etc...)
    Input:
      inputFile: path to wav file to track
      danceStyle: optional, Style enum indicating type of dance music
    Output:
      est_beats: list of beat times in seconds
      """
    #load file
    data, sr = librosa.load(inputFile, sr=None)

    # calculate window parameters based on sample rate
    window_len = int(2 ** np.ceil(np.log2(WINDOW_TIME * sr)))
    hop_len = int(HOP_TIME * sr)

    # novelty function, calculate log flux for low and high frequencies separately
    lf_logflux = logFiltSpecFlux(data, window_len, hop_len, sr, mel_low=True)
    hf_logflux = logFiltSpecFlux(data, window_len, hop_len, sr, mel_low=False)
    mixed_logflux = logFiltSpecFlux(data, window_len, hop_len, sr, mel_cutoff=0, mel_low=False)

    if danceStyle == None:
        tau_0 = TAU0
    else:
        tau_0 = tempo_dict[danceStyle]

    # when calculating autocorrelation, use mean tempo of style in weight function (tau_0 parameter)
    lf_auto = autocorrelate_weighted(lf_logflux, hop_dur=HOP_TIME, tau_0=tau_0)
    hf_auto = autocorrelate_weighted(hf_logflux, hop_dur=HOP_TIME, tau_0=tau_0)
    mixed_auto = autocorrelate_weighted(mixed_logflux, hop_dur=HOP_TIME, tau_0=tau_0)

    #combine peaks from low and high freq autocorrelation to calculate final tempo
    combined_auto = hf_auto + lf_auto 
    global_tempo = tempo_from_auto(combined_auto, HOP_TIME)

    # certain styles perform better on low vs high frequency onset information
    if danceStyle == None:
        onset = mixed_logflux
    else:
        if(danceStyle == Style.SAMBA or danceStyle == Style.VIENNESE):
            onset = lf_logflux
        elif(danceStyle == Style.RUMBA or danceStyle == Style.TANGO):
            onset = hf_logflux
        else:
            onset = mixed_logflux

    # run beat estimation algorithm using tempo estimation
    est_beats = calculate_beats(onset, global_tempo)

    return est_beats, None