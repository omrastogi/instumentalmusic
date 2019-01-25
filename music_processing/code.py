#-------------------------------------------------------------------------------
# Purpose:  To process audio file using audio_processing
# Author:      Om Rastogi
# Created:     25-11-2018
# Library Needed: numpy, pandas, scipy, sklearn, python_speech_features
# Authors Note: This program has an accuracy of about  80% yet.
#               This program has a scope of improvement in terms of algorithm and svm.
#               Presently I have used 2 dimensional, linear svm with only two features.
#               The audio feature is dependent on python_speech_features library.
#-------------------------------------------------------------------------------
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import base
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import pickle
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import struct
import numpy as np
import math
import wave
import os


def audio_file_rewind(audio_file):
    audio_file.rewind()

def find_peak(array):
#This function has the algorithm to find the peaks in fourier transform of the audio array
    length = len(array)
    cnt=[]
    flag= -1
    var=0
    imax =0
    for i in range(1,length):
            if array[i-1]<array[i]:
                if flag == -1:
                    flag = 1
            if array[i-1]>array[i]:
                if flag == 1:
                    flag = -1
                    cnt.append(i-1)
    if len(cnt)==1:
        imax = cnt[0]
    else:
        maxm = np.max(array)
        mid = maxm/8
        for i in range(0, len(cnt)):
            if array[cnt[i]] > mid:
                imax = cnt[i+5]
                break
    return imax
def freq_find(f):
#This function matches the freqency with a note
    notes=["C0","C#0","D0","D#0","E0","F0","F#0","G0","G#0","A0","A#0","B0","C1",
    "C#1","D1","D#1","E1","F1","F#1","G1","G#1","A1","A#1","B1","C2","C#2","D2",
    "D#2","E2","F2","F#2","G2","G#2","A2","A#2","B2","C3","C#3","D3","D#3","E3",
    "F3","F#3","G3","G#3","A3","A#3","B3","C4","C#4","D4","D#4","E4","F4","F#4","G4",
    "G#4","A4","A#4","B4","C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5",
    "B5","C6","C#6","D6","D#6","E6","F6","F#6","G6","G#6","A6","A#6","B6","C7","C#7",
    "D7","D#7","E7","F7","F#7","G7","G#7","A7","A#7","B7","C8","C#8","D8","D#8","E8",
    "F8","F#8","G8","G#8","A8","A#8","B8"]
    freq=[16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96,
    27.50,29.14,30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,
    49.00,51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,
    87.31,92.50,98.00,103.83,110.00,116.54,123.47,130.81,138.59,
    146.83,155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,
    246.94,261.63,277.18,293.66,311.13,329.23,349.23,369.99,392.00,
    415.30,440.00,466.16,493.88,523.25,554.37,587.33,622.25,659.25,
    698.46,739.99,783.99,830.61,880.00,932.33,987.77,1046.50,1108.73,
    1174.66,1244.51,1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,
    1864.66,1975.53,2093.00,2217.46,2349.32,2489.02,2637.02,2793.83,
    2959.96,3135.96,3322.44,3520.00,3729.31,3951.07,4186.01,4434.92,
    4698.63,4978.03,5274.40,5587.65,5919.91,6271.93,6644.88,7040.00,
    7458.62,7902.13]
    Detected_Note = 0
    for i in range(0,len(notes)):
        if (f < freq[i]):
            k=i
            break
    c=(freq[k]+freq[k-1])/2
    if f>c:
        Detected_Note = notes[k]

    else:
        Detected_Note = notes[k-1]


    return Detected_Note


def detect_note_duration(audio_file):
    length = audio_file.getnframes()
    fs = audio_file.getframerate()
    sound_wave = np.zeros(length)
    for i in range (0,length):
        data = audio_file.readframes(1)
        data = struct.unpack("<h", data)
        sound_wave[i] = int(data[0])
    sound_wave = np.divide(sound_wave, float(2**15))
    audio_file_rewind(audio_file)

    thresh = 0.0001
    arr = [[0,0]]
    k = int(0.001*length)                                                       #considering sample length of 1000th of the sound length
    j=1
    for i in range(0,length/k):
        samp = sound_wave[k*i : k*(i+1)]                                        #sampling the audio wave.
        avg = np.mean(np.square(samp))                                          #taking the squared mean of the sample.
        if avg < thresh:                                                        #comparing avg with the thresh value to find silence.
            if arr[j-1][1]==k*i:                                                #if two consecutive sample are silence than
                arr[j-1][1] = k*(i+1)                                           #they should be mearged.

            else:
                arr.append([k*i,k*(i+1)])
                j+=1
    arr.append([len(sound_wave),len(sound_wave)])

    arr1=[]
    for i in range(0,len(arr)-1):
        arr1.append ([arr[i][1],arr[i+1][0]])                                   #filtering the silence to get notes

    for i in range(0,len(arr1)):
        for j in range(0,len(arr1[i])):
            arr1[i][j]=float(arr1[i][j])/44100                                  #converting the value from frames --> seconds
    Note_durations=[]
    for a in arr1:
        if a[1]-a[0] > 0.02:                                                    #any note duration less than 0.02sec ain't considered, because of the small sample length
            Note_durations.append([round(a[0],2),round(a[1],2)])
    return Note_durations


def featuresplot(signal, rate):                                                 #This function will extract audio features like mfcc and spectral desity.
#The function will return mean of 2nd coeff. of mfcc and mean of spectral desity
        mfc= mfcc(signal,rate, nfft=2048)
        spec_centeroid = logfbank(signal,rate, nfft=2048)
        mfc_2nd_coeff = []
        for i in range(0, len(mfc)):
                mfc_2nd_coeff.append(mfc[i][2])
        mean_2nd_coeff_mfc = np.mean(mfc_2nd_coeff)

        mean_cent = []
        for i in range(0, len(spec_centeroid)):
            mean_cent.append(np.mean(spec_centeroid[i]))
        overall_centroid_mean = np.mean(mean_cent)
        return mean_2nd_coeff_mfc, overall_centroid_mean



def svm_train_predict(m,c):
#This library takes the mean mfcc and mean spectral density to train svm
#and returns the instrument used
    with open ('Aditional_files/sample_data.pkl','rb') as pickle_file:
        dataset = pickle.load(pickle_file)                                      #This will upload the dataset

    lst = ['flute','piano','trumpet','violin']
    dataset = np.array(dataset)
    le = LabelEncoder()
    data = pd.DataFrame(dataset)
    X, Y = data.iloc[:,1:], dataset[:,0]
    Y = le.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
#SVM is trained
    clf = svm.SVC(kernel = 'linear', C = 2.0)
    clf.fit(X_train, y_train)
    #acc = clf.score(X_test, y_test)
    index = clf.predict([[m , c]])
    index = int(index)
    instrument = lst[index]
    return instrument



############################### Your Code Here #############################################

def Instrument_identify(wavefile):
	Instruments = []
	Detected_Notes = []
	Onsets = []
	length = wavefile.getnframes()
        fs = wavefile.getframerate()
        sound_wave = np.zeros(length)
        for i in range (0,length):
            data = wavefile.readframes(1)
            data = struct.unpack("<h", data)
            sound_wave[i] = int(data[0])
        signal = sound_wave                                                         #extracting signal for sound feature extraction
        sound_wave = np.divide(sound_wave, float(2**15))                            #extracing the sound_wave
        audio_file_rewind(audio_file)
        Note_durations= detect_note_duration(audio_file)                            #getting the note durations

        for a in  Note_durations:
            new_wave = sound_wave[int(a[0]*44100):int(a[1]*44100)]                  #taking out the notes from the audio.
            new_wave = np.asarray(new_wave)
            samp = len(new_wave)
            m,c = featuresplot(new_wave, fs)
            inst = svm_train_predict(m,c)

            Instruments.append(inst)

            freq_domain = abs(np.fft.rfft(new_wave))
            irange= find_peak(freq_domain)
            peak_range = freq_domain[0:(irange)]
            imax = np.argmax(peak_range)
            f = imax*fs/samp                                                        #determining the frequency of the extracted notes
            Detected_Notes.append(freq_find(f))                                     #appending the detected note
            Onsets.append(a[0])                                                     #appending the onset

	return Instruments, Detected_Notes, Onsets


############################### Main Function #############################################

if __name__ == "__main__":

	#   Instructions
	#   ------------
	#   Do not edit this function.

	# code for checking output for single audio file

	file_name = "Aditional_files/Audio_1.wav"
	audio_file = wave.open(file_name)

	Instruments, Detected_Notes, Onsets = Instrument_identify(audio_file)

	print("\n\tInstruments = "  + str(Instruments))
	print("\n\tDetected Notes = " + str(Detected_Notes))
        print("\n\tOnsets = " + str(Onsets))
	# code for checking output for all audio files




