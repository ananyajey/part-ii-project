#mp3 to wav
#fft on wav
#spectrogram

from os import path
import os
import struct
import wave
from pydub import AudioSegment
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np
from scipy.fftpack import fft
import labels
from labels import *
from PIL import Image


def processing(raw_folderpath, processed_folderpath):
    """
    Parameters
    ----------
    raw_folderpath: Path
        The path to the folder containing the raw data files
    
    processed_folderpath: Path
        The path to the folder which will contain the processed data
    """
    # TODO: implement


def mp3_to_wav(filepath):
    sound = AudioSegment.from_mp3(filepath)
    new_path = "data/output/" + Path(filepath).stem + ".wav"
    return sound.export(new_path, format="wav")


def wav_to_spectrogram(wav_filepath, data_folderpath, window='hann', nperseg=256, nfft=None, return_onesided=True, mode=None): 

    """
    Parameters
    ----------
    wav_filepath: Path
        The path to a WAVE format audio file.
    
    data_folderpath: Path
        The path to the folder in which the spectrogram will be stored.

    window: str
        Desired window to use when computing the Short Term Fourier Transform.
        Window types: 'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris',
        'nottall', 'barthann', 'cosine', 'exponential', 'tukey', 'taylor', 'lanczos', 'kaiser', 'kaiser_bessel_derived', 'gaussian',
        'general_cosine', 'general_gaussian', 'general_hamming', 'dpss', 'chebwin'
    
    nperseg:
        Length of each segment in the STFT.

    nfft:
        Length of the FFT used. By default, nfft = nperseg.
    
    return_onesided: bool


    """
    # STFT config
    if nfft == None:
        nfft = nperseg

    

    # Open the file and perform a Fast Fourier Transform
    with wave.open(wav_filepath, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        channel_count = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

        clip_length = 4
        img_height = 5
        img_width = img_height*frame_count/(sample_rate*clip_length) # img_height*song_length/clip_length

        byte_data = wav_file.readframes(frame_count)
        audio_data = np.frombuffer(byte_data, dtype=np.int16).reshape((frame_count, channel_count)).T
        #audio_data_unpacked = np.array(struct.unpack("h" * wav_file.getnframes(), wav_data_1))
        #sample_rate, data_arr = wavfile.read(wav_filepath)

        #x = fft(audio_data[:,0])
    
    name = os.path.basename(wav_filepath)[:-4]
    genre = os.path.basename(os.path.split(wav_filepath)[0])
    uri = get_uri(name)
    
    if (len(audio_data) == 1): # use channel count?
        frequencies, times, spectrogram = signal.stft(x=audio_data[0], fs=sample_rate)#, window=window, nperseg=nperseg, nfft=nfft)
        fig = plt.figure(frameon=False, dpi = None)
        im = plt.pcolormesh(times, frequencies, spectrogram, shading='auto')
        fig.colorbar(im)
        fig.savefig((data_folderpath + "/" + os.path.basename(wav_filepath).split('.')[0] + ".png"), bbox_inches = "tight", pad_inches = 0)

    elif (len(audio_data) == 2):
        frequencies_ch1, times_ch1, spectrogram_ch1 = signal.stft(x=audio_data[0], fs=sample_rate, window=window, nperseg=nperseg, nfft=nfft, return_onesided=True)#, mode='complex')
        psd_ch1 = 10*np.log10(abs(spectrogram_ch1))
        fig_ch1 = plt.figure(frameon=False, figsize=(img_width, img_height), dpi=100)
        im_1 = plt.pcolormesh(times_ch1, frequencies_ch1, psd_ch1, shading='auto')
        plt.axis('off')
        plt.close()
        #fig_ch1.colorbar(im_1)

        #plt.show()

        frequencies_ch2, times_ch2, spectrogram_ch2 = signal.stft(x=audio_data[1], fs=sample_rate)#, window=window, nperseg=nperseg, nfft=nfft)#, return_onesided=True), mode='magnitude')
        psd_ch2 = 10*np.log10(abs(spectrogram_ch2)+1)
        fig_ch2 = plt.figure(frameon=False, figsize=(img_width, img_height), dpi=100)
        im_2 = plt.pcolormesh(times_ch2, frequencies_ch2, psd_ch2, shading='auto')
        plt.axis('off')
        plt.close()
        #fig_ch2.colorbar(im_2)

        #plt.show()

        fig_ch1.savefig((data_folderpath + "/" + uri + "_ch1.png"), bbox_inches = "tight", pad_inches = 0, dpi = 100)
        fig_ch2.savefig((data_folderpath + "/" + uri + "_ch2.png"), bbox_inches = "tight", pad_inches = 0, dpi = 100)
    
    else:
        raise ValueError('Invalid number of channels')
    
    return

    # Divide audio into n second sections. Plot and save spectrograms
    frequencies, times, spectrogram = signal.spectrogram(x=audio_data[int(3*sample_rate/16):int(sample_rate/4),0], fs=sample_rate, return_onesided=True, mode='complex')
    fig = plt.figure(frameon=False, dpi = None) #specify pixels?

    if ((spectrogram.imag == 0j).all()): # sanity check
        im = plt.pcolormesh(times, frequencies, spectrogram.real, shading='auto')
    else:
        #raise ValueError()
        im = plt.pcolormesh(times, frequencies, spectrogram.real, shading='auto')


def wav_to_chromagram(wav_filepath, data_folderpath): 

    """
    Parameters
    ----------
    wav_filepath: Path
        The path to a WAVE format audio file.
    
    data_folderpath: Path
        The path to the folder in which the chromagram will be stored.
    """
    # TODO: implement


def wav_to_MFCC(wav_filepath, data_folderpath): 

    """
    Parameters
    ----------
    wav_filepath: Path
        The path to a WAVE format audio file.
    
    data_folderpath: Path
        The path to the folder in which the MFCC will be stored.
    """
    # TODO: implement

    with wave.open(wav_filepath, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        channel_count = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

        clip_length = 4
        img_height = 0.5
        img_width = img_height*frame_count/(sample_rate*clip_length) # img_height*song_length/clip_length

        byte_data = wav_file.readframes(frame_count)
        audio_data = np.frombuffer(byte_data, dtype=np.int16).reshape((frame_count, channel_count)).T


    audio_data = audio_data[0,sample_rate:sample_rate*7]

    audio = audio_data/np.max(np.abs(audio_data))

    def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
        # hop_size in ms
        
        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num,FFT_size))
        
        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
        
        return frames

    hop_size = 15 #ms
    FFT_size = 2048

    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)

    window = signal.get_window("hann", FFT_size, fftbins=True)

    audio_win = audio_framed * window

    audio_winT = np.transpose(audio_win)

    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

    audio_fft = np.transpose(audio_fft)

    audio_power = np.square(np.abs(audio_fft))

    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 10

    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def met_to_freq(mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = freq_to_mel(fmin)
        fmax_mel = freq_to_mel(fmax)
        
        print("MEL min: {0}".format(fmin_mel))
        print("MEL max: {0}".format(fmax_mel))
        
        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = met_to_freq(mels)
        
        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs
    
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
        
        for n in range(len(filter_points)-2):
            filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
        
        return filters
    
    filters = get_filters(filter_points, FFT_size)

    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis]

    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)

    def dct(dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num,filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)
        
        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
            
        return basis

    dct_filter_num = 40

    dct_filters = dct(dct_filter_num, mel_filter_num)

    cepstral_coefficents = np.dot(dct_filters, audio_log)

    print(cepstral_coefficents[:, 0])
    
    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
    plt.imshow(cepstral_coefficents, aspect='auto', origin='lower')

    plt.show()
    return
    
       
def wav_to_cochleagram(wav_filepath, data_folderpath): 

    """
    Parameters
    ----------
    wav_filepath: Path
        The path to a WAVE format audio file.
    
    data_folderpath: Path
        The path to the folder in which the cochleagram will be stored.
    """
    # TODO: implement


def generate_images(raw_folderpath, processed_folderpath, image_type):
    """
    Parameter
    ---------
    raw_folderpath:Path
        The path to the folder containing raw audio files to be converted into images
    
    processed_folderpath: Path
        The path to the folder which will contain the generated images
    
    image_type
        The type of graph (spectrogram, chromagram, MFCC, cochleagram) to be generated
    """
    # TODO: implement


def augment(folderpath, graph_type):
    """
    Parameters
    ----------
    folderpath: Path
        The path to the data required to be augmented
    
    graph_type: String
        The type of visualization contained in the folder folderpath (spectrogram, chromagram, MFCC, cochleagram)
    """
    # TODO: implement


def split_images(folderpath, savepath):
    for filename in os.listdir(folderpath):
        if filename.endswith('.png'):
            #with open(os.path.join(folderpath, filename)) as file:
            im = Image.open(os.path.join(folderpath, filename))
            imgwidth, imgheight = im.size
            print("{2} ___ width: {0}, height: {1}".format(imgwidth, imgheight, filename))
            for i in range(0, imgwidth, imgheight):
                box = (i, 0, i+imgheight, imgheight)
                a = im.crop(box)
                a.save("{0}/{1}_{2}.png".format(savepath, filename[:-4], int(i/imgheight)))

                

#if samples.ndim >  1:
#    samples = samples[:,0]





#wav_to_spectrogram("data/raw/baroque/Bach, JS_ Brandenburg Concerto No. 2 in F Major, BWV 1047_ I. — _ Johann Sebastian Bach, Mark Bennet.wav", "data/spectrograms")

#split_images("data/spectrograms", "data/spectrograms/out")

# PSEUDOCODE
# ----------
# rawdataset = "<path>" (path to folder containing audio files)
# dataset = processing(rawdataset)
# 
# CONCATENATE INTO N-SONG AUDIOS
#
# n = <int> (number of songs to be grouped)
# audio_concat = []
# buffer = [<buffer values>]
#
# for file in dataset:
#   for i in range n:
#       if file is mp3:
#           wav_file = wav file conversion of mp3
#       else:
#           wav_file = file
#       audio_data = np.array(struct.unpack("h" * wav_file.getnframes(), wav_file.readframes(wav_file.getnframes())))[frames] (unpack and crop to be an integer number of data samples)
#       np.append(audio_concat, audio_data)
#       if i != (n-1):
#           np.append(audio_concat, buffer)
#
#   save into new folder "concat_audios"
# 
# CONVERT TO VISUALIZATION TYPES
#
# for visual in ["spectrogram", "chromagram", "mfcc", "cochleagram"]:
#   for file in concat_audios:
#       convert file to visual
#       split into sections
#       sanity check
#       save to correct folder