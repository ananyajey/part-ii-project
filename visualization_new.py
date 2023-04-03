import io
import wave
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import cupy as cp
import os
import threading

pps = 344

def wav_to_spectrogram(wav_filepath, pps = pps):
    #print(os.path.basename(wav_filepath))
    with wave.open(wav_filepath, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        channel_count = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

        byte_data = wav_file.readframes(frame_count)
        audio_data = np.frombuffer(byte_data, dtype=np.int16).reshape((frame_count, channel_count)).T

    out = []
    
    for i in range(2):
        #print(os.path.basename(wav_filepath), i)
        f, t, s = signal.stft(x = audio_data[i], fs = sample_rate)

        s_log = np.log(np.abs(s) + 1)

        scale = 255.0 / np.max(s_log)

        s_norm = (s_log * scale).astype(np.uint8)

        im = Image.fromarray(s_norm, mode='L')

        width, height = im.size

        new_width = int(pps * frame_count/sample_rate)

        im_resized = im.resize((new_width, height))



        #with lock:
        '''fig = plt.figure(frameon=False)#, figsize=(frame_count/sample_rate, 5), dpi=150) # figsize??
        #im = plt.pcolormesh(t, f, np.log(np.abs(s) + 1))
        im = plt.imshow(np.log(np.abs(s) + 1))
        plt.axis('off')
        plt.close()

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        im = im.crop(im.getbbox()) # remove whitespace

        width, height = im.size

        final_im = im.resize((pps * int(frame_count / sample_rate), height))
        '''
        out.append(im_resized)
    return out

def wav_to_chromagram(wav_filepath, pps = pps):
    with wave.open(wav_filepath, "rb") as wav_file:
        frame_count = wav_file.getnframes()
        channel_count = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        y, sr = librosa.load(wav_filepath, sr = None, mono = False)
        
        byte_data = wav_file.readframes(frame_count)
        audio_data = np.frombuffer(byte_data, dtype=np.int16).reshape((frame_count, channel_count)).T


    out = []

    for i in range(2):
        chromagram = librosa.feature.chroma_stft(y=y[i], sr=sr)

        S = np.abs(librosa.stft(y[i]))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)



        #img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax[1])


        '''
        '''


        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)

        width, height = im.size

        final_im = im.resize((pps * int(frame_count / sample_rate), height))

        out.append(final_im)



    #S = np.abs(librosa.stft(y))
    #chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    x = 0


    



def wav_to_mfcc():
    return

def wav_to_cochleagram():
    return


def split(image, id, clip_len, shift_len, resize_dim = None, pps = pps):
    '''
    input - image opject
    crop into squares
    resize
    reshape and remove opacity channel
    '''
    width, height = image.size
    left = 0
    right = left + clip_len * pps
    top = 0
    bottom = height
    output = {}
    while (right <= width):
        img = (image.crop((left, top, right, bottom)))

        if(resize_dim != None):
            img = img.resize((resize_dim, resize_dim))
        
        img_arr = np.array(img)/255.0
        #img_arr_reshape = np.moveaxis(img_arr, -1, 0)
        #output_img = img_arr_reshape[:3]
        #output_img = np.array(img)
        output_label = "{0}_{1}_{2}".format(id, int(left/pps), int(right/pps))
        output[output_label] = (img_arr/255.0).reshape((img_arr.shape[0], img_arr.shape[1], 1))
        #print(left, right)
        left += shift_len * pps
        right += shift_len * pps

    return output
        




for item in os.listdir(r"C:\Users\anany\Cambridge\Part II Project\data\raw\rock"):

    p = os.path.join(r"C:\Users\anany\Cambridge\Part II Project\data\raw\rock", item)

        
    asdf = wav_to_spectrogram(p)

    jkl = split(asdf[0], "aaa", 10, 10)

    print(item, "done")


