#vector = ['classical', 'rap', 'pop', 'hiphop', 'jazz'] #.....

import numpy as np
#import PIL
from PIL import Image
import os
import random



def image_to_array(image_filepath):
    """
    Parameters
    ----------
    image_filepath: Path
        The path to the image files
    
    Returns
    -------
    img_array, img_label
        Tuple containing the converted numpy array of the image and its feature vector
    """

    def get_vector(genre):
        genre_set = ['pop', 'jazz', 'rock', 'hip hop', 'baroque']
        return [int(genre == item) for item in genre_set]

    image = Image.open(image_filepath)
    img_array = np.asarray(image)
    img_label = get_vector(os.path.split(os.path.split(image_filepath)[0])[1])
    return img_array, img_label


def load_data(data_folderpath):
    """
    Parameters
    ----------
    
    """
    data = {}
    for item in os.listdir(data_folderpath):
        item_path = os.path.join(data_folderpath, item)
        if os.path.isdir(item_path):
            data.update(load_data(item_path))
        if os.path.isfile(item_path):
            a, l = image_to_array(item_path)
            data[a] = l
    
    return data

def split_data(data_dict, percentage = 0.8):
    random.shuffle(data_dict)
    return data_dict


#x, y = image_to_array("data\images_initial/baroque/2O348yjmVxXiR8UkiBkZ1O__ch1.png")

#print(x)
#print(y)

sample_dict = {}
sample_dict[0] = 0
sample_dict[1] = 1
sample_dict[2] = 2
sample_dict[3] = 3
sample_dict[4] = 4

print(split_data(sample_dict))











# FFT stuff




'''import numpy as np
import matplotlib.pyplot as plt

def stft(x, fs, window, overlap):
    hop_size = window - overlap
    window = np.hanning(window)
    X = np.array([np.fft.fft(window * x[i:i+window]) for i in range(0, len(x)-window, hop_size)])
    return X

fs = 44100 # sample rate
T = 5.0 # length of signal in seconds
t = np.linspace(0, T, int(T*fs), endpoint=False)
x = 0.5 * np.sin(2*np.pi*440*t) + np.sin(2*np.pi*880*t)

X = stft(x, fs, 2048, 1024)
f, ax = plt.subplots(figsize=(4,4))
ax.matshow(np.abs(X[:,:400]), origin='lower', aspect='auto', cmap='viridis')
ax.set_xlabel('Frame')
ax.set_ylabel('Frequency [Hz]')
plt.show()




import cmath
import math

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    return [even[k] + cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)] + \
           [even[k] - cmath.exp(-2j*cmath.pi*k/N)*odd[k] for k in range(N//2)]

T = 5.0 # length of signal in seconds
fs = 44100 # sample rate
t = [i/fs for i in range(int(T*fs))]
x = [0.5 * math.sin(2*math.pi*440*t_i) + math.sin(2*math.pi*880*t_i) for t_i in t]

X = fft(x)
X_magnitude = [math.sqrt(c.real**2 + c.imag**2) for c in X]
X_magnitude = X_magnitude[:len(X_magnitude)//2]'''




import numpy as np
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


print(DFT_slow([1, 2, 3]))
print(np.fft.fft([1,2,3]))
























'''

    
vector = ['baroque', 'metal']

labels = {}

def get_uri_manual(name):
    uri_dict = {"12 Sonatas, Op. 16_ Sonata quarta_ II. Presto" : '28Vpqa5FchyGjEhvOg3e4A',
                "Bach, JS_ Brandenburg Concerto No. 5 in D Major, BWV 1050_ II. Affettuoso" : '2O348yjmVxXiR8UkiBkZ1O',
                "Canon in D major" : '0yBv2qXnyiNGniYOWaOZsX',
                "Castor et Pollux (1754 version)_ Act II Scene 1_ (Troupe de Spartiates)" : '738A3ZFofgnwVG0w47Rru7',
                "Concerto in G Major, Op. 5 No. 4_ III. Allegro" : '7ACCi3YrAUSPf5uSSvlbn4',
                "Der Schulmeister_ Recitativo_ Der Schulmeister" : '3NiBGP0vaz3CaQo1bM2Tmi', 
                "Messiah, HWV 56 _ Pt. 2_ _Hallelujah" : '4TNCQyG2gmepsGoeLdRKn4',
                "Telemann_ Trumpet Concerto in D Major, TWV 51_D7_ I. Adagio" : '5ZyeUQe55N3FdUCs1sOuW3',
                "Toccata and Fugue in D Minor, BWV 565_ I. Toccata" : '4HE2Ex0bjbj3YNXmV01OoM',
                "Violin Sonata in B-Flat Major, Op. 5 No. 2_ IV. Adagio" : '68R6JdZFUTgI2e1MfpD7ko',
                "Vivaldi_ The Four Seasons, Violin Concerto in F Minor, Op. 8 No. 4, RV 297 _Winter_ I. Allegro non m_1" : '0ON4FYmS4Zch1NV0lhv9hX'
    }
    return uri_dict[name]

def add_label(uri, genre):
    if genre == 'baroque':
        labels[uri] = [1,0]
    elif genre == 'metal':
        labels[uri] = [0,1]

def get_label(uri):
    return labels[uri]

def get_info(uri):
# TODO: implement
    return

#add_label('012', 'baroque')
#print(labels)


'''