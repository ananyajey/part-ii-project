#vector = ['classical', 'rap', 'pop', 'hiphop', 'jazz'] #.....

import numpy as np
import PIL
from PIL import Image
import os



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

x, y = image_to_array("data\images_initial/20s3FayrStM8GY0m4dRFsw__ch1.png")

print(x)
print(y)





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