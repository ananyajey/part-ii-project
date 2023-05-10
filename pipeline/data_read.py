import os
import csv
import random
import math
from visualization_new import *
from unidecode import unidecode

def get_genres(audio_folderpath):
    genre_id = {}

    for genre in os.listdir(audio_folderpath):
        i = 1
        while (genre[:i] in genre_id.values() and i <= len(genre)):
            i += 1
        genre_id[genre] = genre[:i]
    
    return genre_id


# audio dict (initial)
def get_audio_dict(audio_folderpath, genre_id):
    audio_dict = {}

    for genre in genre_id:
        files = os.listdir(os.path.join(audio_folderpath, genre))
        count = 0
        # TODO: given an audio dictionary, find what max count for a genre is
        for file in files:
            filepath = os.path.join(audio_folderpath, genre, file)
            id = genre_id[genre] + '{:03d}'.format(count)
            count += 1
            audio_dict[id] = filepath
    
    return audio_dict

# audio dict (initial)
def create_csv(csv_filepath, audio_dict):
    with open(csv_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(list(audio_dict.items()))


# audio dict (after initial)
def get_audio_dict_csv(csv_filepath):
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        #next(reader)
        audio_dict = {row[0]: row[1] for row in reader}
    
    return audio_dict

