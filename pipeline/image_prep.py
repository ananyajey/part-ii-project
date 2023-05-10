import os
import csv
import random
import math
from visualization_new import *
from unidecode import unidecode
import tensorflow as tf
from tensorflow.keras import datasets, layers, models



# SPLIT FUNCTIONS

def split_train_test(id_list, ratio):
    random.shuffle(id_list)
    split_index = int(ratio * len(id_list))

    train_set = id_list[:split_index]
    test_set = id_list[split_index:]

    return train_set, test_set


def split_k_fold(id_list, n_batches):
    random.shuffle(id_list)
    batches = []
    index = 0
    batch_size = math.ceil(len(id_list)/n_batches)
    
    for i in range(n_batches - 1):
        batches.append(id_list[index:index+batch_size])
        index += batch_size
    batches.append(id_list[index:len(id_list)-1])

    return batches


def get_image_func(image_type):
    if image_type == "spectrogram":
        return wav_to_spectrogram
    elif image_type == "chromagram":
        return wav_to_chromagram
    elif image_type == "mfcc":
        return wav_to_mfcc
    elif image_type == "cochleagram":
        return wav_to_cochleagram
    else:
        return



def get_labels(id_list, label_list):
    labels = np.array([label_list.index(x[0]) for x in id_list])
    labels = labels.reshape((len(labels), 1))
    return labels


def image_split_label(id, audio_dict, clip_length, shift_length, resize_dim, label_list, image_type = None, image_folderpath = None):
    '''
    If folderpath is None, convert the audio file. If a folderpath is provided, obtain the images from this path.
    Returns
    -------
    ids: list
    imgs: numpy array
    labels: numpy array
    '''
    ids = []
    imgs = None
    labels = None

    # CONVERT/GET
    if image_folderpath == None:
        to_image = get_image_func(image_type)
        filepath = audio_dict[id]
        images = to_image(filepath)

    # SPLIT
    for i in range(2):
        if image_folderpath == None:
            img = images[i]
        else:
            filepath = os.path.join(image_folderpath, f"{id}_ch{i}.png")
            assert(os.path.exists(filepath)), f"This filepath does not  exist: {filepath}"
            img = Image.open(filepath)
        img_dict = split(img, id + "_ch" + str(i), clip_length, shift_length, resize_dim)
        
        # Set/update audio sample id list
        ids.extend(list(img_dict.keys()))

        # Set/update array of image arrays
        if imgs is None:
            imgs = np.array(list(img_dict.values()))
        else:
            imgs = np.append(imgs, np.array(list(img_dict.values())), axis = 0)

    # LABEL
    labels = get_labels(ids, label_list=label_list)

    return ids, imgs, labels


def special_character_check(audio_folderpath):
    for genre in os.listdir(audio_folderpath):
        files = os.listdir(os.path.join(audio_folderpath, genre))
        for file in files:
            if file != unidecode(file):
                filepath = os.path.join(audio_folderpath, genre, file)
                if unidecode(filepath) != filepath:
                    print(filepath)


def convert_and_save(id_list, audio_dict, image_type, image_folderpath):
    count = 0
    to_image = get_image_func(image_type)
    for id in id_list:
    # CONVERT
        count += 1
        print(f"{count}/{len(id_list)}    {id}", end="")
        filepath = audio_dict[id]
        images = to_image(filepath)

        # SAVE
        for i in range(2):
            images[i].save(os.path.join(image_folderpath, f"{id}_ch{i}.png"))
        print("   DONE")


def return_data(id_list, audio_dict, clip_length, shift_length, resize_dim, label_list, image_type = None, image_folderpath=None):
    ids = None
    imgs = None
    labels = None


    count = 0
    for id in id_list:

        # Convert to images, split, and get labels
        split_ids, split_imgs, split_labels = image_split_label(id, audio_dict=audio_dict, clip_length=clip_length, shift_length=shift_length, resize_dim=resize_dim, label_list=label_list, image_type=image_type, image_folderpath=image_folderpath)
        
        # Set/update audio sample id list
        if ids is None:
            ids = split_ids
        else:
            ids = np.append(ids, split_ids, axis = 0)

        # Set/update array of image arrays
        if imgs is None:
            imgs = np.array(split_imgs)
        else:
            imgs = np.append(imgs, split_imgs, axis = 0)

        # Set/update array of image labels
        if labels is None:
            labels = split_labels
        else:
            labels = np.append(labels, split_labels, axis = 0)

        count += 1
    
    # Shuffle
    indices = random.sample(range(len(ids)), len(ids))
    ids = [ids[i] for i in indices]
    imgs = imgs[indices]
    labels = labels[indices]
    
    return ids, imgs, labels




def generate_data(song_ids, audio_dict, clip_length, shift_length, resize_dim, label_list, image_type = None, image_folderpath = None, gen_batch_size = 1, model_batch_size = None, mode = "train"):
    '''
    Parameters
    ----------
    id_list
        List of song IDs to be converted into images

    folderpath
        Path sa
    
    n
        Number of songs to convert each iteration
    
    mode
        "generate" or "return"

    Returns
    -------
    imgs
        Arrays of converted images

    labels
        Respective labels for converted images in imgs

    '''
    out_ids = []
    out_imgs = None
    out_labels = None
    n_count = 0
    id_count = 0

    if mode == "train":
    
        while True:
            # convert 1 song
            id = song_ids[id_count]
            #print(id_count, id)
            
            new_ids, new_imgs, new_labels = image_split_label(id, audio_dict=audio_dict, clip_length=clip_length, shift_length=shift_length, resize_dim=resize_dim, label_list=label_list, image_type=image_type, image_folderpath=image_folderpath)

            out_ids.extend(new_ids)

            if out_imgs is None:
                out_imgs = new_imgs
                out_labels = new_labels
            else:
                out_imgs = np.append(out_imgs, new_imgs, axis=0)
                out_labels = np.append(out_labels, new_labels, axis=0)

            # update bigcount and smallcount
            n_count += 1
            id_count += 1

            #print(id_count, id)

            if (n_count % gen_batch_size == 0) or (id_count == len(song_ids)):
                assert(len(out_ids) == len(out_imgs) == len(out_labels)), "Something went wrong when generating, the dimensions of ids, imgs, and labels do not match." # sanity check
                perm = np.random.permutation(len(out_ids))
                out_imgs = out_imgs[perm]
                out_labels = out_labels[perm]
                if model_batch_size is None:
                    yield out_imgs, out_labels
                else:
                    while len(out_labels) >= model_batch_size:
                        yield out_imgs[:model_batch_size], out_labels[:model_batch_size]
                        out_imgs = out_imgs[model_batch_size:]
                        out_labels = out_labels[model_batch_size:]

                out_ids = []
                out_imgs = None
                out_labels = None
            
                if id_count == len(song_ids):
                    id_count = 0
                    n_count = 0
                    random.shuffle(song_ids)

    if mode == "test":
        while id_count < len(song_ids):
            # convert 1 song
            id = song_ids[id_count]
            print(id_count, id)
            
            new_ids, new_imgs, new_labels = image_split_label(id, image_type=image_type, image_folderpath=image_folderpath, audio_dict=audio_dict, clip_length=clip_length, shift_length=shift_length, resize_dim=resize_dim, label_list=label_list)

            if out_imgs is None:
                out_imgs = new_imgs
            else:
                out_imgs = np.append(out_imgs, new_imgs, axis=0)

            n_count += 1
            id_count += 1

            if n_count % gen_batch_size == 0:
                yield out_imgs

                out_imgs = None                
            
            if id_count == len(song_ids):
                if out_imgs is not None:
                    yield out_imgs
    


def get_clip_ids(song_ids, audio_dict, clip_length, shift_length):
    clip_id_list = []
    for id in song_ids:
        filepath = audio_dict[id]
        new_ids = []
        with wave.open(filepath, "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()

            audio_len = frame_count/sample_rate
        for i in range(2):
            left = 0.
            right = left + clip_length
            while right <= audio_len:
                new_id = f"{id}_ch{i}_{left}_{right}"
                new_ids.append(new_id)
                left += shift_length
                right += shift_length
        clip_id_list = clip_id_list + new_ids
    
    return clip_id_list


def createModel(input_shape, type=1):
    if type == 1:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(5))
    
    return model