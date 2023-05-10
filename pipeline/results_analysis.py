from IPython.display import Audio
import librosa
import numpy as np
import torch
import os
from pydub import AudioSegment



def get_confusion_matrix(vote_ids, vote_array, label_list):
    if type(vote_array) == torch.Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output = torch.zeros((len(label_list), len(label_list))).to(device)
    else:
        output = np.zeros((len(label_list), len(label_list)))
    for i in range(len(vote_ids)):
        id = vote_ids[i]
        actual_class = id[0]
        actual_index = label_list.index(actual_class)
        output[:, actual_index] += vote_array[i]

    return output


def play_clip(clip_id, audio_dict):
    info = clip_id.split('_')
    song_id = info[0]
    channel = int(info[1][-1])
    start_time = float(info[2])
    end_time = float(info[3])

    filepath = audio_dict[song_id]
    y, sr = librosa.load(filepath)

    audio_segment = y[int(start_time * sr) : int(end_time * sr)]
    print(song_id)
    print(os.path.basename(filepath))
    print(print(f"{int(start_time//60):02}:{int(start_time%60):02}  -  {int(end_time//60):02}:{int(end_time%60):02}"))

    return Audio(audio_segment, rate = sr)


def save_clip(clip_id, audio_dict, save_folderpath):
    info = clip_id.split('_')
    song_id = info[0]
    channel = int(info[1][-1])
    start_time = float(info[2])
    end_time = float(info[3])
    
    filepath = audio_dict[song_id]
    audio = AudioSegment.from_file(filepath, format="wav")
    clip = audio[start_time*1000:end_time*1000]

    clip.export(os.path.join(save_folderpath, f"{song_id}_{start_time}_{end_time}.wav"), format="wav")




def classified_as(id_list, pred_classes, actual_classes, pred_class, actual_class):
    output = []
    for i in range(len(id_list)):
        id = id_list[i]
        pred = pred_classes[i]
        actual = actual_classes[i]
        if (pred == pred_class):
            if (actual == actual_class):
                output.append(id)
    
    return output  


def get_song_votes(song_ids, clip_ids, clip_vote_array):
    if type(clip_vote_array) == torch.Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        song_vote_array = torch.zeros((len(song_ids), len(clip_vote_array[0]))).to(device)
    else:
        song_vote_array = np.zeros((len(song_ids), len(clip_vote_array[0])))
    for i in range(len(clip_ids)):
        clip_id = clip_ids[i]
        song_id = clip_id.split('_')[0]
        index = song_ids.index(song_id)
        song_vote_array[index] = song_vote_array[index] + clip_vote_array[i]
    
    return song_vote_array

def classification_by_confidence(clip_ids, clip_prob_array, clip_pred_classes, clip_actual_classes, actual_class, confident=True, misclassified=False, n=10):
    max_probs, _ = torch.max(clip_prob_array, dim=1)
    
    if confident:
        confidence_index_order = torch.argsort(max_probs, descending=True).cpu().numpy()
    else:
        confidence_index_order = torch.argsort(max_probs, descending=False).cpu().numpy()

    if misclassified:
        condition = (clip_pred_classes != actual_class) & (clip_actual_classes == actual_class)
    else:
        condition = (clip_pred_classes == actual_class) & (clip_actual_classes == actual_class)
    
    true_indices = torch.nonzero(condition).cpu().numpy()
    confidence_indices = np.where(np.isin(confidence_index_order, true_indices))[0]

    clip_ids_indices = confidence_index_order[confidence_indices]

    clip_ids_indices_top = clip_ids_indices[:n]
    output_clip_ids = [clip_ids[i] for i in (clip_ids_indices_top)]

    return output_clip_ids, clip_ids_indices_top