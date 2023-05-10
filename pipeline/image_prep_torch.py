import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pipeline.image_prep import *
import torch.nn.functional as F





class GenDataset(Dataset):
    def __init__(self, song_ids, audio_dict, clip_length, shift_length, resize_dim, label_list, image_type=None, image_folderpath=None, gen_batch_size = 1, mode = "train"):
        # generate_data_parameters
        self.song_ids = song_ids
        self.audio_dict = audio_dict
        self.clip_length = clip_length
        self.shift_length = shift_length
        self.resize_dim = resize_dim
        self.label_list = label_list
        self.image_type = image_type
        self.image_folderpath = image_folderpath
        self.gen_batch_size = gen_batch_size
        self.mode = mode

        # generator function keep-track
        self.gen_func = generate_data_torch(song_ids=self.song_ids, audio_dict=self.audio_dict, clip_length=self.clip_length, shift_length=self.shift_length, resize_dim=self.resize_dim, label_list=self.label_list, image_type=self.image_type, image_folderpath=self.image_folderpath, gen_batch_size = self.gen_batch_size, mode = self.mode)
        self.clip_ids = []
        self.images = None
        self.labels = np.empty((0, 1))
        self.image_offset = 0

    def __len__(self):
        len = 0
        for id in self.song_ids:
            filepath = self.audio_dict[id]
            with wave.open(filepath, "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                audio_len = frame_count/sample_rate
                clip_count = int((audio_len - self.clip_length + self.shift_length)/self.shift_length) * 2
                len += clip_count
        return len
    
    def __getitem__(self, index):
        # reset if index = 0
        if index == 0:
            self.clip_ids = []
            self.images = None
            self.labels = np.empty((0, 1))
            self.image_offset = 0

        # check if imgs is empty
        if self.images is None:
            # get ids, imgs, labels
            ids, imgs, labels = next(self.gen_func)
            
            # update self values for all these
            self.clip_ids.extend(ids)
            self.images = imgs
            self.labels = np.vstack((self.labels, labels))
        
        # GETTING TUPLE
        # imgs[index-offset]
        # labels[index]
        out_image = self.images[index - self.image_offset]
        out_label = self.labels[index]

        # if imgs is last one
        if (index - self.image_offset) == (self.images.shape[0] - 1):
            # update img_offset
            self.image_offset += self.images.shape[0]
            # empty imgs
            self.images = None

        out_image = torch.tensor(out_image).permute(2, 0, 1).float()
        out_label = torch.tensor(out_label[0]).long()
        
        return out_image, out_label


class Model(torch.nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * (image_dim // 4) * (image_dim // 4), 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 5)
    
    def forward(self, input):
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def generate_data_torch(song_ids, audio_dict, clip_length, shift_length, resize_dim, label_list, image_type = None, image_folderpath = None, gen_batch_size = 1, mode = "train"):
    
    out_ids = []
    out_imgs = None
    out_labels = None
    n_count = 0
    id_count = 0

    random.shuffle(song_ids)
    
    while True:
        # convert 1 song
        id = song_ids[id_count]
        
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

        if (n_count % gen_batch_size == 0) or (id_count == len(song_ids)):
            assert(len(out_ids) == len(out_imgs) == len(out_labels)), "Something went wrong when generating, the dimensions of ids, imgs, and labels do not match." # sanity check
            
            if mode == "train":
                perm = np.random.permutation(len(out_ids))
                out_imgs = out_imgs[perm]
                out_labels = out_labels[perm]
                out_ids = [out_ids[i] for i in perm]

            yield out_ids, out_imgs, out_labels

            out_ids = []
            out_imgs = None
            out_labels = None
        
            if id_count == len(song_ids):
                id_count = 0
                n_count = 0
                if mode == "train":
                    random.shuffle(song_ids)


