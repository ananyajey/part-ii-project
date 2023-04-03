import os
from visualization import *



raw_data_path = "C:/Users/anany/Cambridge\Part II Project\data/raw"
unsplit_data_path = "C:/Users/anany\Cambridge\Part II Project\data/test"
split_data_path = "C:/Users/anany\Cambridge\Part II Project\data/test2"

# convert then split, ignore nonsplit images
# create method that takes that data and returns like mnist ((x_train, y_train, x_test, y_test))


#generate_images(raw_data_path, unsplit_data_path, "spectrogram")
#split_images(unsplit_data_path, split_data_path)
#label_images(unsplit_data_path)




dataset_path = unsplit_data_path
folders = os.listdir(dataset_path)
print(folders)

label_dict = {i:folders.index(i) for i in folders}
print(label_dict)


def load_data(data_folderpath, split):
    folders = os.listdir(dataset_path)
    label_dict = {i:folders.index(i) for i in folders}

    def get_one_hot(label_dict, label):
        v = np.zeros(len(label_dict))
        v[label_dict[label]] += 1
        return v
    
    x = []
    y = []

    # TODO: load into x and y

    for genre in os.listdir(data_folderpath):
        vector = get_one_hot(label_dict, genre)
        for image in os.listdir(os.path.join(data_folderpath, genre)):
            # TODO: get img array
            image = Image.open(os.path.join(data_folderpath, genre, image))
            data = np.asarray(image)
            #label = str(np.where(np.asarray(labels)==os.path.basename(os.path.normpath(data_folderpath)))[0])
            #labels[data] = label
            
            img_array = []
            x.append(data)
            y.append(vector)

    #perm = np.random.permutation(len(x))
    #x = np.asarray(x)[perm]
    #y = np.asarray(y)[perm]


    cutoff = int(len(x) * split)
    x_train = x[:cutoff]
    x_test = x[cutoff:]

    y_train = y[:cutoff]
    y_test = y[cutoff:]

    return (x_train, y_train), (x_test, y_test)




(a, b), (c, d) = load_data(unsplit_data_path, 0.7)
temp = 0
