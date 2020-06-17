# Generate lst files given the folder containing sub-folders of images
import os
import random

folder = '<>/Cats_and_Dogs'
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

# Create a train and test lst file
f_train = open("my_lst_train_file.lst", "w")
f_test = open("my_lst_test_file.lst", "w")
image_index = 0
label_index = 0
label_encoding = {}
num_samples = len(subfolders)
for name in subfolders:
    label = name.split('/')[-1]
    label_encoding[label_index] = label
    onlyfiles = [f for f in os.listdir(name) if os.path.isfile(os.path.join(name, f))]
    for image_names in onlyfiles:
        # Randomly write 20% of the files into test lst
        if (random.uniform(0, 1) < 0.8):
            f_train.write(str(image_index) + '\t'+ str(label_index) + '\t'+ label+ '/' + image_names + '\n')
        else:
            f_test.write(str(image_index) + '\t'+ str(label_index) + '\t'+ label+ '/' + image_names + '\n')
        image_index += 1
    label_index += 1
f_train.close()
f_test.close()
print(label_encoding)