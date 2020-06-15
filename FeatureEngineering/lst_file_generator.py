# Generate lst files given the folder containing sub-folders of images
import os
folder = '<your_path>/Cats_and_Dogs'
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

# Create a lst file
f = open("my_lst_file.lst", "w")
image_index = 0
label_index = 0
label_encoding = {}
for name in subfolders:
    label = name.split('/')[-1]
    label_encoding[label_index] = label
    onlyfiles = [f for f in os.listdir(name) if os.path.isfile(os.path.join(name, f))]
    for image_names in onlyfiles:
        f.write(str(image_index) + '\t'+ str(label_index) + '\t'+ label+ '/' + image_names + '\n')
        image_index += 1
    label_index += 1
f.close()
print(label_encoding)