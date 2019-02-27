import os
from sklearn.model_selection import train_test_split as tts

# Prepare classification configuration files (train.txt, test.txt).
# Splitting the dataset in some ways create 3 problems: one 4 classes, two 2 classes.

absp = '/media/data10T_1/datasets/Voynov/Histology_4K_Tifs/'
folders = ['ovary_n', 'thyroid_n', 'ovary_t', 'thyroid_t']

png_imgs = []
labels = []
for i, folder in enumerate(folders):
    ldir = os.listdir(os.path.join(absp, folder))
    png_imgs_dir = [os.path.join(absp, folder, d) for d in ldir if d.endswith('png')]
    png_imgs += png_imgs_dir
    labels += [i] * len(png_imgs_dir)

print(len(png_imgs))
print(len(labels))
print(set(labels))

def compose_line(f, l):
    return f + ' ' + str(l) + '\n'

# 4 classes configuration
absp = '../__histology_tifs_4classes/'
train_x, test_x, train_y, test_y = tts(png_imgs, labels, test_size=0.2, stratify=labels, random_state=42)
file = open(os.path.join(absp, 'train.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(train_x, train_y)])
file = open(os.path.join(absp, 'test.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(test_x, test_y)])

# 2 classes configuration ovary
absp = '../__histology_tifs_2classes_ovary/'
ov_imgs = [t[0] for t in zip(png_imgs, labels) if t[1] % 2 == 0]
ov_labels = [t[1] // 2 for t in zip(png_imgs, labels) if t[1] % 2 == 0]
train_x, test_x, train_y, test_y = tts(ov_imgs, ov_labels, test_size=0.2, stratify=ov_labels, random_state=42)
file = open(os.path.join(absp, 'train.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(train_x, train_y)])
file = open(os.path.join(absp, 'test.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(test_x, test_y)])

# 2 classes configuration ovary
absp = '../__histology_tifs_2classes_thyroid/'
ov_imgs = [t[0] for t in zip(png_imgs, labels) if t[1] % 2 == 1]
ov_labels = [t[1] // 2 for t in zip(png_imgs, labels) if t[1] % 2 == 1]
train_x, test_x, train_y, test_y = tts(ov_imgs, ov_labels, test_size=0.2, stratify=ov_labels, random_state=42)
file = open(os.path.join(absp, 'train.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(train_x, train_y)])
file = open(os.path.join(absp, 'test.txt'), 'w')
file.writelines([compose_line(*t) for t in zip(test_x, test_y)])
