import os
from os import path as p
import shutil
from PIL import Image

l = os.listdir('E:\PROJECTS\LUNGS\___Aorta\AortaRazv\\norm')

pp = 'E:\PROJECTS\LUNGS\___Aorta\AortaRazv'
for n in l:
    ps = p.join(pp, 'norm', n)
    img = Image.open(ps)
    if img.width != 256 or img.height != 256:
        print(n)
        img.close()
        shutil.move(ps, p.join(pp, 'original_norm'))

