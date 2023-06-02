import os
import numpy as np

import numpy as np

from imageio import imwrite

from utils import ensure_dir


def save_set_of_images(path, images):
    if not os.path.exists(path):
        os.mkdir(path)

    images = images.transpose([0, 2, 3, 1])
    images = (np.clip(images, 0, 1) * 255).astype('uint8')

    for i, img in enumerate(images):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        imwrite(os.path.join(path, '%08d.png' % i), img)

def save_images_to_npy(path, images):
    images = images.transpose([0, 2, 3, 1])
    images = (np.clip(images, 0, 1) * 255).astype('uint8')

    np.save(path, images)