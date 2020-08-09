import numpy as np
from skimage import exposure

def image_from_separated_rgb(image_r, image_g, image_b, clip_min_pct=2, clip_max_pct=98, equalize_hist=True):
    '''
    return array(m, n, 3) of RGB image (0-255)
    '''
    image_b = np.clip(image_b, np.percentile(image_b, clip_min_pct), np.percentile(image_b, clip_max_pct))
    image_g = np.clip(image_g, np.percentile(image_g, clip_min_pct), np.percentile(image_g, clip_max_pct))
    image_r = np.clip(image_r, np.percentile(image_r, clip_min_pct), np.percentile(image_r, clip_max_pct))

    if equalize_hist == True:
        image_b = exposure.equalize_hist(image_b)
        image_g = exposure.equalize_hist(image_g)
        image_r = exposure.equalize_hist(image_r)

    image_b = (image_b - image_b.min()) / (image_b.max() - image_b.min())
    image_g = (image_g - image_g.min()) / (image_g.max() - image_g.min())
    image_r = (image_r - image_r.min()) / (image_r.max() - image_r.min())

    image_b = image_b * 255
    image_g = image_g * 255
    image_r = image_r * 255

    # image = make_lupton_rgb(image_r, image_g, image_b, stretch=5, Q=5)
    image = np.array(list(zip(image_r.flatten(), image_g.flatten(), image_b.flatten())), dtype=np.int16).reshape(*image_r.shape, 3)
    return image