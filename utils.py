import os
import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage.transform import rescale
from scipy.ndimage import zoom

from args import parse_args


def load_mri(dirs):
    mri_mask_mltply = []
    side = dirs['SIDE']

    mask_dir = list(dirs['imgs'].values[0].split(","))# dirs['imgs'].values[0]

    #Flipping sides
    if dirs['SIDE'].values[0] == 2:
        mask_dir.reverse()

    masks = load(mask_dir, axis=(1, 2, 0), SIDE=side)

    # mri_dir = dirs['img_dir'].values[0]
    # mris = load(mri_dir, axis=(1, 2, 0), SIDE=side, mri_flg=True)

    # mri_mask_mltply.append(np.array(masks) * np.array(mris))

    mri_mask_mltply.append(np.array(masks))

    return mri_mask_mltply[0]


def load(files, axis=(0, 1, 2), SIDE=1, mri_flg=False, n_jobs=1):
    newlist = []
    for file in files:
        if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.tif'):
            try:
                int(file[-7:-4])
                newlist.append(file)
            except ValueError:
                continue
    files = newlist[:]  # replace list
    # Load images
    data = Parallel(n_jobs=n_jobs)(delayed(read_image)(file, SIDE, mri_flg) for file in files)
    # Transpose array
    # if axis != (0, 1, 2):
        # data = np.transpose(np.array(data), axis)
        # data = zoom(data, (0.5, 1, 1, 1), order=0) #3ch
        # data = zoom(data, (0.6, 1, 1), order=0) #1ch
        # return data![](../data/All/lateral/9000296/R/047.png)

    return np.array(data)


def read_image(f, SIDE=1, mri_flg=False, original_format=False):
    args = parse_args()

    if original_format:
        return cv2.imread(f, -1)
    else:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        try:
            c1 = np.zeros((img.shape))
        except:
            print(f)

        # img = img[110:img.shape[0] - 65, :]

        c1 = np.zeros((img.shape))
        c2 = np.zeros((img.shape))
        c3 = np.zeros((img.shape))

        if args.tissue == 'Bones':
            img[img == 1] = 1
            img[img == 3] = 1
            img[img != 1] = 0
            # c1[img == 1] = 1
            # c2[img == 3] = 1

        elif args.tissue == 'Cartilages':  # Beside menisci
            c1[img == 2] = 1
            c2[img == 4] = 1
            c3[img == 5] = 1
            c3[img == 6] = 1

        elif args.tissue == 'All':
            img[img == 1] = 1
            img[img == 3] = 1
            img[img == 2] = 2
            img[img == 4] = 2
            img[img == 5] = 3
            img[img == 6] = 3
            # c1[img == 1] = 1
            # c1[img == 3] = 1
            # c2[img == 2] = 1
            # c2[img == 4] = 1
            # c3[img == 5] = 1
            # c3[img == 6] = 1

        elif args.tissue == 'BoneMenisci':
            img[img == 2] = 0
            img[img == 4] = 0
            img[img == 1] = 1
            img[img == 3] = 1
            img[img == 5] = 2
            img[img == 6] = 2
            # c1[img == 1] = 1
            # c2[img == 3] = 1
            # c3[img == 5] = 1
            # c3[img == 6] = 1

        elif args.tissue == 'BoneCartilage':
            img[img == 5] = 0
            img[img == 6] = 0
            img[img == 1] = 1
            img[img == 3] = 1
            img[img == 2] = 2
            img[img == 4] = 2
            # c1[img == 1] = 1
            # c2[img == 3] = 1
            # c3[img == 2] = 1
            # c3[img == 4] = 1

        else:
            c1[img == 5] = 1
            c1[img == 6] = 1
            c2[img == 2] = 1
            c2[img == 4] = 1

        c1 = rescale(c1, 0.5, anti_aliasing=False, order=0)
        c2 = rescale(c2, 0.5, anti_aliasing=False, order=0)
        c3 = rescale(c3, 0.5, anti_aliasing=False, order=0)

        img_3ch = np.stack([c1, c2, c3])

        '''
        if img is None:
            print('This img is None:', f)
        if ~mri_flg:
            # img[img != 0] = 1
            # img[img == 5] = 0
            chert = 1
        '''
        '''
        try:
            img = rescale(img, 0.5, anti_aliasing=False)
        except:
            print(f)
        # print(len(img))'''
        img = rescale(img, 0.5, anti_aliasing=False)

        return img[None]  # img_3ch

