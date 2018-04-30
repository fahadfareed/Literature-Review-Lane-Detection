
from moviepy.editor import VideoFileClip
import numpy as np
import os
from detect import detect
import scipy.misc
import csv
from os import listdir as ls
from utils import y_

def extract_frames(moviedir, times, imgdir):
    """
    Extract specific 'times' from a 'movedir' and save
    frames in 'imagedir'.
    """
    clip = VideoFileClip(moviedir)
    for t in times:
        imgpath = os.path.join(imgdir, '{}.jpeg'.format(t))
        clip.save_frame(imgpath, t)
        print(imgpath[:-5])

def get_grid(coordinates):
    
    grid = np.zeros(shape=(11, 64))

    for c in coordinates:
        idx = np.abs(y_ - c[1]).argmin()
        x = int(c[0] / 20.0)
        grid[idx][x] = 1

    return [g for gr in grid for g in gr]

def save_coordinates(imgdir, times, outdir=None, save=None):
    """
    Find coordinates of lanes from images in 'imagedir'
    corresponding to specific 'times'. If 'times' is None,
    recover times from a csv. 

    Save output images in 'outdir' if the parameter is
    not None.

    Save coordinates in csv format if 'save' is not None.
    """
    coordinates, grids = [], []

    for t in times:
        try:
            imgpath = os.path.join(imgdir, '{}.jpeg'.format(t))
            frame = scipy.misc.imread(imgpath)

            print(imgpath[:-5])
            coords, out = detect(imgpath[:-5])
            grid = get_grid(coords)

            if outdir is not None:
                outpath = os.path.join(outdir, '{}.jpeg'.format(t))
                scipy.misc.imsave(outpath, out)

            coordinates.append(coords)
            grids.append(grid)

        except:
            continue

    rows = zip(times, grids)

    if 'save' is not None:
        with open('../data_grid.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

movie = '../videos/training_16.mp4'
imgdir = '../images'
outdir = '../images/outputs'
paths = [float(x[:-5]) for x in ls('../images/outputs') if x[-5:] == '.jpeg']

save_coordinates('../images_new', times=paths, save='True')

"""
import time as t
for p in paths:
    t1 = t.time()
    detect(imgdir + '/' + str(p))
    print("Time taken: ", t.time() - t1)
"""

#get_grid(detect('../images/40.1'))