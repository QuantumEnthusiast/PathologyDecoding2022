import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from PIL import Image
import numpy as np
import features.config as c

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
         'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG',
            'PuOr', 'RdGy', 'RdBu','RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm',
            'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']

def generate_spectogram_image(EEG, file_name, fmin, fmax, cmap='viridis'):

    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(1, 1, 1)
    plt.specgram(x=EEG, NFFT=c.NFFT,
                 window=mlab.window_hanning,
                 Fs=c.fs,
                 noverlap=c.NOVERLAP, cmap=cmap)  # 'Paired')
    plt.axis('off')
    plt.ylim([fmin, fmax])
    imgName = file_name + '.png'
    fig.savefig(imgName)
    plt.close()  # do not show output

    # crop spectrogram
    image = Image.open(imgName).convert("L")
    width, height = image.size

    M = np.asarray(image).astype('int32')
    cut_x = []
    cut_y = []
    for i in range(width):
        if np.sum(M[:, i]) != np.sum([255] * height):
            cut_x.append(i)
    for j in range(height):
        if np.sum(M[j]) != np.sum([255] * width):
            cut_y.append(j)

    cut_x1 = cut_x[0]
    cut_x2 = cut_x[-1]
    cut_y1 = cut_y[0]
    cut_y2 = cut_y[-1]
    # box=(left, upper, right, lower)
    image = image.crop((cut_x1, cut_y1, cut_x2, cut_y2))
    image.save(imgName, 'PNG')
    print('Spectrogram image is successfully saved')

    return imgName