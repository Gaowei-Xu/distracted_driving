from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from cStringIO import StringIO
import PIL
import cv2

def plot_dist_merge(probs):
    classes = (
              'c0: safe driving / talking',
              'c1: operating the radio/makeup' ,
              'c2: dialing' ,
              'c3: drinking / reaching behind' ,
              'c4: texting'
            )


    x = np.arange(len(classes))


    plt.figure(figsize=(6.4, 2.4))
    ax1 = plt.axes([0.45, 0.1, 0.50, 0.84])

    ax1.barh(x, probs, align='center', color='g', alpha = 0.6)
    plt.ylim(-1, 5)
    plt.xlim(-0, 1)

    plt.xlabel('probs', size = 14, color='k')
    plt.ylabel('', size = 14, color='k')
    ax1.set_yticks(x)
    ax1.set_yticklabels(classes)


    buffer_ = StringIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    ar = np.asarray(image)
    buffer_.close()
    plt.close('all')

    return ar


    #plt.savefig('plot.png', dpi=100)





def plot_dist(probs):
    classes = ('c0: safe driving',
              'c1: texting - right',
              'c2: talking on the phone - right',
              'c3: texting - left',
              'c4: talking on the phone - left',
              'c5: operating the radio',
              'c6: drinking',
              'c7: reaching behind',
              'c8: hair and makeup',
              'c9: talking to passenger'
            )


    x = np.arange(len(classes))


    plt.figure(figsize=(6.4, 2.4))
    ax1 = plt.axes([0.45, 0.1, 0.50, 0.84])

    ax1.barh(x, probs, align='center', color='g', alpha = 0.6)
    plt.ylim(-1, 10)
    plt.xlim(-0, 1)

    plt.xlabel('probs', size = 14, color='k')
    plt.ylabel('', size = 14, color='k')
    ax1.set_yticks(x)
    ax1.set_yticklabels(classes)


    buffer_ = StringIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    ar = np.asarray(image)
    buffer_.close()
    plt.close('all')

    return ar












if __name__ == '__main__':
    probs = np.array([0.1, 0.2, 0.05, 0.15, 0.06, 0.04, 0.1, 0.1, 0.1, 0.1])
    ar = plot_dist(probs)

    cv2.imwrite("plot.png", ar)


