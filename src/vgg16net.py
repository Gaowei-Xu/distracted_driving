import cv2
import sys, os
import caffe
import numpy as np
#import lmdb
import random as rn
import time

BATCH_SIZE = 512
ALEX_DATA_WIDTH = 224
ALEX_DATA_HEIGHT = 224

class Vgg16netDetector:
    """Detector"""
    model_path = None
    net = None
    transformer = None

    def __init__(self, path, model_path, deploy_path, mean_binary_path, buse_cvmat):
        self._init_net(path, model_path, deploy_path, mean_binary_path, buse_cvmat)


    def _init_net(self, path, model_path, deploy_path, mean_binary_path, b_use_cvmat):
        caffe.set_mode_gpu()
        caffe.set_device(0)

        self.model_path = model_path
        self.net = caffe.Net(os.path.join(path, deploy_path),
                    os.path.join(path, model_path),
                    caffe.TEST)

        #blob = caffe.proto.caffe_pb2.BlobProto()
        #data = open( os.path.join(path, mean_binary_path), 'rb' ).read()
        #blob.ParseFromString(data)
        #arr = np.array(caffe.io.blobproto_to_array(blob))

        #init transformer
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))

        self.transformer.set_mean('data', np.load(os.path.join(path, mean_binary_path)).mean(1).mean(1))
        print
        if not b_use_cvmat:
           self.transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR
           self.transformer.set_raw_scale('data', 255.0) # caffe.io.load_image() load image as 0-1

    def predict(self, imagelist):
        return self.net_forward(imagelist)


    def predict_singleimg(self, img):
        self.net.blobs['data'].reshape(1,3,ALEX_DATA_WIDTH,ALEX_DATA_HEIGHT)
        self.net.blobs['data'].data[0] = self.transformer.preprocess('data', img)

        out = self.net.forward()

        label = out['prob'][0].argmax()
        return (label, out['prob'][0])


    def predict_randomcrops(self, imglst):
        self.net.blobs['data'].reshape(len(imglst),3,ALEX_DATA_WIDTH,ALEX_DATA_HEIGHT)
        for i , imgdata in enumerate(imglst):
            self.net.blobs['data'].data[i] = self.transformer.preprocess('data', imgdata)
        out = self.net.forward()
        return out['prob']

    """
    def net_forward(self, imagelst):
        '''
        :param image: input image
        :return:
            probability : probability of prediction as true
        '''
        print "cnn forward processing ", len(imagelst), " windows "

        idx = len(imagelst) / BATCH_SIZE                            # BATCH_SIZE = 512
        remain_count = len(imagelst) - idx * BATCH_SIZE
        prediction = []

        #forward and return prediction back
        #out = self.net.forward()
        #prediction = [x[1] for x in out['prob']]

        for it in range(idx + 1):
            #fixme: it is better to split into bulks.
            if it < idx:
                sub_images = imagelst[it * BATCH_SIZE : (it + 1) * BATCH_SIZE]
                self.net.blobs['data'].reshape(BATCH_SIZE,3,ALEX_DATA_WIDTH,ALEX_DATA_HEIGHT)
            else :
                if remain_count == 0 :
                    break
                self.net.blobs['data'].reshape(remain_count,3,ALEX_DATA_WIDTH,ALEX_DATA_HEIGHT)
                sub_images = imagelst[it * BATCH_SIZE : ]

            # push windows int data blob
            for i, img in enumerate(sub_images):
                self.net.blobs['data'].data[i] = self.transformer.preprocess('data', img)

            out = self.net.forward()

            print out['prob'][0].argmax()

            label_file = "/home/victor/caffe/data/ilsvrc12/synset_words.txt"
            labels = np.loadtxt(label_file, str, delimiter='\t')

            print 'output label ', labels[out['prob'][0].argmax()]

            top5 = out['prob'][0].argsort()[::-1][:5]
            print "top5", zip(out['prob'][0][top5], labels[top5])

        """







