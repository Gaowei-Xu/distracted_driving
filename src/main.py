from vgg16net import Vgg16netDetector
import os
import cv2
import caffe
import pandas as pd
import numpy as np
from random import randint
import time
import logging
from plotdist import plot_dist
from time import gmtime, strftime
from para_cfg import ParaCfg
from logger import Logger
from forward import Forward


def np_to_string(arr):
    xstr = ''
    for i in range(len(arr)):
        if i == 0:
            xstr = xstr +str(arr[i])
        else:
            xstr = xstr + ',' +str(arr[i])
    return xstr


def get_random_crop_box(imgsize, crop_size, num):
    xlist = list()
    for i in range(num):
        _x = randint(0, imgsize-crop_size)
        _y = randint(0, imgsize-crop_size)
        _w = crop_size
        _h = crop_size
        xlist.append((_x, _y, _w, _h))
    return xlist


def vgg16_eval_full_test(IMAGE_PATH, outfile, model_file, deploy_file, mean_binary_file):

    timestamp_start = time.time()

    #init net
    vggnet = Vgg16netDetector(VGGNET_PATH,
            model_path = model_file,
            deploy_path = deploy_file,  mean_binary_path = mean_binary_file, buse_cvmat=True)

    xoutfile = open(outfile, 'w');

    timestamp_loadvgg = time.time()

    imgs = os.listdir(IMAGE_PATH)
    imgs_num = len(imgs)

    for i, xfile in enumerate(os.listdir(IMAGE_PATH)):
        image_color = cv2.imread(os.path.join(IMAGE_PATH, xfile))
        predict_label, probs = vggnet.predict_singleimg(image_color)
        xoutfile.write(xfile + np_to_string(probs)+"\n")

    xoutfile.close()
    timestamp_end = time.time()

    print "Loading vggnet uses " + str(timestamp_loadvgg - timestamp_start) + " seconds."
    print "Forward uses totally " + str(timestamp_end - timestamp_loadvgg) + " seconds."
    print "Forward time consumption: " + str((timestamp_end - timestamp_loadvgg)*1000.0/imgs_num) + " MilliSecond per image."

    print "Done"



def vgg16_eval_val_randomcrops_score(imgpath, outfile, NUMBER_OF_CROPS):

    timestamp_start = time.time()

    #init cifarnet
    vggnet = Vgg16netDetector(VGGNET_PATH,
            model_path = "vgg16_finetune_iter_15000_hingeloss.caffemodel",
            deploy_path= "deploy.prototxt",  mean_binary_path="imagenet_mean.npy", buse_cvmat=True)

    xoutfile = open(outfile, 'w');

    timestamp_loadvgg = time.time()

    imgs = os.listdir(IMAGE_PATH)
    imgs_num = len(imgs)

    for i, xfile in enumerate(os.listdir(IMAGE_PATH)):
        image_color = cv2.imread(os.path.join(imgpath, xfile))
        croplst = get_random_crop_box(256, 224, NUMBER_OF_CROPS)
        imglst = list()
        for box in croplst:
            _x, _y, _w, _h = box
            imglst.append(image_color[_y:_y+_h, _x:_x+_w])

        probs = vggnet.predict_randomcrops(imglst)

        #compute average score
        rows, cols = probs.shape
        avg_probs = np.zeros((1, cols))
        for i in range(rows):
            avg_probs = avg_probs + probs[i:i+1]
        avg_probs = avg_probs/rows
        avg_probs = avg_probs.flatten()
        #print avg_probs
        xoutfile.write(xfile + np_to_string(probs)+"\n")

        predict_label = avg_probs.argmax()


    xoutfile.close()
    timestamp_end = time.time()

    print "Loading vggnet uses " + str(timestamp_loadvgg - timestamp_start) + " seconds."
    print "Forward uses totally " + str(timestamp_end - timestamp_loadvgg) + " seconds."
    print "Forward time consumption: " + str((timestamp_end - timestamp_loadvgg)*1000.0/imgs_num) + " MilliSecond per image (numofcrops = " + str(NUMBER_OF_CROPS) + ")."
    print "Forward time consumption: " + str((timestamp_end - timestamp_loadvgg)*1000.0/imgs_num/NUMBER_OF_CROPS) + " MilliSecond per frame."
    print "Done"



def forward_webcam():
    return

def forward_local_imageset():
    return


def main():
    para_cfg_obj = ParaCfg('../config/appConfig.cfg')

    logger_obj = Logger(para_cfg_obj.LOGFILE_PATH)
    track_logger = logger_obj.get_logger()

    para_cfg_obj.to_log(track_logger)

    forward_obj = Forward(track_logger, para_cfg_obj)
    forward_obj.run()

    #vgg16_eval_full_test(IMAGE_PATH, OUTPUT_FILE, 0)

    #vgg16_eval_val_randomcrops_score(IMAGE_PATH, OUTPUT_FILE, NUMBER_OF_CROPS)

    #local_video_forward(VIDEO_PATH, 0, OUTPUT_FILE)

if __name__ == '__main__':
    main()
