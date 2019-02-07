from para_cfg import ParaCfg
from logger import Logger
import time
from vgg16net import Vgg16netDetector
import os
import cv2
import caffe
import pandas as pd
import numpy as np
from random import randint
from plotdist import plot_dist_merge

class Forward:
    def __init__(self, track_logger, para_cfg_obj):
        self.net_load_start = time.time()

        # init net
        self.vggnet = Vgg16netDetector( para_cfg_obj.NET_BASE_PATH,
                                        para_cfg_obj.MODEL_PATH,
                                        para_cfg_obj.DEPLOY_PATH,
                                        para_cfg_obj.MEAN_BINARY_PATH,
                                        buse_cvmat = True )

        self.para = para_cfg_obj
        self.logger = track_logger
        self.net_load_end = time.time()
        self.logger.info('Loading net model and weights uses ' + str(self.net_load_end - self.net_load_start) + ' seconds.')

    def run(self):
        if self.para.INPUT_TYPE == 'webcam':
            self.run_webcam()
        elif self.para.INPUT_TYPE == 'imageset':
            self.run_local_imageset()
        elif self.para.INPUT_TYPE == 'video':
            self.run_local_video()
        else:
            self.logger.error('INPUT_TYPE invalid, please check the configure file.')
        return


    # private methods
    def __get_random_crop_box(self, imgsize, crop_size, num):
        xlist = list()
        for i in range(num):
            _x = randint(0, imgsize-crop_size)
            _y = randint(0, imgsize-crop_size)
            _w = crop_size
            _h = crop_size
            xlist.append((_x, _y, _w, _h))
        return xlist


    def __np_to_string(self, arr):
        xstr = ''
        for i in range(len(arr)):
            if i == 0:
                xstr = xstr +str(arr[i])
            else:
                xstr = xstr + ',' +str(arr[i])
        return xstr


    def __merge_prob(self, probs):
        # merge the 10 classes into 5 classes
        '''
        c0 --------> c0 and c9                        c1 --------> c5 and c8
        c2 --------> c2 and c4                        c3 --------> c6 and c7
        c4 --------> c1 and c3
        '''
        probs_5classes = [ probs[0]+probs[9], probs[5]+probs[8], probs[2]+probs[4], probs[6]+probs[7], probs[1]+probs[3] ]
        return probs_5classes



    def run_local_video(self):
        timestamp_start = time.time()

        stride = self.para.STRIDE


        cap = cv2.VideoCapture(self.para.VIDEO_PATH)

        processed_frame_index = 0
        frame_count = 0

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == False:
                break

            frame_count += 1

            if frame_count % stride != 0:
                continue

            processed_frame_index += 1
            crop_img = frame[0:720, 240:1280]                               # crop the size to 640 x 480
            img_in = cv2.resize(crop_img, (self.para.FRAME_WIDTH, self.para.FRAME_HEIGHT))
            predict_label, probs = self.vggnet.predict_singleimg(img_in)    # forward


            probs_5classes = self.__merge_prob(probs)
            predict_label_5classes = np.argmax(probs_5classes)

            frame_name = str(frame_count) + "_c" + str(predict_label_5classes) + ".jpg"

            if self.para.GENERATE_RAW_FRAME:
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_PATH, frame_name), img_in)

            if self.para.GENERATE_FRAME_WITH_PROB:
                dist_probs_img = plot_dist_merge(probs_5classes)
                dist_probs_img_resized = cv2.resize(dist_probs_img[:,:,0:3], (self.para.FRAME_WIDTH, self.para.FRAME_HEIGHT/2))
                merge_image = np.concatenate((img_in, dist_probs_img_resized), axis = 0)
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_WITH_PROB_PATH, frame_name), merge_image)

            if self.para.SHOW_FRAME_REALTIME:
                if self.para.GENERATE_FRAME_WITH_PROB:
                    cv2.imshow('frame', merge_image)
                else:
                    cv2.imshow('frame', img_in)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.para.GENERATE_LOGFILE:
                self.logger.info(frame_name + "\t" + self.__np_to_string(probs_5classes) + "\t" + str(predict_label_5classes))

        timestamp_end = time.time()

        self.logger.info("Forward uses totally " + str(timestamp_end - timestamp_start) + " seconds.")
        self.logger.info("Forward time consumption: " + str((timestamp_end - timestamp_start)*1000.0/processed_frame_index) + " MilliSecond per frame.")
        self.logger.info("DONE")

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()
        return



    def run_local_imageset(self):
        timestamp_start = time.time()
        image_path = self.para.IMAGE_SET_BASE_PATH

        imgs = os.listdir(image_path)
        imgs_num = len(imgs)

        for i, xfile in enumerate(os.listdir(image_path)):
            img_in = cv2.imread(os.path.join(image_path, xfile))

            t_start = time.time()
            predict_label, probs = self.vggnet.predict_singleimg(img_in)
            t_end = time.time()
            self.logger.info("Processing " + str(xfile) + ' uses ' + str(t_end-t_start) + 's...')

            probs_5classes = self.__merge_prob(probs)
            predict_label_5classes = np.argmax(probs_5classes)

            img_name = xfile.split('.jpg')[0] + "_c" + str(predict_label_5classes) + ".jpg"

            if self.para.GENERATE_RAW_FRAME:
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_PATH, img_name), img_in)

            if self.para.GENERATE_FRAME_WITH_PROB:
                dist_probs_img = plot_dist_merge(probs_5classes)
                dist_probs_img_resized = cv2.resize(dist_probs_img[:,:,0:3], (self.para.FRAME_WIDTH, self.para.FRAME_HEIGHT/2))
                merge_image = np.concatenate((img_in, dist_probs_img_resized), axis = 0)
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_WITH_PROB_PATH, img_name), merge_image)

            if self.para.SHOW_FRAME_REALTIME:
                if self.para.GENERATE_FRAME_WITH_PROB:
                    cv2.imshow('frame', merge_image)
                else:
                    cv2.imshow('frame', img_in)

                # delay
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        timestamp_end = time.time()

        self.logger.info("Forward uses totally " + str(timestamp_end - timestamp_start) + " seconds.")
        self.logger.info("Forward time consumption (including IO): " + str((timestamp_end - timestamp_start)*1000.0/imgs_num) + " MilliSecond per frame.")
        self.logger.info("DONE")
        return




    def run_webcam(self):
        stride = self.para.STRIDE

        processed_frame_index = 0
        frame_count = 0

        cap = cv2.VideoCapture(0)

        while(True):
            t_start = time.time()
            ret, frame = cap.read()

            if ret == False:
                break

            frame_count += 1
            if frame_count % stride != 0:
                continue

            processed_frame_index += 1
            img_in = cv2.resize(frame, (self.para.FRAME_WIDTH, self.para.FRAME_HEIGHT))
            predict_label, probs = self.vggnet.predict_singleimg(img_in)    # forward
            t_end = time.time()
            self.logger.info("Processing frame " + str(processed_frame_index) + ' (including IO) uses ' + str(t_end-t_start) + 's...')

            probs_5classes = self.__merge_prob(probs)
            predict_label_5classes = np.argmax(probs_5classes)

            frame_name = str(frame_count) + "_c" + str(predict_label_5classes) + ".jpg"

            if self.para.GENERATE_RAW_FRAME:
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_PATH, frame_name), img_in)

            if self.para.GENERATE_FRAME_WITH_PROB:
                dist_probs_img = plot_dist_merge(probs_5classes)
                dist_probs_img_resized = cv2.resize(dist_probs_img[:,:,0:3], (self.para.FRAME_WIDTH, self.para.FRAME_HEIGHT/2))
                merge_image = np.concatenate((img_in, dist_probs_img_resized), axis = 0)
                cv2.imwrite(os.path.join(self.para.OUTPUT_FRAMES_WITH_PROB_PATH, frame_name), merge_image)

            if self.para.SHOW_FRAME_REALTIME:
                if self.para.GENERATE_FRAME_WITH_PROB:
                    cv2.imshow('frame', merge_image)
                else:
                    cv2.imshow('frame', img_in)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.para.GENERATE_LOGFILE:
                self.logger.info(frame_name + "\t" + self.__np_to_string(probs_5classes) + "\t" + str(predict_label_5classes))


        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()
        return

