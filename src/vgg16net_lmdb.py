
from vgg16net import Vgg16netDetector
import os 
import cv2
import caffe 
import pandas as pd
import numpy as np
import lmdb
from random import randint, shuffle
from vgg16net_process import random_effect_process
from util import draw_rect
import re

#alexnet_path = "/home/victor/caffe/examples/alexnet"

train_image_path = "/home/guizi/data/kaggle/drive/train"

RCNN_TRAIN_IMG_PATH = "/home/guizi/result/rcnn/drive/pimgs_train"
RAW_TRAIN_IMG_PATH = "/home/guizi/data/kaggle/drive/train"
db_path = "/home/victor/caffe/examples/ddrive/data_original/org_train_lmdb"

keys = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
LMDB_DATA_COFF = 224*224*3*10

def get_random_crop_box(imgsize, crop_size, num):
    xlist = list()
    for i in range(num):
        _x = randint(0, imgsize-crop_size)
        _y = randint(0, imgsize-crop_size)
        _w = crop_size
        _h = crop_size
        xlist.append((_x, _y, _w, _h))
    return xlist 

def get_random_crop_box_v2(limitbox , crop_size, num):
    xlist = list()
    x_start, x_end, y_start, y_end = limitbox
    for i in range(num):
        _x = randint(x_start, x_end)
        _y = randint(y_start, y_end)
        _w = crop_size
        _h = crop_size
        xlist.append((_x, _y, _w, _h))
    return xlist 

def get_fixed_crops_data(img, crop_size):
    imglst = list()
    #1, resize to crop_size
    imglst.append(cv2.resize(img, (crop_size, crop_size)))
    #4 cornor crops and 1 center crops
    boxlst = [(0,0), (31, 31), (0,31), (31,0), (16,16) ]
    for box in boxlst:
        _x, _y = box
        ximg = img[_y:_y+crop_size, _x:_x+crop_size]
        imglst.append(ximg)
    return imglst
    
def create_db(dbpath, dbsize):
    _lmdb = lmdb.open(dbpath, map_size= dbsize*LMDB_DATA_COFF)
    _txn = _lmdb.begin(write=True)
    return (_lmdb, _txn, 0)

def db_push(xMat, txn, label, index, imgname):
    #split into B, G, R channels
    rgbMat = cv2.split(xMat)
    #print xMat.shape
    b_arr = np.asarray(rgbMat[0])
    g_arr = np.asarray(rgbMat[1])
    r_arr = np.asarray(rgbMat[2])

    data = np.array([b_arr, g_arr, r_arr])

    #convert to array_to_datum
    datum = caffe.io.array_to_datum(data, label)

    #<key, value> , key=index + imgname, value=datum.data
    str_id = '{:08}'.format(index)+"_"+imgname
    txn.put(str_id.encode('ascii'),datum.SerializeToString()) 
    
def get_label_from_filename(xstr):
    dirstr = xstr.strip().split("/")[-2]
    label = re.findall(r'\d+', dirstr)[0]
    return int(label)            
    
def generate_lmdb():
    # get list of files and do shuffle
    filelist = list()
    for xdir in os.listdir(train_image_path):
        dir_path = os.path.join(train_image_path, xdir)
        for xfile in os.listdir(dir_path):
            filelist.append(os.path.join(dir_path, xfile))
    print "total number of files ", len(filelist)    
    
    shuffle(filelist)
    
    boxlist = list()
    for mfile in filelist:
        for box in get_radom_crop_box(256, 224, 5):
            boxlist.append((mfile, box)) 
    shuffle(boxlist)
    
    print "total number of box list ", len(boxlist)    
    
    #create lmdb 
    db, dbtxn, dbindex = create_db(db_path, len(boxlist))
    
    # for each image , resize, get random box
    for i, item in enumerate(boxlist):
        mfile, box = item
        print "processing ", mfile
        img = cv2.imread(mfile)
        img = cv2.resize(img, (256, 256))

        _x, _y, _w, _h = box
        cropimg = img[_y:_y+_h, _x:_x+_w]
        db_push(cropimg, dbtxn, 
                get_label_from_filename(mfile), 
                dbindex, mfile.strip().split("/")[-1])
        dbindex = dbindex+1
    #commit db and close
    dbtxn.commit()
    db.close()
    print "lmdb convertion Done"

def generate_lmdb_shuffle():
    # get list of files and do shuffle
    filelist = list()
    for xdir in os.listdir(train_image_path):
        dir_path = os.path.join(train_image_path, xdir)
        for xfile in os.listdir(dir_path):
            filelist.append(os.path.join(dir_path, xfile))
    print "total number of files ", len(filelist)    
    
    #create lmdb 
    db, dbtxn, dbindex = create_db(db_path, len(filelist)*5)
    
    # for each image , resize, get random box
    for i, mfile in enumerate(filelist):
        print "processing ", mfile
        img = cv2.imread(mfile)
        img = cv2.resize(img, (256, 256))
        for box in get_radom_crop_box(256, 224, 5):
            _x, _y, _w, _h = box
            cropimg = img[_y:_y+_h, _x:_x+_w]
            db_push(cropimg, dbtxn, 
                    get_label_from_filename(mfile), 
                    dbindex, mfile.strip().split("/")[-1])
            dbindex = dbindex+1
    #commit db and close
    dbtxn.commit()
    db.close()
    print "lmdb convertion Done"


def shuffle_train_file(trainfile,  outfile, epochs):
    xfile = open(trainfile, 'r')
    orglist = xfile.readlines()
    xlist = list()
    for i in range(epochs):
        tmplist = orglist
        shuffle(tmplist)
        xlist.extend(tmplist)
    print len(xlist) , 'after ',  epochs, "shuffules"
    
    ofile = open(outfile, 'w')
    for line in xlist:
        ofile.write(line)
    ofile.close()

def data_augmentation(trainfile, db_path, epochs):
            
    # get list of files and do shuffle b/w epochs
    xfile = open(trainfile, 'r')
    orglist = xfile.readlines()
    xlist = list()
    for i in range(epochs):
        tmplist = orglist
        shuffle(tmplist)
        xlist.extend(tmplist)
    print len(xlist) , 'after ',  epochs, "shuffules"
            
    #create lmdb 
    db, dbtxn, dbindex = create_db(db_path, len(xlist))
    
    fixedcroplst = [(-1, -1), (0,0), (31, 31), (0,31), (31,0), (16,16) ]
    
    # for each image , resize, get random crop and select one effect
    for i, xline in enumerate(xlist):
        imgfile = xline.strip().split(" ")[0]
        label   = int(xline.strip().split(" ")[-1]) 
        if not os.path.isfile(os.path.join(RCNN_TRAIN_IMG_PATH ,imgfile)):
            continue
        #print "processing ", os.path.join(RCNN_TRAIN_IMG_PATH ,imgfile)
        #load and resize
        img = cv2.imread(os.path.join(RCNN_TRAIN_IMG_PATH ,imgfile))
        img = cv2.resize(img, (256, 256))
        #get crop from list
        _x, _y = fixedcroplst[randint(0, len(fixedcroplst)-1)]
        if _x < 0:
            cropimg = cv2.resize(img, (224, 224))
        else:
            cropimg = img[_y:_y+224, _x:_x+224]
        #pick random effect
        ximg = random_effect_process(cropimg)
        #push to db
        db_push(ximg,  dbtxn, 
                label, dbindex, imgfile+'_'+str(_x)+'+'+str(_y))
        dbindex = dbindex+1
        #commit once for each epoch
        if i%1000==0:
           print "committing to db each 1k", i
           dbtxn.commit()
           dbtxn = db.begin(write=True)

    #commit db and close
    dbtxn.commit()
    db.close()
    print "data augmentation lmdb convertion Done"    


def data_to_lmdb(trainfile, db_path, epochs, flags):
    boxlst = [(0,0), (31, 31), (0,31), (31,0), (16,16) ]
    # get list of files and do shuffle b/w epochs
    xfile = open(trainfile, 'r')
    orglist = xfile.readlines()
    xlist = list()
    for i in range(len(boxlst)):
        tmplist = orglist
        shuffle(tmplist)
        xlist.extend(tmplist)
    print len(xlist) , 'after ',  epochs, "shuffules"
    
    #create lmdb 
    db, dbtxn, dbindex = create_db(db_path, len(xlist))
    
    # for each image , resize, get random crop and select one effect
    for i, xline in enumerate(xlist):
        imgfile = xline.strip().split(" ")[0]
        label   = int(xline.strip().split(" ")[-1]) 
        
        #print "processing ", os.path.join(RCNN_TRAIN_IMG_PATH ,imgfile)
        if not os.path.isfile(os.path.join(RCNN_TRAIN_IMG_PATH, imgfile)):
            continue
        #load and resize
        img = cv2.imread(os.path.join(RCNN_TRAIN_IMG_PATH ,imgfile))
        img = cv2.resize(img, (256, 256))
        if flags == 'VariableWidth':
            _x, _y = boxlst[randint(0, len(boxlst)-1)]
            _w = randint(224, 256-_x-1) 
            _h = randint(224, 256-_y-1) 
            cropimg = img[_y:_y+_h, _x:_x+_w]
            cropimg = cv2.resize(cropimg, (224, 224))
        elif flags == 'FixedCrops':
            #fixed crops
            _x, _y = boxlst[randint(0, len(boxlst)-1)]
            cropimg = img[_y:_y+224, _x:_x+224]
        

        #push to db
        db_push(cropimg,  dbtxn, 
                label, dbindex, imgfile)
        
        dbindex = dbindex+1
        #commit once for each epoch
        if i%1000==0:
           print "committing to db : ", i
           dbtxn.commit()
           dbtxn = db.begin(write=True)

    #commit db and close
    dbtxn.commit()
    db.close()
    print "data augmentation lmdb convertion Done"  
    
def convert_train_file(trainfile, outfile):
    xfile = open(trainfile, 'r')
    xout = open(outfile, 'w')
    for line in xfile.readlines():
        filename = line.strip().split(" ")[0]
        label = line.strip().split(" ")[-1]
        subpath = filename[0:2]
        imgname =  filename[3:]
        
        xout.write(subpath+'_'+imgname+" "+label+"\n")
    xout.close()    

def convert_shuffle(infile, outfile):
    xfile = open(infile, 'r')
    xout = open(outfile, 'w')
    xlist = xfile.readlines()
    shuffle(xlist)
    for line in xlist:
        filename = line.strip().split(" ")[0]
        label = line.strip().split(" ")[-1]
        xout.write(filename+" "+label+"\n")
    
    '''
    for line in xlist:
        filename = line.strip().split(" ")[0]
        label = line.strip().split(" ")[-1]
        subpath = filename[0:2]
        imgname =  filename[3:]
        
        xout.write(subpath+'/'+imgname+" "+label+"\n")
    '''
    xout.close()    


def merge_file_shuffle(ina, inb, out):
    xina = open(ina, 'r')
    xinb = open(inb, 'r')
    xout = open(out, 'w')
    
    xlist = xina.readlines()
    xlist.extend(xinb.readlines())
    shuffle(xlist)
    
    for line in xlist:
        filename = line.strip().split(" ")[0]
        label = line.strip().split(" ")[-1]
        xout.write(filename+" "+label+"\n")
    
    xina.close()
    xinb.close()
    xout.close()
    
if __name__ == "__main__":
   
    data_augmentation("/home/victor/caffe/examples/ddrive/data_rcnn_driver/train.txt",
                     "/home/victor/caffe/examples/ddrive/data_rcnn_driver/xaugdata_train_v2_lmdb",
                     6)
   
    '''
    merge_file_shuffle("/home/victor/caffe/examples/ddrive/data_rcnn_driver/train.txt",
                       "/home/victor/caffe/examples/ddrive/data_rcnn_driver/val.txt",
                       "/home/victor/caffe/examples/ddrive/data_rcnn_driver/fulltrain.txt",)
    '''
    '''
    shuffle_train_file("/home/victor/caffe/examples/ddrive/data_org/train_org.txt",
                       "/home/victor/caffe/examples/ddrive/data_org/train_org_5.txt", 5)
    '''
    '''
    convert_train_file(
                    "/home/victor/caffe/examples/vgg1024/data_raw_driver/train.txt",
                    "/home/victor/caffe/examples/ddrive/data_rcnn_driver/train.txt")
                       
    convert_train_file(
                    "/home/victor/caffe/examples/vgg1024/data_raw_driver/val.txt",
                    "/home/victor/caffe/examples/ddrive/data_rcnn_driver/val.txt"
                    )
    '''
    '''              
    data_to_lmdb("/home/victor/caffe/examples/ddrive/data_rcnn_driver/train.txt",
                  "/home/victor/caffe/examples/ddrive/data_rcnn_driver/xcrops_train_lmdb", 
                  5, 'FixedCrops')
    '''
      
    '''           
    data_to_lmdb("/home/victor/caffe/examples/vgg1024/data_raw_driver/train.txt",
                  "/home/victor/caffe/examples/vgg1024/data_raw_driver/xcrops_train_lmdb", 
                  5, 'FixedCrops')
    '''
    #generate_lmdb()
    
    #shuffle_train_file("/home/victor/caffe/examples/ddrive/data_rcnn_shuffle/train.txt",
    #                    "/home/victor/caffe/examples/ddrive/data_rcnn_shuffle/train_10shuffle.txt", 10)