""" Save images and corresponding labels as TFRecord files
    http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle
from random import seed

import numpy as np
import cv2 as cv
import pandas as pd
import tensorflow as tf
import os
import sys

sys.path.append('../')
import pose_detection.nnpose_zoning as config
from image_processing import transformations as imgtf
from tf_neural_net import commons as com


def set_test_mode():
    global tfmode, npmode, modeDir
    npmode, tfmode = False, False
    if len(sys.argv) == 2:
        runmode = sys.argv[1]

        if not (runmode == '-np' or '-tf'):
            print('Run-mode must be -np: numpy format, -tf: tfRecorder format')
            sys.exit()
        else:
            if runmode == '-tf':
                tfmode = True
                modeDir = 'tfrecord'
            elif runmode == '-np':
                npmode = True
                modeDir = 'numpy'
    else:
        print('Specify mode for running script. e.g. -tf')
        sys.exit()


# returns symetric zoneid
def get_symetric_pair(zid):
    if zid == 0: return 2
    if zid == 2: return 0
    if zid == 1: return 3
    if zid == 3: return 1
    if zid == 4: return None
    if zid == 5: return 6
    if zid == 6: return 5
    if zid == 7: return 9
    if zid == 8: return None
    if zid == 9: return 7
    if zid == 10: return 11
    if zid == 11: return 10
    if zid == 12: return 13
    if zid == 13: return 12
    if zid == 14: return 15
    if zid == 15: return 14
    if zid == 16: return None


def get_frame_visual_order(zoneid):
    validframes = validZonesPerFrame[:, zoneid]
    # next represents fid to start from, step represents which direction to move
    next, step = zoneSortOrder[zoneid + 1]
    order = []
    for i in range(16):
        fid = next % 16
        if validframes[fid]:
            order.append(fid)
        next += step

    return np.array(order)


# sort cropped images in order that zone region appears to be rotating
def sort_by_visual_order(zonelist, zoneid):
    order = get_frame_visual_order(zoneid)
    assert (len(order) == len(zonelist))
    sortlist = []
    for i in range(len(order)):
        nextFrame = order[i]
        frameName = 'Frame{}_'.format(nextFrame)
        for file in filter (lambda x: frameName in x, zonelist): file
        if file:
            sortlist.append(file)
    return sortlist


def get_expected_frames_per_zonegroup():
    global framesPerZone, validZonesPerFrame
    validZonesPerFrame = config.get_valid_zone_per_frame()
    validzones = validZonesPerFrame.astype(np.int32)
    framesPerZone = np.sum(validzones, axis=0) # shape=(17)
    zoneidsInGroupid = {0:[0,2], 1:[1,3], 2:[4], 3:[5,6], 4:[7,9], 5:[8], 6:[10,11], 7:[12,13], 8:[14,15], 9:[16]}
    expFrmCnt = []
    for g in range(10):
        zonesInGrp = zoneidsInGroupid.get(g)
        if len(zonesInGrp) == 2:
            assert (framesPerZone[zonesInGrp[0]] == framesPerZone[zonesInGrp[1]])
        expFrmCnt.append(framesPerZone[zonesInGrp[0]])
    return np.array(expFrmCnt)



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))   # Convert data to features

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) # Convert data to features


def gather_files_of_scan_and_zone(pendingList, enof):
    '''
    filter all filenames linked by the same scanID and zone (hence views from valid frames)
    :param pendingList: list of all remaining filenames to filter from
    :param enof:        expected number of frames for given zone
    :return:            prefix and sorted list of all files with same scanid and zone number
    '''
    # filter out all cropped images for chosen scan and zone
    # filename: < tag + '-' + scanid + '_' + bodyZone + '-' + frameName + '_' + zoneLabel + '.png' >
    scanImgsOfZone = []
    file = pendingList[0]
    sindex = file.find('-') + 1
    eindex = sindex + file[sindex:].find('-')
    prefix = file[sindex: eindex]
    for s in filter(lambda x: prefix in x, pendingList): scanImgsOfZone.append(s)
    assert (len(scanImgsOfZone) == enof)

    # arrange order of images
    sindex = prefix.find('Zone')
    zid = int(prefix[sindex + 4:]) - 1
    return prefix, sort_by_visual_order(scanImgsOfZone, zid)


def parse_to_npfile(rdir, wdir, nametag, cin, enof):
    '''
    Encode images and labels in numpy format
    :param rdir:    directory to read zone cropped images from
    :param wdir:    directory to write numpy file to
    :param nametag: name of file
    :param cin:     class (threat or no-threat) index
    :param enof:    expected number of frames for zone groups
    :return:        nothing returned
    '''
    allZoneGrpImgList = os.listdir(rdir)
    numOfPendingFiles = len(allZoneGrpImgList)
    seed(numOfPendingFiles)
    shuffle(allZoneGrpImgList)
    numofscans = int(len(allZoneGrpImgList) / enof)
    scanIndx = 0

    # initialize data and label arrays
    fImages = np.zeros(shape=(numofscans, enof, imgH, imgW), dtype=np.float32)
    fnzLbls = np.zeros(shape=(numofscans, enof + 1), dtype=np.int32)

    while numOfPendingFiles > 0:
        prefix, sortedZoneList = gather_files_of_scan_and_zone(allZoneGrpImgList, enof)
        # get and set class of zone given scanid & zoneid
        cell = df_label.loc[df_label['Id'] == prefix, 'Probability'].values[0]
        znClass = int(cell)
        fnzLbls[scanIndx, enof] = znClass
        classCounter[cin, 10, znClass] += 1 # Bookkeeping: Update counter
        
        # read each cropped image and corresponding label
        for i in range(enof):
            filename = sortedZoneList[i]
            cropImg = cv.imread(os.path.join(rdir, filename), 0)
            assert (cropImg.shape == (imgH, imgW))
            fImages[scanIndx, i, :, :] = cropImg.astype(np.float32)
            ein = filename.find('.')
            imgLabel = int(filename[ein - 1: ein])
            fnzLbls[scanIndx, i] = imgLabel
            classCounter[cin, zin, imgLabel] += 1   # Bookkeeping: Update counter
            allZoneGrpImgList.remove(filename)      # remove file from main list

        scanIndx += 1
        numOfPendingFiles = len(allZoneGrpImgList)
        if numOfPendingFiles % 100*enof == 0:
            print('{} pending files for {} data'.format(len(allZoneGrpImgList), nametag))
            sys.stdout.flush()

    # write numpy files
    imgfile = os.path.join(wdir, nametag + '_imgs.npy')
    lblfile = os.path.join(wdir, nametag + '_lbls.npy')
    np.save(imgfile, fImages)
    np.save(lblfile, fnzLbls)
    sys.stdout.flush()


# encode data as TFRecord
def parse_to_tfrecord(rdir, tfrFile, mode, cin, enof):
    '''
    Encode images and labels in tfRecord format
    :param rdir:    directory to read zone cropped images from
    :param tfrFile: tfRecord file to be created
    :param mode:    String, mode of data. eg. 'train/', 'valdn/'
    :param cin:     class (threat or no-threat) index
    :param enof:    expected number of frames for zone groups
    :return:        nothing is returned
    '''
    allZoneGrpImgList = os.listdir(rdir)
    numOfPendingFiles = len(allZoneGrpImgList)
    seed(numOfPendingFiles)
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tfrFile)

    while numOfPendingFiles > 0:
        fImages = np.zeros(shape=(enof, imgH, imgW), dtype=np.uint8)
        fnzLbls = np.zeros(shape=(enof + 1), dtype=np.int32)

        prefix, sortedZoneList = gather_files_of_scan_and_zone(allZoneGrpImgList, enof)
        # get and set class of zone given scanid & zoneid
        cell = df_label.loc[df_label['Id'] == prefix, 'Probability'].values[0]
        znClass = int(cell)
        fnzLbls[enof] = znClass
        classCounter[cin, 10, znClass] += 1 # Bookkeeping: Update counter

        for i in range(enof):
            filename = sortedZoneList[i]
            cropImg = cv.imread(os.path.join(rdir, filename), 0)
            assert (cropImg.shape == (imgH, imgW))
            fImages[i, :, :] = cropImg.astype(np.float32)
            ein = filename.find('.')
            cropImgLabel = int(filename[ein - 1: ein])
            fnzLbls[i] = cropImgLabel
            classCounter[cin, zin, cropImgLabel] += 1   # Bookkeeping: Update counter
            allZoneGrpImgList.remove(filename)          # remove file from main list

        # Create a feature
        featureString = cv.imencode('.png', fImages.reshape(enof*imgH, imgW))[1].tostring()
        frLabelString = fnzLbls.tostring()
        feature = {mode + 'fnzLbls': _bytes_feature(tf.compat.as_bytes(featureString)),
                   mode + 'fImages': _bytes_feature(tf.compat.as_bytes(frLabelString))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        numOfPendingFiles = len(allZoneGrpImgList)
        if numOfPendingFiles % 100 * enof == 0:
            print('{} pending files for {} data'.format(len(allZoneGrpImgList), mode))
            sys.stdout.flush()

    writer.close()
    sys.stdout.flush()


def main():  # def main(unused_argv):
    global imgW, imgH, channels, zin, classCounter, df_label, zoneSortOrder
    set_test_mode()
    imgW, imgH, channels = 112, 112, 1 #***
    tsa_csv = '../../Data/tsa_psc/stage1_labels.csv'
    df_label = pd.read_csv(tsa_csv)
    homeDir = os.path.abspath('../../../Passenger-Screening-Challenge/Data/aps_images/zone_pose_nn_test/')
    com.create_dir(os.path.join(homeDir, 'dataformat'))
    wDir = os.path.join(homeDir, 'dataformat', modeDir)
    com.create_dir(wDir)
    expframes = get_expected_frames_per_zonegroup()
    zoneSortOrder = {1: (1, -1), 2: (1, -1), 3: (15, 1), 4: (15, 1), 5: (4, -1), 6: (1, -1),
                     7: (15, 1), 8: (1, -1), 9: (13, 1), 10:(15, 1), 11:(2, -1), 12:(14, 1),
                     13:(2, -1), 14:(14, 1), 15:(2, -1), 16:(14, 1), 17:(5, 1)}
    zone_groups = ['Bicep', 'Forearm', 'Chest', 'Rib_Cage_and_Abs', 'Upper_Hip_or_Thigh',
                   'Groin', 'Lower_Thigh', 'Calf', 'Ankle_Bone', 'Upper_Back']

    if __name__ == "__main__":  # i.e. if this script is run.
        # (2: train, eval, 10: num of groups, 2: nothreat, threat label)
        classCounter = np.zeros(shape=(2, 11, 2), dtype=np.int32)

        for zin in range(len(zone_groups)):
            zone_g = zone_groups[zin]
            print("\n", zone_g)
            trainDir = os.path.join(homeDir, "train", zone_g)
            valdnDir = os.path.join(homeDir, "validation", zone_g)

            if tfmode:
                ext = '.tfrecords'
                trnRecFile = os.path.join(wDir, 'train_' + zone_g + ext)
                valRecFile = os.path.join(wDir, 'validation_' + zone_g + ext)
                parse_to_tfrecord(trainDir, trnRecFile, 'train/', 0, expframes[zin])
                parse_to_tfrecord(valdnDir, valRecFile, 'valdn/', 1, expframes[zin])

            elif npmode:
                parse_to_npfile(trainDir, wDir, 'train_' + zone_g, 0, expframes[zin])
                parse_to_npfile(valdnDir, wDir, 'validation_' + zone_g, 1, expframes[zin])

        file = os.path.join(homeDir, 'zonegroup_counter.npy')
        np.save(file, classCounter)


main()
