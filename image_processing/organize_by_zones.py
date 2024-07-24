'''
    Organize train set and evaluation set by common zones to be trained by zone-specialized neural networks
'''

import pandas as pd
import numpy as np
import cv2 as cv
import sys
import os

sys.path.append('../')
import pose_detection.grid_zoning_norm as grid
import neural_net.commons as com

# create directory if it does not exist
def create_dir(path):
    if os.path.exists(path) == False:
        os.mkdir(path)

def create_zonepair_subdirectories(dir, pairing):
    for zone in range(17):
        create_dir(dir + pairing.get(zone))

def generate_zone_dataset(rdir, wdir, df, pairing, fliplist):
    list = os.listdir(rdir)
    for i in range(len(list)):
        dir = list[i]
        scanid = dir[:dir.find('_')]

        for frameNum in range(16):
            # check to see if frame is marked
            frameName = 'Frame' + str(frameNum)
            cell = df.loc[df['ID'] == scanid, frameName].values
            if len(cell) > 0 and cell != "N/M":
                refPtList = eval(cell[0])
                refPtList = grid.parse_refpt_list(refPtList)
            else:
                refPtList = []

            filepath = os.path.join(rdir, dir, str(frameNum) + '.png')
            frame = grid.read_image(filepath)

            for zoneid in range(17):
                zoneCoord = grid.get_zone_coord(frameNum, zoneid)
                if zoneCoord is not None:
                    x1, y1 = zoneCoord[0], zoneCoord[1]
                    x2, y2 = x1 + zoneCoord[2], y1 + zoneCoord[3]
                    zoneLabel = com.get_label(y1, y2, x1, x2, refPtList)
                    zoneImage = grid.crop(frame, zoneCoord)
                    if zoneid in fliplist:
                        zoneImage = np.fliplr(zoneImage)
                    filename = scanid + '_' + str(zoneid+1) + '-' + frameName + '_' + str(zoneLabel) + '.png'
                    zone_dir = wdir + pairing.get(zoneid) + '/'
                    zoneImage = cv.resize(zoneImage, (112, 112), interpolation=cv.INTER_AREA)
                    cv.imwrite(zone_dir + filename, zoneImage)

        if i % 100 == 0:
            print('{}/{} scans parsed'.format(i, len(list)))


def main():
    grid.initialize_global_variables()

    marked_csv = '../../Data/tsa_psc/stage1_labels_1_marked.csv'
    ROOT_DIR = '../../../Passenger-Screening-Challenge/Data/aps_images/dataset/'
    TRAIN_DIR = ROOT_DIR + 'train_set/'
    VALID_DIR = ROOT_DIR + 'validation_set/'
    homeDir = '../../../Passenger-Screening-Challenge/Data/aps_images/zone_pose_nn/'
    homeTrain = homeDir + 'train/'
    homeValid = homeDir + 'valid/'
    create_dir(homeDir)
    create_dir(homeValid)
    create_dir(homeTrain)
    zoneidFlip = [2, 3, 6, 9, 11, 13, 15]
    df = pd.read_csv(marked_csv)
    zoneidPairing = {0: 'Bicep', 1: 'Forearm', 2: 'Bicep', 3: 'Forearm', 4: 'Chest', 5: 'Rib_Cage_and_Abs',
                     6: 'Rib_Cage_and_Abs', 7: 'Upper_Hip_or_Thigh', 8: 'Groin', 9: 'Upper_Hip_or_Thigh',
                     10: 'Lower_Thigh', 11: 'Lower_Thigh', 12: 'Calf', 13: 'Calf', 14: 'Ankle_Bone',
                     15: 'Ankle_Bone', 16: 'Upper_Back'}

    create_zonepair_subdirectories(homeTrain, zoneidPairing)
    create_zonepair_subdirectories(homeValid, zoneidPairing)

    generate_zone_dataset(TRAIN_DIR, homeTrain, df, zoneidPairing, zoneidFlip)
    generate_zone_dataset(VALID_DIR, homeValid, df, zoneidPairing, zoneidFlip)


main()
