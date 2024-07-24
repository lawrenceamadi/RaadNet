# imports
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv


# Divide the available space on an image into 16 sectors. In the [0] image these
# zones correspond to the TSA threat zones.  But on rotated images, the slice
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)

# sector 5 is used for both threat zone 5 and 17
sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32)

sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

# crop dimensions, upper left x, y, width, height
sector_crop_list = [[ 88, 149, 112, 112], # sector 1
                    [ 88,  69, 112, 112], # sector 2
                    [330, 149, 112, 112], # sector 3
                    [350,  80, 100, 112], # sector 4
                    [166, 188, 180, 112], # sector 5/17
                    [144, 274, 112, 112], # sector 6
                    [256, 274, 112, 112], # sector 7
                    [143, 364, 112, 112], # sector 8
                    [200, 364, 112, 112], # sector 9
                    [275, 364, 112, 112], # sector 10
                    [144, 443, 112,  90], # sector 11
                    [256, 443, 112,  90], # sector 12
                    [144, 518, 112,  90], # sector 13
                    [256, 518, 112,  90], # sector 14
                    [144, 570, 112,  90], # sector 15
                    [256, 570, 112,  90], # sector 16
                   ]
'''sector_crop_list = [[ 50,  50, 250, 250], # sector 1
                    [  0,   0, 250, 250], # sector 2
                    [ 50, 250, 250, 250], # sector 3
                    [0,   250, 250, 250], # sector 4 **changed
                    [150, 150, 250, 250], # sector 5/17
                    [200, 100, 250, 250], # sector 6
                    [200, 150, 250, 250], # sector 7
                    [250,  50, 250, 250], # sector 8
                    [250, 150, 250, 250], # sector 9
                    [300, 200, 250, 250], # sector 10
                    [400, 100, 250, 250], # sector 11
                    [350, 200, 250, 250], # sector 12
                    [410,   0, 250, 250], # sector 13
                    [410, 200, 250, 250], # sector 14
                    [410,   0, 250, 250], # sector 15
                    [410, 200, 250, 250], # sector 16
                   ]'''

'''
    The zone_slice_list() is used to isolate each zone by masking out the rest of the body. 
    Once you have the isolated threat zone. The zone_crop_list() is used to crop the image 
    to 250 x 250 and as best as possible to center the pixels of interest in the frame.
'''

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_slice_list = [ [ # threat zone 1
                      sector01_pts, sector01_pts, sector01_pts, None,
                      None, None, sector03_pts, sector03_pts,
                      sector03_pts, sector03_pts, sector03_pts,
                      None, None, sector01_pts, sector01_pts, sector01_pts ],

                    [ # threat zone 2
                      sector02_pts, sector02_pts, sector02_pts, None,
                      None, None, sector04_pts, sector04_pts,
                      sector04_pts, sector04_pts, sector04_pts, None,
                      None, sector02_pts, sector02_pts, sector02_pts ],

                    [ # threat zone 3
                      sector03_pts, sector03_pts, sector03_pts, sector03_pts,
                      None, None, sector01_pts, sector01_pts,
                      sector01_pts, sector01_pts, sector01_pts, sector01_pts,
                      None, None, sector03_pts, sector03_pts ],

                    [ # threat zone 4
                      sector04_pts, sector04_pts, sector04_pts, sector04_pts,
                      None, None, sector02_pts, sector02_pts,
                      sector02_pts, sector02_pts, sector02_pts, sector02_pts,
                      None, None, sector04_pts, sector04_pts ],

                    [ # threat zone 5
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      None, None, None, None,
                      None, None, None, None ],

                    [ # threat zone 6
                      sector06_pts, None, None, None,
                      None, None, None, None,
                      sector07_pts, sector07_pts, sector06_pts, sector06_pts,
                      sector06_pts, sector06_pts, sector06_pts, sector06_pts ],

                    [ # threat zone 7
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts,
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts,
                      None, None, None, None,
                      None, None, None, None ],

                    [ # threat zone 8
                      sector08_pts, sector08_pts, None, None,
                      None, None, None, sector10_pts,
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts,
                      sector08_pts, sector08_pts, sector08_pts, sector08_pts ],

                    [ # threat zone 9
                      sector09_pts, sector09_pts, sector08_pts, sector08_pts,
                      sector08_pts, None, None, None,
                      sector09_pts, sector09_pts, None, None,
                      None, None, sector10_pts, sector09_pts ],

                    [ # threat zone 10
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts,
                      sector10_pts, sector08_pts, sector10_pts, None,
                      None, None, None, None,
                      None, None, None, sector10_pts ],

                    [ # threat zone 11
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts,
                      None, None, sector12_pts, sector12_pts,
                      sector12_pts, sector12_pts, sector12_pts, None,
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts ],

                    [ # threat zone 12
                      sector12_pts, sector12_pts, sector12_pts, sector12_pts,
                      sector12_pts, sector11_pts, sector11_pts, sector11_pts,
                      sector11_pts, sector11_pts, sector11_pts, None,
                      None, sector12_pts, sector12_pts, sector12_pts ],

                    [ # threat zone 13
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts,
                      None, None, sector14_pts, sector14_pts,
                      sector14_pts, sector14_pts, sector14_pts, None,
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts ],

                    [ # threat zone 14
                      sector14_pts, sector14_pts, sector14_pts, sector14_pts,
                      sector14_pts, None, sector13_pts, sector13_pts,
                      sector13_pts, sector13_pts, sector13_pts, None,
                      None, None, None, None ],

                    [ # threat zone 15
                      sector15_pts, sector15_pts, sector15_pts, sector15_pts,
                      None, None, sector16_pts, sector16_pts,
                      sector16_pts, sector16_pts, None, sector15_pts,
                      sector15_pts, None, sector15_pts, sector15_pts ],

                    [ # threat zone 16
                      sector16_pts, sector16_pts, sector16_pts, sector16_pts,
                      sector16_pts, sector16_pts, sector15_pts, sector15_pts,
                      sector15_pts, sector15_pts, sector15_pts, None,
                      None, None, sector16_pts, sector16_pts ],

                    [ # threat zone 17
                      None, None, None, None,
                      None, None, None, None,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]


# Each element in the zone_crop_list contains the sector to use in the call to roi()
zone_crop_list =  [ [ # threat zone 1
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None,
                      None, None, sector_crop_list[2], sector_crop_list[2],
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], None,
                      None, sector_crop_list[0], sector_crop_list[0],
                      sector_crop_list[0] ],

                    [ # threat zone 2
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None,
                      None, None, sector_crop_list[3], sector_crop_list[3],
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
                      None, None, sector_crop_list[1], sector_crop_list[1],
                      sector_crop_list[1] ],

                    [ # threat zone 3
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2],
                      sector_crop_list[2], None, None, sector_crop_list[0],
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0],
                      sector_crop_list[0], sector_crop_list[0], None, None,
                      sector_crop_list[2], sector_crop_list[2] ],

                    [ # threat zone 4
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
                      sector_crop_list[3], None, None, sector_crop_list[1],
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1],
                      sector_crop_list[1], sector_crop_list[1], None, None,
                      sector_crop_list[3], sector_crop_list[3] ],

                    [ # threat zone 5
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4],
                      None, None, None, None, None, None, None, None ],

                    [ # threat zone 6
                      sector_crop_list[5], None, None, None, None, None, None, None,
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[5],
                      sector_crop_list[5], sector_crop_list[5], sector_crop_list[5],
                      sector_crop_list[5], sector_crop_list[5] ],

                    [ # threat zone 7
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                      sector_crop_list[6], sector_crop_list[6],
                      None, None, None, None, None, None, None, None ],

                    [ # threat zone 8
                      sector_crop_list[7], sector_crop_list[7], None, None, None,
                      None, None, sector_crop_list[9], sector_crop_list[9],
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                      sector_crop_list[7], sector_crop_list[7], sector_crop_list[7],
                      sector_crop_list[7] ],

                    [ # threat zone 9
                      sector_crop_list[8], sector_crop_list[8], sector_crop_list[7],
                      sector_crop_list[7], sector_crop_list[7], None, None, None,
                      sector_crop_list[8], sector_crop_list[8], None, None, None,
                      None, sector_crop_list[9], sector_crop_list[8] ],

                    [ # threat zone 10
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[7],
                      sector_crop_list[9], None, None, None, None, None, None, None,
                      None, sector_crop_list[9] ],

                    [ # threat zone 11
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10],
                      sector_crop_list[10], None, None, sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], None, sector_crop_list[10],
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10] ],

                    [ # threat zone 12
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], None, None,
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11] ],

                    [ # threat zone 13
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12],
                      sector_crop_list[12], None, None, sector_crop_list[13],
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                      sector_crop_list[13], None, sector_crop_list[12],
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],

                    [ # threat zone 14
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                      sector_crop_list[13], sector_crop_list[13], None,
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[12],
                      sector_crop_list[12], sector_crop_list[12], None, None, None,
                      None, None ],

                    [ # threat zone 15
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
                      sector_crop_list[14], None, None, sector_crop_list[15],
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                      None, sector_crop_list[14], sector_crop_list[14], None,
                      sector_crop_list[14], sector_crop_list[14] ],

                    [ # threat zone 16
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
                      sector_crop_list[14], sector_crop_list[14], None, None, None,
                      sector_crop_list[15], sector_crop_list[15] ],

                    [ # threat zone 17
                      None, None, None, None, None, None, None, None,
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4] ] ]


def get_cropping_configuration():
    return zone_crop_list


def get_subject_zone_label(zone_num, df):
    #------------------------------------------------------------------------------------------------
    # get_subject_zone_label(zone_num, df):    gets a label for a given subject and zone
    # zone_num:                                a 0 based threat zone index
    # df:                                      a df like that returned from get_subject_labels(...)
    # returns:                                 [0,1] if contraband is present, [1,0] if it isnt
    #-----------------------------------------------------------------------------------------------

    # Dict to convert a 0 based threat zone index to the text we need to look up the label
    zone_index = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6',
                  6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12',
                  12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
                  16: 'Zone17'
                 }
    # get the text key from the dictionary
    key = zone_index.get(zone_num)

    # select the probability value and make the label
    if df.loc[df['Zone'] == key]['Probability'].values[0] == 1:
        # threat present
        return 1 #[0,1]
    else:
        #no threat present
        return 0 #[1,0]


def roi(img, vertices):
    #-----------------------------------------------------------------------------------------
    # roi(img, vertices):              uses vertices to mask the image
    # img:                             the image to be masked
    # vertices:                        a set of vertices that define the region of interest
    # returns:                         a masked image
    #-----------------------------------------------------------------------------------------
    # blank mask
    masked = np.zeros_like(img)
    if vertices is not None:
        # fill the mask
        y1, y2 = vertices[0][1], vertices[2][1]
        x1, x2 = vertices[0][0], vertices[1][0]
        masked[y1:y2, x1:x2, :] = img[y1:y2, x1:x2]

    return masked


def get_zone_coord(frameNum, zoneid):
    return zone_crop_list[zoneid][frameNum]


def parse_refpt_list(refPtList):
    return refPtList


def crop(img, crop_list):
    #-----------------------------------------------------------------------------------------
    # crop(img, crop_list):                uses vertices to mask the image
    # img:                                 the image to be cropped
    # crop_list:                           a crop_list entry with [x , y, width, height]
    # returns:                             a cropped image
    #-----------------------------------------------------------------------------------------

    x1, y1 = crop_list[0], crop_list[1]
    x2, y2 = x1 + crop_list[2], y1 + crop_list[3]
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def read_image(file):
    img = cv.imread(file)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def initialize_global_variables():
    print('Fixed grid system requires no global variable initialization...')
