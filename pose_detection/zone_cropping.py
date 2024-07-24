# imports
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
import pandas as pd
import sys


# Divide the available space on an image into 16 sectors. In the [0] image these
# zones correspond to the TSA threat zones.  But on rotated images, the slice
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0,160],[200,160],[200,240],[0,240]], np.int32)
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
sector_crop_list = [[ 88, 150, 112, 112], # sector 1   **149
                    [ 88,  70, 112, 112], # sector 2   **69
                    [330, 150, 112, 112], # sector 3
                    [330,  70, 112, 112], # sector 4
                    [166, 190, 180,  90], # sector 5/17 **188-112
                    [144, 275, 112, 112], # sector 6
                    [256, 275, 112, 112], # sector 7
                    [144, 370, 112, 112], # sector 8    **364
                    [200, 370, 112, 112], # sector 9
                    [275, 370, 112, 112], # sector 10
                    [144, 475, 112,  60], # sector 11   **443-90
                    [256, 475, 112,  60], # sector 12
                    [144, 530, 112,  75], # sector 13   **518-90
                    [256, 530, 112,  75], # sector 14
                    [144, 600, 112,  60], # sector 15   **570-90
                    [256, 600, 112,  60], # sector 16
                   ]

# crop dimensions, upper left x, y, width, height. Favors center of frame
center_crop_list = [[200, 150, 112, 112], # sector 1**** frame: 12
                    [180,  70, 112, 112], # sector 2**** frame: 12
                    [200, 150, 112, 112], # sector 3**** frame: 4
                    [180,  70, 112, 112], # sector 4**** frame: 4
                    [166, 190, 180,  90], # sector 5/17
                    [166, 275, 180, 112], # sector 6**** frame: 10, 11, 12, 13
                    [166, 275, 180, 112], # sector 7**** frame: 4, 5, 6
                    [166, 370, 180, 112], # sector 8**** frame: 10, 11, 12, 13
                    [200, 370, 112, 112], # sector 9
                    [166, 370, 180, 112], # sector 10*** frame: 4, 5, 6
                    [200, 475, 112,  60], # sector 11*** frame: 10, 11, 12
                    [200, 475, 112,  60], # sector 12*** frame: 4, 5, 6
                    [200, 530, 112,  75], # sector 13*** frame: 11
                    [200, 530, 112,  75], # sector 14*** frame: 5
                    [200, 600, 112,  60], # sector 15*** frame: 11
                    [200, 600, 112,  60], # sector 16*** frame: 5
                   ]

# crop dimensions, upper left x, y, width, height. Favors right side of frame
rightr_crop_list = [[256, 150, 112, 112], # sector 1**** frame: 11
                    [276,  70, 112, 112], # sector 2**** frame: 11
                    [276, 150, 112, 112], # sector 3**** frame: 3
                    [256,  70, 112, 112], # sector 4**** frame: 3
                    [256, 190, 112,  90], # sector 5/17* frame: 5, 6*, 12, 13
                    [276, 275, 112, 112], # sector 6**** frame: 7
                    [166, 275, 112, 112], # sector 7**** frame: 7
                    [276, 370, 112, 112], # sector 8**** frame: 7
                    [230, 370, 112, 112], # sector 9**** frame: 7, 14
                    [166, 370, 112, 112], # sector 10*** frame: 7
                    [230, 475, 112,  60], # sector 11
                    [230, 475, 112,  60], # sector 12*** frame: 3
                    [276, 530, 112,  75], # sector 13*** frame: 6
                    [220, 530, 112,  75], # sector 14*** frame: 4, 10
                    [276, 600, 112,  60], # sector 15*** frame: 1, 6
                    [220, 600, 112,  60], # sector 16*** frame: 4
                   ]


# crop dimensions, upper left x, y, width, height. Favors left side of frame
lefter_crop_list = [[144, 150, 112, 112], # sector 1**** frame: 13
                    [124,  70, 112, 112], # sector 2**** frame: 13
                    [144, 150, 112, 112], # sector 3**** frame: 5
                    [124,  70, 112,  90], # sector 4**** frame: 5
                    [166, 190, 112, 112], # sector 5/17* frame: 4, 10, 11
                    [236, 275, 112, 112], # sector 6**** frame: 9
                    [124, 275, 112, 112], # sector 7**** frame: 9
                    [236, 370, 112, 112], # sector 8**** frame: 9
                    [170, 370, 112, 112], # sector 9**** frame: 3, 9
                    [144, 370, 112, 112], # sector 10*** frame: 9
                    [170, 475, 112,  60], # sector 11*** frame: 13
                    [170, 475, 112,  60], # sector 12
                    [180, 530, 112,  75], # sector 13*** frame: 6, 12
                    [124, 530, 112,  75], # sector 14*** frame: 10
                    [180, 600, 112,  60], # sector 15*** frame: 12
                    [124, 600, 112,  60], # sector 16*** frame: 10, 15
                   ]


# Each element in the zone_crop_list contains the sector to use in the call to roi()
modi_crop_list =  [ [ # threat zone 1
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None,
                      None, None, None, sector_crop_list[2],
                      sector_crop_list[2], sector_crop_list[2], rightr_crop_list[0], rightr_crop_list[0],
                      center_crop_list[0], lefter_crop_list[0], sector_crop_list[0], sector_crop_list[0] ],

                    [ # threat zone 2
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None,
                      None, None, None, sector_crop_list[3],
                      sector_crop_list[3], sector_crop_list[3], rightr_crop_list[3], rightr_crop_list[1],
                      center_crop_list[1], lefter_crop_list[1], sector_crop_list[1], sector_crop_list[1] ],

                    [ # threat zone 3
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], rightr_crop_list[2],
                      center_crop_list[2], lefter_crop_list[2], sector_crop_list[0], sector_crop_list[0],
                      sector_crop_list[0], sector_crop_list[0], None, None,
                      None, None, sector_crop_list[2], sector_crop_list[2] ],

                    [ # threat zone 4
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], rightr_crop_list[3],
                      center_crop_list[3], lefter_crop_list[3], sector_crop_list[1], sector_crop_list[1],
                      sector_crop_list[1], sector_crop_list[1], None, None,
                      None, None, sector_crop_list[3], sector_crop_list[3] ],

                    [ # threat zone 5
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      lefter_crop_list[4], None, None, None,
                      None, None, None, None,
                      rightr_crop_list[4], rightr_crop_list[4], sector_crop_list[4], sector_crop_list[4] ],

                    [ # threat zone 6
                      sector_crop_list[5], sector_crop_list[5], None, None,
                      None, None, None, rightr_crop_list[5],
                      sector_crop_list[6], lefter_crop_list[5], center_crop_list[5], center_crop_list[5],
                      center_crop_list[5], center_crop_list[5], sector_crop_list[5], sector_crop_list[5] ],

                    [ # threat zone 7
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                      center_crop_list[6], center_crop_list[6], center_crop_list[6], rightr_crop_list[6],
                      sector_crop_list[5], lefter_crop_list[6], None, None,
                      None, None, None, sector_crop_list[6] ],

                    [ # threat zone 8
                      sector_crop_list[7], sector_crop_list[7], None, None,
                      None, None, None, rightr_crop_list[7],
                      sector_crop_list[9], lefter_crop_list[7], center_crop_list[7], center_crop_list[7],
                      center_crop_list[7], center_crop_list[7], sector_crop_list[7], sector_crop_list[7] ],

                    [ # threat zone 9
                      sector_crop_list[8], sector_crop_list[8], lefter_crop_list[8], lefter_crop_list[8],
                      None, None, None, rightr_crop_list[8],
                      sector_crop_list[8], lefter_crop_list[8], None, None,
                      None, rightr_crop_list[8], rightr_crop_list[8], sector_crop_list[8] ],

                    [ # threat zone 10
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                      center_crop_list[9], center_crop_list[9], center_crop_list[9], rightr_crop_list[9],
                      sector_crop_list[7], lefter_crop_list[9], None, None,
                      None, None, None, sector_crop_list[9] ],

                    [ # threat zone 11
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], None,
                      None, None, sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], center_crop_list[10], center_crop_list[10],
                      center_crop_list[10], lefter_crop_list[10], sector_crop_list[10], sector_crop_list[10] ],

                    [ # threat zone 12
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], rightr_crop_list[11],
                      center_crop_list[11], center_crop_list[11], center_crop_list[11], sector_crop_list[10],
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], None,
                      None, None, sector_crop_list[11], sector_crop_list[11] ],

                    [ # threat zone 13
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], None,
                      None, None, rightr_crop_list[12], sector_crop_list[13],
                      sector_crop_list[13], sector_crop_list[13], rightr_crop_list[13], center_crop_list[12],
                      lefter_crop_list[12], sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],

                    [ # threat zone 14
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                      rightr_crop_list[13], center_crop_list[13], lefter_crop_list[12], sector_crop_list[12],
                      sector_crop_list[12], sector_crop_list[12], lefter_crop_list[13], None,
                      None, None, sector_crop_list[13], sector_crop_list[13] ],

                    [ # threat zone 15
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], None,
                      None, None, rightr_crop_list[14], sector_crop_list[15],
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], center_crop_list[14],
                      lefter_crop_list[14], sector_crop_list[14], sector_crop_list[14], lefter_crop_list[15] ],

                    [ # threat zone 16
                      sector_crop_list[15], rightr_crop_list[14], sector_crop_list[15], sector_crop_list[15],
                      rightr_crop_list[15], center_crop_list[15], sector_crop_list[14], sector_crop_list[14],
                      sector_crop_list[14], sector_crop_list[14], lefter_crop_list[15], None,
                      None, None, sector_crop_list[15], sector_crop_list[15] ],

                    [ # threat zone 17
                      None, None, None, None,
                      None, rightr_crop_list[4], rightr_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4], lefter_crop_list[4], lefter_crop_list[4],
                      None, None, None, None ] ]

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


def display(img, title="Output"):
    cv.imshow(title, img)
    stay = True
    while stay:
        key = cv.waitKey(10)
        if key == ord('e'):
            sys.exit()
        if key == ord('n'):
            #cv.destroyAllWindows()
            stay = False

def get_top_pixel(timg):
    # returns 1D (size = num of columns) ndarray containing row indices of first occurrence for each column
    row_indices = np.argmax(timg==255, axis=0)
    # Note: 0 is returned per column if no 255 pixel was found in column.
    # therefore, to make np.argmin work set all 0s to max:660 so np.argmin does not pick them
    row_indices[row_indices == 0] = 660
    # returns 1D (size = 1) ndarray containing col indices of first occurrence
    x = np.argmin(row_indices)
    y = row_indices[x]
    return y

def get_x_bounds(timg, type, start):
    # Focus on feet
    img = timg
    if type == 'right':
        img = np.fliplr(timg)  # flip image horizontally
    # returns 1D (size = num of rows) ndarray containing col indices of first occurrence for each row
    col_indices = np.argmax(img[start:, :]==255, axis=1)
    # Note: 0 is returned per column if no 255 pixel was found in column.
    # therefore, to make np.argmin work set all 0s to max:660 so np.argmin does not pick them
    col_indices[col_indices == 0] = 660
    # returns 1D (size = 1) ndarray containing col indices of first occurrence
    y = np.argmin(col_indices)
    x = col_indices[y]
    return x

def color_to_binary(img):
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gbimg = cv.GaussianBlur(gimg, (5,5), 0)
    ret, timg = cv.threshold(gbimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return timg

def transpixel(x, y):
    global scaleFactor
    o_x, o_y = x * scaleFactor, y * scaleFactor
    x_margin = ((imgW * scaleFactor) - imgW) / 2
    y_margin = (imgH * scaleFactor) - imgH
    t_x, t_y = o_x - x_margin - x_shift, o_y - y_margin # - x_shift
    return int(t_x), int(t_y)

# draws box from coordinates in list
def draw_box(image, list, mode):
    for x in range(len(list)):
        box = list[x]
        tx1, ty1 = box[0][0], box[0][1]
        tx2, ty2 = box[1][0], box[1][1]
        if mode == 'normalize':
            tx1, ty1 = transpixel(tx1, ty1)
            tx2, ty2 = transpixel(tx2, ty2)
        cv.rectangle(image, (tx1,ty1), (tx2,ty2), (0, 0, 255), 2)

# draws already marked regions in image and comparison region if any in frame
def draw_marked_regions(image, list, mode='original'):
    imgclone = np.copy(image)
    # draw marked regions
    draw_box(imgclone, list, mode)
    return imgclone

def standard_resize(img):
    global scaleFactor
    aspectRatio = img.shape[1] / img.shape[0]
    new_width = int(aspectRatio * standard_H)
    #width = min(new_width, bodysize[1])
    scaleFactor = standard_H / img.shape[0]
    rimg = cv.resize(img, (new_width, standard_H), interpolation=cv.INTER_AREA)
    return rimg


def transtand(img, y_top, leftmost, rightmost):
    crop_top = max(0, y_top-margin)
    crop_lef = max(0, leftmost-margin)
    crop_rig = min(511, rightmost+margin)
    focus = img[crop_top:, crop_lef:crop_rig, :]
    sized = standard_resize(focus)
    padding = imgW - sized.shape[1]
    leftpad = int(padding / 2)
    rightpad = leftpad + sized.shape[1]
    slate = np.copy(background)
    slate[10:, leftpad:rightpad] = sized
    return slate


def normalize(img):
    global x_shift
    timg = color_to_binary(img)
    topmost = get_top_pixel(timg)
    x_rightmost = imgW - get_x_bounds(timg, 'right', 0)
    x_leftmost = get_x_bounds(timg, 'left', 0)
    cv.line(timg, (0, topmost), (imgW, topmost), 255)
    cv.line(timg, (x_rightmost, 0), (x_rightmost, imgH), 255)
    cv.line(timg, (x_leftmost, 0), (x_leftmost, imgH), 255)
    display(timg, 'Border')

    simg = transtand(img, topmost, x_leftmost, x_rightmost)
    x_shift = ((x_leftmost - (imgW-x_rightmost)) * scaleFactor) / 2
    transformed = draw_marked_regions(simg, refPtList, mode='normalize')
    display(transformed, 'Standard')
    return simg

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


def crop(img, crop_list):
    #-----------------------------------------------------------------------------------------
    # crop(img, crop_list):                uses vertices to mask the image
    # img:                                 the image to be cropped
    # crop_list:                           a crop_list entry with [x , y, width, height]
    # returns:                             a cropped image
    #-----------------------------------------------------------------------------------------
    if crop_list is not None:
        x_coord = crop_list[0]
        y_coord = crop_list[1]
        width = crop_list[2]
        height = crop_list[3]
        cropped_img = img[y_coord:y_coord+height, x_coord:x_coord+width]
        return cropped_img
    else:
        return np.zeros(shape=(250,250), dtype=np.uint8)


if __name__ == "__main__":
    global imgW, imgH, margin, standard_H, background, refPtList

    if len(sys.argv) == 3:
        startFrame = int(sys.argv[1])
        startZone = int(sys.argv[2]) - 1
    else:
        startFrame = 0
        startZone = 0

    csvfile = '../../Data/tsa_psc/stage1_labels_1_marked.csv'
    #scanid = '0ae548ce1d028459baf4a1d7dae807f3'#'0e34d284cb190a79e3316d1926061bc3'#'0a54f0d947f8fdc614d2a6031ecbb661'
    scanid = '0a27d19c6ec397661b09f7d5998e0b14'
    scan = '../../../Passenger-Screening-Challenge/Data/aps_images/full_image_threat/' + scanid +'/'
    nameZoneIndex = {0:'Zone1: Right Bicep', 1:'Zone2: Right Forearm', 2:'Zone3: Left Bicep', 3:'Zone4: Left Forearm',
                     4:'Zone5: Upper Chest', 5:'Zone6: Right Rib Cage and Abs', 6:'Zone7: Left Rib Cage and Abs',
                     7:'Zone8: Upper Right Hip/Thigh', 8:'Zone9: Groin', 9:'Zone10: Upper Left Hip/Thigh',
                     10:'Zone11: Lower Right Thigh', 11:'Zone12: Lower Left Thigh', 12:'Zone13: Right Calf',
                     13:'Zone14: Left Calf', 14:'Zone15: Right Ankle Bone', 15:'Zone16: Left Ankle Bone',
                     16:'Zone17: Upper Back'}

    cv.namedWindow("Frame")
    cv.namedWindow("Border")
    cv.namedWindow("Standard")
    cv.namedWindow("Mask1")
    cv.namedWindow("Crop1")
    cv.namedWindow("Mask2")
    cv.namedWindow("Crop2")
    cv.moveWindow("Frame", 0, 200)
    cv.moveWindow("Standard", 520, 200)
    #cv.moveWindow("Border", 2400, 200)
    #cv.moveWindow("Mask1", 1100, 200)
    #cv.moveWindow("Mask2", 1820, 200)
    #cv.moveWindow("Crop1", 3100, 50)
    #cv.moveWindow("Crop2", 3100, 400)
    cv.moveWindow("Border", 1040, 200)
    cv.moveWindow("Mask1", 4000, 200)
    cv.moveWindow("Mask2", 4520, 200)
    cv.moveWindow("Crop1", 5040, 200)
    cv.moveWindow("Crop2", 5040, 550)

    imgW, imgH = 512, 660
    margin = 50
    standard_H = 650
    background = cv.imread('../../Data/tsa_psc/background.png')
    df = pd.read_csv(csvfile)

    for frameNum in range(startFrame, 16):
        # check to see if frame is already marked, if so draw marked regions
        frameName = 'Frame' + str(frameNum)
        print(frameName)
        cell = df.loc[df['ID'] == scanid, frameName].values
        if len(cell) > 0 and cell != "N/M":
            refPtList = eval(cell[0])
        else:
            refPtList = []

        frame = cv.imread(scan + str(frameNum) + '.png')
        original = draw_marked_regions(frame, refPtList)
        display(original, 'Frame')

        norm = normalize(frame)

        for zone in range(startZone, 17):
            zoneName = nameZoneIndex.get(zone)
            #zone = 4 if zone == 16 else zone

            mask = roi(frame, zone_slice_list[zone][frameNum])
            cv.putText(mask, zoneName, (20, 600), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv.LINE_AA)
            display(mask, 'Mask1')
            cropzone = crop(frame, zone_crop_list[zone][frameNum])
            display(cropzone, 'Crop1')

            mask = roi(norm, zone_slice_list[zone][frameNum])
            cv.putText(mask, zoneName, (20, 600), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv.LINE_AA)
            display(mask, 'Mask2')
            cropzone = crop(norm, modi_crop_list[zone][frameNum])
            display(cropzone, 'Crop2')

            startZone = 0
        startFrame = 0


#main()
