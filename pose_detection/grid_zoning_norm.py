# imports
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv


# crop dimensions
# from upper left:  [  x,   y, wdt, hgt]
sector_crop_list = [[ 70, 150, 150, 112], #  0: zone 1,     frame:  0,  1, 14, 15|  7,  8,  9
                    [ 70,  70, 128, 128], #  1: zone 2,     frame:  0,  1, 14, 15|  7,  8,  9
                    [292, 150, 150, 112], #  2: zone 3,     frame:  7,  8,  9|  0,  1,  2, 15
                    [314,  70, 128, 128], #  3: zone 4,     frame:  7,  8,  9|  0,  1,  2, 15
                    [166, 190, 180,  90], #  4: zone 5/17,  frame:  0,  1,  2,  3, 13, 14, 15
                    [144, 275, 112, 112], #  5: zone 6,     frame:  0,  1, *2, 13, 15|  8
                    [256, 275, 112, 112], #  6: zone 7,     frame:  8|  0,  1,  2,*14, 15
                    [144, 370, 112, 112], #  7: zone 8,     frame:  0,  1,  2, 13, 15|  8
                    [200, 370, 112, 112], #  8: zone 9,     frame:  0,  1,  8, 15
                    [256, 370, 112, 112], #  9: zone 10,    frame:  8|  0,  1,  2,*14, 15
                    [144, 470, 112,  70], # 10: zone 11,    frame:  0,  1, 12, 13|  8
                    [256, 470, 112,  70], # 11: zone 12,    frame:  8,  9|  0,  1,  2,  3, 15
                    [144, 530, 112,  75], # 12: zone 13,    frame:  0,  1, 12, 13|  8
                    [256, 530, 112,  75], # 13: zone 14,    frame:  8,  9|  0,  1,  2,  3, 15
                    [144, 590, 112,  70], # 14: zone 15,    frame:  0,  1, 12, 13|  8
                    [256, 590, 112,  70], # 15: zone 16,    frame:  8,  9|  0,  1,  2,  3, 15
                   ]

# from upper left:  [  x,   y, wdt, hgt]. Favors center of frame
center_crop_list = [[142, 150, 112, 112], #  0: zone 1,     frame: 12| *2
                    [142,  70, 150, 112], #  1: zone 2,     frame: 12| *2
                    [258, 150, 112, 112], #  2: zone 3,     frame:  4|*14
                    [220,  70, 150, 112], #  3: zone 4,     frame:  4|*14
                    [166, 190, 180,  90], #  4: zone 5/17,  frame:  7,  8,  9
                    [166, 275, 180, 112], #  5: zone 6,     frame: 10, 11, 12
                    [166, 275, 180, 112], #  6: zone 7,     frame:  4,  5,  6
                    [166, 370, 180, 112], #  7: zone 8,     frame: 10, 11, 12
                    [200, 370, 112, 112], #  8: zone 9,     frame:
                    [166, 370, 180, 112], #  9: zone 10,    frame:  4,  5,  6
                    [200, 470, 112,  70], # 10: zone 11,    frame: 10, 11
                    [200, 470, 112,  70], # 11: zone 12,    frame:  5
                    [200, 530, 112,  75], # 12: zone 13,    frame: 10, 11
                    [200, 530, 112,  75], # 13: zone 14,    frame:  5
                    [200, 590, 112,  70], # 14: zone 15,    frame: 10, 11
                    [200, 590, 112,  70], # 15: zone 16,    frame:  5
                   ]

# from upper left:  [  x,   y, wdt, hgt]. Favors right side of frame
rightr_crop_list = [[256, 150, 150, 112], #  0: zone 1,     frame: 10
                    [278,  70, 128, 128], #  1: zone 2,     frame: 10
                    [295, 150, 112, 112], #  2: zone 3,     frame:  3
                    [279,  70, 128, 128], #  3: zone 4,     frame:  3
                    [220, 190, 150,  90], #  4: zone 5/17,  frame:  4|  5,  6
                    [276, 275, 112, 112], #  5: zone 6,     frame: *6, 7
                    [166, 275, 112, 112], #  6: zone 7,     frame:  7
                    [276, 370, 112, 112], #  7: zone 8,     frame:  6,  7
                    [230, 370, 112, 112], #  8: zone 9,     frame:  7, 13, 14
                    [166, 370, 112, 112], #  9: zone 10,    frame:  7
                    [280, 470, 112,  70], # 10: zone 11,    frame:  6,  7
                    [166, 470, 112,  70], # 11: zone 12,    frame:  6,  7
                    [280, 530, 112,  75], # 12: zone 13,    frame:  6,  7
                    [166, 530, 112,  75], # 13: zone 14,    frame:  6,  7
                    [280, 590, 112,  70], # 14: zone 15,    frame:  6,  7
                    [166, 590, 112,  70], # 15: zone 16,    frame:  6,  7
                   ]

# from upper left:  [  x,   y, wdt, hgt]. Favors left side of frame
lefter_crop_list = [[105, 150, 112, 112], #  0: zone 1,     frame: 13
                    [105,  70, 128, 128], #  1: zone 2,     frame: 13
                    [142, 150, 150, 112], #  2: zone 3,     frame:  6
                    [142,  70, 128, 128], #  3: zone 4,     frame:  6
                    [142, 190, 150,  90], #  4: zone 5/17,  frame: 12| 10, 11
                    [120, 275, 112, 112], #  5: zone 6,     frame: 14
                    [144, 275, 112, 112], #  6: zone 7,     frame:  9,*10
                    [120, 370, 112, 112], #  7: zone 8,     frame: 14
                    [170, 370, 112, 112], #  8: zone 9,     frame:  2,  3,  9
                    [144, 370, 112, 112], #  9: zone 10,    frame:  9,*10
                    [120, 470, 112,  70], # 10: zone 11,    frame: 14, 15
                    [144, 470, 112,  70], # 11: zone 12,    frame:  9, 10
                    [120, 530, 112,  75], # 12: zone 13,    frame: 14, 15
                    [144, 530, 112,  75], # 13: zone 14,    frame:  9, 10
                    [120, 590, 112,  70], # 14: zone 15,    frame: 14, 15
                    [144, 590, 112,  70], # 15: zone 16,    frame:  9, 10
                   ]

# from upper left:  [  x,   y, wdt, hgt]
shiftr_crop_list = [[200, 150, 150, 112], #  0: zone 1,     frame: 11
                    [222,  70, 128, 128], #  1: zone 2,     frame: 11
                    [162, 150, 150, 112], #  2: zone 3,     frame:  5
                    [162,  70, 128, 128], #  3: zone 4,     frame:  5
                    [142, 190, 150,  90], #  4: zone 5/17,  frame:
                    [256, 275, 112, 112], #  5: zone 6,     frame:  9
                    [188, 275, 180, 112], #  6: zone 7,     frame:  3
                    [256, 370, 112, 112], #  7: zone 8,     frame:  9
                    [170, 370, 112, 112], #  8: zone 9,     frame:
                    [188, 370, 180, 112], #  9: zone 10,    frame:  3
                    [166, 470, 112,  70], # 10: zone 11,    frame:  2
                    [234, 470, 112,  70], # 11: zone 12,    frame:  4, 14
                    [166, 530, 112,  75], # 12: zone 13,    frame:  2
                    [234, 530, 112,  75], # 13: zone 14,    frame:  4, 14
                    [166, 590, 112,  70], # 14: zone 15,    frame:  2
                    [234, 590, 112,  70], # 15: zone 16,    frame:  4, 14
                    ]


# Each element in the zone_crop_list contains the sector to use in the call to roi()
# modified from zone_crop_list
modi_crop_list =  \
      [ [ # threat zone 1: R-Bicep
          sector_crop_list[0], sector_crop_list[0], center_crop_list[0], None, # index 2 is a dummy
          None, None, None, sector_crop_list[2],
          sector_crop_list[2], sector_crop_list[2], rightr_crop_list[0], shiftr_crop_list[0],
          center_crop_list[0], lefter_crop_list[0], sector_crop_list[0], sector_crop_list[0] ],

        [ # threat zone 2: R-Forearm
          sector_crop_list[1], sector_crop_list[1], center_crop_list[1], None, # index 2 is a dummy
          None, None, None, sector_crop_list[3],
          sector_crop_list[3], sector_crop_list[3], rightr_crop_list[1], shiftr_crop_list[1],
          center_crop_list[1], lefter_crop_list[1], sector_crop_list[1], sector_crop_list[1] ],

        [ # threat zone 3: L-Bicep
          sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], rightr_crop_list[2],
          center_crop_list[2], shiftr_crop_list[2], lefter_crop_list[2], sector_crop_list[0],
          sector_crop_list[0], sector_crop_list[0], None, None,
          None, None, center_crop_list[2], sector_crop_list[2] ], # index 14 is a dummy

        [ # threat zone 4: L-Forearm
          sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], rightr_crop_list[3],
          center_crop_list[3], shiftr_crop_list[3], lefter_crop_list[3], sector_crop_list[1],
          sector_crop_list[1], sector_crop_list[1], None, None,
          None, None, center_crop_list[3], sector_crop_list[3] ], # index 14 is a dummy

        [ # threat zone 5: Chest
          sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
          rightr_crop_list[4], None, None, None,
          None, None, None, None,
          lefter_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4] ],

        [ # threat zone 6: R-Abdomen
          sector_crop_list[5], sector_crop_list[5], sector_crop_list[5], None, # index 2 & 6 are dummies
          None, None, rightr_crop_list[5], rightr_crop_list[5],
          sector_crop_list[6], shiftr_crop_list[5], center_crop_list[5], center_crop_list[5],
          center_crop_list[5], sector_crop_list[5], lefter_crop_list[5], sector_crop_list[5] ],

        [ # threat zone 7: L-Abdomen
          sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], shiftr_crop_list[6],
          center_crop_list[6], center_crop_list[6], center_crop_list[6], rightr_crop_list[6],
          sector_crop_list[5], lefter_crop_list[6], lefter_crop_list[6], None,
          None, None, sector_crop_list[6], sector_crop_list[6] ], # index 10 & 14 are dummies

        [ # threat zone 8: R-Hip
          sector_crop_list[7], sector_crop_list[7], sector_crop_list[7], None, # index 2 & 6 are dummies
          None, None, rightr_crop_list[7], rightr_crop_list[7],
          sector_crop_list[9], shiftr_crop_list[7], center_crop_list[7], center_crop_list[7],
          center_crop_list[7], sector_crop_list[7], lefter_crop_list[7], sector_crop_list[7] ],

        [ # threat zone 9: Groin
          sector_crop_list[8], sector_crop_list[8], lefter_crop_list[8], lefter_crop_list[8],
          None, None, None, rightr_crop_list[8],
          sector_crop_list[8], lefter_crop_list[8], None, None,
          None, rightr_crop_list[8], rightr_crop_list[8], sector_crop_list[8] ],

        [ # threat zone 10: L-Hip
          sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], shiftr_crop_list[9],
          center_crop_list[9], center_crop_list[9], center_crop_list[9], rightr_crop_list[9],
          sector_crop_list[7], lefter_crop_list[9], lefter_crop_list[9], None,
          None, None, sector_crop_list[9], sector_crop_list[9] ], # index 10 & 14 are dummies

        [ # threat zone 11: R-Thigh
          sector_crop_list[10], sector_crop_list[10], shiftr_crop_list[10], None,
          None, None, rightr_crop_list[10], rightr_crop_list[10],
          sector_crop_list[11], sector_crop_list[11], center_crop_list[10], center_crop_list[10],
          sector_crop_list[10], sector_crop_list[10], lefter_crop_list[10], lefter_crop_list[10] ],

        [ # threat zone 12: L-Thigh
          sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
          shiftr_crop_list[11], center_crop_list[11], rightr_crop_list[11], rightr_crop_list[11],
          sector_crop_list[10], lefter_crop_list[11], lefter_crop_list[11], None,
          None, None, shiftr_crop_list[11], sector_crop_list[11] ],

        [ # threat zone 13: R-Calf
          sector_crop_list[12], sector_crop_list[12], shiftr_crop_list[12], None,
          None, None, rightr_crop_list[12], rightr_crop_list[12],
          sector_crop_list[13], sector_crop_list[13], center_crop_list[12], center_crop_list[12],
          sector_crop_list[12], sector_crop_list[12], lefter_crop_list[12], lefter_crop_list[12] ],

        [ # threat zone 14: L-Calf
          sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
          shiftr_crop_list[13], center_crop_list[13], rightr_crop_list[13], rightr_crop_list[13],
          sector_crop_list[12], lefter_crop_list[13], lefter_crop_list[13], None,
          None, None, shiftr_crop_list[13], sector_crop_list[13] ],

        [ # threat zone 15: R-Ankle
          sector_crop_list[14], sector_crop_list[14], shiftr_crop_list[14], None,
          None, None, rightr_crop_list[14], rightr_crop_list[14],
          sector_crop_list[15], sector_crop_list[15], center_crop_list[14], center_crop_list[14],
          sector_crop_list[14], sector_crop_list[14], lefter_crop_list[14], lefter_crop_list[14] ],

        [ # threat zone 16: L-Ankle
          sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
          shiftr_crop_list[15], center_crop_list[15], rightr_crop_list[15], rightr_crop_list[15],
          sector_crop_list[14], lefter_crop_list[15], lefter_crop_list[15], None,
          None, None, shiftr_crop_list[15], sector_crop_list[15] ],

        [ # threat zone 17: Back
          None, None, None, None,
          None, rightr_crop_list[4], rightr_crop_list[4], center_crop_list[4],
          center_crop_list[4], center_crop_list[4], lefter_crop_list[4], lefter_crop_list[4],
          None, None, None, None ] ]


modi_crop_dict = {'RBp' : modi_crop_list[0],  'RFm' : modi_crop_list[1],
                  'LBp' : modi_crop_list[2],  'LFm' : modi_crop_list[3],
                  'UCh' : modi_crop_list[4],
                  'RAb' : modi_crop_list[5],  'LAb' : modi_crop_list[6],
                  'URTh': modi_crop_list[7],  'ULTh': modi_crop_list[9],
                  'Gr'  : modi_crop_list[8],
                  'LRTh': modi_crop_list[10], 'LLTh': modi_crop_list[11],
                  'RCf' : modi_crop_list[12], 'LCf' : modi_crop_list[13],
                  'RAk' : modi_crop_list[14], 'LAk' : modi_crop_list[15],
                  'UBk' : modi_crop_list[16]}


def get_cropping_configuration():
    return modi_crop_list


def get_subject_zone_label(zone_num, df):
    #-------------------------------------------------------------------------------------------
    # get_subject_zone_label(zone_num, df): gets a label for a given subject and zone
    # zone_num:                             a 0 based threat zone index
    # df:                                   a df like that returned from get_subject_labels(...)
    # returns:                              [0,1] if contraband is present, [1,0] if it isnt
    #-------------------------------------------------------------------------------------------

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

def display(img, title="Output"):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_zone_coord(frameNum, zoneid):
    coord = modi_crop_list[zoneid][frameNum]
    coord = np.array(coord) - [xoff, yoff, 0, 0] # because imgH is not 660
    return coord

def frame_proir_zone_coord(frameNum, zone_name):
    return modi_crop_dict[zone_name][frameNum]


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


def color_to_binary(img):
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gbimg = cv.GaussianBlur(gimg, (5,5), 0)
    ret, timg = cv.threshold(gbimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return timg


def transpixel(x, y):
    global scaleFactor, x_shift
    o_x, o_y = x * scaleFactor, y * scaleFactor
    x_margin = ((imgW * scaleFactor) - imgW) / 2
    y_margin = (imgH * scaleFactor) - imgH
    t_x, t_y = o_x - x_margin - x_shift, o_y - y_margin # - x_shift
    return int(t_x), int(t_y)

# draws box from coordinates in list
def parse_refpt_list(list):
    transList = []
    for x in range(len(list)):
        box = list[x]
        tx1, ty1 = box[0][0], box[0][1]
        tx2, ty2 = box[1][0], box[1][1]
        tx1, ty1 = transpixel(tx1, ty1)
        tx2, ty2 = transpixel(tx2, ty2)
        transList.append([(tx1, ty1), (tx2, ty2)])
    return transList

def get_top_pixel(timg):
    # returns 1D (size = num of columns) ndarray containing
    # row indices of first occurrence for each column
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
    # returns 1D (size = num of rows) ndarray containing
    # col indices of first occurrence for each row
    col_indices = np.argmax(img[start:, :]==255, axis=1)
    # Note: 0 is returned per column if no 255 pixel was found in column.
    # therefore, to make np.argmin work set all 0s to max:660 so np.argmin does not pick them
    col_indices[col_indices == 0] = 660
    # returns 1D (size = 1) ndarray containing col indices of first occurrence
    y = np.argmin(col_indices)
    x = col_indices[y]
    return x

def get_aspect_ratio(img, y_top, leftmost, rightmost, rep):
    shrink = step * rep
    crop_top = max(0, y_top - margin)
    crop_lef = max(0, leftmost - (margin - shrink))
    crop_rig = min(510, rightmost + (margin - shrink)) + 1
    #print(crop_top, crop_lef, crop_rig, rep)
    focus = img[crop_top:, crop_lef:crop_rig, :]
    ratio = focus.shape[1] / focus.shape[0]
    return ratio, focus

def standard_resize(img, aspectRatio):
    global scaleFactor
    new_width = int(aspectRatio * standard_H)
    if new_width > imgW:
        new_width = imgW
        print('**Warning: Extreme measure used to shrink and fir image')
    scaleFactor = standard_H / img.shape[0]
    rimg = cv.resize(img, (new_width, standard_H), interpolation=cv.INTER_AREA)
    return rimg

def transtand(img, y_top, leftmost, rightmost):
    aspectRatio, rc = maxAspectRatio + 1, 0
    while aspectRatio > maxAspectRatio and rc < maxRepeat:
        aspectRatio, focus = get_aspect_ratio(img, y_top, leftmost, rightmost, rc)
        rc += 1
    #print(focus.shape)
    sized = standard_resize(focus, aspectRatio)
    #print (sized.shape)
    padding = imgW - sized.shape[1]
    leftpad = int(padding / 2)
    rightpad = leftpad + sized.shape[1]
    slate = np.copy(background)
    slate[6:, leftpad:rightpad] = sized
    return slate

def spatial_normalization(img):
    global x_shift
    timg = color_to_binary(img)
    topmost = get_top_pixel(timg)
    x_rightmost = imgW - get_x_bounds(timg, 'right', 0)
    x_leftmost = get_x_bounds(timg, 'left', 0)
    simg = transtand(img, topmost, x_leftmost, x_rightmost)
    x_shift = ((x_leftmost - (imgW-x_rightmost)) * scaleFactor) / 2
    return simg

def read_image(file):
    img = cv.imread(file)
    return cv.cvtColor(spatial_normalization(img), cv.COLOR_BGR2GRAY)

def initialize_global_variables(wdt=512, hgt=656):
    global imgW, imgH, xoff, yoff, margin, standard_H, maxAspectRatio, maxRepeat, step, background
    imgW, imgH = wdt, hgt
    xoff, yoff = 512 - imgW, 660 - imgH
    margin, step = 50, 5
    maxRepeat = int(margin / step)
    standard_H = 650
    maxAspectRatio = imgW / standard_H
    background = cv.imread('../../Data/tsa_psc/background.png')
