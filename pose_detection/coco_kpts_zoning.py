import numpy as np

def valid_zone_per_16frames():
    # Zones:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17    # Frames
    config = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 0
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 1
              [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 2
              [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # 3
              [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 4
              [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],  # 5
              [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 6
              [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 7
              [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
              [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9
              [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 10
              [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # 11
              [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # 12
              [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0],  # 13
              [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],  # 14
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]  # 15
    return np.array(config, dtype=np.bool)

# def valid_keypt_per_16frames():
#     # Keypts:  1   2   3   4   5   6   7   8   9  10  11  12  13    # Frames
#     #          Ns RSh REb RWr LSh LEb LWr RHp RKe RAk LHp LKe LAk
#     config = [[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 0
#               [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 1
#               [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 2
#               [1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1],  # 3
#               [1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1],  # 4
#               [1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1],  # 5
#               [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 6
#               [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 7
#               [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 8
#               [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 9
#               [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1],  # 10
#               [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0],  # 11
#               [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0],  # 12
#               [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0],  # 13
#               [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1],  # 14
#               [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]  # 15
#     A = np.array(config, dtype=np.bool)
#     # transform shape=(16, 13) to shape=(13, 16)
#     return np.transpose(A) # equivalent to: np.rot90(np.fliplr(A))
def valid_kypts_per_16frames():
    # Key_pts: 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    #          Nk RSh REb RWr LSh LEb LWr RHp RKe RAk LHp LKe LAk MHp Hd   # Frames
    config = [[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 0
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 1
              [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1],  # 2*
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  1],  # 3**
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1],  # 4***
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  1],  # 5**
              [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1],  # 6*
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 7
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 8
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 9
              [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  1],  # 10*
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  1],  # 11**
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1],  # 12***
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  1],  # 13**
              [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  1],  # 14*
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]  # 15
    A = np.array(config, dtype=np.bool)
    # transform shape=(16, 13) to shape=(13, 16)
    return np.transpose(A) # equivalent to: np.rot90(np.fliplr(A))

# def valid_keypt_per_64frames():
#     config = np.ones(shape=(64, 13), dtype=np.int32)
#     config[[13, 14, 15, 16, 17, 18, 19, 20],  1] = 0  # RSh (kpt:2)
#     config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  2] = 0  # REb (kpt:3)
#     config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  3] = 0  # RWr (kpt:4)
#     config[[44, 45, 46, 47, 48, 49, 50, 51],  4] = 0  # LSh (kpt:5)
#     config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  5] = 0  # LEb (kpt:6)
#     config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  6] = 0  # LWr (kpt:7)
#     config[[13, 14, 15, 16, 17, 18, 19, 20],  7] = 0  # RHp (kpt:8)
#     config[[13, 14, 15, 16, 17, 18, 19, 20],  8] = 0  # RKe (kpt:9)
#     config[[13, 14, 15, 16, 17, 18, 19, 20],  9] = 0  # RAk (kpt:10)
#     config[[44, 45, 46, 47, 48, 49, 50, 51], 10] = 0  # LHp (kpt:11)
#     config[[44, 45, 46, 47, 48, 49, 50, 51], 11] = 0  # LKe (kpt:12)
#     config[[44, 45, 46, 47, 48, 49, 50, 51], 12] = 0  # LAk (kpt:13)
#     A = np.array(config, dtype=np.bool)
#     # transform shape=(64, 13) to shape=(13, 64)
#     return np.transpose(A)
def valid_kypts_per_64frames():
    config = np.ones(shape=(64, 15), dtype=np.int32)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  1] = 0  # RSh (kpt:2)
    config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  2] = 0  # REb (kpt:3)
    config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  3] = 0  # RWr (kpt:4)
    config[[44, 45, 46, 47, 48, 49, 50, 51],  4] = 0  # LSh (kpt:5)
    config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  5] = 0  # LEb (kpt:6)
    config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  6] = 0  # LWr (kpt:7)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  7] = 0  # RHp (kpt:8)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  8] = 0  # RKe (kpt:9)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  9] = 0  # RAk (kpt:10)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 10] = 0  # LHp (kpt:11)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 11] = 0  # LKe (kpt:12)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 12] = 0  # LAk (kpt:13)
    config[[13, 14, 15, 17, 18, 19, 20, 44, 45, 46, 47, 49, 50, 51], 13] = 0  # MHp (kpt:14)***
    A = np.array(config, dtype=np.bool)
    # transform shape=(64, 13) to shape=(13, 64)
    return np.transpose(A)


def set_config(padConfig, frameW, frameH):
    global pad, frameWdt, frameHgt, NO_COORD
    pad = padConfig
    frameWdt = frameW
    frameHgt = frameH
    NO_COORD = np.array([0, 0, 0, 0, 0])


def confined_within_boundary(coord):
    '''
    Adjust coordinates (crop window location and size) to stay within image boundary
    :param coord:   crop coordinate vector: [xLeft, yTop, wdt, hgt]
    :return:        adjusted coordinate with tag: 1 at 4th index
    '''
    xLeft, yTop, wdt, hgt = coord
    assert (xLeft <= frameWdt and wdt >= 0)
    assert (yTop <= frameHgt and hgt >= 0)

    if xLeft < 0:
        # shift window to the right and alter (shrink wdt) crop window size
        wdt = max(0, wdt + xLeft)       # smaller width
        xLeft = 0
    if (xLeft + wdt) > frameWdt:
        # shift window to the left by altering (shrink wdt) crop window size
        wdt = max(0, frameWdt - xLeft)  # smaller width
    if yTop < 0:
        # shift window down and alter (shrink hgt) crop window size
        hgt = max(0, hgt + yTop)        # smaller height
        yTop = 0
    if (yTop + hgt) > frameHgt:
        # shift window up by altering (shrink hgt) crop window size
        hgt = max(0, frameHgt - yTop)   # smaller height

    assert (0 <= xLeft <= frameWdt and 0 <= (xLeft + wdt) <= frameWdt)
    assert (0 <= yTop <= frameHgt and 0 <= (yTop + hgt) <= frameHgt)
    return [xLeft, yTop, wdt, hgt, 1]


def locate_zone_1(fid, k2, k3, reliableKeypt): # Right Bicep
    if reliableKeypt[1] and reliableKeypt[2]:
        xMin, yMin = min(k2[0], k3[0]), min(k2[1], k3[1])
        wdt, hgt = abs(k2[0] - k3[0]), abs(k2[1] - k3[1])
        xMin, wdt = xMin - pad[0][0], wdt + pad[0][1]
        yMin, hgt = yMin - pad[0][2], hgt + pad[0][3]
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_2(fid, k3, k4, reliableKeypt): # Right Forearm
    if reliableKeypt[2] and reliableKeypt[3]:
        xMin, yMin = min(k3[0], k4[0]), min(k3[1], k4[1])
        wdt, hgt = abs(k3[0] - k4[0]), abs(k3[1] - k4[1])
        xMin, wdt = xMin - pad[1][0], wdt + pad[1][1]
        yMin, hgt = yMin - pad[1][2], hgt + pad[1][3]
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_3(fid, k5, k6, reliableKeypt): # Left Bicep
    if reliableKeypt[4] and reliableKeypt[5]:
        xMin, yMin = min(k5[0], k6[0]), min(k5[1], k6[1])
        wdt, hgt = abs(k5[0] - k6[0]), abs(k5[1] - k6[1])
        xMin, wdt = xMin - pad[0][0], wdt + pad[0][1]
        yMin, hgt = yMin - pad[0][2], hgt + pad[0][3]
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_4(fid, k6, k7, reliableKeypt): # Left Forearm
    if reliableKeypt[5] and reliableKeypt[6]:
        xMin, yMin = min(k6[0], k7[0]), min(k6[1], k7[1])
        wdt, hgt = abs(k6[0] - k7[0]), abs(k6[1] - k7[1])
        xMin, wdt = xMin - pad[1][0], wdt + pad[1][1]
        yMin, hgt = yMin - pad[1][2], hgt + pad[1][3]
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_517(fid, k2, k5, k8, k11, reliableKeypt): # Chest and Back TODO: eliminate unnecessary min() & max()
    if 0 <= fid <= 2 or 7 <= fid <= 9 or fid >= 14: # chest or back, full visibility
        if reliableKeypt[1] and reliableKeypt[4] and reliableKeypt[7] and reliableKeypt[10]:
            xMin, yMin = min(k2[0], k5[0]) - pad[2][0], min(k2[1], k5[1]) - pad[2][2]
            wdt = abs(k2[0] - k5[0]) + pad[2][1]
            hgt = max(abs(k2[1] - k8[1]), int(abs(k5[1] - k11[1]))) * 0.3 + pad[2][3]
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 3 <= fid <= 5: # chest, right side hidden
        if reliableKeypt[4] and reliableKeypt[10]:
            xMin, yMin = k5[0] - 150, min(k5[1], k11[1]) - pad[2][2]
            wdt, hgt = 150, int(abs(k5[1] - k11[1]) * 0.3 + pad[2][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif fid == 6: # back, right side partially hidden
        if reliableKeypt[4] and reliableKeypt[7] and reliableKeypt[10]:
            xMin, yMin = k5[0] - pad[2][0], min(k5[1], k11[1]) - pad[2][2]
            wdt, hgt = abs(k5[0] - k8[0]) + pad[2][1], int(abs(k5[1] - k11[1]) * 0.3 + pad[2][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif fid == 10:  # back, left side partially hidden
        if reliableKeypt[1] and reliableKeypt[7] and reliableKeypt[10]:
            xMin, yMin = k11[0] - pad[2][0], min(k2[1], k8[1]) - pad[2][2]
            wdt, hgt = abs(k2[0] - k11[0]) + pad[2][1], int(abs(k2[1] - k8[1]) * 0.3 + pad[2][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif fid == 11: # back, left side hidden
        if reliableKeypt[1] and reliableKeypt[7]:
            xMin, yMin = k2[0] - 150, min(k2[1], k8[1]) - pad[2][2]
            wdt, hgt = 150, int(abs(k2[1] - k8[1]) * 0.3 + pad[2][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif fid == 12 or fid == 13: # chest, left side hidden
        if reliableKeypt[1] and reliableKeypt[7]:
            xMin, yMin = k2[0], min(k2[1], k8[1]) - pad[2][2]
            wdt, hgt = 150, int(abs(k2[1] - k8[1]) * 0.3 + pad[2][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_6(fid, k1, k2, k8, reliableKeypt): # Right Arbs
    if 0 <= fid <= 1 or fid == 15:  # front fully visible
        if reliableKeypt[0] and reliableKeypt[1] and reliableKeypt[7]:
            xMin, yMin = k1[0] - 150, int(k8[1] - abs(k2[1] - k8[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150, int(abs(k2[1] - k8[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 7 <= fid <= 10: # back visible
        if reliableKeypt[0] and reliableKeypt[1] and reliableKeypt[7]:
            xMin, yMin = k1[0], int(k8[1] - abs(k2[1] - k8[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150, int(abs(k2[1] - k8[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 11 <= fid <= 14: # right side visible
        if reliableKeypt[1] and reliableKeypt[7]:
            xMin, yMin = k2[0] - pad[3][0], int(k8[1] - abs(k2[1] - k8[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150 + pad[3][1], int(abs(k2[1] - k8[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_7(fid, k1, k5, k11, reliableKeypt): # Left Arbs
    if 0 <= fid <= 1 or fid == 15:  # front fully visible
        if reliableKeypt[0] and reliableKeypt[4] and reliableKeypt[10]:
            xMin, yMin = k1[0], int(k11[1] - abs(k5[1] - k11[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150, int(abs(k5[1] - k11[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 6 <= fid <= 9: # back visible
        if reliableKeypt[0] and reliableKeypt[4] and reliableKeypt[10]:
            xMin, yMin = k1[0] - 150, int(k11[1] - abs(k5[1] - k11[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150, int(abs(k5[1] - k11[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 2 <= fid <= 5: # left side visible
        if reliableKeypt[4] and reliableKeypt[10]:
            xMin, yMin = k5[0] + pad[3][0] - 150 - pad[3][1], int(k11[1] - abs(k5[1] - k11[1]) * 0.6 - pad[3][2])
            wdt, hgt = 150 + pad[3][1], int(abs(k5[1] - k11[1]) * 0.6 + pad[3][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_8(fid, k8, k9, reliableKeypt): # Right Hip
    if reliableKeypt[7] and reliableKeypt[8]:
        waist_knee_len = abs(k8[1] - k9[1])
        xMin, yMin = min(k8[0], k9[0]) - pad[4][0], k8[1] - pad[4][2]
        wdt, hgt = abs(xMin - max(k8[0], k9[0])) + pad[4][0], int(waist_knee_len * 0.75 + pad[4][3])
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_9(fid, k8, k9, k11, k12, reliableKeypt): # Groin Region
    # assumes k8[1] == k11[1] && k9[1] == k12[1]
    if 0 <= fid <= 1 or 7 <= fid <= 9 or fid == 15: # front or back visible
        if reliableKeypt[7] and reliableKeypt[8] and reliableKeypt[10]:
            xMin, yMin = min(k8[0], k11[0]) + pad[4][1], k8[1] - pad[4][2]
            wdt, hgt = abs(k8[0] - k11[0]) - pad[4][1], int(abs(k8[1] - k9[1]) * 0.75 + pad[4][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 2 <= fid <= 3: # front (left side) partially visible
        if reliableKeypt[10] and reliableKeypt[11]:
            xMin, yMin = k11[0] - 120, k11[1] - pad[4][2]
            wdt, hgt = 120, int(abs(k11[1] - k12[1]) * 0.75 + pad[4][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])
    elif 13 <= fid <= 14: # front (right side) partially visible
        if reliableKeypt[7] and reliableKeypt[8]:
            xMin, yMin = k8[0], k8[1] - pad[4][2]
            wdt, hgt = 120, int(abs(k8[1] - k9[1]) * 0.75 + pad[4][3])
            return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_10(fid, k11, k12, reliableKeypt): # Left Hip
    if reliableKeypt[10] and reliableKeypt[11]:
        waist_knee_len = abs(k11[1] - k12[1])
        xMin, yMin = min(k11[0], k12[0]) - pad[4][0], k11[1] - pad[4][2]
        wdt, hgt = abs(xMin - max(k11[0], k12[0])) + pad[4][0], int(waist_knee_len * 0.75 + pad[4][3])
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_11(fid, k8, k9, reliableKeypt): # Right Thigh
    if reliableKeypt[7] and reliableKeypt[8]:
        waist_knee_len = abs(k8[1] - k9[1])
        xMin, yMin = min(k8[0], k9[0]) - pad[5][0], int(k9[1] - waist_knee_len * 0.25 - pad[5][2])
        wdt, hgt = abs(xMin - max(k8[0], k9[0])) + pad[5][0], int(waist_knee_len * 0.5)
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_12(fid, k11, k12, reliableKeypt): # Left Thigh
    if reliableKeypt[10] and reliableKeypt[11]:
        waist_knee_len = abs(k11[1] - k12[1])
        xMin, yMin = min(k11[0], k12[0]) - pad[5][0], int(k12[1] - waist_knee_len * 0.25 - pad[5][2])
        wdt, hgt = abs(xMin - max(k11[0], k12[0])) + pad[5][0], int(waist_knee_len * 0.5)
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_13(fid, k9, k10, reliableKeypt): # Right Calf
    if reliableKeypt[8] and reliableKeypt[9]:
        knee_ankle_len = abs(k9[1] - k10[1])
        xMin, yMin = min(k9[0], k10[0]) - pad[5][0], int(k9[1] + knee_ankle_len * 0.33 - pad[5][2])
        wdt, hgt = abs(xMin - max(k9[0], k10[0])) + pad[5][0], abs(min(k10[1], (frameHgt - 60)) - yMin)
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_14(fid, k12, k13, reliableKeypt): # Left Calf
    if reliableKeypt[11] and reliableKeypt[12]:
        knee_ankle_len = abs(k12[1] - k13[1])
        xMin, yMin = min(k12[0], k13[0]) - pad[5][0], int(k12[1] + knee_ankle_len * 0.33 - pad[5][2])
        wdt, hgt = abs(xMin - max(k12[0], k13[0])) + pad[5][0], abs(min(k13[1], (frameHgt - 60)) - yMin)
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_15(fid, k9, k10, reliableKeypt): # Right Ankle
    if reliableKeypt[8] and reliableKeypt[9]:
        xMin, yMin = min(k9[0], k10[0]) - pad[5][0], min(k10[1], (frameHgt - 60))
        wdt, hgt = abs(xMin - max(k9[0], k10[0])) + pad[5][0], frameHgt - yMin
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD

def locate_zone_16(fid, k12, k13, reliableKeypt): # Left Ankle
    if reliableKeypt[11] and reliableKeypt[12]:
        xMin, yMin = min(k12[0], k13[0]) - pad[5][0], min(k13[1], (frameHgt - 60))
        wdt, hgt = abs(xMin - max(k12[0], k13[0])) + pad[5][0], frameHgt - yMin
        return confined_within_boundary([xMin, yMin, wdt, hgt])

    return NO_COORD
