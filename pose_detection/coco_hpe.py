import numpy as np
import cv2 as cv
import sys
import os


class CocoHPE(object):
    '''
        Class implements human pose estimation using pre-trained Caffe COCO model
        Pose estimation includes keypoint estimation and part affinity.
        Currently using opencv dnn implementation
    '''

    def __init__(self, validKyptsInFrames=None, framesPerScan=64):
        self.validKptvsFrm = validKyptsInFrames # shape=(13, 16/64)
        poseModelDir = "../../../repos_unofficial/RTMPPE/openpose-master/models/"
        cocoWeights = "pose/coco/pose_iter_440000.caffemodel"
        cocoProto = "pose/coco/pose_deploy_linevec.prototxt"
        protoFile = os.path.join(poseModelDir, cocoProto)
        weightsFile = os.path.join(poseModelDir, cocoWeights)
        if os.path.exists(protoFile) and os.path.exists(weightsFile):
            # Initialize opencv poseNN by Reading the network into Memory
            self._net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
        else:
            print('Error! protoFile: {} or weightsFile: {} does not exist'.
                  format(protoFile, weightsFile))
            sys.exit()

        self.COCO_KEYPOINTS_ALL = 19  # originally 19 keypoints
        self.COCO_KEYPOINTS_IMP = 14 # of important keypoints starting from 0, alt. COCO_KEYPOINTS
        self.COCO_KEYPOINTS_INT = 13 # 13 keypoints of interest, alt. MAIN_KEYPOINTS
        self.FRAMES_PER_SCAN = framesPerScan
        self.NUM_OF_ZONES = 17

        self.STICK_FIGURE_PARTS = \
            ["RShoulder", "LShoulder", "RArm", "RForearm", "LArm", "LForearm",
             "RAbdomen", "RThigh", "RLeg", "LAbdomen", "LThigh", "LLeg"]

        self.ALLiNDX_TO_INTiNDX = \
            {0:None, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10,
             12:11, 13:12, 14:None, 15:None, 16:None, 17:None, 18:None}

        self.KPTLABEL_COLORCODE = \
            set_colors(['Ns', 'Nk', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
                        'RAk', 'LHp', 'LKe', 'LAk', 'REy', 'LEy', 'REr', 'LEr', 'MHp', 'Bg'])

        self.INDEX_TO_KPT_LABEL = \
            {0:'Ns', 1:'Nk', 2:'RSh', 3:'REb', 4:'RWr', 5:'LSh', 6:'LEb',
             7:'LWr', 8:'RHp', 9:'RKe', 10:'RAk', 11:'LHp', 12:'LKe',
             13:'LAk', 14:'REy', 15:'LEy', 16:'REr', 17:'LEr', 18:'Bg'}

        self.KPT_LABEL_TO_INDEX = dict((v, k) for k, v in
                                       self.INDEX_TO_KPT_LABEL.items()) # flip keys & values

        self.BODY_PART_TO_INDEX = \
            {"Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4, "LShoulder":5,
             "LElbow":6, "LWrist":7, "RHip":8, "RKnee":9, "RAnkle":10, "LHip":11, "LKnee":12,
             "LAnkle":13, "REye":14, "LEye":15, "REar":16, "LEar":17, "Background":18}

        # symmetric keypoints
        self.SYM_BODY_PART_KPTS = \
            {'RShoulder':1, 'RElbow':2, 'RWrist':3, 'RHip': 7, 'RKnee': 8, 'RAnkle': 9,
             'LShoulder':4, 'LElbow':5, 'LWrist':6, 'LHip':10, 'LKnee':11, 'LAnkle':12}

        self.BODY_PART_KPTS_PAIR = \
            {"RShoulder":(1, 2), "LShoulder":(1, 5), "RArm":(2, 3), "RForearm":(3, 4),
             "LArm":(5, 6), "LForearm":(6, 7), "RAbdomen":(1, 8), "RThigh":(8, 9),
             "RLeg":(9, 10), "LAbdomen":(1, 11), "LThigh":(11, 12), "LLeg":(12, 13),
             "NeckNose":(1, 0), "RNoseEye":(0, 14), "REyeEar":(14, 16), "LNoseEye":(0, 15),
             "LEyeEar":(15, 17), "REarShoulder":(2, 16), "LEarShoulder":(5, 17)}

        self.COCO_HEATMAP_PAFVEC = \
            {"RShoulder":(31, 32), "LShoulder":(39, 40), "RArm":(33, 34), "RForearm":(35, 36),
             "LArm":(41, 42), "LForearm":(43, 44), "RAbdomen":(19, 20), "RThigh":(21, 22),
             "RLeg":(23, 24), "LAbdomen":(25, 26), "LThigh":(27, 28), "LLeg":(29, 30),
             "NeckNose":(47, 48), "RNoseEye":(49, 50), "REyeEar":(53, 54), "LNoseEye":(51, 52),
             "LEyeEar":(55, 56), "REarShoulder":(37, 38), "LEarShoulder": (45, 46)}



    def pose_stick_figure(self, image, xKpts, yKpts, kptsConf, fid=None,
                          kptsScore=None, kptsLwgt=None, kptsNdft=None, kptsDrift=None,
                          header='aps', thick=1, xstart=0, ystart=0,
                          lineColor=(255, 255, 255), label=True, invKpts=True, invEdge=True):
        '''
        Draw pose stick figure on image. xKpts, yKpts, and kptsConf must all be the same shape
        :param image:   single image (color/gray) that is the same scale as keypoint coordinates
        :param xKpts:   x coordinate of interested keypoints, shape=(13), dtype=np.int32
        :param yKpts:   y coordinate of interested keypoints, shape=(13), dtype=np.int32
        :param kptsConf:keypoint confidence score, shape=(13), dtype=np.float32
        :param fid:     the frame ID for keypoints. Must be set if invKpts or invEdge is False
        :param label:   boolean variable indicating whether or not to label keypoints
        :param invKpts: whether or not to display invalid keypoints in frame
        :param invEdge: whether to display invalid parts. edge between at least 1 of 2 invalid kpts
        :return:        copied image that was altered
        '''
        #assert (xKpts.shape==yKpts.shape and yKpts.shape==kptsConf.shape)
        # copy image and convert to colored if necessary
        imgBGR = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.shape[2]==1 else np.copy(image)
        column = '    cnf' if kptsScore is None else '    cnf dft wgt ndf scr'
        if label and isinstance(fid, int):
            x_pos = xstart + 5
            y_pos = ystart + 100
            cv.putText(imgBGR, header, (x_pos, y_pos), cv.FONT_HERSHEY_PLAIN,
                       1, lineColor, 1, lineType=cv.LINE_AA)
            cv.putText(imgBGR, column, (x_pos, y_pos + 20), cv.FONT_HERSHEY_PLAIN,
                       0.8, lineColor, 1, lineType=cv.LINE_AA)

        # draw lines (edges) between two keypoints
        for limb in self.STICK_FIGURE_PARTS:
            kA, kB = self.BODY_PART_KPTS_PAIR[limb]
            kptA, kptB = self.ALLiNDX_TO_INTiNDX[kA], self.ALLiNDX_TO_INTiNDX[kB]
            #assert (isinstance(kptA, int) and isinstance(kptB, int))
            if invEdge or (self.validKptvsFrm[kptA][fid] and self.validKptvsFrm[kptB][fid]):
                partA = (xKpts[kptA], yKpts[kptA])
                partB = (xKpts[kptB], yKpts[kptB])
                cv.line(imgBGR, partA, partB, (0, 0, 0), thickness=(thick + 2))
                cv.line(imgBGR, partA, partB, lineColor, thickness=thick)

        # highlight keypoints
        for kpt in range(xKpts.shape[0]):
            if invKpts or self.validKptvsFrm[kpt][fid]:
                point = (xKpts[kpt], yKpts[kpt])
                labl, conf, score, lwgt, ndft, drift  = None, None, None, None, None, None
                if label:
                    conf = kptsConf[kpt]
                    labl = self.INDEX_TO_KPT_LABEL[kpt + 1]
                    if kptsScore is not None:
                        score = kptsScore[kpt]
                        lwgt = kptsLwgt[kpt]
                        ndft = kptsNdft[kpt]
                        drift = kptsDrift[kpt]
                self.pinpoint_keypoint(imgBGR, point, xstart, ystart, kpt,
                                       conf, lwgt, drift, ndft, score, kptLabel=labl)

        return imgBGR


    def pinpoint_keypoint(self, image, point, xstart, ystart, kptID, confidence=None,
                          lwgt=None, drift=None, ndft=None, score=None, kptLabel=None, tagDot=True):
        '''
        Highlight or mark a keypoint and append text label if provided
        :param image:       single image (must be colored) and same scale as keypoint coordinates
        :param point:       pixel point: (x, y) to keypoint
        :param kptID:       keypoint ID
        :param confidence:  confidence score of keypoint generated or passed from hpe
        :param kptLabel:    label of the keypoint indicating what body joint/part
        :return:            changes are made to passed image inplace so nothing is returned
        '''
        shadowColor = (0, 0, 0)
        keypntColor = self.KPTLABEL_COLORCODE[kptLabel] if kptLabel else (0, 0, 0)
        cv.circle(image, point, 7, shadowColor, thickness=-1, lineType=cv.FILLED)
        cv.circle(image, point, 5, keypntColor, thickness=-1, lineType=cv.FILLED)
        if confidence and tagDot:
            txt = '{:.2f}'.format(confidence).strip('0') # remove leading and trailing zeros
            loc = (point[0] - 8, point[1] - 8)
            cv.putText(image, txt, loc, cv.FONT_HERSHEY_PLAIN,
                       0.8, shadowColor, 2, lineType=cv.LINE_AA)
            cv.putText(image, txt, loc, cv.FONT_HERSHEY_PLAIN,
                       0.8, keypntColor, 1, lineType=cv.LINE_AA)
        if kptLabel and confidence:
            lablLoc = (xstart + 5, ystart + (kptID * 14) + 140)
            conf_txt = '{:.2f}'.format(confidence).strip('0') # remove leading and trailing zeros
            if score is not None:
                lwgt_txt = '{:.2f}'.format(lwgt).strip('0')
                drift_txt = '{}'.format(drift)
                ndft_txt = '{:.2f}'.format(ndft).strip('0')
                score_txt = '{:.2f}'.format(score).strip('0')
                txt = "{:<3} {:<3} {:>3} {:<3} {:<3} {:<3}".\
                    format(kptLabel, conf_txt, drift_txt, lwgt_txt, ndft_txt, score_txt)
            else: txt = '{:<3} {:<3}'.format(kptLabel, conf_txt)
            cv.putText(image, txt, lablLoc, cv.FONT_HERSHEY_PLAIN,
                       0.8, keypntColor, 1, lineType=cv.LINE_AA)
        elif confidence:
            infoLoc = (point[0] - 15, point[1] - 5)
            txt = " {}".format(round(confidence, 2))
            cv.putText(image, txt, infoLoc, cv.FONT_HERSHEY_SIMPLEX,
                       0.4, shadowColor, 2, lineType=cv.LINE_AA)
            cv.putText(image, txt, infoLoc, cv.FONT_HERSHEY_SIMPLEX,
                       0.4, keypntColor, 1, lineType=cv.LINE_AA)


    def decipher_cvdnn_pt(self, pt, hpeWdt, hpeHgt, imgWdt=256, imgHgt=336):
        '''
        Scale the point to fit on the original image, then take center of scaled pixel box
        :param pt:      point (x, y) coordinate
        :param hpeWdt:  width of output image by opencv's dnn Caffe model
        :param hpeHgt:  height of output image by opencv's dnn Caffe model
        :param imgWdt:  width to scale to
        :param imgHgt:  height to scale to
        :return:        point in center of scaled pixel box
        '''
        wShift, hShift = imgWdt / (2 * hpeWdt), imgHgt / (2 * hpeHgt)
        x = pt[0] * (imgWdt / hpeWdt)
        y = pt[1] * (imgHgt / hpeHgt)
        return (x + wShift, y + hShift)


    def keypoints_scores_and_pts(self, singleOut, keyPointsNum, faceKptsNum, W, H, imgW, imgH):
        '''
        Extract interested keypoint coordinates and scores from a given image
        :param singleOut:   shrunk ndarray from hpe for a single image, shape=(57, 42, 32)
        :param keyPointsNum:number of interested keypoints, eg. 13
        :param fid:         frame number ot ID
        :param W:           width of output image by opencv's dnn Caffe model
        :param H:           height of output image by opencv's dnn Caffe model
        :param imgW:        width to scale to
        :param imgH:        height to scale to
        :return:            vector of confidence, x, and y coordinates, shapes=(keyPointsNum)
        '''
        kyptConf = np.zeros(shape=(keyPointsNum), dtype=np.float32)
        xptCoord = np.zeros(shape=(keyPointsNum), dtype=np.float32)
        yptCoord = np.zeros(shape=(keyPointsNum), dtype=np.float32)
        faceKpts = np.zeros(shape=(faceKptsNum, 3), dtype=np.float32)

        for i in range(self.COCO_KEYPOINTS_ALL - 1):
            # confidence map of corresponding body's part.
            confMap = singleOut[i, :, :]
            # Find global maxima of the probMap.
            conf = np.max(confMap)
            indicies = np.unravel_index(np.argmax(confMap), confMap.shape)
            pt = (indicies[1], indicies[0])
            x, y = self.decipher_cvdnn_pt(pt, W, H, imgWdt=imgW, imgHgt=imgH)

            if i in range(1, keyPointsNum + 1):
                # interested keypoints
                #assert (1 <= i <= 13)
                kypt_indx = i - 1
                kyptConf[kypt_indx] = conf
                xptCoord[kypt_indx] = x
                yptCoord[kypt_indx] = y
            else:
                # face keypoint
                face_indx = max(0, i - keyPointsNum)
                #assert (i in [0, 14, 15, 16/64, 17])
                faceKpts[face_indx] = [x, y, conf]

        return xptCoord, yptCoord, kyptConf, faceKpts


    def interested_keypoints_and_confidence(self, output, imgW, imgH):
        '''
        Extract and record interested kpts coordinates and confidence for given scan (16/64 frames)
            output.shape = (batch, 57, 42, 32) : (# of images, 19 KP + 2*19 PAF, Map Hgt, Map Wdt)
            return ndarray shape = (13, 16/64) : (keyPointsNum, batch)
        :param output:  n dimensional array from model, shape=(batch, 57, 42, 32)
        :param imgW:    width of image that was passed to hpe
        :param imgH:    height of image that was passed to hpe
        :return:        ndarray of confidence, x, and y coordinates, shapes=(13, 16/64)
        '''

        batch = output.shape[0]
        H = output.shape[2]
        W = output.shape[3]
        keyPtsNum = self.COCO_KEYPOINTS_INT
        facePtNum = self.COCO_KEYPOINTS_ALL - keyPtsNum - 1 # Nose, Eyes, and Ears

        keypt_conf = np.zeros(shape=(keyPtsNum, batch), dtype=np.float32)
        x_pt_coord = np.zeros(shape=(keyPtsNum, batch), dtype=np.float32)
        y_pt_coord = np.zeros(shape=(keyPtsNum, batch), dtype=np.float32)
        face_kypts = np.zeros(shape=(facePtNum, batch, 3), dtype=np.float32)

        for fid in range(batch):
            xVec, yVec, cVec, face = \
                self.keypoints_scores_and_pts(output[fid], keyPtsNum, facePtNum, W, H, imgW, imgH)
            x_pt_coord[:, fid], y_pt_coord[:, fid], keypt_conf[:, fid] = xVec, yVec, cVec
            face_kypts[:, fid] = face

        return x_pt_coord, y_pt_coord, keypt_conf, face_kypts


    def get_keypoint_scores_only(self, output, keyPointsNum):
        score = np.zeros(shape=(keyPointsNum - 1), dtype=np.float32)
        for i in range(1, keyPointsNum):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            # Find global maxima of the probMap.
            minVal, prob, minLoc, pt = cv.minMaxLoc(probMap)
            score[i - 1] = prob
        return score


    def feed_to_pose_nn(self, images, wdt=256, hgt=336):
        '''
        Feed the input to the pre-trained model and returns the output
            index i in first dimension represents results for image i in blob
        :param images: input image or array of images
        :param wdt:    width of input image(s)
        :param hgt:    height of input image(s)
        :return:       array of arrays of output from model
        '''

        # Prepare the frame to be fed to the network
        if images.ndim == 3:
            inpBlob = cv.dnn.blobFromImage(images, 1.0/255, (wdt, hgt),
                                           (0, 0, 0), swapRB=False, crop=False)
        else:
            inpBlob = cv.dnn.blobFromImages(images, 1.0/255, (wdt, hgt),
                                            (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared object as the input blob of the network
        self._net.setInput(inpBlob)

        # make predictions and parse keypoints
        output = self._net.forward()

        return output


def set_colors(colorKeys, step=15, pad=7):
    '''
    Set a unique, distinct color (BGR) for each color key and return dictionary of key-to-color map
        Note: infinite loop may occur in current implementation
    :param colorKeys: list of color keys
    :return:          dictionary of key-to-color map
    '''
    from random import seed
    from random import randint
    colorKeys.sort()
    seed(len(colorKeys))
    keyToColorMap = {}
    usedColors = [(0, 0, 0), (255, 255, 255)] # Black cannot be used
    b, g, r = usedColors[0]
    for i, key in enumerate(colorKeys):
        j = i
        while (b, g, r) in usedColors:
            s = (step * j) % 220
            b, g, r = randint(s, 256), randint(s, 256), randint(s, 256)
            '''
            s = max(0, (step * j) % 255)
            e = s + 34
            l, a, b = randint(1, 255), randint(1, e), randint(s, 255)
            pixels = np.repeat(np.asarray([l, a, b]), 100, axis=0)
            pixels = np.reshape(pixels, (10, 10, 3)).astype(np.uint8)
            b, g, r = cv.cvtColor(pixels, cv.COLOR_LAB2BGR)[0, 0, :]
            b, g, r = int(b), int(g), int(r)
            '''
            j += 1
        keyToColorMap[key] = (b, g, r)
        for b_idx in range(-pad, pad + 1):
            for g_idx in range(-pad, pad + 1):
                for r_idx in range(-pad, pad + 1):
                    usedColors.append((b + b_idx, g + g_idx, r + r_idx))
    return keyToColorMap


def set_spaced_colors(colorKeys):
    # evenly spaced colors
    n = len(colorKeys)
    maxValue = 16581375  # 255**3
    minValue = 8000 # 20**3
    interval = int((maxValue - minValue) / n)
    hxColors = [hex(I)[2:].zfill(6) for I in range(minValue, maxValue, interval)]
    colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in hxColors]
    return dict(zip(colorKeys, colors))

