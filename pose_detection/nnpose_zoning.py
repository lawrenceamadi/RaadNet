'''
    Refine pose estimated keypoints using TSA dataset priors
    - Swap symmetric keypoints that are not in proper order relative to each other
    - Reset y-coordinates of low-confident/outlier keypoints
'''

import numpy as np
np.set_printoptions(precision=2, threshold=np.inf, floatmode='fixed', linewidth=150)

import pandas as pd
import ast
import sys
import os

sys.path.append('../')
from pose_detection import coco_hpe
from pose_detection import coco_kpts_zoning as znkp
from image_processing import transformations as imgtf



def show_stict_figure(title, images, xCoord, yCoord, kptConf, invEdge=False, invKpts=True):
    '''
    Display pose estimation stick figure of scan images
    :param title:   name of images being shown
    :param images:  ndarray of scan images or frames
    :param xCoord:  ndarray of x coordinates of keypoints in all frames
    :param yCoord:  ndarray of y coordinates of keypoints in all frames
    :param kptConf: ndarray of keypoint confidence
    :param invKpts: whether or not to display invalid keypoints in frame
    :param invEdge: whether to display invalid parts. ie, edge between at least 1 of 2 invalid kpts
    :return: nothing
    '''
    for fin in range(N_FRAMES):
        stick = \
            hpe.pose_stick_figure(images[fin], xCoord[:, fin].astype(np.int),
                                  yCoord[:, fin].astype(np.int),
                                  kptConf[:, fin], fid=fin, invEdge=invEdge, invKpts=invKpts)
        imgtf.displayWindow(stick, winName=title, x=2) #2000


def log(msg, mode=1, writeToFile=True):
    '''
    Log runtime messages to logfile
    :param msg:     message text to log
    :param mode:    message type indicating which logfile to attach msg to
    :param wrtToFile: indicating whether to write buffer to file
    :return:        nothing is returned
    '''
    print(msg)
    if writeToFile and LOG_INFO:
        with open(_logfile[mode], 'a+') as log:
            print(msg, file=log)
            log.close()


def eliminate_outliers(initPoints, fid, outlierThresh=50):
    '''
    Find and remove outlier points
    :param initPoints:      initial points
    :param fid              frame ID
    :param outlierThresh:   outlier threshold distance
    :return:                list of inlier points, list may be empty if outlierThresh is too small
    '''
    inliers = initPoints.tolist()
    repeat = True

    while repeat:
        points = inliers
        outlierIndx = []

        # find outliers
        for i in range(len(points)):
            otherPoints = points.copy()
            pt = otherPoints.pop(i)
            assert (len(otherPoints) == len(points) - 1)
            distToOthers = np.abs(np.array(otherPoints) - pt)
            closestDist = np.min(distToOthers)
            if closestDist > outlierThresh:
                outlierIndx.append(i)

        # remove outliers
        repeat = True if len(outlierIndx) > 0 else False
        for i, indx in enumerate(outlierIndx):
            outlier = inliers.pop(indx - i)
            log('-Outlier: face keypoint outlier. scan:{}, frame:{}, faceKptIndx:{},'
                ' xValue:{}'.format(_scanid, fid, indx, outlier), mode=2)

    return inliers


def facemask(faceCoord, faceKptConf, neckYCoord, neckMargin=15):
    '''
    Generate face mask for the f=N_FRAMES using
        Nose, Eyes, and Ears x coordinate and Necks' y coordinate
    :param faceCoord: shape=(5,16/64,2). (x,y) scaled coordinates of Nose, Eyes, & Ears
    :param faceKptConf: shape=(5,16/64), dtype=np.float32. confidence of Nose, Eyes, and Ears
    :param neckYCoord: best y-coord of neck. Possibly refined and set to y coord with highest prob
    :param neckMargin: neck y-coordinate allowance
    :return: return ndarray, shape=(16/64, FrameH, FrameW), dtype=np.int32 of computed face masks
    '''
    assert (faceCoord.shape==(5,N_FRAMES,2) and faceKptConf.shape==(5,N_FRAMES))
    faceMasks = np.ones(shape=(N_FRAMES,FRM_HGT,FRM_WDT), dtype=np.uint8) # was np.int32

    yBestIndices = np.argmax(faceKptConf, axis=1)  # frm-idx with highest score, for each kpt: (5)
    yBestKCoords = faceCoord[(0, 1, 2, 3, 4), yBestIndices, 1] # yNose, yREy, yLEy, yREr, yLEr
    fmskYmax = neckYCoord - neckMargin
    fmskYmid = int((np.min(yBestKCoords) + np.max(yBestKCoords)) / 2)
    fmskYhgt = 2 * max(0, neckYCoord - fmskYmid)
    fmskYmin = neckYCoord - fmskYhgt

    for fid in range(N_FRAMES):
        if not (fid in [4, 12]):
            xFaceKpts = eliminate_outliers(faceCoord[:, fid, 0], fid)
            if len(xFaceKpts) > 0:
                fmskXmin, fmskXmax = np.min(xFaceKpts), np.max(xFaceKpts) + 1
                #print(fid, fmskXmin, fmskXmax, fmskYmin, fmskYmid, fmskYmax, xFaceKpts)
                faceMasks[fid, fmskYmin: fmskYmax, fmskXmin: fmskXmax] = 0
    return faceMasks


def refine_keypoint_coord_v1(refine_x=False, refine_y=True, ankle_y=10):
    '''
    Refine keypoint coordinates by changing coordinates keypoints that have low confidence
        - ie. keypoints with confidence less than CONF_THRESHOLD.
        - y-coordinate of each kpt in ALL frames are replaced with
          the y-coordinate of the most confident kpt
    :param refine_x: boolean indicating whether or not to make changes to X coordinate if necessary
    :param refine_y: boolean indicating whether or not to make changes to Y coordinate
    :return:        keypoint coordinates of shape=(15, 16/64, 2), may or may not be refined
    '''
    # TODO: Instead of using fixed threshold, compute outliers based on
    #  deviation from most of the confidence scores.

    # copy over x_coord of keypoints (unreliable entries will be noted and addressed later in loop)
    refinedkptCoord = np.zeros(shape=(N_KPTS, N_FRAMES, 2), dtype=np.int32)
    refinedkptCoord[:, :, 0] = _xptCoord
    refinedkptCoord[:, :, 1] = _yptCoord

    # initialize reliable keypoints by running threshold test
    confidentKpts = _kyptConf>=CONF_THRESHOLD # shape=(15, 16/64) dtype=np.bool

    # eliminate keypoints with score < threshold (and invalid keypoints) by setting score to 0
    # notice in get_threshold_configuration() invalid keypoints will have scores < threshold
    # alternative: refined_scores = np.where(_kyptConf < CONF_THRESHOLD, 0, _kyptConf)
    refinedScores = np.where(confidentKpts, _kyptConf, 0)
    yBestIndicies = np.argmax(refinedScores, axis=1) # fid with highest score, for each kpt: (15,)
    # todo: 2 sets of yBestIndices for front & back view frames
    perKptScoreSum = np.sum(refinedScores, axis=1) # scores of each kpt summed across frames: (15,)

    for kid in range(N_KPTS):
        if refine_y:
            symPair = Y_REF_SYM_PAIR.get(kid) # None if kid is not a symmetric kpt
            if symPair is None:
                # when given kpt does not belong to a symmetric pair, set
                # y-coordinate of kpt in all frames to that of the most confident kpt
                bestKyptYcoord = _yptCoord[kid, yBestIndicies[kid]]
            else:
                # when given kpt belongs to a symmetric pair, set y-coordinate
                # of kpt in all frames to that of the most confident symmetric kpt
                kidA, kidB = symPair # symmetric keypoint indexes
                assert (kidA==kid or kidB==kid)
                idx = 0 if refinedScores[kidA, yBestIndicies[kidA]] > \
                           refinedScores[kidB, yBestIndicies[kidB]] else 1
                opt = symPair[idx]
                bestKyptYcoord = _yptCoord[opt, yBestIndicies[opt]]
            # set(broadcast) the best Y_Coord across all frames for each keypoint
            refinedkptCoord[kid, :, 1] = bestKyptYcoord
            if perKptScoreSum[kid] == 0:
                log('-Odd, None of {} (kpt:{:<3}) confidence score is above threshold '
                    'in all frames of scan:{} as a consequence y_coord of keypoint at '
                    'frame 0 will be used for guestimation..'
                    .format(COLLECTED_KPTS[kid], kid, _scanid), mode=2)

    # fix (set) the y_coord of Right(9) and Left(12) Ankles
    if ankle_y > 0: refinedkptCoord[(9, 12), :, 1] = IMG_HGT - ankle_y
    return refinedkptCoord


def refine_keypoint_coord_v2(refine_x=False, refine_y=True, ankle_y=10):
    '''
    Refine keypoint coordinates by changing coordinates keypoints that have low confidence
        - ie. keypoints with confidence less than CONF_THRESHOLD.
        - for each kpt, the y-coordinate of low confident frame kpt are replaced with
          the y-coordinate of the most confident kpt (or symmetric kpt)
    :param refine_x: boolean indicating whether or not to make changes to X coordinate if necessary
    :param refine_y: boolean indicating whether or not to make changes to Y coordinate
    :return: keypoint coordinates of shape=(15, 16/64, 2), may or may not be refined
    '''
    # TODO: Instead of using fixed threshold, compute outliers based on
    #  deviation from most of the confidence scores.

    # copy over x_coord of keypoints (unreliable entries will be noted and addressed later in loop)
    refinedkptCoord = np.zeros(shape=(N_KPTS, N_FRAMES, 2), dtype=np.int32)
    refinedkptCoord[:, :, 0] = _xptCoord
    refinedkptCoord[:, :, 1] = _yptCoord

    # initialize reliable keypoints by running threshold test
    confidentKpts = _kyptConf>=CONF_THRESHOLD # shape=(15, 16/64) dtype=np.bool

    # eliminate keypoints with score < threshold (and invalid keypoints) by setting score to 0
    # notice in get_threshold_configuration() invalid keypoints will have scores < threshold
    # alternative: refined_scores = np.where(_kyptConf < CONF_THRESHOLD, 0, _kyptConf)
    refinedScores = np.where(confidentKpts, _kyptConf, 0)
    yBestIndicies = np.argmax(refinedScores, axis=1) # fid with highest score, for each kpt: (15,)
    perKptScoreSum = np.sum(refinedScores, axis=1) # scores of each kpt summed across frames: (15,)

    for kid in range(N_KPTS):
        if refine_y:
            # set y-coordinate of non-confident kpts to that of the most confident kpt
            # or that of most confident symmetric kpt, if there is one
            bestKyptYcoord = None
            if perKptScoreSum[kid]==0:
                # no frame with confident enough kpt,
                # so check if symmetric kpt has a confident estimation
                symPair = Y_REF_SYM_PAIR.get(kid) # None if kid is not a symmetric kpt
                if symPair is not None:
                    # when given kpt belongs to a symmetric pair, set y-coordinate
                    # of non-confident kpts to that of the most confident symmetric kpt
                    kidA, kidB = symPair # symmetric keypoint indexes
                    assert (kidA==kid or kidB==kid)
                    sym_kid = kidA if kid==kidB else kidB
                    if perKptScoreSum[sym_kid] > 0:
                        bestKyptYcoord = _yptCoord[sym_kid, yBestIndicies[sym_kid]]

                log('-Odd, None of {} (kpt:{:<3}) confidence score is above threshold '
                    'in all frames of scan:{} as a consequence y_coord of keypoint at '
                    'frame 0 will be used for guestimation..'
                    .format(COLLECTED_KPTS[kid], kid, _scanid), mode=2)
            else:
                bestKyptYcoord = _yptCoord[kid, yBestIndicies[kid]]

            # set y-coord of non-confident kpts to that of the most confident kpt
            if bestKyptYcoord is not None:
                refinedkptCoord[kid, :, 1] = \
                    np.where(confidentKpts[kid], refinedkptCoord[kid, :, 1], bestKyptYcoord)

    # fix (set) the y_coord of Right(9) and Left(12) Ankles
    if ankle_y>0: refinedkptCoord[(9, 12), :, 1] = IMG_HGT - ankle_y
    return refinedkptCoord


def refine_keypoint_coord_v3(refine_x=False, refine_y=True, ankle_y=10):
    '''
    Refine keypoint coordinates by changing coordinates keypoints that have low confidence
        - low confident keypoints are defined as lower-bound-outliers (prob < Q1 - 1.5*IQR).
        - for each kpt, the y-coordinate of low confident frame kpt are replaced with
          the aggregate y-coordinate of the confident (inlier) kpts
    :param refine_x: boolean indicating whether or not to make changes to X coordinate if necessary
    :param refine_y: boolean indicating whether or not to make changes to Y coordinate
    :return: keypoint coordinates of shape=(15, 16/64, 2), may or may not be refined
    '''
    global _aggOutlierBookKeep

    # copy over x_coord of keypoints (unreliable entries will be noted and addressed later in loop)
    outlierBookKeep = np.zeros(shape=(N_KPTS, 3), dtype=np.float32)  # (n_in, n_out, out_thresh)
    refinedkptCoord = np.zeros(shape=(N_KPTS, N_FRAMES, 2), dtype=np.int32)
    refinedkptCoord[:, :, 0] = _xptCoord
    refinedkptCoord[:, :, 1] = _yptCoord

    for kid in range(N_KPTS):
        # initialize reliable keypoints by computing inliers vs. outlier per kpt set
        q1, q3 = np.quantile(_kyptConf[kid], [0.25, 0.75])  # compute Q1 & Q3
        iqr = q3 - q1  # compute inter-quartile-range
        lowOutlierThresh = q1 - 1.5*iqr
        inlierKpts = _kyptConf[kid]>=lowOutlierThresh
        inlierKptsIndexes = np.argwhere(inlierKpts)

        if refine_y:
            # set y-coordinate of non-confident kpts to that of the aggregate of confident kpts
            kptAggYcoord = np.mean(refinedkptCoord[kid, inlierKptsIndexes, 1])
            kptAggYcoord = np.around(kptAggYcoord, 0)
            refinedkptCoord[kid, :, 1] = \
                np.where(inlierKpts, refinedkptCoord[kid, :, 1], kptAggYcoord)

            n_inliers = np.sum(inlierKpts.astype(np.int32))
            n_outliers = N_FRAMES - n_inliers
            outlierBookKeep[kid] = [n_inliers, n_outliers, lowOutlierThresh]

    # book-keeping
    _aggOutlierBookKeep += outlierBookKeep
    log('--yref: scanid:{}\n\tn_inliers:{} - n_outliers:{} - outlier_thresh:{}'.
        format(_scanid, np.int32(outlierBookKeep[:,0]),
               np.int32(outlierBookKeep[:,1]), np.around(outlierBookKeep[:,2], 2)))

    # fix (set) the y_coord of Right(9) and Left(12) Ankles
    if ankle_y>0: refinedkptCoord[(9, 12), :, 1] = IMG_HGT - ankle_y
    return refinedkptCoord


def scale_back_keypoint_coord(keyptCoord):
    '''
    Scale keypoint coordinates back to the original FRM_WDT x FRM_HGT
        and translate to center. Also ensure coordinates are integers
    :param keyptCoord:  unscaled (IMG_WDT x IMG_HGT) keypoint coordinates shape: (15, 16/64, 2)
    :return: scaled (FRM_WDT x FRM_HGT) keypoint coordinates shape: (15, 16/64, 2), dtype=np.int32
    '''
    d1, d2, d3 = keyptCoord.shape
    A = np.ones(shape=(d1, d2, d3 + 1), dtype=np.float32)
    A[:, :, :2] = keyptCoord # last dimension must be of size 3
    scaled = np.dot(A, SCALE_TRANS_MATRIX)
    assert (scaled.shape[:2]==keyptCoord.shape[:2])  # eg.(15, 16/64, 3)
    assert (scaled.shape[2]==keyptCoord.shape[2]+1)  # eg.(15, 16/64, 3)
    scaled = scaled[:, :, :2]
    assert (np.all(np.abs(np.multiply(keyptCoord, SCALE_VECTOR) - scaled) <= 1))
    return scaled.astype(np.int32)


def swap(kpt_in1, kpt_in2, f_in):
    '''
    Perform in-place swap between two symmetric keypoints that was detected to be flipped
    :param kpt_in1: index of the first keypoint
    :param kpt_in2: index of the second keypoint
    :param f_in:    frame ID that both keypoints belong to
    :return:        changes are made to global variables, hence nothing is returned
    '''
    global _kyptConf, _xptCoord, _yptCoord
    # swap symmetric keypoints that are flipped. Make sure they have different xcoord positions
    if _xptCoord[kpt_in1][f_in] != _xptCoord[kpt_in2][f_in]:
        log('--swap: scanid:{}, frame:{:>2}, {:>3}({:>3}) <-> ({:>3}){:<3}'.
            format(_scanid, f_in, COLLECTED_KPTS[kpt_in1], _xptCoord[kpt_in1][f_in],
                   _xptCoord[kpt_in2][f_in], COLLECTED_KPTS[kpt_in2]))
        ctemp, xtemp, ytemp = \
            _kyptConf[kpt_in1][f_in], _xptCoord[kpt_in1][f_in], _yptCoord[kpt_in1][f_in]
        _kyptConf[kpt_in1][f_in] = _kyptConf[kpt_in2][f_in]
        _xptCoord[kpt_in1][f_in] = _xptCoord[kpt_in2][f_in]
        _yptCoord[kpt_in1][f_in] = _yptCoord[kpt_in2][f_in]
        _kyptConf[kpt_in2][f_in] = ctemp
        _xptCoord[kpt_in2][f_in], _yptCoord[kpt_in2][f_in] = xtemp, ytemp


def swap_flipped_symmetric_kypts(x_coord, fid):
    '''
    Discover symmetric keypoints that are horizontally flipped
        (due to inaccuracy of hpe) and swap them
    :param x_coord: x coordinates of keypoints to inspect
    :param fid: frame ID to inspect
    :return: child function is called if flip is detected, hence nothing is returned
    '''
    # shape of params: (15, 17)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RShoulder']][fid] < x_coord[SYM_BP_KPTS['LShoulder']][fid]):
        swap(SYM_BP_KPTS['RShoulder'], SYM_BP_KPTS['LShoulder'], fid)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RElbow']][fid] < x_coord[SYM_BP_KPTS['LElbow']][fid]):
        swap(SYM_BP_KPTS['RElbow'], SYM_BP_KPTS['LElbow'], fid)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RWrist']][fid] < x_coord[SYM_BP_KPTS['LWrist']][fid]):
        swap(SYM_BP_KPTS['RWrist'], SYM_BP_KPTS['LWrist'], fid)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RHip']][fid] < x_coord[SYM_BP_KPTS['LHip']][fid]):
        swap(SYM_BP_KPTS['RHip'], SYM_BP_KPTS['LHip'], fid)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RKnee']][fid] < x_coord[SYM_BP_KPTS['LKnee']][fid]):
        swap(SYM_BP_KPTS['RKnee'], SYM_BP_KPTS['LKnee'], fid)
    if ORIENTATION[fid] is not None and ORIENTATION[fid] != \
            (x_coord[SYM_BP_KPTS['RAnkle']][fid] < x_coord[SYM_BP_KPTS['LAnkle']][fid]):
        swap(SYM_BP_KPTS['RAnkle'], SYM_BP_KPTS['LAnkle'], fid)


def hpe_keypoints_of_interests(rdir, sdir, channels, display=False):
    '''
    Performs foreground extraction and pass images to hpe to estimate keypoints of interest
    :param rdir:    dataset home directory. ie train, eval, or val set
    :param sdir:    scan directory name
    :param channels:cropped images' channels
    :param display: boolean indicating whether to visualize results of this function
    :return:        ndarrays of keypoint coordinates and confidence
    '''
    global _fid, _scan_images
    _scan_images = np.zeros(shape=(N_FRAMES, FRM_HGT, FRM_WDT, channels), dtype=np.uint8)
    scanSegmFrms = np.zeros(shape=(N_FRAMES, IMG_HGT, IMG_WDT, channels), dtype=np.uint8)

    for _fid in range(N_FRAMES):
        isSide = True if _fid is 4 or 12 else False
        vSplit = True if _fid in [2, 3, 5, 6, 10, 11, 13, 14] else False
        filename = str(_fid) + ".png"
        path = os.path.join(rdir, sdir, filename)
        orgframe = imgtf.read_image(path, channels, 'BGR')
        segframe = imgtf.separate_foreground(orgframe, IMG_WDT, IMG_HGT,
                                             side=isSide, vsplit=vSplit, display=False)
        _scan_images[_fid] = orgframe
        scanSegmFrms[_fid] = segframe

    # COCO model
    output = hpe.feed_to_pose_nn(scanSegmFrms)  # shape: (16/64, 57, 42, 32)
    assert (output.shape == (N_FRAMES, 57, 42, 32))
    # for 15 relevant keypoints and 16/64 frames. shape of each array below is (15, 16/64)
    xptCoord, yptCoord, kyptConf, faceKypts = \
        hpe.interested_keypoints_and_confidence(output, IMG_WDT, IMG_HGT)
    if display: show_stict_figure('hpe keypoints', scanSegmFrms, xptCoord, yptCoord, kyptConf)
    return xptCoord, yptCoord, kyptConf, faceKypts


def touchup_and_scale_hpekpts(scan_id, refine_x=False, refine_y=True, ankle_y=10,
                              get_face_mask=True, scale_back=True, display=False):
    global _scanid
    _scanid = scan_id
    # swap symmetric keypoints if flipped
    for fin in range(N_FRAMES):
        swap_flipped_symmetric_kypts(_xptCoord, fin)
    # retain changes to keypoint confidence after swapping symmetric keypoints
    kyptConf = np.around(_kyptConf, N_DECIMALS)
    # kptCoord shape: (15, 16/64, 2), where 2 -> (x, y)
    if REFINE_MODE=='v1':
        kptCoord = refine_keypoint_coord_v1(refine_x, refine_y, ankle_y)
    elif REFINE_MODE=='v2':
        kptCoord = refine_keypoint_coord_v2(refine_x, refine_y, ankle_y)
    else: kptCoord = refine_keypoint_coord_v3(refine_x, refine_y, ankle_y)
    assert (np.all(kptCoord >= 0) and
            np.all(kptCoord[:, :, 0] <= IMG_WDT) and
            np.all(kptCoord[:, :, 1] <= IMG_HGT))

    if scale_back:
        # kptCoord shape: (15, 16/64, 2), where 2 -> (x, y)
        kptCoord = scale_back_keypoint_coord(kptCoord)
        assert (np.all(kptCoord >= 0) and
                np.all(kptCoord[:, :, 0] <= FRM_WDT) and
                np.all(kptCoord[:, :, 1] <= FRM_HGT))

    faceMasks = None
    if get_face_mask:
        faceMasks = facemask(scale_back_keypoint_coord(_faceKypts[:, :, :2]),
                             _faceKypts[:, :, 2], kptCoord[0, 0, 1])
    if display:
        show_stict_figure('refined keypoints', _scan_images,
                          kptCoord[:, :, 0], kptCoord[:, :, 1], _kyptConf)
    return kptCoord, kyptConf, faceMasks


def set_global_variables(xptCoord, yptCoord, kyptConf, faceKypts):
    global _xptCoord, _yptCoord, _kyptConf, _faceKypts
    _xptCoord, _yptCoord, _kyptConf, _faceKypts = xptCoord, yptCoord, kyptConf, faceKypts
    assert (np.all(0 <= _xptCoord))
    assert (np.all(0 <= _yptCoord))
    assert (np.all(0 <= _kyptConf))
    assert (np.all(_xptCoord <= IMG_WDT))
    assert (np.all(_yptCoord <= IMG_HGT))
    assert (np.all(_kyptConf <= 1))


def refine_estimated_keypoints(prefix, refine_x=False, refine_y=True, ankle_y=10):
    global _aggOutlierBookKeep
    print('\n', prefix)
    kpt_df = pd.read_csv(os.path.join(hpeDir, '{}.csv'.format(prefix)))
    tag1 = ("%.1f"%CONF_THRESHOLD).replace('0.','.') if REFINE_MODE in ['v1','v2'] else '.x'
    ref_tag = 'ref{}{}{}{}{}'.format(tag1, int(refine_x), int(refine_y), ankle_y, REFINE_MODE)
    writecsv = os.path.join(hpeDir, '{}-{}.csv'.format(prefix, ref_tag))
    log('\n\n{}\n'.format(ref_tag))
    xptCoord = np.zeros(shape=(COMBO_KPTS, N_FRAMES), dtype=np.int32)
    yptCoord = np.zeros(shape=(COMBO_KPTS, N_FRAMES), dtype=np.int32)
    probConf = np.zeros(shape=(COMBO_KPTS, N_FRAMES), dtype=np.float32)
    if REFINE_MODE=='v3':
        _aggOutlierBookKeep = np.zeros(shape=(N_KPTS, 3), dtype=np.float32)

    # Compute distance
    n_scans = kpt_df.shape[0]
    for (index, row) in kpt_df.iterrows():
        scanid = row['scanID']
        #print('{:>6}. {}'.format(index+1, scanid))
        # read original hpe
        for fid in range(N_FRAMES):
            columnName = 'Frame{}'.format(fid)
            predKptDict = ast.literal_eval(row[columnName])

            # collect estimated keypoints
            for kpt_idx, kpt in ESTIMATED_KPTS.items():
                predKptMeta = predKptDict[kpt]
                xptCoord[kpt_idx, fid] = max(0, predKptMeta[0])
                yptCoord[kpt_idx, fid] = max(0, predKptMeta[1])
                probConf[kpt_idx, fid] = predKptMeta[2]

            # derive some other keypoints from estimated keypoints
            for kpt_idx, agg_kpt_idxs in EXTRA_KPTS_AGG.items():
                xptCoord[kpt_idx, fid] = np.mean(xptCoord[agg_kpt_idxs, fid])
                yptCoord[kpt_idx, fid] = np.mean(yptCoord[agg_kpt_idxs, fid])
                probConf[kpt_idx, fid] = np.mean(probConf[agg_kpt_idxs, fid])

        if np.max(probConf)>1: print('\tconfidence {} > 1'.format(np.max(probConf)))
        probConf = np.clip(probConf, 0, 1)
        set_global_variables(xptCoord[:15,:], yptCoord[:15,:], probConf[:15,:], None)

        kptCoord, kyptConf, _ = \
            touchup_and_scale_hpekpts(scanid, refine_x=refine_x, refine_y=refine_y, # ankle_y was 15
                                      ankle_y=ankle_y, get_face_mask=False, scale_back=False)

        # log refined hpe keypoints collection to dataframe
        for fid in range(N_FRAMES):
            columnName = 'Frame{}'.format(fid)
            refKptDict = dict()
            for kid, kpt in COLLECTED_KPTS.items():
                refKptDict[kpt] = (kptCoord[kid,fid,0], kptCoord[kid,fid,1], kyptConf[kid,fid])
            kpt_df.at[index, columnName] = str(refKptDict) # faster than loc

        if (index+1)%100==0 or (index+1)==n_scans:
            print('\n{:>4}/{} SCANS PASSED..'.format(index+1, n_scans))

        # reset vaiables
        xptCoord *= 0
        yptCoord *= 0
        probConf *= 0

    kpt_df.to_csv(writecsv, encoding='utf-8', index=False)
    if REFINE_MODE=='v3':
        _aggOutlierBookKeep /= n_scans  # aggregate computed as average
        log('\n All scan aggregate of {}:\n\tn_inliers:{}\n\tn_outliers:{}\n\toutlier_thresh:{}'.
            format(ref_tag, np.around(_aggOutlierBookKeep[:,0], 1),
                   np.around(_aggOutlierBookKeep[:,1], 1), np.around(_aggOutlierBookKeep[:,2], 2)))


def initialize_global_variables(n_frames=16, n_kpts=15, conf_thresh=0.1,
                                img_shape=(256, 336), n_dec=4, keep_log=False):
    global hpe, SCALE_VECTOR, FRM_WDT, FRM_HGT, IMG_WDT, IMG_HGT, SCALE_TRANS_MATRIX, \
        ROOT_DIR, CONF_THRESHOLD, REFINE_X, REFINE_Y, LOG_INFO, Y_REF_SYM_PAIR, \
        SYM_BP_KPTS, ORIENTATION, N_FRAMES, NUM_OF_ZONES, N_DECIMALS, \
        N_KPTS, COMBO_KPTS, EXTRA_KPTS_AGG, ESTIMATED_KPTS, COLLECTED_KPTS

    LOG_INFO = keep_log
    FRM_WDT, FRM_HGT = 512, 660 # Dimension of images on disk. important verify before running***
    IMG_WDT, IMG_HGT = img_shape
    REFINE_X, REFINE_Y = True, True
    SCALE_VECTOR = np.array([FRM_WDT/IMG_WDT, FRM_HGT/IMG_HGT])
    SCALE_TRANS_MATRIX = np.array([[FRM_WDT/IMG_WDT, 0, FRM_WDT/(2*IMG_WDT)],
                                   [0, FRM_HGT/IMG_HGT, FRM_HGT/(2*IMG_HGT)],
                                   [0, 0, 1]])
    N_FRAMES = n_frames
    NUM_OF_ZONES = 17
    N_KPTS = n_kpts # this is the subset of coco-hpe kpts and derived kpts
    COMBO_KPTS = 20  # this includes both relevant coco-hpe kpts and derived kpts
    EXTRA_KPTS_AGG = {0:[1, 4], 13:[7, 10], 14:[15, 16, 17, 18, 19]}  # 0:Neck, 13:Mid-Hip, 14:Head
    CONF_THRESHOLD = conf_thresh
    N_DECIMALS = n_dec

    if N_FRAMES == 16:
        # Orientation of body parts: xRight < xLeft
        ORIENTATION = [True,  True,  True,  True,  None, False, False, False,
                       False, False, False, False, None, True,  True,  True]
        FRAME_VALID_KYPTS = znkp.valid_kypts_per_16frames()  # ***CHECK
        ROOT_DIR = '../../../datasets/tsa/aps_images/dataset/'
    else:
        assert (N_FRAMES == 64)
        ORIENTATION = [True,  True,  True,  True,  True,  True,  True,  True,
                       True,  True,  True,  True,  True,  True,  True,  True,
                       None,  False, False, False, False, False, False, False,
                       False, False, False, False, False, False, False, False,
                       False, False, False, False, False, False, False, False,
                       False, False, False, False, False, False, False, False,
                       None,  True,  True,  True,  True,  True,  True,  True,
                       True,  True,  True,  True,  True,  True,  True,  True]
        FRAME_VALID_KYPTS = znkp.valid_kypts_per_64frames()
        ROOT_DIR = '../../../datasets/tsa/a3daps_images/dataset/'

    hpe = coco_hpe.CocoHPE(FRAME_VALID_KYPTS, framesPerScan=N_FRAMES)
    SYM_BP_KPTS = hpe.SYM_BODY_PART_KPTS
    Y_REF_SYM_PAIR = {1:(1, 4), 4:(1, 4), 7:(7, 10), 10:(7, 10),
                      8:(8, 11), 11:(8, 11), 9:(9, 12), 12:(9, 12)}
    ESTIMATED_KPTS = {1:'RSh', 2:'REb',  3:'RWr',  4:'LSh',  5:'LEb',  6:'LWr',
                      7:'RHp', 8:'RKe',  9:'RAk',  10:'LHp', 11:'LKe', 12:'LAk',
                      15:'Ns', 16:'REy', 17:'LEy', 18:'REr', 19:'LEr'}
    COLLECTED_KPTS = {1:'RSh', 2:'REb',  3:'RWr',  4:'LSh',  5:'LEb',  6:'LWr',
                      7:'RHp', 8:'RKe',  9:'RAk',  10:'LHp', 11:'LKe', 12:'LAk',
                      0:'Nk',  13:'MHp', 14:'Hd'}


if __name__ == "__main__":
    global _logfile

    # Setup and configurations
    subset = 'all_sets'  #***
    ds_ext = 'a3daps' #***
    thresh = 0.3 #+++
    REFINE_MODE = 'v3'
    net_configs = ['w32_256x192-rgb_wfp']
    # ['w48_256x192', 'res50_256x192', 'res101_256x192', 'res152_256x192']
    # N_SCANS = 2635 #1147

    n_frames = 16 if ds_ext=='aps' else 64
    initialize_global_variables(n_frames=n_frames, conf_thresh=thresh, img_shape=(512, 660))
    hpeDir = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/'.format(ds_ext)
    _logfile = {1:os.path.join(hpeDir, 'log-{}-swap.log'.format(subset)),
                2:os.path.join(hpeDir, 'log-{}-opt.log'.format(subset))}

    for config in net_configs:
        refine_estimated_keypoints('{}-{}'.format(subset, config))
        refine_estimated_keypoints('{}-{}-opt'.format(subset, config))



"""
Notes
 1. All scan aggregate of ref.x0110v3:
                         Nk   RSh  REb  RWr  LSh  LEb  LWr  RHp  RKe  RAk  LHp  LKe  LAk  MHp  Hd
        n_inliers:     [61.7 61.4 60.2 61.7 61.4 60.3 61.8 62.3 63.1 63.6 62.4 63.1 63.5 62.3 62.6] 
        n_outliers:    [ 2.3  2.6  3.7  2.3  2.6  3.7  2.2  1.6  0.8  0.4  1.6  0.9  0.5  1.7  1.3]
        outlier_thresh:[0.30 0.29 0.30 0.27 0.29 0.30 0.27 0.27 0.24 0.19 0.27 0.24 0.20 0.28 0.33]
"""