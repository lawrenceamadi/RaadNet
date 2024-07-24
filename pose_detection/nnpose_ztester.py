'''
    Test different sub routines from foreground extraction and pose detection to pose reconstruction
'''

import pandas as pd
import numpy as np
import cv2 as cv
import time
import sys
import ast
import os
from scipy.spatial import distance

sys.path.append('../')
from pose_detection import nnpose_zoning as npoz
from pose_detection import coco_kpts_zoning as znkp
from image_processing import transformations as imgtf


def inspect(imgDir, npyDir, samples=27):
    dfw = pd.read_csv(DUMMY_CSV)
    # 'cocoGeneratedKpts_A.npy' 'cocoGenKptsRefined_A.npy'
    readNpy = os.path.join(npyDir, 'cocoGenKptsRefinReconCorr_B.npy')
    scanKeypoints = np.load(readNpy)  # shape=(NUM_OF_SCANS, ALL_KPTS, FRAMES_PER_SCAN, 3)

    for index, row in dfw.iterrows():
        if index < samples:
            scanid = row['scanID']
            print('{:>4}. {}'.format(index, scanid))
            scan_images = np.zeros(shape=(FRAMES_PER_SCAN, 660, 512, 3), dtype=np.uint8)

            for fid in range(FRAMES_PER_SCAN):
                filename = str(fid) + ".png"
                path = os.path.join(imgDir, scanid, filename)
                scan_images[fid] = imgtf.read_image(path, 3, 'BGR', size=None)#, cropout=(npoz.OFF_X, npoz.OFF_Y))
                #scan_images[fid] = cv.resize(orgframe, (npoz.IMG_WDT, npoz.IMG_HGT), interpolation=cv.INTER_AREA)

            xCoord = scanKeypoints[index, :, :, 0] + npoz.OFF_X# 1: 14
            yCoord = scanKeypoints[index, :, :, 1] + npoz.OFF_Y
            kptConf = scanKeypoints[index, :, :, 2]

            npoz.show_stict_figure(scan_images, xCoord, yCoord, kptConf, invEdge=False, invKpts=True)


def test_hpe_keypoints(imgDir, csvDir, npyDir, samples=27):
    writeNpy = os.path.join(npyDir, 'cocoGeneratedKpts_A.npy')
    writeCsv = os.path.join(csvDir, 'cocoGenKptsScaled_B.csv')
    dfw = pd.read_csv(DUMMY_CSV)
    scanKeypoints = np.zeros(shape=(NUM_OF_SCANS, ALL_KPTS, FRAMES_PER_SCAN, 3), dtype=np.float32)
    start_time = time.time()

    for index, row in dfw.iterrows():
        if index < samples:
            scanid = row['scanID']
            scanNum = index + 1
            xptCoord, yptCoord, kyptConf, faceKypt = npoz.hpe_keypoints_of_interests(imgDir, scanid, channels=3)
            kptCoord = np.zeros(shape=(npoz.hpe.COCO_KEYPOINTS_INT, FRAMES_PER_SCAN, 2), dtype=xptCoord.dtype)
            kptCoord[:, :, 0] = xptCoord
            kptCoord[:, :, 1] = yptCoord
            kptCoord = npoz.scale_back_keypoint_coord(kptCoord)
            fceCoord = npoz.scale_back_keypoint_coord(faceKypt[:, :, :2])

            for fid in range(FRAMES_PER_SCAN):
                columnName = 'Frame{}'.format(fid)
                kptDict = {}
                for kid in range(ALL_KPTS):
                    if kid in faceKptIndicies:
                        idx = faceIndexMap[kid]
                        xPtA, yPtA, ptConfA = faceKypt[idx][fid][0], faceKypt[idx][fid][1], faceKypt[idx][fid][2]
                        xPtB, yPtB, ptConfB = fceCoord[idx][fid][0], fceCoord[idx][fid][1], faceKypt[idx][fid][2]
                    else:
                        idx = kid - 1
                        assert (0 <= idx <= 12)
                        xPtA, yPtA, ptConfA = xptCoord[idx][fid], yptCoord[idx][fid], kyptConf[idx][fid]
                        xPtB, yPtB, ptConfB = kptCoord[idx][fid][0], kptCoord[idx][fid][1], kyptConf[idx][fid]

                    kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid]
                    kptDict[kpt] = (xPtB + npoz.OFF_X, yPtB + npoz.OFF_Y, round(ptConfB, 3))
                    scanKeypoints[index, kid, fid] = [xPtA, yPtA, ptConfA]
                dfw.loc[dfw['scanID'] == scanid, columnName] = str(kptDict)
            dfw.loc[dfw['scanID'] == scanid, 'Status'] = 'set'

            if scanNum % 20 == 0:
                runtime = time.time() - start_time
                print('{:>4} SCANS PASSED..\t{:<6} seconds per scan'.format(scanNum, round(runtime / scanNum, 3)))
                dfw.to_csv(writeCsv, encoding='utf-8', index=False)
                np.save(writeNpy, scanKeypoints)

    runtime = time.time() - start_time
    print("\nRuntime: {} minutes\nAvg runtime: {} seconds per scan".format(
                        round(runtime / 60, 3), round(runtime / samples, 3)))
    dfw.to_csv(writeCsv, encoding='utf-8', index=False)
    np.save(writeNpy, scanKeypoints)


def test_refined_kpts(csvDir, npyDir, samples=27):
    readNpy = os.path.join(npyDir, 'cocoGeneratedKpts_A.npy')
    writeNpy1 = os.path.join(npyDir, 'ccocoGenKptsRefined_A.npy')
    writeNpy2 = os.path.join(npyDir, 'cocoScanFaceMasks_A.npy')
    writeCsv = os.path.join(csvDir, 'cocoGenKptsRefined_B.csv')
    dfw = pd.read_csv(DUMMY_CSV)
    scanKeypoints = np.load(readNpy) # shape=(NUM_OF_SCANS, ALL_KPTS, FRAMES_PER_SCAN, 3)
    refScanKeypts = np.zeros(shape=(NUM_OF_SCANS, 13, FRAMES_PER_SCAN, 3), dtype=np.float32)
    scanFaceMasks = np.ones(shape=(NUM_OF_SCANS, FRAMES_PER_SCAN, npoz.FRM_HGT, npoz.FRM_WDT), dtype=np.uint8)
    start_time = time.time()

    for index, row in dfw.iterrows():
        if index < samples:
            scanid = row['scanID']
            scanNum = index + 1

            xptCoord = scanKeypoints[index, 1: 14, :, 0]
            yptCoord = scanKeypoints[index, 1: 14, :, 1]
            kyptConf = scanKeypoints[index, 1: 14, :, 2]
            faceKypt = scanKeypoints[index, faceKptIndicies]

            npoz.set_global_variables(xptCoord, yptCoord, kyptConf, faceKypt)
            kptCoord, scanFaceMasks[index] = npoz.touchup_and_scale_hpekpts(scanid)

            for fid in range(FRAMES_PER_SCAN):
                columnName = 'Frame{}'.format(fid)
                kptDict = {}
                for idx in range(13):
                    xPt, yPt, ptConf = kptCoord[idx][fid][0], kptCoord[idx][fid][1], kyptConf[idx][fid]
                    kpt = npoz.hpe.INDEX_TO_KPT_LABEL[idx + 1]
                    kptDict[kpt] = (xPt + npoz.OFF_X, yPt + npoz.OFF_Y, round(ptConf, 3))
                    refScanKeypts[index, idx, fid] = [xPt, yPt, ptConf]
                dfw.loc[dfw['scanID'] == scanid, columnName] = str(kptDict)
            dfw.loc[dfw['scanID'] == scanid, 'Status'] = 'set'

            if scanNum % 100 == 0:
                runtime = time.time() - start_time
                print('{:>4} SCANS PASSED..\t{:<6} seconds per scan'.format(scanNum, round(runtime / scanNum, 3)))
                dfw.to_csv(writeCsv, encoding='utf-8', index=False)
                np.save(writeNpy1, refScanKeypts)
                np.save(writeNpy2, scanFaceMasks)

    runtime = time.time() - start_time
    print("\nRuntime: {} minutes\nAvg runtime: {} seconds per scan".format(
        round(runtime / 60, 3), round(runtime / samples, 3)))
    dfw.to_csv(writeCsv, encoding='utf-8', index=False)
    np.save(writeNpy1, refScanKeypts)
    np.save(writeNpy2, scanFaceMasks)


def test_recontruction(imgDir, csvDir, npyDir, type, samples=27):
    readNpy1 = os.path.join(npyDir, 'ccocoGenKptsRefined_A.npy')
    readNpy2 = os.path.join(npyDir, 'cocoScanFaceMasks_A.npy')
    writeNpy = os.path.join(npyDir, 'cocoGenKptsRefinRecon' + type + '_B.npy')
    writeCsv = os.path.join(csvDir, 'cocoGenKptsRefinRecon' + type + '_B.csv')
    dfw = pd.read_csv(DUMMY_CSV)
    refScanKeypts = np.load(readNpy1) # shape=(NUM_OF_SCANS, 13, FRAMES_PER_SCAN, 3)
    scanFaceMasks = np.load(readNpy2) # shape=(NUM_OF_SCANS, FRAMES_PER_SCAN, FRM_HGT, FRM_WDT)
    reconScanKpts = np.zeros(shape=(NUM_OF_SCANS, 13, FRAMES_PER_SCAN, 3), dtype=np.float32)
    start_time = time.time()

    for index, row in dfw.iterrows():
        if index < samples:
            scanid = row['scanID']
            scanNum = index + 1
            scan_images = np.zeros(shape=(FRAMES_PER_SCAN, npoz.FRM_HGT, npoz.FRM_WDT, 3), dtype=np.uint8)
            for fid in range(FRAMES_PER_SCAN):
                filename = str(fid) + ".png"
                path = os.path.join(imgDir, scanid, filename)
                scan_images[fid] = imgtf.read_image(path, 3, 'BGR', size=None, cropout=(npoz.OFF_X, npoz.OFF_Y))

            faceMasks = scanFaceMasks[index]
            xptCoord = refScanKeypts[index, :, :, 0].astype(np.int32)
            yptCoord = refScanKeypts[index, :, :, 1].astype(np.int32)
            kyptConf = refScanKeypts[index, :, :, 2]

            # reconstruct optimal keypoints
            kptCoord, reconConf = npoz.kmt.recon_from_hpe_kpts(scan_images, xptCoord, yptCoord, kyptConf, faceMasks,
                                                      recInv=False, displayPath=False)
            for fid in range(FRAMES_PER_SCAN):
                columnName = 'Frame{}'.format(fid)
                kptDict = {}
                for idx in range(13):
                    xPt, yPt, ptConf = kptCoord[idx][fid][0], kptCoord[idx][fid][1], reconConf[idx][fid]
                    kpt = npoz.hpe.INDEX_TO_KPT_LABEL[idx + 1]
                    kptDict[kpt] = (xPt + npoz.OFF_X, yPt + npoz.OFF_Y, round(ptConf, 3))
                    reconScanKpts[index, idx, fid] = [xPt, yPt, ptConf]
                dfw.loc[dfw['scanID'] == scanid, columnName] = str(kptDict)
            dfw.loc[dfw['scanID'] == scanid, 'Status'] = 'set'

            if scanNum % 1 == 0:  # 5, 50
                runtime = time.time() - start_time
                print('{:>4} SCANS PASSED..\t{:<6} seconds per scan'.format(scanNum, round(runtime / scanNum, 3)))
                dfw.to_csv(writeCsv, encoding='utf-8', index=False)
                np.save(writeNpy, reconScanKpts)

    runtime = time.time() - start_time
    print("\nRuntime: {} minutes\nAvg runtime: {} seconds per scan".format(
                        round(runtime / 60, 3), round(runtime / samples, 3)))
    dfw.to_csv(writeCsv, encoding='utf-8', index=False)
    np.save(writeNpy, reconScanKpts)


#---------------------------------------------------------------------------------------------------

def kpt_distance_anaylsis(poseCsvPaths):
    # Note: cocoKptsMarkings is done at 512x660
    '''df0 = pd.read_csv(os.path.join(csvDir, 'cocoKptsMarkings.csv'))
    df1 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsScaled_B.csv'))
    df2 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsRefined_B.csv'))
    df3 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsRefinReconCorr_B.csv'))
    df4 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsRefinReconSift_B.csv'))
    df5 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsRefinReconSurf_B.csv'))
    df6 = pd.read_csv(os.path.join(csvDir, 'cocoGenKptsRefinReconOrb_B.csv'))
    df7 = pd.read_csv(os.path.join(hpeDir, 'train_set_deepHRNet.csv'))
    df8 = pd.read_csv(os.path.join(hpeDir, 'train_set_deepHRNet_ref.csv'))
    df9 = pd.read_csv(os.path.join(hpeDir, 'train_set_deepHRNet_opt.csv'))
    df10 = pd.read_csv(os.path.join(hpeDir, 'train_set_deepHRNet_opt_ref.csv'))
    dfs = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]'''

    dfs = list()
    dfLabels = list()
    for (dir, csv_file) in poseCsvPaths:
        df = pd.read_csv(os.path.join(dir, csv_file))
        dfs.append(df)
        s_idx = min(0, csv_file.find('-') + 1)
        label = csv_file[s_idx: csv_file.find('.csv')]
        dfLabels.append(label)
    df0 = dfs[0]

    nkpts = 12
    status, group = ['complete'], ['Train']
    df0 = df0[df0.Status.isin(status) & df0.Group.isin(group)]
    df0.reset_index(drop=True, inplace=True)
    samples = df0.shape[0]
    kptDist = np.zeros(shape=(samples, len(dfs), nkpts, FRAMES_PER_SCAN, 3), dtype=np.int32)
    # Compute distance
    for did, df in enumerate(dfs):
        for (index, row) in df0.iterrows():
            scanid = row['scanID']
            for fid in range(FRAMES_PER_SCAN):
                columnName = 'Frame{}'.format(fid)
                trueKptDict = ast.literal_eval(row[columnName])
                r_df = df.loc[df['scanID'] == scanid, columnName]
                r_df.reset_index(drop=True, inplace=True)
                predKptDict = ast.literal_eval(r_df[0])
                for kid in range(nkpts):
                    kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid + 2] # Skip Nk
                    truePt = trueKptDict.get(kpt, None)
                    if truePt is not None:
                        assert (len(truePt) == 2)
                        predPt = np.array([predKptDict[kpt][0], predKptDict[kpt][1]])
                        if did == 7: predPt += [4, -4]
                        xDiff = abs(truePt[0] - predPt[0])
                        yDiff = abs(truePt[1] - predPt[1])
                        dist = np.sqrt(xDiff**2 + yDiff**2)
                        pDist = distance.euclidean(truePt, predPt)
                        assert (dist == pDist)
                        kptDist[index, did, kid, fid] = [xDiff, yDiff, pDist]

            if (index + 1) % 25 == 0:  # 50
                print('{:>4} SCANS PASSED..'.format(index + 1))

    #compute_error_old(kptDist, dfs, nkpts, samples)
    compute_error_new(kptDist, dfs, nkpts, samples, dfLabels)


def compute_error_old(kptDist, dfs, nkpts, samples):
    FRAME_VALID_KYPTS = znkp.valid_keypt_per_16frames() #***CHECK
    #np.save(os.path.join(npyDir, 'true-vs-cocoGen.npy'), kptDist)
    avgKptDist = np.mean(kptDist, axis=0, dtype=np.float64) # (6, 13, 16, 3)
    dftypes = ['TrueKpt', 'CocoGen', 'Refined', 'CorrRecon', 'SiftRecon', 'SurfRecon',
               'OrbRecon', 'HRNet', 'HRNetSwap', 'HRNetOpt', 'HRNetOptSwap']
    columns = [#'Comparison', 'Nk', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
               'Comparison', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
               'RAk', 'LHp', 'LKe', 'LAk', 'Avg']
    df_x = pd.DataFrame(columns=columns)
    df_y = pd.DataFrame(columns=columns)
    df_d = pd.DataFrame(columns=columns)

    for i in range(len(dfs)):
        df_x.at[i, 'Comparison'] = dftypes[i]
        df_y.at[i, 'Comparison'] = dftypes[i]
        df_d.at[i, 'Comparison'] = dftypes[i]
        print('{}. DataFrame: {}'.format(i + 1, dftypes[i]))
        xAvgSum, yAvgSum, dAvgSum = 0, 0, 0
        for k in range(nkpts):
            xSum = np.sum(avgKptDist[i, k, :, 0])
            ySum = np.sum(avgKptDist[i, k, :, 1])
            dSum = np.sum(avgKptDist[i, k, :, 2])
            kid = npoz.hpe.ALLiNDX_TO_INTiNDX[k + 2]  # ***CHECK
            num = np.sum(FRAME_VALID_KYPTS[kid].astype(np.int32))
            xAvg, yAvg, dAvg = xSum / num, ySum / num, dSum / num
            xAvgSum += xAvg
            yAvgSum += yAvg
            dAvgSum += dAvg
            kpt = npoz.hpe.INDEX_TO_KPT_LABEL[k + 2] #***
            df_x.at[i, kpt] = round(xAvg, 2)
            df_y.at[i, kpt] = round(yAvg, 2)
            df_d.at[i, kpt] = round(dAvg, 2)
        df_x.at[i, 'Avg'] = round(xAvgSum / nkpts, 2)
        df_y.at[i, 'Avg'] = round(yAvgSum / nkpts, 2)
        df_d.at[i, 'Avg'] = round(dAvgSum / nkpts, 2)

    writeCsv1 = os.path.join(csvDir, str(samples) + '_old_distance_xAxis.csv')
    writeCsv2 = os.path.join(csvDir, str(samples) + '_old_distance_yAxis.csv')
    writeCsv3 = os.path.join(csvDir, str(samples) + '_old_distance_euclidean.csv')
    df_x.to_csv(writeCsv1, encoding='utf-8', index=False)
    df_y.to_csv(writeCsv2, encoding='utf-8', index=False)
    df_d.to_csv(writeCsv3, encoding='utf-8', index=False)


def compute_error_new(kptDist, dfs, nkpts, samples, dfLabels):
    FRAME_VALID_KYPTS = znkp.valid_keypt_per_16frames() #***CHECK
    #np.save(os.path.join(npyDir, 'true-vs-cocoGen.npy'), kptDist)
    #avgKptDist = np.mean(kptDist, axis=0, dtype=np.float64) # (6, 13, 16, 3)
    ###dfLabels = ['TrueKpt', 'CocoGen', 'Refined', 'CorrRecon', 'SiftRecon', 'SurfRecon',
    ###           'OrbRecon', 'HRNet', 'HRNetSwap', 'HRNetOpt', 'HRNetOptSwap']
    columns = [#'Comparison', 'Nk', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
               'Comparison', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
               'RAk', 'LHp', 'LKe', 'LAk', 'Avg']
    df_x = pd.DataFrame(columns=columns)
    df_y = pd.DataFrame(columns=columns)
    df_d = pd.DataFrame(columns=columns)

    # kptDist.shape = (samples, len(dfs), nkpts, 16, 3)

    for i in range(len(dfs)):
        df_x.at[i, 'Comparison'] = dfLabels[i]
        df_y.at[i, 'Comparison'] = dfLabels[i]
        df_d.at[i, 'Comparison'] = dfLabels[i]
        print('{}. DataFrame: {}'.format(i + 1, dfLabels[i]))
        xAvgSum, yAvgSum, dAvgSum = 0, 0, 0
        for k in range(nkpts):
            xSum = np.sum(kptDist[:, i, k, :, 0])
            ySum = np.sum(kptDist[:, i, k, :, 1])
            dSum = np.sum(kptDist[:, i, k, :, 2])
            kid = npoz.hpe.ALLiNDX_TO_INTiNDX[k + 2] #***CHECK
            num = np.sum(FRAME_VALID_KYPTS[kid].astype(np.int32)) # todo: check if kpt index is respected
            cnt = samples * num
            xAvg, yAvg, dAvg = xSum / cnt, ySum / cnt, dSum / cnt
            xAvgSum += xAvg
            yAvgSum += yAvg
            dAvgSum += dAvg
            kpt = npoz.hpe.INDEX_TO_KPT_LABEL[k + 2] #***
            df_x.at[i, kpt] = round(xAvg, 2)
            df_y.at[i, kpt] = round(yAvg, 2)
            df_d.at[i, kpt] = round(dAvg, 2)
        df_x.at[i, 'Avg'] = round(xAvgSum / nkpts, 2)
        df_y.at[i, 'Avg'] = round(yAvgSum / nkpts, 2)
        df_d.at[i, 'Avg'] = round(dAvgSum / nkpts, 2)

    writeCsv1 = os.path.join(csvDir, str(samples) + '_new_distance_xAxis.csv')
    writeCsv2 = os.path.join(csvDir, str(samples) + '_new_distance_yAxis.csv')
    writeCsv3 = os.path.join(csvDir, str(samples) + '_new_distance_euclidean.csv')
    df_x.to_csv(writeCsv1, encoding='utf-8', index=False)
    df_y.to_csv(writeCsv2, encoding='utf-8', index=False)
    df_d.to_csv(writeCsv3, encoding='utf-8', index=False)


def visualize_csv_kpts_v1(csvDir):
    # Note: cocoKptsMarkings is done at 512x660
    imgDir = '../../../datasets/tsa/aps_images/dataset/train/'
    df0 = pd.read_csv(os.path.join(csvDir, 'cocoKptsMarkings.csv'))
    df7 = pd.read_csv('../../../repos_official/deephrnet/output/deepHRNet.csv')
    dfs = [df7]

    # Compute distance
    for did, df in enumerate(dfs):
        for index, row in df0.iterrows():
            if row['Status'] == 'complete' and row['Group'] == 'Train':
                scanid = row['scanID']
                scanNum = index + 1
                for fid in range(FRAMES_PER_SCAN):
                    gt_xKpts = np.zeros((13), np.int32)
                    gt_yKpts = np.zeros((13), np.int32)
                    gt_kConf = np.ones((13), np.float32)
                    xKpts = np.zeros((13), np.int32)
                    yKpts = np.zeros((13), np.int32)
                    kConf = np.zeros((13), np.float32)

                    columnName = 'Frame{}'.format(fid)
                    trueKptDict = ast.literal_eval(row[columnName])
                    predKptDict = ast.literal_eval(df.loc[df['scanID'] == scanid,
                                                          columnName][index])
                    for kid in range(12):
                        kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid + 2]  # Skip Ns & Nk
                        xKpts[kid + 1], yKpts[kid + 1], kConf[kid + 1] = predKptDict[kpt]
                        if trueKptDict.get(kpt) is not None:
                            gt_xKpts[kid + 1], gt_yKpts[kid + 1] = trueKptDict[kpt]

                    xKpts[0] = min(xKpts[1], xKpts[4]) + abs(xKpts[1] - xKpts[4]) // 2
                    yKpts[0] = min(yKpts[1], yKpts[4]) + abs(yKpts[1] - yKpts[4]) // 2
                    kConf[0] = (kConf[1] + kConf[4]) / 2
                    gt_xKpts[0], gt_yKpts[0] = trueKptDict['Nk']

                    xKpts += 4 # ie. 8 - 4
                    yKpts -= 4
                    image = cv.imread(os.path.join(imgDir, scanid, '{}.png'.format(fid)))
                    stick_figure(image, fid, gt_xKpts, gt_yKpts, gt_kConf, 'gt', wait=1)
                    stick_figure(image, fid, xKpts, yKpts, kConf, 'pred', wait=0)

                if scanNum % 25 == 0:  # 50
                    print('{:>4} SCANS PASSED..'.format(scanNum))


def visualize_csv_kpts_v2(kpt_csv, subset, n_frames=16, delay=0,
                          combo=None, show_main=True, show_neig=False):
    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}.csv'
    aps_df = pd.read_csv(path_template.format('aps', kpt_csv))
    a3d_df = pd.read_csv(path_template.format('a3daps', kpt_csv))
    co_color = [(64, 64, 64), (255, 255, 255)]
    kpt_df = list()
    if combo is not None:
        for i, co_tag in enumerate(combo):
            path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}-{}.csv'
            kpt_df.append(pd.read_csv(path_template.format('aps', kpt_csv, co_tag)))
            if i == 0:
                path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}/{}-{}.csv'
                vel_df = pd.read_csv(path_template.format('aps', 'drift', kpt_csv, co_tag))
    img_dir_template = '../../../datasets/tsa/{}_images/dataset/{}'
    aps_img_dir = os.path.join(img_dir_template.format('aps', subset))
    a3d_img_dir = os.path.join(img_dir_template.format('a3daps', subset))
    scan_ids_list = os.listdir(aps_img_dir)

    i = 0
    while scan_ids_list[i] != '0fc066d8ab1c5a6a42b636c1fc5876a6': i += 1

    while True:
        scanid = scan_ids_list[i]
        browse_scan_frames = True
        aps0_fid = 0
        while browse_scan_frames:
            a3d1_fid = (aps0_fid * 4) % 64
            a3d2_fid = (aps0_fid * 4 - 1) % 64
            a3d3_fid = (aps0_fid * 4 + 1) % 64
            aps_img = cv.imread(os.path.join(aps_img_dir, scanid, '{}.png'.format(aps0_fid)))
            a3d_img = cv.imread(os.path.join(a3d_img_dir, scanid, '{}.png'.format(a3d1_fid)))
            aps_img = imgtf.simple_foreground_extract(aps_img)
            aps_img = cv.applyColorMap(aps_img, cv.COLORMAP_HOT)
            image = cv.addWeighted( a3d_img, 0.8, aps_img, 0.2, 0)
            f_image = np.zeros((660, 1000, 3), dtype=np.uint8)
            f_image[0:660, 0:512, :] = image
            text = '{:>4}. scanid: {}'.format(i + 1, scanid)
            cv.putText(f_image, text, (520, 50), cv.FONT_HERSHEY_PLAIN,
                       1, (0, 127, 127), 1, lineType=cv.LINE_AA)
            column_name = 'Frame{}'.format(aps0_fid)
            cell = vel_df.loc[vel_df['scanID'] == scanid, column_name]
            frm_score_meta = eval(cell.values[0])  # eval() or ast.literal_eval()

            if show_main:
                # draw aps stick figure
                f_image = draw_pose(f_image, aps_df, scanid, aps0_fid, aps0_fid, frm_score_meta,
                                    3, 'aps', x_start=520, y_start=0, color=(0, 255, 200))
                # draw first a3daps stick figure
                f_image = draw_pose(f_image, a3d_df, scanid, a3d1_fid, aps0_fid, frm_score_meta,
                                    0, 'a3daps', x_start=715, y_start=0, color=(0, 0, 255))
            if show_neig:
                # draw second a3daps stick figure
                f_image = draw_pose(f_image, a3d_df, scanid, a3d2_fid, aps0_fid, frm_score_meta,
                                    1, 'a3daps', x_start=520, y_start=300, color=(0, 127, 255))
                # draw third a3daps stick figure
                f_image = draw_pose(f_image, a3d_df, scanid, a3d3_fid, aps0_fid, frm_score_meta,
                                    2, 'a3daps', x_start=715, y_start=300, color=(0, 255, 255))
            if combo is not None:
                for idx, co_tag in enumerate(combo):
                    # draw a3daps stick figure
                    y = idx * 300
                    f_image = draw_pose(f_image, kpt_df[idx], scanid, aps0_fid, aps0_fid, None,
                                None, co_tag, x_start=910, y_start=y, color=co_color[idx], thick=2)
            # display image
            cv.imshow('Joint Frames', f_image)
            key = cv.waitKey(delay)
            if key == ord('n'): aps0_fid = (aps0_fid + 1) % n_frames # next frame
            elif key == ord('b'): aps0_fid = (aps0_fid - 1) % n_frames # previous frame
            elif key == ord('m'): # next scan
                browse_scan_frames = False
                i += 1
            elif key == ord('v'): # previous scan
                browse_scan_frames = False
                i -= 1
            elif key == ord('q'): sys.exit()


def draw_pose(f_image, df, scanid, fid, aps_fid, frm_score_meta,
              fsm_idx, ext_txt, x_start, y_start, color, thick=1):
    x_kpts = np.zeros((13), np.int32)
    y_kpts = np.zeros((13), np.int32)
    k_conf = np.zeros((13), np.float32)
    if frm_score_meta is None:
        score, l_wgt, n_dft, drift = None, None, None, None
    else:
        score = np.zeros((13), np.float32)  # adjusted confidence and limb drift score
        l_wgt = np.zeros((13), np.float32)
        n_dft = np.zeros((13), np.float32)
        drift = np.zeros((13), np.int32)
    column_name = 'Frame{}'.format(fid)
    cell = df.loc[df['scanID'] == scanid, column_name]
    frm_kpts_meta = eval(cell.values[0])  # eval() or ast.literal_eval()
    for kid in range(13):
        kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid + 1]  # Skip Ns
        x_kpts[kid], y_kpts[kid], k_conf[kid] = frm_kpts_meta[kpt]
        # order: [a3d1_kpts, a3d2_kpts, aps0_kpts]
        if frm_score_meta is not None:
            frm_txt = 'Frm:{:>2}'.format(fid)
            header = '{:<6} {}'.format(ext_txt, frm_txt)
            kpt_score_meta = frm_score_meta[kpt]
            score[kid] = kpt_score_meta[0][fsm_idx]
            l_wgt[kid] = kpt_score_meta[1][fsm_idx]
            n_dft[kid] = kpt_score_meta[2][fsm_idx]
            drift[kid] = kpt_score_meta[3][fsm_idx]
        else: header = ext_txt

    f_image = npoz.hpe.pose_stick_figure(f_image, x_kpts, y_kpts, k_conf, aps_fid,
                                         score, l_wgt, n_dft, drift, header, thick=thick,
                                         xstart=x_start, ystart=y_start, lineColor=color,
                                         label=True, invKpts=False, invEdge=False)
    return f_image


def stick_figure(image, fid, xKpts, yKpts, kConf, winName, wait=1):
    '''
    trans = np.array([[ 0.375, -0., 0.   ],
                      [ 0.,  0.375, 4.25 ]])
    #trans = np.array([[0.48, -0., -26.88],
    #                  [0., 0.48, -30.4]])
    image = cv.warpAffine(image,
                          trans,
                          (192, 256),
                          flags=cv.INTER_LINEAR)
    '''
    image = npoz.hpe.pose_stick_figure(image, xKpts, yKpts, kConf, fid=fid,
                                       label=True, invKpts=False, invEdge=False)
    cv.imshow(winName, image)
    if cv.waitKey(wait) == ord('q'): sys.exit()


def aps_best_from_combo(kpt_csv, tag, n_frames=16, n_cycles=1, conf_alpha=0.5,
                        fair_aps=-0.1, conf_beta=(0, -0.05, -0.05, -0.5)):
    print('\n', kpt_csv)
    assert (0 < conf_alpha <= 1)
    conf_beta = np.asarray(conf_beta)
    fair_beta = conf_beta
    fair_beta[-1] = fair_aps
    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}.csv'
    aps_df = pd.read_csv(path_template.format('aps', kpt_csv))
    a3d_df = pd.read_csv(path_template.format('a3daps', kpt_csv))
    kpt_wdf = aps_df.copy()
    for col in kpt_wdf.columns:
        if col != 'scanID': kpt_wdf[col].values[:] = ''
    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}-{}.csv'
    wrt_kpt_csv = path_template.format('aps', kpt_csv, tag)
    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}/{}-{}.csv'
    wrt_vdt_csv = path_template.format('aps', 'drift', kpt_csv, tag)
    vdt_wdf = aps_df.copy()
    limb_kpt_parent = {'RWr': 'REb', 'REb': 'RSh', 'RSh': 'RHp',
                       'LWr': 'LEb', 'LEb': 'LSh', 'LSh': 'LHp',
                       'RAk': 'RKe', 'RKe': 'RHp', 'RHp': 'RSh',
                       'LAk': 'LKe', 'LKe': 'LHp', 'LHp': 'LSh'}

    for (index, row) in aps_df.iterrows():
        scanid = row['scanID']
        for idx in range(n_frames * n_cycles):
            if idx % n_frames == 0: cycle = int(idx / n_frames)

            if cycle % 2 == 0:
                # forward pass
                aps0_fid = idx % n_frames
                lft_drift = True
                rgt_drift = False
            else:
                # backward pass
                aps0_fid = n_frames - (idx % n_frames ) - 1
                lft_drift = False
                rgt_drift = True

            if idx > 0:
                lft_aps0_fid = (aps0_fid - 1) % n_frames
                lft_frm_opt_kpts = get_kpt_meta_v1(lft_aps0_fid, scanid, kpt_wdf)
            else: lft_frm_opt_kpts = None
            if idx > 14:
                rgt_aps0_fid = (aps0_fid + 1) % n_frames
                rgt_frm_opt_kpts = get_kpt_meta_v1(rgt_aps0_fid, scanid, kpt_wdf)
            else: rgt_frm_opt_kpts = None

            frm_opt_kpts, kpts_drift_meta = \
                select_optimal_keypoints(aps0_fid, scanid, aps_df, a3d_df, limb_kpt_parent,
                                         lft_frm_opt_kpts, rgt_frm_opt_kpts, conf_beta, fair_beta,
                                         conf_alpha, lft_drift=lft_drift, rgt_drift=rgt_drift)
            column_name = 'Frame{}'.format(aps0_fid)
            kpt_wdf.at[index, column_name] = str(frm_opt_kpts) # faster than loc
            vdt_wdf.at[index, column_name] = str(kpts_drift_meta)

        if (index + 1) % 50 == 0:
            print('{:>4} SCANS PASSED..'.format(index + 1))

    kpt_wdf.to_csv(wrt_kpt_csv, encoding='utf-8', index=False)
    vdt_wdf.to_csv(wrt_vdt_csv, encoding='utf-8', index=False)


def select_optimal_keypoints(aps0_fid, scanid, aps_df, a3d_df, limb_kpt_parent, lft_frm_opt_kpts,
                             rgt_frm_opt_kpts, conf_beta, fair_beta, alpha, merge_logic=0,
                             lft_drift=True, rgt_drift=False, aps_next_neig=1, a3d_next_neig=4):
    optimal_kpts = dict()
    kpt_scr_meta = dict()
    a3d1_fid = (aps0_fid * 4) % 64
    a3d2_fid = (a3d1_fid - 1) % 64 #***(aps0_fid * 4 - 1) % 64
    a3d3_fid = (a3d1_fid + 1) % 64 #***(aps0_fid * 4 + 1) % 64
    aps_valid_kpts_per_frm = znkp.valid_keypt_per_16frames()
    a3d_valid_kpts_per_frm = znkp.valid_keypt_per_64frames()

    cum_valid_kpts = combine_validity([aps0_fid], [a3d1_fid, a3d2_fid, a3d3_fid], merge_logic,
                                      aps_valid_kpts_per_frm, a3d_valid_kpts_per_frm)
    aps0_kpts = get_kpt_meta_v1(aps0_fid, scanid, aps_df, cum_valid_kpts)
    a3d1_kpts = get_kpt_meta_v1(a3d1_fid, scanid, a3d_df, cum_valid_kpts)
    a3d2_kpts = get_kpt_meta_v1(a3d2_fid, scanid, a3d_df, cum_valid_kpts)
    a3d3_kpts = get_kpt_meta_v1(a3d3_fid, scanid, a3d_df, cum_valid_kpts)
    equal_frames = [4, 12] #*** [3, 4, 11, 12] [3, 4, 5, 11, 12, 13]
    beta = fair_beta if aps0_fid in equal_frames else conf_beta

    if lft_drift:
        lft_aps_fid = (aps0_fid - aps_next_neig) % 16
        lft_a3d_fid = (a3d1_fid - a3d_next_neig) % 64 #***(lft_aps0_fid * 4) % 64
        #assert (np.all(aps_valid_kpts_per_frm[:, lft_aps_fid] == a3d_valid_kpts_per_frm[:, lft_a3d_fid]))
        lft_beta = fair_beta if lft_aps_fid in equal_frames else conf_beta
        lft_cum_valid_kpts = combine_validity([lft_aps_fid], [lft_a3d_fid], merge_logic,
                                              aps_valid_kpts_per_frm, a3d_valid_kpts_per_frm)
        if lft_frm_opt_kpts is None:
            assert (aps0_fid == 0)
            # left neighboring frames
            lft_aps_kpts = get_kpt_meta_v1(lft_aps_fid, scanid, aps_df, lft_cum_valid_kpts)
            lft_a3d_kpts = get_kpt_meta_v1(lft_a3d_fid, scanid, a3d_df, lft_cum_valid_kpts)
        else: # compute velocity from opt. keypoint in previous frame, for all frames except 0
            assert (0 < aps0_fid <= 15)
            lft_frm_opt_kpts = filter_valid_kpts(lft_frm_opt_kpts, lft_cum_valid_kpts)
            lft_aps_kpts = lft_frm_opt_kpts
            lft_a3d_kpts = lft_frm_opt_kpts
    if rgt_drift:
        rgt_aps_fid = (aps0_fid + aps_next_neig) % 16
        rgt_a3d_fid = (a3d1_fid + a3d_next_neig) % 64 #***(rgt_aps0_fid * 4) % 64
        #assert (np.all(aps_valid_kpts_per_frm[:, rgt_aps_fid] == a3d_valid_kpts_per_frm[:, rgt_a3d_fid]))
        rgt_beta = fair_beta if rgt_aps_fid in equal_frames else conf_beta
        rgt_cum_valid_kpts = combine_validity([rgt_aps_fid], [rgt_a3d_fid], merge_logic,
                                              aps_valid_kpts_per_frm, a3d_valid_kpts_per_frm)
        if rgt_frm_opt_kpts is None: # use frame 0 already established optimal keypoints
            assert (aps0_fid < 15)
            rgt_aps_kpts = get_kpt_meta_v1(rgt_aps_fid, scanid, aps_df, rgt_cum_valid_kpts)
            rgt_a3d_kpts = get_kpt_meta_v1(rgt_a3d_fid, scanid, a3d_df, rgt_cum_valid_kpts)
        else:
            assert (0 <= aps0_fid <= 15)
            rgt_frm_opt_kpts = filter_valid_kpts(rgt_frm_opt_kpts, rgt_cum_valid_kpts)
            rgt_aps_kpts = rgt_frm_opt_kpts
            rgt_a3d_kpts = rgt_frm_opt_kpts

    for kid in range(13):
        kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid + 1]  # Skip Nose
        # order is important: [a3d1_kpts, a3d2_kpts, a3d3_kpts, aps0_kpts]
        kpts_set = np.asarray([a3d1_kpts[kpt], a3d2_kpts[kpt], a3d3_kpts[kpt], aps0_kpts[kpt]])
        kpts_score = kpts_set[:, 3] # index 3: contains confidence if valid kpt, 0 otherwise
        ignore_drift = kpt in ['RSh', 'LSh', 'Nk'] and aps0_fid in equal_frames
        p_kpt = limb_kpt_parent.get(kpt, None)

        # verify valid limb pairs for drifts
        if lft_drift:
            # check for valid limb pair
            aps_lft_vlp = is_valid_limb_pair(kpt, p_kpt, aps0_kpts, lft_aps_kpts)
            a3d_lft_vlp = is_valid_limb_pair(kpt, p_kpt, a3d1_kpts, lft_a3d_kpts)
            lft_vlp = aps_lft_vlp and a3d_lft_vlp
        else: lft_vlp = True
        if rgt_drift:
            # check for valid limb pair
            aps_rgt_vlp = is_valid_limb_pair(kpt, p_kpt, aps0_kpts, rgt_aps_kpts)
            a3d_rgt_vlp = is_valid_limb_pair(kpt, p_kpt, a3d1_kpts, rgt_a3d_kpts)
            rgt_vlp = aps_rgt_vlp and a3d_rgt_vlp
        else: rgt_vlp = True
        vlp = lft_vlp and rgt_vlp
        if vlp:
            p_kpts_set = np.asarray([a3d1_kpts[p_kpt], a3d2_kpts[p_kpt],
                                     a3d3_kpts[p_kpt], aps0_kpts[p_kpt]])
            p_kpts_score = p_kpts_set[:, 3] # index 3: contains confidence if valid kpt, 0 otherwise
            limb_wgt = (alpha * kpts_score) + ((1 - alpha) * p_kpts_score)
            # assert (np.all(limb_wgt == (kpts_score + p_kpts_score) / 2))
        else: limb_wgt = kpts_score
        limb_wgt = limb_wgt + beta  # Note, limb_wgt is only relevant when drift is computed
        cum_limb_weight = [limb_wgt]
        limbs_cum_drift = list()

        if lft_drift:
            # limb left drift
            aps0_lft_dft, aps0_lft_wgt = limb_drift(kpt, p_kpt, aps0_kpts, lft_aps_kpts, vlp, alpha)
            a3d1_lft_dft, a3d1_lft_wgt = limb_drift(kpt, p_kpt, a3d1_kpts, lft_a3d_kpts, vlp, alpha)
            a3d2_lft_dft, a3d2_lft_wgt = limb_drift(kpt, p_kpt, a3d2_kpts, lft_a3d_kpts, vlp, alpha)
            a3d3_lft_dft, a3d3_lft_wgt = limb_drift(kpt, p_kpt, a3d3_kpts, lft_a3d_kpts, vlp, alpha)
            # computed drift is noted only when limb weight in
            # neighboring frame for all (a3d1, a3d2, a3d3, aps0) is valid
            if aps0_lft_wgt * a3d1_lft_wgt * a3d2_lft_wgt * a3d3_lft_wgt > 0 and not ignore_drift:
                limb_lft_dft = [a3d1_lft_dft, a3d2_lft_dft, a3d3_lft_dft, aps0_lft_dft]
                limbs_cum_drift.append(limb_lft_dft)
                lft_limb_wgt = np.asarray([a3d1_lft_wgt, a3d2_lft_wgt, a3d3_lft_wgt, aps0_lft_wgt])
                lft_limb_wgt = lft_limb_wgt + lft_beta
                cum_limb_weight.append(lft_limb_wgt)

        if rgt_drift:
            # limb right drift
            aps0_rgt_dft, aps0_rgt_wgt = limb_drift(kpt, p_kpt, aps0_kpts, rgt_aps_kpts, vlp, alpha)
            a3d1_rgt_dft, a3d1_rgt_wgt = limb_drift(kpt, p_kpt, a3d1_kpts, rgt_a3d_kpts, vlp, alpha)
            a3d2_rgt_dft, a3d2_rgt_wgt = limb_drift(kpt, p_kpt, a3d2_kpts, rgt_a3d_kpts, vlp, alpha)
            a3d3_rgt_dft, a3d3_rgt_wgt = limb_drift(kpt, p_kpt, a3d3_kpts, rgt_a3d_kpts, vlp, alpha)
            # computed drift is noted only when limb weight in
            # neighboring frame for all (a3d1, a3d2, a3d3, aps0) is valid
            if aps0_rgt_wgt * a3d1_rgt_wgt * a3d2_rgt_wgt * a3d3_rgt_wgt > 0 and not ignore_drift:
                limb_rgt_dft = [a3d1_rgt_dft, a3d2_rgt_dft, a3d3_rgt_dft, aps0_rgt_dft]
                limbs_cum_drift.append(limb_rgt_dft)
                rgt_limb_wgt = np.asarray([a3d1_rgt_wgt, a3d2_rgt_wgt, a3d3_rgt_wgt, aps0_rgt_wgt])
                rgt_limb_wgt = rgt_limb_wgt + rgt_beta
                cum_limb_weight.append(rgt_limb_wgt)

        if len(limbs_cum_drift) > 0:
            cum_limb_weight = np.asarray(cum_limb_weight)
            cum_limb_weight = np.clip(cum_limb_weight, a_min=0, a_max=1)
            avg_limb_weight = np.mean(cum_limb_weight, axis=0)
            limbs_avg_drift = np.mean(limbs_cum_drift, axis=0)
            # denominator = np.max(np.sum(limbs_avg_drift), 1e-7)
            denominator = np.max(limbs_avg_drift) + np.min(limbs_avg_drift) + 1e-7
            norm_limb_drift = 1.0 - limbs_avg_drift / denominator
            kpts_score = avg_limb_weight * norm_limb_drift
        else:
            zeros = np.zeros(4)
            avg_limb_weight, norm_limb_drift, limbs_avg_drift = zeros, zeros, zeros
            kpts_score = limb_wgt

        opt_idx = np.argmax(kpts_score)
        opt_kpt_meta = kpts_set[opt_idx]
        optimal_kpts[kpt] = (int(opt_kpt_meta[0]), int(opt_kpt_meta[1]), opt_kpt_meta[2])
        kpt_scr_meta[kpt] = tuple([list(np.around(kpts_score, 4)),
                                   list(np.around(avg_limb_weight, 4)),
                                   list(np.around(norm_limb_drift, 4)),
                                   list(np.around(limbs_avg_drift, 0))])
    return optimal_kpts, kpt_scr_meta


def limb_drift(mainkpt, neigkpt, mainfrm_kpts, neigfrm_kpts, is_valid_limb_par, alpha):
    # returns kpt velocity drift and keypoint's confidence score in neighboring frame
    # todo: when kpt in frame is invalid
    #  when kpt in neighboring frame is invalid
    #  when neighboring kpt in frame is invalid
    #  when neighboring kpt in neighboring frame is invalid

    # index 3: contains confidence if valid kpt, 0 otherwise
    neigfrm_mainkpt_wgt = neigfrm_kpts[mainkpt][3]
    # Note, direction of change is important
    neigfrm_mainkpt_x, neigfrm_mainkpt_y = neigfrm_kpts[mainkpt][:2]
    mainfrm_mainkpt_x, mainfrm_mainkpt_y = mainfrm_kpts[mainkpt][:2]
    mainkpt_chg_x = neigfrm_mainkpt_x - mainfrm_mainkpt_x  # change in x-axis or x-velocity
    mainkpt_chg_y = neigfrm_mainkpt_y - mainfrm_mainkpt_y  # change in y-axis or y-velocity
    mainkpt_loc_chg = np.sqrt(mainkpt_chg_x ** 2 + mainkpt_chg_y ** 2)

    if is_valid_limb_par:
        # index 3: contains confidence if valid kpt, 0 otherwise
        neigfrm_neigkpt_wgt = neigfrm_kpts[neigkpt][3]
        # when either keypoints' confidence in neighboring frame is 0, drift is irrelevant
        #***if neigfrm_mainkpt_wgt * neigfrm_neigkpt_wgt <= 0: return 0, 0
        mainfrm_mainkpt_wgt = mainfrm_kpts[mainkpt][3]
        mainfrm_neigkpt_wgt = mainfrm_kpts[neigkpt][3]
        assert (mainfrm_mainkpt_wgt * mainfrm_neigkpt_wgt > 0) #***
        assert (neigfrm_mainkpt_wgt * neigfrm_neigkpt_wgt > 0) #***
        neig_limb_wgt = (alpha * neigfrm_mainkpt_wgt) + ((1 - alpha) * neigfrm_neigkpt_wgt)
        #***neig_limb_wgt = (neigfrm_mainkpt_wgt + neigfrm_neigkpt_wgt) / 2
        # when considering parent keypoint (eg., mainkpt: RAk then neigkpt: RKe)
        neigfrm_neigkpt_x, neigfrm_neigkpt_y = neigfrm_kpts[neigkpt][:2]
        mainfrm_neigkpt_x, mainfrm_neigkpt_y = mainfrm_kpts[neigkpt][:2]
        neigkpt_chg_x = neigfrm_neigkpt_x - mainfrm_neigkpt_x  # change in x-axis or x-velocity
        neigkpt_chg_y = neigfrm_neigkpt_y - mainfrm_neigkpt_y  # change in y-axis or y-velocity
        neigkpt_loc_chg = np.sqrt(neigkpt_chg_x ** 2 + neigkpt_chg_y ** 2)

        limb_locatn_chg = (mainkpt_loc_chg + neigkpt_loc_chg) // 2 #***
        limb_drift_x = neigkpt_chg_x - mainkpt_chg_x # x-component of limb drift vector
        limb_drift_y = neigkpt_chg_y - mainkpt_chg_y # y-component of limb drift vector
        limb_orient_chg = np.sqrt(limb_drift_x ** 2 + limb_drift_y ** 2)
        limb_drift = limb_locatn_chg + limb_orient_chg #***mainkpt_loc_chg + limb_orient_chg
    else:
        # when keypoint's confidence in neighboring frame is 0, drift is irrelevant
        if neigfrm_mainkpt_wgt == 0: return 0, 0
        # index 3: contains confidence if valid kpt, 0 otherwise
        neig_limb_wgt = neigfrm_mainkpt_wgt
        limb_drift = mainkpt_loc_chg

    return limb_drift, neig_limb_wgt

def get_kpt_meta_v1(fid, scanid, df, frm_valid_kpts=None):
    column_name = 'Frame{}'.format(fid)
    cell = df.loc[df['scanID'] == scanid, column_name]
    frm_kpts_meta = eval(cell.values[0])
    if frm_valid_kpts is not None:
        return filter_valid_kpts(frm_kpts_meta, frm_valid_kpts)
    return frm_kpts_meta

def get_kpt_meta_v2(fid, scanid, df, valid_kpts_per_frm=None):
    column_name = 'Frame{}'.format(fid)
    cell = df.loc[df['scanID'] == scanid, column_name]
    frm_kpts_meta = eval(cell.values[0])
    if valid_kpts_per_frm is not None:
        frm_valid_kpts = valid_kpts_per_frm[:, fid]
        return filter_valid_kpts(frm_kpts_meta, frm_valid_kpts)
    return frm_kpts_meta

def filter_valid_kpts(frm_kpts_meta, frm_valid_kpts):
    for kid in range(13):
        kpt = npoz.hpe.INDEX_TO_KPT_LABEL[kid + 1]  # Skip Nose and Neck
        kpt_meta = frm_kpts_meta[kpt]
        score = kpt_meta[2] if frm_valid_kpts[kid] else 0
        frm_kpts_meta[kpt] = kpt_meta + tuple([score])
    return frm_kpts_meta

def combine_validity(aps_fids, a3d_fids, logi_mode, aps_valid_kpts_per_frm, a3d_valid_kpts_per_frm):
    if logi_mode == 0: truth_logic = np.logical_and
    elif logi_mode == 1: truth_logic = np.logical_or
    aps_valid_kpts = aps_valid_kpts_per_frm[:, aps_fids[0]]
    for i in range(1, len(aps_fids)):
        aps_valid_kpts = truth_logic(aps_valid_kpts, aps_valid_kpts_per_frm[:, aps_fids[i]])
    a3d_valid_kpts = a3d_valid_kpts_per_frm[:, a3d_fids[0]]
    for i in range(1, len(a3d_fids)):
        a3d_valid_kpts = truth_logic(a3d_valid_kpts, a3d_valid_kpts_per_frm[:, a3d_fids[i]])
    return truth_logic(aps_valid_kpts, a3d_valid_kpts)

def is_valid_limb_pair(mainkpt, neigkpt, mainfrm_kpts, neigfrm_kpts):
    valid_limb_pair = False
    if neigkpt is not None:
        neigfrm_mainkpt_wgt = neigfrm_kpts[mainkpt][3]
        mainfrm_mainkpt_wgt = mainfrm_kpts[mainkpt][3]
        mainfrm_neigkpt_wgt = mainfrm_kpts[neigkpt][3]
        neigfrm_neigkpt_wgt = neigfrm_kpts[neigkpt][3]
        wgt = neigfrm_mainkpt_wgt * neigfrm_neigkpt_wgt * mainfrm_mainkpt_wgt * mainfrm_neigkpt_wgt
        if wgt > 0: valid_limb_pair = True
    return valid_limb_pair

'''
def pixel_to_meters(df_x, df_y, nkpts, samples):
    x_m = 1. / 512 # 512 pixels - 1 meter
    y_m = 2.0955 / 660 # 660 pixels - 2.0955 meters
    x_mm = x_m * 1000
    y_mm = y_m * 1000
    dftypes = ['TrueKpt', 'CocoGen', 'Refined', 'CorrRecon', 'SiftRecon', 'SurfRecon',
               'OrbRecon', 'HRNet', 'HRNetSwap', 'HRNetOpt', 'HRNetOptSwap']
    columns = ['Comparison', 'RSh', 'REb', 'RWr', 'LSh', 'LEb', 'LWr', 'RHp', 'RKe',
               'RAk', 'LHp', 'LKe', 'LAk', 'Avg']
    df_d = pd.DataFrame(columns=columns)

    for i in range(len(dftypes)):
        df_d.at[i, 'Comparison'] = dftypes[i]
        print('{}. DataFrame: {}'.format(i + 1, dftypes[i]))
        dAvgSum = 0
        for k in range(nkpts):
            kpt = npoz.hpe.INDEX_TO_KPT_LABEL[k + 2] #***
            dist = np.sqrt((df_x.at[i, kpt] * x_mm)**2 + (df_y.at[i, kpt] * y_mm)**2)
            dAvgSum += dist
            df_d.at[i, kpt] = round(dist, 2)
        df_d.at[i, 'Avg'] = round(dAvgSum / nkpts, 2)

    writeCsv = os.path.join(csvDir, str(samples) + '_new_distance_euclidean_mm.csv')
    df_d.to_csv(writeCsv, encoding='utf-8', index=False)
'''

if __name__ == "__main__":
    global DUMMY_CSV, NUM_OF_SCANS, FRAMES_PER_SCAN, ALL_KPTS, faceKptIndicies, faceIndexMap, csvDir
    subset = 'all_sets'  #***
    ds_ext = 'a3daps' #***
    thresh = 0.3
    FRAMES_PER_SCAN = 16 if ds_ext=='aps' else 64
    NUM_OF_SCANS = 2635 #1147 #***
    #npoz.initialize_global_variables(kernel=49, step=3, nframes=FRAMES_PER_SCAN, conf_thresh=thresh,
    #                            imgShape=(512, 660), matcher='corr', descType='None', keepLog=False)
    npoz.initialize_global_variables(n_frames=FRAMES_PER_SCAN,
                                     conf_thresh=thresh, img_shape=(512, 660))

    #imgDir = '../../../PSCNets/Data/a3daps_images/dataset/train_set/'
    csvDir = '../../Data/tsa_psc/kpts_csvs'
    #npyDir = '../../../PSCNets/Data/kpts_npys'
    hpeDir = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/'.format(ds_ext)

    #DUMMY_CSV = os.path.join(csvDir, 'cocoKptsMarkings_dummy.csv')
    #ALL_KPTS = npoz.hpe.COCO_KEYPOINTS_ALL - 1
    #faceKptIndicies = [0, 14, 15, 16, 17]  # Nose, Eyes, and Ears
    #faceIndexMap = {0: 0, 14: 1, 15: 2, 16: 3, 17: 4}

    #test_hpe_keypoints(imgDir, csvDir, npyDir)
    #test_refined_kpts(csvDir, npyDir)
    ##test_recontruction(imgDir, csvDir, npyDir, type='Corr') # step=3, matcher='corr', descType='None' 29.152 sec
    #test_recontruction(imgDir, csvDir, npyDir, type='Sift') # step=6, matcher='desc', descType='sift' 316.123 sec
    #test_recontruction(imgDir, csvDir, npyDir, type='Surf') # step=6, matcher='desc', descType='surf' 35.553 sec
    #test_recontruction(imgDir, csvDir, npyDir, type='Orb') # step=6, matcher='desc', descType='orb' 38.552 sec

    # csv_paths = [(csvDir, 'cocoKptsMarkings.csv'),
    #              (csvDir, 'cocoGenKptsScaled_B.csv'), (csvDir, 'cocoGenKptsRefined_B.csv'),
    #              (csvDir, 'cocoGenKptsRefinReconCorr_B.csv'), (csvDir, 'cocoGenKptsRefinReconSift_B.csv'),
    #              (csvDir, 'cocoGenKptsRefinReconSurf_B.csv'), (csvDir, 'cocoGenKptsRefinReconOrb_B.csv'),
    #              (hpeDir, 'all_sets-w32_256x192.csv'), (hpeDir, 'all_sets-w32_256x192-ref.csv'),
    #              (hpeDir, 'all_sets-w32_256x192-opt.csv'), (hpeDir, 'all_sets-w32_256x192-opt-ref.csv'),
    #              (hpeDir, 'all_sets-w32_256x192-ref-30.csv'), (hpeDir, 'all_sets-w32_256x192-opt-ref-30.csv'),
    #              (hpeDir, 'all_sets-w32_256x192-opt-ref-30-velcomb_v1.csv'), (hpeDir, 'all_sets-w32_256x192-opt-ref-30-velcomb.csv'),
    #              (hpeDir, 'all_sets-w32_256x192-opt-ref-30-bav1.csv'), (hpeDir, 'all_sets-w32_256x192-opt-ref-30-bav2.csv')]

    # #kpt_distance_anaylsis(csv_paths)
    # #print('All Done! :)\n')
    # #inspect(imgDir, npyDir)
    # #visualize_csv_kpts_v1(csvDir)
    # visualize_csv_kpts_v2('all_sets-w32_256x192-opt-ref-30', 'train_set', combo=['velcomb','bav2'])
    #'''

    ####net_configs = ['w32_256x192']#, 'w48_256x192',
    ####               #'res50_256x192', 'res101_256x192', 'res152_256x192']
    ####for config in net_configs:
    ####    refine_keypoints('{}-{}'.format(subset, config))
    ####    refine_keypoints('{}-{}-opt'.format(subset, config))

    # tag = 'velcomb'
    # for config in net_configs:
    #     #aps_best_from_combo('{}-{}-ref-30'.format(subset, config), tag, n_cycles=1)
    #     aps_best_from_combo('{}-{}-opt-ref-30'.format(subset, config), tag, n_cycles=1)
    # #kpt_distance_anaylsis(csv_paths)
    #'''
    '''
    df_x = pd.read_csv(os.path.join(csvDir, '23_new_distance_xAxis.csv'))
    df_y = pd.read_csv(os.path.join(csvDir, '23_new_distance_yAxis.csv'))
    pixel_to_meters(df_x, df_y, 12, 23)
    '''