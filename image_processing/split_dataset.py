'''
    1. create spreadsheet that splits dataset train/validation/test sets in a stratified manner on hit configuration
    2. using spreadsheet dataset are separated into corresponding subset directories
'''

import numpy as np
import pandas as pd
import sys, os
import datetime
from distutils.dir_util import copy_tree
from sklearn.utils import shuffle


def create_dir(path):
    # create directory if it does not exist
    if os.path.exists(path) == False:
        os.mkdir(path)

def print_format(rowName, array):
    return '{:<18} '.format(rowName) + \
           '{0[0]:<8} {0[1]:<5} {0[2]:<5} {0[3]:<5} {0[4]:<5} {0[5]:<5} {0[6]:<5} {0[7]:<5} ' \
           '{0[8]:<5} {0[9]:<5} {0[10]:<5} {0[11]:<5} {0[12]:<5} {0[13]:<5} {0[14]:<5} {0[15]:<5} ' \
           '{0[16]:<5} {0[17]:<5}'.format(array)
        
def log(msg, openMode="a+"):
    '''
    Log runtime messages to logfile
    :param msg:     message text to log
    :return:        nothing is returned
    '''
    f = open(_logFile, openMode)
    f.write(msg + '\n')
    f.close()
    print(msg)


def separate_to_directories(distributionCsvFile, dataFormat):
    dataDir = os.path.join('../../../Passenger-Screening-Challenge/Data', dataFormat, 'dataset')
    readDir = os.path.join(dataDir, 'set_classified_1147')
    trnSDir = os.path.join(dataDir, 'train')
    valSDir = os.path.join(dataDir, 'valid')
    dfDistr = pd.read_csv(distributionCsvFile)
    create_dir(trnSDir)
    create_dir(valSDir)
    print('\nGrouping Scans...')

    for index, row in dfDistr.iterrows():
        scanid = row['scanID']
        set = row['Group']
        readScanDir = os.path.join(readDir, scanid)
        copyOver = True
        if set == 'Train':
            wrtScanDir = os.path.join(trnSDir, scanid)
        elif set == 'Validation':
            wrtScanDir = os.path.join(valSDir, scanid)
        else:
            copyOver = False
            print('Sacn: {} in unrecognized group: {}'.format(scanid, set))

        if copyOver:
            create_dir(wrtScanDir)
            copy_tree(readScanDir, wrtScanDir)

        print('\t{:>4}. {}'.format(index + 1, scanid))


def hits_per_zone(df):
    hits = np.zeros(shape=(18), dtype=np.int32)  # index 0: scanHits, index 1 - 17: zones 1 - 17
    for i in range(hits.shape[0]):
        zHits = df[_zoneNames[i]].values
        hits[i] = np.sum(zHits)
    return hits


def filter_dataframe(hits):
    '''
    Filter data samples that fit the hit configuration
    :param hits:    list of zones with probability == 1
    :return:        filtered dataFrame, the index of the original _hitdf is maintained
    '''
    zoneHits = np.zeros(shape=(18), dtype=np.int32)  # index 0: scanHits, index 1 - 17: zones 1 - 17
    if hits:
        zoneHits[hits] = 1
    return _hitdf[(_hitdf[_zoneNames[1]] == zoneHits[1]) &
                 (_hitdf[_zoneNames[2]] == zoneHits[2]) &
                 (_hitdf[_zoneNames[3]] == zoneHits[3]) &
                 (_hitdf[_zoneNames[4]] == zoneHits[4]) &
                 (_hitdf[_zoneNames[5]] == zoneHits[5]) &
                 (_hitdf[_zoneNames[6]] == zoneHits[6]) &
                 (_hitdf[_zoneNames[7]] == zoneHits[7]) &
                 (_hitdf[_zoneNames[8]] == zoneHits[8]) &
                 (_hitdf[_zoneNames[9]] == zoneHits[9]) &
                 (_hitdf[_zoneNames[10]] == zoneHits[10]) &
                 (_hitdf[_zoneNames[11]] == zoneHits[11]) &
                 (_hitdf[_zoneNames[12]] == zoneHits[12]) &
                 (_hitdf[_zoneNames[13]] == zoneHits[13]) &
                 (_hitdf[_zoneNames[14]] == zoneHits[14]) &
                 (_hitdf[_zoneNames[15]] == zoneHits[15]) &
                 (_hitdf[_zoneNames[16]] == zoneHits[16]) &
                 (_hitdf[_zoneNames[17]] == zoneHits[17]) ]


def stratified_split(hits=None, splitRatio=0.666):
    '''
    Separate into 2 sets, data samples with same zone hit configuration according to Split Ratio
    :param hits:    list of zones with probability == 1
    :return:        nothing is returned
    '''
    global _numOfConfigs, _numOfSamples, _trnSetCount, _valSetCount
    numOfHits = len(hits) if hits else 0
    df = shuffle(filter_dataframe(hits), random_state=_numOfConfigs)
    size = len(df.index)
    trnCnt, valCnt = 0, 0

    if size > 0:
        trnCnt = max(1, int(round(size * splitRatio)))
        valCnt = size - trnCnt
        for i, dfHitIndex in enumerate(df.index):
            groupSet = 'Train' if i < trnCnt else 'Validation'
            _hitdf.loc[dfHitIndex, 'Group'] = groupSet
            #_hitdf.at[dfHitIndex, 'Group'] = groupSet

        _trnSetCount += trnCnt
        _valSetCount += valCnt

    log('\tHit Size: {:<2} Zone Configuration: {:<13} Data Samples: {:<4} Train: {:<3} '
        'Validation: {:<3}'.format(numOfHits, str(hits), size, trnCnt, valCnt))
    _numOfSamples += size
    _numOfConfigs += 1


def split_k_folds(dfSubsetIndexes, n, k):
    global _mainQueue, _kFoldsBagCounts

    if n > 0:
        # init data-structures and variables
        kQueue = _mainQueue[k]
        assert (len(kQueue) == k), 'k: {}, kQueue: {}'.format(k, kQueue)
        headQueue, tailQueue = [], []
        kFoldColumn = 'k{}_Folds'.format(k)
        sidx, eidx = 0, 0

        # k-fold parameters
        m = int(np.floor(n / k))
        r = n - m*k

        # separate data samples into k bags
        for i in range(k):
            bag_indx = kQueue.pop(0)
            c = m+1 if r>0 else m
            if c > 0:
                sidx = eidx
                eidx = sidx + c
                _kFoldsBagCounts[k][bag_indx] += c
                # set chosen data samples at indexes to bag id
                for dfHitIndex in dfSubsetIndexes[sidx: eidx]:
                    #_hitdf.loc[dfHitIndex, kFoldColumn] = bag_indx + 1
                     _hitdf.at[dfHitIndex, kFoldColumn] = bag_indx + 1

            if r > 0:
                tailQueue.append(bag_indx)
                r -= 1
            else:
                headQueue.append(bag_indx)

        assert (eidx == n)
        assert (len(headQueue) + len(tailQueue) == k), '{}, {}, {}'.format(k, headQueue, tailQueue)
        _mainQueue[k] = headQueue + tailQueue


def kfolds_split(hits=None):
    global _numOfConfigs, _numOfSamples
    hitSize = len(hits) if hits else 0
    df = shuffle(filter_dataframe(hits), random_state=_numOfConfigs)
    n = len(df.index)

    msg = '\tHit Size: {:<2} Zone Configuration: {:<13} Data Samples: {:<4}'\
            .format(hitSize, str(hits), n)

    for k in K_FOLDS:
        split_k_folds(df.index, n, k)
        if n > 0:
            template = ' {}-Bags: {' + ':<{}'.format(k*5) + '} '
            msg += template.format(k, str(_kFoldsBagCounts[k]))
            assert (np.max(_kFoldsBagCounts[k]) - np.min(_kFoldsBagCounts[k]) <= 2), msg

    log(msg)
    _numOfSamples += n
    _numOfConfigs += 1


def next_zone_candidates(start, last=18):
    # return the list of zones 1-17 after start
    return np.arange(start, last).tolist()


def recurse_combination_config(zoneConfig, zoneCandidates, configSizeMax):
    configSize = len(zoneConfig)
    assert (0 <= configSize <= configSizeMax)
    if configSize == configSizeMax:
        #stratified_split(hits=zoneConfig, splitRatio=_splitRatio)
        kfolds_split(hits=zoneConfig)
    else:
        for candZone in zoneCandidates: # for zones: candZone to 17
            newZoneConfig = zoneConfig.copy()
            newZoneConfig.append(candZone)
            newCandidates = next_zone_candidates(candZone + 1)
            recurse_combination_config(newZoneConfig, newCandidates, configSizeMax)


def n_combination_r(rSize):
    assert (rSize > 0)
    initZoneConfig = []
    zoneCandidates = next_zone_candidates(1)
    recurse_combination_config(initZoneConfig, zoneCandidates, rSize)


def distribute_by_hit_combinations(hitcsv, wrtcsv):
    global _hitdf, _zoneNames, _numOfConfigs, _numOfSamples, \
        _splitRatio, _trnSetCount, _valSetCount, _mainQueue, _kFoldsBagCounts, K_FOLDS
    _numOfConfigs = 0
    _numOfSamples = 0

    # stratified split variables
    _trnSetCount = 0
    _valSetCount = 0
    _splitRatio = 0.75

    _hitdf = pd.read_csv(hitcsv)
    maxScanHit = np.max(_hitdf['ScanHits'].values)
    _zoneNames = {0: 'ScanHits', 1: 'R-Bicep_z1', 2: 'R-Forearm_z2', 3: 'L-Bicep_z3',
                  4: 'L-Forearm_z4', 5: 'Chest_z5', 6: 'R-Abdomen_z6', 7: 'L-Abdomen_z7',
                  8: 'R-Hip_z8', 9: 'Groin_z9', 10: 'L-Hip_z10', 11: 'R-Thigh_z11',
                  12: 'L-Thigh_z12', 13: 'R-Calf_z13', 14: 'L-Calf_z14',
                  15: 'R-Ankle_z15', 16: 'L-Ankle_z16', 17: 'Back_z17'}

    # k-folds split variables
    K_FOLDS = [3, 5, 7]
    K_COLUMNS = list()
    _mainQueue = dict()
    _kFoldsBagCounts = dict()
    for k in K_FOLDS:
        _mainQueue[k] = list(range(0, k)) # python queue
        K_COLUMNS.append('k{}_Folds'.format(k))
        _kFoldsBagCounts[k] = np.zeros(k, np.int32)

    # copy over old group set column
    setdf = pd.read_csv('../../Data/tsa_psc/train_set_distribution.csv')
    for i, row in setdf.iterrows():
        _hitdf.loc[_hitdf['scanID'] == row['scanID'], 'groupSet'] = row['Group']

    # refresh logfile
    log('File Created: {}'.format(datetime.datetime.now()), openMode="w")

    # split no-hit cases
    log('\nProcessing {} Combination {}'.format(17, 0))
    #stratified_split(splitRatio=_splitRatio)
    kfolds_split()

    # split hit cases
    for r in range(1, maxScanHit + 1):
        log('\nProcessing {} Combination {}'.format(17, r))
        n_combination_r(r)

    columns_oi = ['scanID', 'groupSet'] + K_COLUMNS
    grpdf = _hitdf[columns_oi]
    grpdf.to_csv(wrtcsv, encoding='utf-8', index=False)


    msg = '\nSummary:' \
          '\nMaximum scan hit rate is:         {}'.format(maxScanHit) + \
          '\nTotal number of combinations:     {}'.format(_numOfConfigs) + \
          '\nProcessed data samples count:     {}'.format(_numOfSamples)

    for i, k in enumerate(K_FOLDS):
        msg += '\n\nk{}_Folds; number of threats,'.format(k)
        for b in range(k):
            b_id = b + 1
            hitsPerZone = hits_per_zone(_hitdf[_hitdf[K_COLUMNS[i]] == b_id])
            hitRate = np.int32(np.around((hitsPerZone[1:] * 100) / _kFoldsBagCounts[k][b], 0))
            msg += '\n    Bag {}:\n\tper zone: {}, total: {},\n\tper zone threat hit %: {}'.\
                    format(b_id, str(hitsPerZone[1:]), hitsPerZone[0], str(hitRate))

    log(msg)


if __name__ == "__main__":
    global _logFile
    _logFile = '../../Data/tsa_psc/kfolds_distribution.txt'
    wrtcsv = '../../Data/tsa_psc/kfolds_distribution.csv'
    hitcsv = '../../Data/tsa_psc/stage1ScanHit.csv'
    distribute_by_hit_combinations(hitcsv, wrtcsv)
    #separate_to_directories(wrtcsv, 'a3daps_images')
