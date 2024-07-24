import pandas as pd
import os
import json
import shutil

# read .csv into pandas data frame, modify it and returns data frame
def readcsv(path):
    df = pd.read_csv(path)
    df['NoZoneId'] = " "
    df['Zone'] = 0
    df['Marked'] = 0
    for index, row in df.iterrows():
        idzone = row['Id'].split('_')   #returns list of id and zone separated
        df.set_value(index,'NoZoneId',idzone[0])
        df.set_value(index, 'Zone', idzone[1][4:])
    #print df.head()
    return df


# creates data frame with ID = scan ids and RectDiagCoords
def createcsv(rootdir, initial):
    listofdir = os.listdir(rootdir)
    # create new data frame
    df2 = pd.DataFrame(listofdir, columns=['ID'])
    df2['Marked'] = initial
    df2['Frame0'] = "N/M"       # Not Marked
    df2['Frame1'] = "N/M"
    df2['Frame2'] = "N/M"
    df2['Frame3'] = "N/M"
    df2['Frame4'] = "N/M"
    df2['Frame5'] = "N/M"
    df2['Frame6'] = "N/M"
    df2['Frame7'] = "N/M"
    df2['Frame8'] = "N/M"
    df2['Frame9'] = "N/M"
    df2['Frame10'] = "N/M"
    df2['Frame11'] = "N/M"
    df2['Frame12'] = "N/M"
    df2['Frame13'] = "N/M"
    df2['Frame14'] = "N/M"
    df2['Frame15'] = "N/M"
    # print (df2.head())
    return df2


def first_function():
    ''' reads stage1_labels_1.csv and scan Ids from the directory containing all scan images.
        Then generates stage1_labels.txt, stage1_labels_1_modified.csv, and stage1_labels_1_marked.csv.
        Note, before generating the above files, the program copies already existing files into
        /Data/tsa_psc/old_versions.
    '''
    scandir = '../../../Passenger-Screening-Challenge/Data/aps_images/full_image_threat'
    csvfile = '../../Data/tsa_psc/stage1_labels_1.csv'
    modifiedcsvfile = '../../Data/tsa_psc/stage1_labels_1_modified.csv'
    markedcsvfile = '../../Data/tsa_psc/stage1_labels_1_marked.csv'
    jsontxtfile = '../../Data/tsa_psc/stage1_labels.txt'
    oldversionfolder = '../../Data/tsa_psc/old_versions'

    shutil.copy2(modifiedcsvfile, oldversionfolder)  # save copy of old version
    shutil.copy2(markedcsvfile, oldversionfolder)  # save copy of old version
    shutil.copy2(jsontxtfile, oldversionfolder)  # save copy of old version

    dataframe = readcsv(csvfile)
    dataframe.to_csv(modifiedcsvfile, encoding='utf-8', index=False)

    dataframe = createcsv(scandir, 0)
    dataframe.to_csv(markedcsvfile, encoding='utf-8', index=False)

    markings = {}
    s = json.dumps(markings)
    f = open(jsontxtfile, "w")
    f.write(s)

    print("Necessary files are built")


def create_hit_spreadsheet():
    '''
    Reads stage1_labels.csv and reorganizes the entries by creating frame and other features
    '''
    tsacsv = '../../Data/tsa_psc/train_set_labels_v2.csv' #stage1_labels_corrected.csv'
    hitcsv = '../../Data/tsa_psc/stage1ScanHit.csv'
    tsadf = pd.read_csv(tsacsv)
    jointIDs = tsadf['Id'].values.tolist()
    hitdf = pd.DataFrame(columns=['scanID', 'ScanHits', 'R-Bicep_z1', 'R-Forearm_z2', 'L-Bicep_z3',
                                  'L-Forearm_z4', 'Chest_z5', 'R-Abdomen_z6', 'L-Abdomen_z7', 'R-Hip_z8',
                                  'Groin_z9', 'L-Hip_z10', 'R-Thigh_z11', 'L-Thigh_z12', 'R-Calf_z13',
                                  'L-Calf_z14', 'R-Ankle_z15', 'L-Ankle_z16', 'Back_z17'])
    zoneNames = {0: 'ScanHits', 1: 'R-Bicep_z1', 2: 'R-Forearm_z2', 3: 'L-Bicep_z3',
                 4: 'L-Forearm_z4', 5: 'Chest_z5', 6: 'R-Abdomen_z6', 7: 'L-Abdomen_z7',
                 8: 'R-Hip_z8', 9: 'Groin_z9', 10: 'L-Hip_z10', 11: 'R-Thigh_z11',
                 12: 'L-Thigh_z12', 13: 'R-Calf_z13', 14: 'L-Calf_z14',
                 15: 'R-Ankle_z15', 16: 'L-Ankle_z16', 17: 'Back_z17'}
    index = 0
    while(len(jointIDs) > 0):
        entry = jointIDs[0]
        token = entry.split('_')  # returns list of id and zone separated
        scanid = token[0]
        hitdf.loc[index, 'scanID'] = scanid
        hitsum = 0
        for z in range(1, 18):
            zoneTag = '_Zone{}'.format(z)
            dataPnt = scanid + zoneTag
            gtLabel = tsadf.loc[tsadf['Id'] == dataPnt, 'Probability'].values[0]
            hitdf.loc[index, zoneNames[z]] = gtLabel
            hitsum += gtLabel
            jointIDs.remove(dataPnt)

        hitdf.loc[index, 'ScanHits'] = hitsum
        index += 1

    hitdf.to_csv(hitcsv, encoding='utf-8', index=False)


if __name__ == "__main__":
    create_hit_spreadsheet()
