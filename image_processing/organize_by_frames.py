'''This program organizes no-threat images from all scans into 16 frames
   images are stored in the root directory: no_threat_frames
   under 16 subdirectories with names 0 to 16
   Each image in the subdirectory is named: scanid_frame#
'''

import os
import cv2
import pandas as pd


# create the needed directories to save images to
def create_subdirectories(rootdir):
    for i in range(16):
        os.mkdir(rootdir + str(i))
    print ("Sub Directories created")

# copy image from one location to the other
def copy_image(src, dest):
    img = cv2.imread(src)
    cv2.imwrite(dest, img)

# copy scans from no_threat directory
def sort_no_threat_dir(rdir, wdir):
    print ("Organizing files from no-threat directory..")
    listofscans = os.listdir(rdir)
    counter = 0
    for scan in listofscans:
        counter = counter + 1
        for frame in range(16):
            src = rdir + scan + '/' + str(frame) + '.png'
            dest = wdir + str(frame) + '/' + scan + '_f' + str(frame) + '.png'
            copy_image(src, dest)

        if counter % 10 == 0:
            print (str(counter)+" scans organized")

    print (str(counter*16)+" frames from no-threat directory organized")

# copy scans from threat directory
def sort_threat_dir(rdir, wdir, file):
    print ("Organizing files from threat directory..")
    df = pd.read_csv(file)
    listofscans = os.listdir(rdir)
    counter = 0
    for scan in listofscans:
        for frame in range(16):
            cell = df.loc[df['ID'] == scan, "Frame"+str(frame)].values[0]
            if cell == "N/M": # implies the frame has no threat
                counter = counter + 1
                src = rdir + scan + '/' + str(frame) + '.png'
                dest = wdir + str(frame) + '/' + scan + '_f' + str(frame) + '.png'
                copy_image(src, dest)

                if counter % 100 == 0:
                    print(str(counter) + " frames organized")

    print (str(counter)+" frames from threat directory organized")


def main():
    w_root_dir = "../../../Passenger-Screening-Challenge/Data/aps_images/no_threat_frames/"
    r_nthreat_dir = "../../../Passenger-Screening-Challenge/Data/aps_images/full_image_no_threat/"
    r_threat_dir = "../../../Passenger-Screening-Challenge/Data/aps_images/full_image_threat/"
    csv = '../../Data/tsa_psc/stage1_labels_1_marked.csv'

    create_subdirectories(w_root_dir)

    # organize scans in no_threat directory
    sort_no_threat_dir(r_nthreat_dir, w_root_dir)

    # organize scans in threat directory
    sort_threat_dir(r_threat_dir, w_root_dir, csv)

    print ("All scans organized into directories")

main()