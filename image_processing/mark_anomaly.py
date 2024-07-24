import sys
import os
import cv2
import json
import time, threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# display image because imshow() doesn't work on opencv 3
def show(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

# read .csv into pandas data frame and returns data frame
def readcsv(path):
    return pd.read_csv(path)

# read text file in json format and save to dictionary
def readjson(path):
    f = open(path, "r")
    s = f.read()
    return json.loads(s)

# save markings of frame
def record_markings(scid, fnum, coords):
    # record in data frame
    dfw.loc[dfw['ID'] == scid, fnum] = str(coords)
    # record in json dictionary
    markings[scid+'_'+fnum] = {
        'scanid': scid,
        'frame': fnum,
        'coords': coords
    }

# edit/change markings of frame
def reset_markings(scid, fnum, coords):
    if len(coords) == 0:
        # remove in data frame by resetting to 'N/M'
        dfw.loc[dfw['ID'] == scid, fnum] = "N/M"
        # remove from json dictionary
        if scid+'_'+fnum in markings:
            del markings[scid+'_'+fnum]
    else: record_markings(scid, fnum, coords)

# mark unique scanID as completed
def mark_completed(scid):
    global change
    dfw.loc[dfw['ID'] == scid, 'Marked'] = 1                # 1 implies marked
    dfr.loc[dfr['NoZoneId'] == scid, 'Marked'] = 1          # 1 implies marked
    change = True
    print ("***Message: scan " + str(scid) + " is marked as completed" \
            "\n            and will be permanently saved in 10 minutes or less")

# permanently writes changes from memory to files
def save_progress():
    global change
    dfw.to_csv(writecsvfile, encoding='utf-8', index=False) # save/write data frame to csv file
    dfr.to_csv(readcsvfile, encoding='utf-8', index=False)  # save/write data frame to csv file
    f = open(jsontextfile, "w")                             # open text file to write
    f.write(json.dumps(markings))                           # write dictionary to text file in json format
    change = False                                          # reset change

# periodically save data frame by writing to csv and json
def periodic_save():
    global change
    if change:
        save_progress()
        print ("***Alert: Periodic save on "+time.ctime())
        #TODO: create a log file and log periodic save times
    # sleep for 10 minutes then recursively call function
    time.sleep(600)
    periodic_save()

# applies transformation matrix on 2D point
def transform_point(xcord, ycord, transMatrix):
    hpoint = transMatrix.dot([xcord, ycord, 1])
    transX = int(hpoint[0] / hpoint[2])
    transY = int(hpoint[1] / hpoint[2])
    return transX, transY

# applies transformation matrix to entire image
def transform_image(img, transMatrix):
    if np.array_equal(transMatrix, np.identity(3)):
        return img
    else:
        rows = img.shape[0]
        cols = img.shape[1]
        return cv2.warpAffine(img, transMatrix[:2, :], (cols, rows))

# draws box from coordinates in list
def draw_box(image, list, color):
    global TM
    for x in range(len(list)):
        box = list[x]
        tx1, ty1 = transform_point(box[0][0], box[0][1], TM)
        tx2, ty2 = transform_point(box[1][0], box[1][1], TM)
        cv2.rectangle(image, (tx1,ty1), (tx2,ty2), color, 2)

# draws already marked regions in image and comparison region if any in frame
def draw_marked_regions(image, list):
    global inspectRegionList, cropimgColor
    imgclone = np.copy(image)
    # draw marked regions
    draw_box(imgclone, list, (0, 0, 255))

    # draw comparison region if any (relevant for runmode=='-t'
    if runmode == '-t':
        draw_box(imgclone, inspectRegionList, cropimgColor)
    return imgclone

# enhance image contrast to make more visible
def enhance_contrast(image, factor):
    imgclone = np.copy(image)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
    lab = cv2.cvtColor(imgclone, cv2.COLOR_BGR2LAB)             # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)                                    # split on 3 different channels
    l2 = clahe.apply(l)                                         # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))                                 # merge channels
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)                 # convert from LAB to BGR

# apply preexisting and current changes to image in view
def alter_image(image, tMatrix, dList):
    # image: must be current contrast image, that way changes to contrast is preserved
    alteredImage = transform_image(image, tMatrix)             # transform image
    alteredImage = draw_marked_regions(alteredImage, dList)    # draw boxes
    return alteredImage

# mark region of concern
def mouse_event(event, x, y, flags, param):
    # grab references to the global variables
    global refPtList, refPt, cropping, img, winInfo, TM, TMList, canvas

    # if the left mouse button is clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        realx, realy = transform_point(x,y,np.linalg.inv(TM))
        refPt = [(realx,realy)]
        cropping = True
        # capture image appearance before marking to be reused during cropping
        canvas = alter_image(img, TM, refPtList)

    # check to see if the left mouse button is released and record the ending
    # (x, y) coordinates and indicate that the cropping operation is finished
    elif event == cv2.EVENT_LBUTTONUP:
        realx, realy = transform_point(x, y, np.linalg.inv(TM))
        refPt.append((realx,realy))
        diagonal = np.linalg.norm(np.asarray(refPt[0])-np.asarray(refPt[1]))
        if diagonal > 10:       # record marking only if diagonal length is above threshold(10)
            refPtList.append(refPt)
        cropping = False

    # keep drawing rectangle while cropping
    if cropping:
        imgclone = np.copy(canvas)
        # show rectangle as it's being drawn
        tx, ty = transform_point(refPt[0][0], refPt[0][1], TM)
        cv2.rectangle(imgclone, (tx, ty), (x, y), (255, 0, 0), 2)
        cv2.imshow(winInfo, imgclone)

    # zoom in or out on region if scroll down or up respectively
    if event == cv2.EVENT_MOUSEWHEEL:
        # for EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling, respectively
        s = 1.5       # scale factor
        # zoom in
        if flags >= 0 and len(TMList)<10:
            TMList.append(TM)
            # translation and scale transformation matrix
            # Notice: T*S*-T is equivalent to -T*S if scale factor(s) = 2
            M = np.float32([[s, 0, x-s*x], [0, s, y-s*y], [0, 0, 1]])
            TM = M.dot(TM)
            altimg = alter_image(img, TM, refPtList)
            cv2.imshow(winInfo, altimg)
        # zoom out
        if flags < 0 and len(TMList)>0:
            TM = TMList.pop()
            altimg = alter_image(img, TM, refPtList)
            cv2.imshow(winInfo, altimg)

# loads all scans, marked or unmarked
def load_all(listofscans, rootdir):
    for i in range(len(listofscans)):
        flaggedframes, flaggedzones = list_flagged_frames(listofscans[i])
        scanposition = str(i+1) + '/' + str(len(listofscans))
        display(rootdir + '/', listofscans[i], scanposition, flaggedframes, flaggedzones, 0)

# loads only marked scans
def load_marked(listofscans, rootdir):
    for i in range(len(listofscans)):
        # only show images that are not yet marked
        if dfw.loc[dfw['ID'] == listofscans[i], 'Marked'].values[0] == 1:
            flaggedframes, flaggedzones = list_flagged_frames(listofscans[i])
            scanposition = str(i+1) + '/' + str(len(listofscans))
            display(rootdir + '/', listofscans[i], scanposition, flaggedframes, flaggedzones, 0)

# default mode: loads only unmarked scans
def load_unmarked(listofscans, rootdir, startIndex):
    for i in range(startIndex, len(listofscans)):
        # only show images that are not yet marked
        cell = dfw.loc[dfw['ID'] == listofscans[i], 'Marked']
        if len(cell) > 0 and cell.values[0] == 0:
            flaggedframes, flaggedzones = list_flagged_frames(listofscans[i])
            scanposition = str(i+1) + '/' + str(len(listofscans))
            display(rootdir + '/', listofscans[i], scanposition, flaggedframes, flaggedzones, 0)
    print('Great! all scans are marked')

# loads starting from the given scan
def load_scan(listofscans, rootdir):
    scanID = input("Enter ID of scan: ")
    notfound = True
    i = 0
    while i<len(listofscans) and notfound:
        if listofscans[i] == scanID:
            notfound = False
            flaggedframes, flaggedzones = list_flagged_frames(listofscans[i])
            scanposition = str(i + 1) + '/' + str(len(listofscans))
            display(rootdir + '/', listofscans[i], scanposition, flaggedframes, flaggedzones, 0)
        i = i+1
    if (notfound): print ("Unable to find Scan")
    else:
        choice = input("Continue to other unmarked scans? Y/N: ")
        if (choice == "Y"):
            load_unmarked(listofscans, rootdir, i)

# loops through all the subdirectories of each scan and builds a list of flagged frames & also creates data frame
def loop_img_dir(rootdir):
    if os.path.exists(rootdir):
        if os.path.isdir(rootdir):
            listofdir = os.listdir(rootdir)
            if runmode == "-a":
                load_all(listofdir, rootdir)
                print ("CONGRATULATIONS!!! All scans have been reviewed and marked!")
            elif runmode == "-m":
                load_marked(listofdir, rootdir)
                print ("All marked scans have been reviewed")
            elif runmode == "-u":
                load_unmarked(listofdir, rootdir, 0)
                print ("CONGRATULATIONS!!! All unmarked scans have been marked!")
            elif runmode == "-s":
                load_scan(listofdir, rootdir)
            else:
                print ("run code without argument or with -a, -m, -u ")
        else:
            print (rootdir+" is not a directory")
    else:
        print (rootdir+" does not exist")

# returns a union list of flagged frames by mapping flagged zones in csv file to frames
def list_flagged_frames(scanID):
    dfsub = dfr.loc[dfr['NoZoneId'] == scanID]
    zones = " "
    uframes = []
    for j, row in dfsub.iterrows():
        uframes = list(set().union(uframes,map_zone_to_frame(row['Zone'])))
        zones = zones+","+str(row['Zone'])
    # frames = sorted(set(map(tuple,frames)), reverse=True) # union all lists
    return uframes, zones # return union set of the list frames, and a string of corresponding zones

# creates a list of all the frames corresponding to a zone
def map_zone_to_frame(zone):
    if zone==1 or zone==2 or zone==11 or zone==13 or zone==15:
        return [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15]    # all except 4
    if zone==3 or zone==4 or zone==12 or zone==14 or zone==16:
        return [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]     # all except 12
    if zone==5:
        return [0,1,2,3,13,14,15]                        # all except 0-4, 12-15
    if zone==6:
        return [0,1,6,7,8,9,10,11,12,13,14,15]          # all except 2-5
    if zone==7:
        return [0,1,2,3,4,5,6,7,8,9,10,15]              # all except  11-4
    if zone==8:
        return [0,1,7,8,9,10,11,12,13,14,15]            # all except 2-6
    if zone==9:
        return [0,1,2,3,5,6,7,8,9,10,11,13,14,15]       # all except 4,12
    if zone==10:
        return [0,1,2,3,4,5,6,7,8,9,15]                 # all except 10-14
    if zone==17:
        return [5,6,7,8,9,10,11]                        # all except 4-12

# locates cropped region in frame and marks it in green box
def identify_cropped_region(originalImg):
    gimg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    gcropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
    w, h = gcropimg.shape[::-1]
    match = cv2.matchTemplate(gimg, gcropimg, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    #cv2.rectangle(enhancedImg, top_left, bottom_right, (0, 255, 0), 2)
    return [top_left, bottom_right]

# show all flagged frames of a scan for marking
def display(scandir, scanid, scanpos, fframes, fzones, start):
    global img, refPtList, winInfo, TMList, TM, inspectRegionList
    print ("----------------------------------------------\nScanID: "+str(scanid)+"\nScan Position: "+str(scanpos))
    numofframes = len(fframes)
    visited = np.zeros(shape=(numofframes), dtype=np.int)
    e = start
    while e<numofframes:
        visited[e] = 1
        frame = 'Frame' + str(fframes[e])
        img = cv2.imread(scandir+scanid+'/'+str(fframes[e])+'.png')
        refPtList = []              # a list of diagonal coordinates of marked regions in a frame
        inspectRegionList = []      # list is only relevant for runmode = -t, otherwise it is always empty
        TMList = []                 # a list of transformation matrix used at different scales
        TM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])    # transformation matrix relevant to zoom feature

        contrasts = []  # list of images with different contrasts
        contrasts.append(img)
        img = enhance_contrast(img, 2.5)

        cv2.destroyWindow(winInfo)
        winInfo = 'Scan:'+str(scanpos)+'  Frame:'+str(fframes[e])+'  Image:'+str(e+1)\
                  +'/'+str(numofframes)+'  Zones:'+str(fzones)
        cv2.namedWindow(winInfo)
        cv2.moveWindow(winInfo, 740, 50) #2740
        cv2.setMouseCallback(winInfo, mouse_event)      # set mouse callback function for window

        # check to see if frame is already marked, if so draw marked regions
        cell = dfw.loc[dfw['ID'] == scanid, frame].values[0]
        if cell != "N/M":
            refPtList = eval(cell)                      # np.array(ast.literal_eval(cell)) doesn't quite work well
        # find region of comparison if image to be inspected is in frame
        if e==start and runmode=='-t':
            inspectRegionList.append(identify_cropped_region(contrasts[0]))
        # if either list is non-empty
        if refPtList or inspectRegionList:
            altimg = draw_marked_regions(img, refPtList)
            cv2.imshow(winInfo, altimg)
        else:
            cv2.imshow(winInfo, img)                    # display the image

        # stay until the 'n' key is pressed
        while True:
            key = cv2.waitKey(1) #& 0xFF                 # wait for a keypress

            # if the 'r' or 'delete' key is pressed, reset the cropping regions
            if key == ord("r") or key%255 == 46:
                del refPtList[:]                        # empty list
                reset_markings(scanid, frame, refPtList)# reset recorded markings of frame
                cv2.imshow(winInfo, img)

            # if the 'backspace'key is pressed undo last marking
            elif key%255 == 8 and len(refPtList)>0:
                del refPtList[len(refPtList)-1]         # delete last list item
                reset_markings(scanid, frame, refPtList)# reset recorded markings of frame
                altimg = draw_marked_regions(img, refPtList)
                cv2.imshow(winInfo, altimg)

            # if the 'n' or '-->' key is pressed, break from the loop and move to the next frame
            elif (key == ord("n") or key%255 == 39):
                if len(str(refPtList)) > 2:                         # don't save if empty, i.e. '[]'
                    record_markings(scanid, frame, refPtList)       # save marked coordinates
                e = (e+1)%numofframes
                break

            # if the 'p' or '<--' key is pressed, break from loop and go back to previous frame
            elif (key == ord("p") or key%255 == 37):
                # you can only go to previous if there is a previous frame in same scan
                if len(str(refPtList)) > 2:                         # don't save if empty, i.e. '[]'
                    record_markings(scanid, frame, refPtList)       # save marked coordinates
                e = e-1 if e>0 else numofframes-1
                break

            # if the 'f' or 'enter' key is pressed, break from loop and go to the next scan
            elif (key == ord("f") or key&255 == 13) and np.sum(visited) == numofframes:
                mark_completed(scanid)                  # scan is Marked only if all frames are reviewed
                e = numofframes + 1
                break

            # if the 'b' or 'arrow up' key is pressed, increase contrast
            elif (key == ord("b") or key%255 == 38) and len(contrasts)<20:
                contrasts.append(img)
                img = enhance_contrast(img, 0.3)
                altimg = alter_image(img, TM, refPtList)
                cv2.imshow(winInfo, altimg)

            # if the 'd' or 'arrow down' key is pressed, reverse contrast effect
            elif (key == ord("d") or key%255 == 40) and len(contrasts)>0:
                img = contrasts.pop()
                altimg = alter_image(img, TM, refPtList)
                cv2.imshow(winInfo, altimg)

            # if the 'e' or 'esc' key is pressed, or window is closed
            # write refPtList, write data frame to csv file and exit the program
            elif key == 27 or key == ord("e") or cv2.getWindowProperty(winInfo,0) < 0:
                print ("***Warning: Terminating Program...")
                if len(str(refPtList)) > 2:                         # don't save if empty, i.e. '[]'
                    record_markings(scanid, frame, refPtList)       # save marked coordinates
                print ("***Warning: current scan will not be marked completed")
                print ("***Message: saving progress...")
                save_progress()
                print ("***Message: session's progress successfully saved")
                cv2.destroyAllWindows()                             # close all open windows
                print ("***Message: Program Terminated")
                sys.exit()

            # if the 'h' or 'space bar' key is pressed, display help message
            elif key == ord("h") or key%255 == 32:
                print ("This program implements a user interface for manually marking suspicious regions " \
                      "in images from the TSA Passenger Screening Data set.\n" \
                      "The program has the following 4 launch modes: \n" \
                      "   -a: launch program to load all scans\n" \
                      "   -m: launch program to load only already marked scans\n" \
                      "   -u OR Default: launch program to load only unmarked scans\n" \
                      "   -s: launch program to load a particular scan and then continue to other scans\n" \
                      "Below are the program's mouse controls\n" \
                      "   mark region: click and drag your mouse over the region to draw a box\n" \
                      "   zoom in/out: use your mousewheel to zoom in or out" \
                      "Below are the program's key controls\n" \
                      "   r/delete: delete all markings done on current image\n" \
                      "   backspace: undo previous(last) marking\n" \
                      "   n/right-arrow: move to next image\n" \
                      "   p/left-arrow: move to previous image\n" \
                      "   f/enter: move forward to next scan when done marking current scan\n" \
                      "   b/up-arrow: enhance contrast (brighten) image\n" \
                      "   d/down-arrow: decrease contrast (darken) image\n" \
                      "   e/esc: forcefully exit the program\n" \
                      "   h/space-bar: display program help information")


# iterates over frames of wrongly predicted cropped regions
def inspect(df, threatScanDir, nothreatScanDir, threatCropDir, nothreatCropDir):
    global cropimg, cropimgColor
    for i, row in df.iterrows():
        filename = row['File_Location']
        realClass = row['Real_Label']
        wrongPred = row['Predicted_Label']

        dash_1_index = filename.find('_')
        scanid = filename[0 : dash_1_index]
        frame = int(filename[dash_1_index + 1 : dash_1_index + 2])
        allFrames = np.arange(16)

        if realClass==1 and wrongPred==0:    # Missed Detection (FP)
            cropimg = cv2.imread(threatCropDir + filename)
            windowName = "Alleged Missed Detection on Frame: " + str(frame)
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 400,400)
            cv2.imshow(windowName, enhance_contrast(cropimg, 2.5))
            cv2.moveWindow(windowName, 1260, 50)
            cropimgColor = (0, 255, 255)
            # Case 1 (Faulty Marking): prediction is correct, cropped image contains no threat
            # Case 2 (Faulty Model): prediction is incorrect, cropped image contains a threat
            flaggedframes, flaggedzones = list_flagged_frames(scanid)
            display(threatScanDir, scanid, i, allFrames, flaggedzones, frame)

        elif realClass==0 and wrongPred==1: # False Alarm (FN)
            cropimg = cv2.imread(nothreatCropDir + filename)
            windowName = "Alleged False Alarm on Frame: " + str(frame)
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 400,400)
            cv2.imshow(windowName, enhance_contrast(cropimg, 2.5))
            cv2.moveWindow(windowName, 1260, 50)
            cropimgColor = (0, 255, 0)  # green
            # Case 1 (Faulty Marking): prediction is correct, cropped image contains a threat
            # Case 2 (Faulty Model): prediction is incorrect, cropped image contains no threat
            display(nothreatScanDir, scanid, i, allFrames, " ", frame)
        cv2.destroyWindow(windowName)



def main():
    '''reads and writes to stage1_labels_1_modified.csv, stage1_labels.txt, stage1_labels_1_marked.csv.
        Implements user interface application for marking images.
    '''
    global runmode, dfr, dfw, markings, readcsvfile, writecsvfile, jsontextfile, winInfo, img, cropping, change

    if len(sys.argv) == 2:
        runmode = sys.argv[1]
    else:
        runmode = "-u"

    readcsvfile = '../../Data/tsa_psc/old_versions/stage1_labels_1_modified.csv'
    dfr = readcsv(readcsvfile)  # pandas data frame
    jsontextfile = '../../Data/tsa_psc/stage1_labels.txt'
    markings = readjson(jsontextfile)   # dictionary in json format

    # Local machine default path: '../../Passenger-Screening-Challenge/Data/aps_images/full_image_threat'
    # scandir = input("Enter path to root directory of scan images: ")    # raw_input
    threat_scans = '../../../datasets/tsa/aps_images/dataset/train_set/'
    nothreat_scans = '../../../Passenger-Screening-Challenge/Data/aps_images/full_image_no_threat/'

    # global variable initializations
    cropping = False    # indicates when marking a region is in progress
    change = False      # indicates whether a write to data frame has occurred since last periodic save
    winInfo = " "       # curates important information about displayed image
    img = cv2.imread('../../Data/tsa_psc/body_zones.png')
    cv2.imshow("Body Zones", img)
    cv2.moveWindow("Body Zones", 50, 50) #2050

    # thread to recursively call save function after every 10 minutes
    periodic_save_thread = threading.Thread(target = periodic_save)
    periodic_save_thread.daemon = True
    periodic_save_thread.start()

    if runmode == '-t':
        imgW, imgH = 112, 112
        image_rootdir = "../../../Passenger-Screening-Challenge/Data/aps_images/"
        nothreat_dir = image_rootdir + str(imgW) + 'x' + str(imgH) + "_cropped_image_no_threat/"
        threat_dir = image_rootdir + str(imgW) + 'x' + str(imgH) + "_cropped_image_threat/"
        model_dir = "../../../Passenger-Screening-Challenge/Data/cnn_model/tsa_cropped_model_13/250/"
        writecsvfile = '../../Data/tsa_psc/stage1_labels_1_marked_combined.csv'
        dfw = readcsv(writecsvfile) # pandas data frame
        csvfile = model_dir + 'pred/incorrect_predictions.csv'
        df = readcsv(csvfile)  # pandas data frame
        inspect(df, threat_scans, nothreat_scans, threat_dir, nothreat_dir)
    else:
        writecsvfile = '../../Data/tsa_psc/old_versions/stage1_labels_1_marked.csv'
        dfw = readcsv(writecsvfile) # pandas data frame
        loop_img_dir(threat_scans)

    save_progress()
    cv2.destroyAllWindows()             # close all open windows
    sys.exit()

main()
