import cv2 as cv
import dlib as db
import numpy as np
import sys
import os
import time
from multiprocessing import Process, Lock, Manager
import threading
from scipy import signal

sys.path.append('../')
from pose_detection import reconstruction as recon
from pose_detection import matcher as match
#from pose_detection import descriptors
from image_processing import transformations as imgtf
from pose_detection import graph_matrix as gmat


class Keypoints3D(object):
    '''
        Match found keypoints in cardinal frames to other frames
    '''
    def __init__(self, imgW, imgH, kernelSize=49, valKptVsFrm=None, step=3, xHalfRange=150, yHalfRange=6,
                 rotAng=22.5, scanRad=6.3, spanTree='max', matcher='corr', descType=None, display=False):
        self.imgW, self.imgH = imgW, imgH
        self.step = step #***
        self.xHalfRange = xHalfRange
        self.kernelSize = kernelSize #***
        #self.xBound, self.yBound = int(kernelSize / 2) + 5, int(kernelSize / 2) + 1
        self.yRange = np.arange(-yHalfRange, yHalfRange + 1, step) # for desired effect eg. [-6, -3, 0, 3, 6] #***
        if yHalfRange % step != 0: print('WARNING: for desired effect, yHalfRange % step == 0')
        self.pointID = 0 # initialized as 0 and will be updated continuously
        self.validKyptsInFrames = valKptVsFrm # shape=(13, 16)
        self.TRACKER_THRESH = 5.0
        self.NO_EDGE_WEIGHT = None
        self.MinST = spanTree == 'min'
        self.MaxST = spanTree == 'max'
        self.INVALID_EDGE_WEIGHT = np.inf if self.MinST else 0

        # other instances
        self.reconObj = recon.OrthRecon(imgW, scanRad, rotAng, step)
        if matcher == 'corr':
            self.matchObj = match.Correlator(imgW, imgH, step, kernelSize)
            self.describe = False
        if matcher == 'desc':
            self.matchObj = match.Descriptor(imgW, imgH, step, kernelSize, descType)
            self.describe = True
            self.descType = descType
            if descType == 'sift': self.vSize = 128
            elif descType == 'surf': self.vSize = 64
            elif descType == 'orb': self.vSize = 32
            else: print('Argument Missing: descriptor type must be specified for Keypoint3D'), sys.exit()

        self.display = display
        if display:
            self.winNames = ['Leftmost_I', 'Left_I', 'Pivot_I', 'Right_I', 'Rightmost_I',
                             'Leftmost_G', 'Left_G', 'Pivot_G', 'Right_G', 'Rightmost_G']
            for i in range(len(self.winNames)):
                x_p = (300 * i) % 1500 - 10 #x_p = (384 * i) % 1920 - 10
                y_p = 410 if i > 4 else -10 #y_p = 510 if i > 4 else -10
                imgtf.create_display_window(self.winNames[i], x_p, y_p, x_size=300, y_size=380)


    def instantiate_recon(self, images, gradmag, gradang, dx, dy, descptr):
        '''
        Instantiate sets of 5 images needed for reconstruction
        :param images:      ndarray containing 5 images
        :param gradients:   ndarray containing 5 image gradients
        :param copy:        whether or not to make a copy of the images
        :return:            nothing is returned
        '''
        llimg, llgrad, llang, lldxy = images[0], gradmag[0], gradang[0], np.stack((dx[0], dy[0]), axis=-1)
        limg, lgrad, lang, ldxy = images[1], gradmag[1], gradang[1], np.stack((dx[1], dy[1]), axis=-1)
        pimg, pgrad, pang, pdxy = images[2], gradmag[2], gradang[2], np.stack((dx[2], dy[2]), axis=-1)
        rimg, rgrad, rang, rdxy = images[3], gradmag[3], gradang[3], np.stack((dx[3], dy[3]), axis=-1)
        rrimg, rrgrad, rrang, rrdxy = images[4], gradmag[4], gradang[4], np.stack((dx[4], dy[4]), axis=-1)
        self.matchObj.set_original_images(pimg, rimg, limg, rrimg, llimg) #***
        self.matchObj.set_gradient_magnitude(pgrad, rgrad, lgrad, rrgrad, llgrad) #***
        self.matchObj.set_gradients_dx_dy(pdxy, rdxy, ldxy, rrdxy, lldxy) #***
        if self.describe: self.matchObj.set_descriptors(descptr[2], descptr[3], descptr[1], descptr[4], descptr[0])


    def instantiate_display(self, images, gradmag, leftmostFid=None):
        '''
        Set images for display purposes
        :param images:      ndarray containing 5 images
        :param gradmag:     ndarray containing 5 image gradients
        :param leftmostFid: frame ID of the leftmost frame
        :return:            nothing is returned
        '''
        llimg, llgrad = images[0], gradmag[0]
        limg, lgrad = images[1], gradmag[1]
        pimg, pgrad = images[2], gradmag[2]
        rimg, rgrad = images[3], gradmag[3]
        rrimg, rrgrad = images[4], gradmag[4]

        self.imgCopy = {'Leftmost_I': np.copy(llimg), 'Left_I': np.copy(limg),
                        'Pivot_I': np.copy(pimg),
                        'Right_I': np.copy(rimg), 'Rightmost_I': np.copy(rrimg),
                        'Leftmost_G': imgtf.cast_to_uint8(llgrad),
                        'Left_G': imgtf.cast_to_uint8(lgrad),
                        'Pivot_G': imgtf.cast_to_uint8(pgrad),
                        'Right_G': imgtf.cast_to_uint8(rgrad), 'Rightmost_G': imgtf.cast_to_uint8(rrgrad)}
        if leftmostFid is not None:
            PT, llf, color = (30, 30), leftmostFid, (255, 255, 255)
            lf, pf, rf, rrf = (llf + 1) % 16, (llf + 2) % 16, (llf + 3) % 16, (llf + 4) % 16
            cv.putText(self.imgCopy['Leftmost_I'], str(llf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Leftmost_G'], str(llf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Left_I'], str(lf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Left_G'], str(lf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Pivot_I'], str(pf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Pivot_G'], str(pf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Right_I'], str(rf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Right_G'], str(rf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Rightmost_I'], str(rrf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)
            cv.putText(self.imgCopy['Rightmost_G'], str(rrf), PT, cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)

        if self.display: imgtf.displayWindows(10, self.winNames, self.imgCopy)


    def setup_reconstruction_from_frame(self, leftmostFid, kFrames, numOfFrames):
        '''
        Choose the chain frames and gradient components to be used for reconstruction
        :param leftmostFid: first frame in the chain of adjacent frames starting from the left
        :param kFrames:     number of frames in the chain
        :param numOfFrames: total number of frames
        :return:            nothing is returned
        '''
        self.reconFids = []
        simages, gradmag, gradang, grad_dx, grad_dy, descptr = [], [], [], [], [], []
        for i in range(kFrames):
            fid = (leftmostFid + i) % numOfFrames
            self.reconFids.append(fid)
            simages.append(self.scnFrmPatch[fid])
            gradmag.append(self.scnGmgPatch[fid])
            gradang.append(self.scnGradang[fid])
            grad_dx.append(self.scnGdxPatch[fid])
            grad_dy.append(self.scnGdyPatch[fid])
            if self.describe: descptr.append(self.scnDescrpt[fid])
        self.instantiate_recon(simages, gradmag, gradang, grad_dx, grad_dy, descptr)

        if self.display:
            simages, gradmag, gradang, grad_dx, grad_dy, descptr = [], [], [], [], [], []
            for i in range(kFrames):
                fid = self.reconFids[i]
                simages.append(self.scanFrames[fid])
                gradmag.append(self.scnGradmag[fid])
                gradang.append(self.scnGradang[fid])
                grad_dx.append(self.scnGrad_dx[fid])
                grad_dy.append(self.scnGrad_dy[fid])
            self.instantiate_display(simages, gradmag, leftmostFid=leftmostFid)


    def progress_display(self, yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt):
        imgtf.mark_point(self.imgCopy['Pivot_I'], (xpv, yM), self.pointID, (255, 255, 255))
        imgtf.mark_point(self.imgCopy['Right_I'], (rxOpt, yM), self.pointID, (255, 255, 255))
        imgtf.mark_point(self.imgCopy['Left_I'], (lxOpt, yM), self.pointID, (255, 255, 255))
        imgtf.mark_point(self.imgCopy['Rightmost_I'], (rrxOpt, yM), self.pointID, (255, 255, 255))
        imgtf.mark_point(self.imgCopy['Leftmost_I'], (llxOpt, yM), self.pointID, (255, 255, 255))

        k = int(self.kernelSize / 2)
        pD = self.imgCopy['Pivot_C'][yM - k: yM + k + 1, xpv - k: xpv + k + 1]
        rD = self.imgCopy['Right_C'][yM - k: yM + k + 1, rxOpt - k: rxOpt + k + 1]
        lD = self.imgCopy['Left_C'][yM - k: yM + k + 1, lxOpt - k: lxOpt + k + 1]
        rrD = self.imgCopy['Rightmost_C'][yM - k: yM + k + 1, rrxOpt - k: rrxOpt + k + 1]
        llD = self.imgCopy['Leftmost_C'][yM - k: yM + k + 1, llxOpt - k: llxOpt + k + 1]
        netD = pD + rD + lD + rrD + llD

        imgtf.draw_grad_vector(self.imgCopy['Pivot_G'], xpv, yM, np.sum(pD[0]), np.sum(pD[1]))
        imgtf.draw_grad_vector(self.imgCopy['Pivot_G'], xpv, yM, np.sum(netD[0]), np.sum(netD[1]), color=[0, 255, 0])
        imgtf.draw_grad_vector(self.imgCopy['Right_G'], rxOpt, yM, np.sum(rD[0]), np.sum(rD[1]))
        imgtf.draw_grad_vector(self.imgCopy['Left_G'], lxOpt, yM, np.sum(lD[0]), np.sum(lD[1]))
        imgtf.draw_grad_vector(self.imgCopy['Rightmost_G'], rrxOpt, yM, np.sum(rrD[0]), np.sum(rrD[1]))
        imgtf.draw_grad_vector(self.imgCopy['Leftmost_G'], llxOpt, yM, np.sum(llD[0]), np.sum(llD[1]))

        imgtf.displayWindows(10, self.winNames, self.imgCopy)


    def center_about_boundaries(self, images, boundary='zero'):
        if np.ndim(images) == 4:
            cnt, hgt, wdt, chn = images.shape
            shape = (cnt, hgt + self.kernelSize, wdt + self.kernelSize, chn)
            imageShape = (hgt + self.kernelSize, wdt + self.kernelSize, chn)
        elif np.ndim(images) == 3:
            cnt, hgt, wdt = images.shape
            shape = (cnt, hgt + self.kernelSize, wdt + self.kernelSize)
            imageShape = (hgt + self.kernelSize, wdt + self.kernelSize)

        paddedImages = np.zeros(shape=shape, dtype=images.dtype)
        for i in range(cnt):
            paddedImages[i] = self.matchObj.add_boundaries(images[i], imageShape, boundary=boundary)
        return paddedImages


    def get_gradient_comps(self, img, k=11, faceMask=None):
        '''
        Computes and returns image gradient
        :param img: colored image
        :param k:   square kernel size
        :return:    image gradient
        '''
        gray = cv.cvtColor(cv.GaussianBlur(img, (k, k), 0), cv.COLOR_BGR2GRAY) #(9, 9)
        mag, ang, dx, dy = imgtf.sobel_gradient(gray) #k=25)
        grad = imgtf.normalize_image(mag)
        if faceMask is not None:
            grad, dx, dy = grad * faceMask, dx * faceMask, dy * faceMask
        return grad, ang, dx, dy


    def get_image_and_gradient(self, filepath):
        img = cv.imread(filepath)
        grad, ang, dx, dy = self.get_gradient_comps(img)
        return img, grad


    def get_images(self, filepath):
        img = cv.imread(filepath)
        mag, ang, dx, dy = self.get_gradient_comps(img)
        return img, mag, ang, dx, dy


    def set_scan_gradient_components(self, faceMasks):
        '''
        Converts original (colored) images to gradients and return array
        :param faceMasks: Array of 16 frames (in order) in a scan, shape=(16, frameH, frameW, 3), dtype=np.uint8
        :return:          array of gradient images of shape=(16, frameH, frameW), dtype=np.float32
        '''
        frames, hgt, wdt, channels = self.scanFrames.shape
        self.scnGradmag = np.zeros(shape=(frames, hgt, wdt), dtype=np.float32)
        self.scnGradang = np.zeros(shape=(frames, hgt, wdt), dtype=np.float32)
        self.scnGrad_dx = np.zeros(shape=(frames, hgt, wdt), dtype=np.float32)
        self.scnGrad_dy = np.zeros(shape=(frames, hgt, wdt), dtype=np.float32)
        for fid in range(frames):
            mag, ang, dx, dy = self.get_gradient_comps(self.scanFrames[fid], faceMask=faceMasks[fid])
            self.scnGradmag[fid] = mag
            self.scnGradang[fid] = ang
            self.scnGrad_dx[fid] = dx
            self.scnGrad_dy[fid] = dy
        self.scnGmgPatch = self.center_about_boundaries(self.scnGradmag, boundary='zero')
        self.scnGdxPatch = self.center_about_boundaries(self.scnGrad_dx, boundary='zero')
        self.scnGdyPatch = self.center_about_boundaries(self.scnGrad_dy, boundary='zero')


    def process_descriptors(self, fidStart, nSize, sharedDict):
        # have to create object for each process else code won't run on windows
        descObj = match.Descriptor(self.imgW, self.imgH, self.step, self.kernelSize, self.descType)
        for fid in range(fidStart, fidStart + nSize):
            frameDesciptor = descObj.describe_image(self.scanFrames[fid], self.vSize)
            with self.lock:
                sharedDict[fid] = frameDesciptor


    def set_image_descriptors(self, nFrms=16, frmsPerProcess=4):
        manager = Manager()
        sharedDict, self.lock = manager.dict(), Lock()
        processes, descriptorList = [], []
        for fidS in range(0, nFrms, frmsPerProcess):
            p = Process(target=self.process_descriptors, args=(fidS, frmsPerProcess, sharedDict))
            processes.append(p), p.start()
        for p in processes: p.join()
        for fid in range(nFrms): descriptorList.append(sharedDict[fid])
        self.scnDescrpt = np.stack(descriptorList)
        #print(self.scnDescrpt)
        #sys.exit()


    # Projection Functions
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def reconstruct_from_right(self, pvKpt, pvtx, xROI, yCands):
        xCands = np.arange(xROI[0], xROI[1], self.step)
        shiftx = not self.describe
        rxCands, lxCands, rrxCands, llxCands = self.reconObj.recon_from_right(pvKpt, yCands, xCands, shiftx)
        bestScore, bestConfig = self.matchObj.get_best_match(pvtx, yCands, rxCands, rrxCands, lxCands, llxCands) #***

        if self.display:
            yM, xpv, rxM, lxM, rrxM, llxM = bestConfig
            imgtf.mark_point(self.imgCopy['Pivot_I'], (xpv, yM), self.pointID, (0, 255, 0))
            imgtf.mark_point(self.imgCopy['Right_I'], (rxM, yM), self.pointID, (0, 255, 0))
            imgtf.mark_point(self.imgCopy['Left_I'], (lxM, yM), self.pointID, (0, 255, 0))
            imgtf.mark_point(self.imgCopy['Rightmost_I'], (rrxM, yM), self.pointID, (0, 255, 0))
            imgtf.mark_point(self.imgCopy['Leftmost_I'], (llxM, yM), self.pointID, (0, 255, 0))
        return bestScore, bestConfig

    def reconstruct_from_left(self, pvKpt, pvtx, xROI, yCands):
        xCands = np.arange(xROI[0], xROI[1], self.step)
        shiftx = not self.describe
        rxCands, lxCands, rrxCands, llxCands = self.reconObj.recon_from_left(pvKpt, yCands, xCands, shiftx)
        bestScore, bestConfig = self.matchObj.get_best_match(pvtx, yCands, rxCands, rrxCands, lxCands, llxCands) #***

        if self.display:
            yM, xpv, rxM, lxM, rrxM, llxM = bestConfig
            imgtf.mark_point(self.imgCopy['Pivot_I'], (pvtx, yM), self.pointID, (255, 0, 0))
            imgtf.mark_point(self.imgCopy['Right_I'], (rxM, yM), self.pointID, (255, 0, 0))
            imgtf.mark_point(self.imgCopy['Left_I'], (lxM, yM), self.pointID, (255, 0, 0))
            imgtf.mark_point(self.imgCopy['Rightmost_I'], (rrxM, yM), self.pointID, (255, 0, 0))
            imgtf.mark_point(self.imgCopy['Leftmost_I'], (llxM, yM), self.pointID, (255, 0, 0))

        return bestScore, bestConfig


    def reconstruct_from_best_descriptor_candidate(self, pvKpt, pivotPt):
        assert (self.describe == True)
        xpv = pivotPt[0]
        xRightRegion = self.xRegion[0]
        yCands = self.yRange + self.matchObj.y_nearest_multiple(pivotPt[1])
        xStart = self.matchObj.x_nearest_multiple(xRightRegion[0])
        xEnd = self.matchObj.x_nearest_multiple(xRightRegion[1]) + 1
        rScore, rConfig = self.reconstruct_from_right(pvKpt, xpv, [xStart, xEnd], yCands)
        yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt = rConfig

        if self.display: self.progress_display(yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt)
        return yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt


    def reconstruct_from_best_correlation_candidate(self, pvKpt, pivotPt):
        assert (self.describe == False)
        xpv = pivotPt[0]
        yCands = self.yRange + pivotPt[1]
        rScore, rConfig = self.reconstruct_from_right(pvKpt, xpv, self.xRegion[0], yCands)
        lScore, lConfig = self.reconstruct_from_left(pvKpt, xpv, self.xRegion[1], yCands)
        yM, xpv, rxM, lxM, rrxM, llxM = rConfig if rScore > lScore else lConfig #***
        # get local optimum in neighborhood
        rxOpt = self.matchObj.right_1storder_local_opt_match(yM, xpv, rxM)
        lxOpt = self.matchObj.left_1storder_local_opt_match(yM, xpv, lxM)
        rrxOpt = self.matchObj.right_2ndorder_local_opt_match(yM, xpv, rxM, rrxM)
        llxOpt = self.matchObj.left_2ndorder_local_opt_match(yM, xpv, lxM, llxM)

        if self.display: self.progress_display(yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt)
        return yM, xpv, rxOpt, lxOpt, rrxOpt, llxOpt


    def track_object(self, trackImgFid, tracker, xpv, displayKey):
        # Update the tracker and request information about the quality of the tracking update
        trackingQuality = tracker.update(self.scanTrack[trackImgFid])
        useDefaultRange = True
        # If the tracking quality is good enough, determine the updated
        # position of the tracked region and draw the rectangle
        # Note. self.xMargin was replaced with self.xBound on 03/18/2019
        if trackingQuality > self.TRACKER_THRESH:
            tracked_position = tracker.get_position()
            x = int(tracked_position.left()) # x could be negative
            y = int(tracked_position.top()) # y could be negative
            w = int(tracked_position.width())
            h = int(tracked_position.height())
            assert (w >= 0 and h >= 0)
            # what if (x + w)==0, this is possible if tracker threshold is set too low
            if x + w > 0 and y + h > 0: # todo: use size threshold other than 0
                useDefaultRange = False
                if self.display: cv.rectangle(self.imgCopy[displayKey], (x, y), (x + w, y + h), (0, 0, 0), thickness=2)
        if useDefaultRange:
            x = xpv - self.xHalfRange
            w = 2 * self.xHalfRange

        return (max(0, x), min(self.imgW, x + w + 1))


    def project_point_to_5_views(self, x, y, halfSz=50):
        imgpt = np.array([x, y])
        p_kpt = np.array([self.reconObj.prep_x(x), y, 0])
        self.pointID += 1
        if self.display:
            imgtf.mark_point(self.imgCopy['Pivot_I'], imgpt, self.pointID)
            cv.imshow('Pivot_I', self.imgCopy['Pivot_I'])

        # Create the tracker for tracking region
        assert (isinstance(x, (int, np.int32)) and isinstance(y, (int, np.int32)) and isinstance(halfSz, int))
        self.xRegion = []
        r_tracker = db.correlation_tracker()
        l_tracker = db.correlation_tracker()
        xDiagA, yDiagA = int(max(0, x - halfSz)), int(max(0, y - halfSz))
        xDiagB, yDiagB = int(min(x + halfSz, self.imgW)), int(min(y + halfSz, self.imgH))
        # todo: 1. try original image for tracking. 2. use clean image not imgCopy that is altered
        llfid, lfid, pvfid, rfid, rrfid = self.reconFids
        r_tracker.start_track(self.scanTrack[pvfid], db.rectangle(xDiagA, yDiagA, xDiagB, yDiagB))
        l_tracker.start_track(self.scanTrack[pvfid], db.rectangle(xDiagA, yDiagA, xDiagB, yDiagB))
        if self.display:
            cv.rectangle(self.imgCopy['Pivot_I'], (xDiagA, yDiagA), (xDiagB, yDiagB), (0, 0, 0), thickness=2)
        self.xRegion.append(self.track_object(rfid, r_tracker, x, 'Right_I'))
        self.xRegion.append(self.track_object(lfid, l_tracker, x, 'Left_I'))

        if self.describe: return self.reconstruct_from_best_descriptor_candidate(p_kpt, imgpt)
        return self.reconstruct_from_best_correlation_candidate(p_kpt, imgpt)


    def get_tags(self, k, mode):
        '''
        returns a list of node tags given mode
        :param k:       number of pixel points that is reconstructed per hpe keypoints
        :param mode:    mode of nodes: 'exit' or 'enter' nodes
        :return:        list of node tags matching mode
        '''
        # tag 0 represents hpe_kpt, tags 1 to k represents reconstructed points, 1:start, kC: middle/origin, k:end
        tags, kC = [0], (k + 1) // 2
        if mode == 'exit':
            start, end = kC + 1, k + 1
        else: # mode == 'enter'
            start, end = 1, kC
        for i in range(start, end):
            tags.append(i)
        return tags


    def next_node_in_group(self, kptID, groupSIN, groupEIN, tag):
        '''
        Find and return the first empty node index within group (ie nodes or keypoints belonging to same frame)
            code is modified to reserve the first index within group for the Original Parent HPE node
        :param kptID:       hpe keypoint ID
        :param groupSIN:    group nodes' start index
        :param groupEIN:    group nodes' end index
        :param tag:         The tag of the node {0, ..,5}
        :return:            new node index in graph
        '''
        startIndex = groupSIN if tag == 0 else groupSIN + 1
        emptyIndex = np.argmin(self.graphNodesData[kptID, startIndex: groupEIN, 0])
        newNodeInd = startIndex + emptyIndex
        assert (self.graphNodesData[kptID, newNodeInd, 0] == -1)
        return newNodeInd


    def corr_pair_matching_score(self, pt1, pt2, adjacentFrameIDPair):
        '''
        Compare two keypoints region and return their matching score (correlation based matching)
        :param pt1: (x, y) pixel point of the first keypoint
        :param pt2: (x, y) pixel point of the second keypoint
        :param adjacentFrameIDPair: (fidA, fidB) frame index of both points
        :return:    computed correlation between keypoint patches
        '''
        xA, yA = pt1
        xB, yB = pt2
        assert (0 <= xA <= self.imgW and 0 <= yA <= self.imgH)
        assert (0 <= xB <= self.imgW and 0 <= yB <= self.imgH)
        fidA, fidB = adjacentFrameIDPair
        k = self.matchObj.kernelHalf
        xA, xB, yA, yB = np.array([xA, xB, yA, yB]) + k # apply boundary shift (+k)
        patchA = self.scnGmgPatch[fidA] #***
        patchB = self.scnGmgPatch[fidB] #***
        dxdyA = np.stack((self.scnGdxPatch[fidA], self.scnGdyPatch[fidA]), axis=-1)
        dxdyB = np.stack((self.scnGdxPatch[fidB], self.scnGdyPatch[fidB]), axis=-1)
        iPatchA = patchA[yA - k: yA + k + 1, xA - k: xA + k + 1]
        iPatchB = patchB[yB - k: yB + k + 1, xB - k: xB + k + 1]
        dPatchA = dxdyA[yA - k: yA + k + 1, xA - k: xA + k + 1]
        dPatchB = dxdyB[yB - k: yB + k + 1, xB - k: xB + k + 1]
        return self.matchObj.corr_pair_match_score(iPatchA, iPatchB, dPatchA, dPatchB) #***


    def desc_pair_matching_score(self, pt1, pt2, adjacentFrameIDPair):
        '''
        Compare two keypoints region and return their matching score (descriptor based matching)
        :param pt1: (x, y) pixel point of the first keypoint
        :param pt2: (x, y) pixel point of the second keypoint
        :param adjacentFrameIDPair: (fidA, fidB) frame index of both points
        :return:    computed desciptor distance between keypoints (float or int)
        '''
        xA, yA = pt1
        xB, yB = pt2
        assert (0 <= xA <= self.imgW and 0 <= yA <= self.imgH)
        assert (0 <= xB <= self.imgW and 0 <= yB <= self.imgH)
        fidA, fidB = adjacentFrameIDPair
        # todo: revise mapping to reflect boundary approach change to code
        xinA, xinB = self.matchObj.xpt_to_desc_xindex(xA), self.matchObj.xpt_to_desc_xindex(xB)
        yinA, yinB = self.matchObj.ypt_to_desc_yindex(yA), self.matchObj.ypt_to_desc_yindex(yB)
        #print('({:>3}, {:>3}), ({:>3}, {:>3})'.format(xA, yA, xB, yB))
        dprA = self.scnDescrpt[fidA, yinA, xinA]
        dprB = self.scnDescrpt[fidB, yinB, xinB]
        return self.matchObj.desc_pair_match_score(dprA, dprB)


    def reconstruct(self, ptOrigin, confidence, validKpt, invalRecon):
        '''
        Orthographic projection and reconstruction of keypoint on 5 adjacent frames
        :param pvFid:       The frame ID from which reconstruction is done
        :param ptOrigin:    pixel keypoint: (x, y) to reconstruct from pivot (center) image
        :param confidence:  confidence score from hpe of parent keypoint (ptOrigin)
        :param validKpt:    boolean indicating whether ptOrigin is from a valid keypoint
        :param invalRecon:  boolean, True: also perform reconstruction of invalid keypoint, otherwise don't
        :return:            list of tuple of tuple with 0:pt (x, y), 1: (confidence, validity)
        '''
        x, y = ptOrigin
        if invalRecon or validKpt:
            y_, x_, rx, lx, rrx, llx = self.project_point_to_5_views(x, y)
            ll = ((llx, y_), (confidence, validKpt))
            l  = ((lx,  y_), (confidence, validKpt))
            pv = ((x_,  y_), (confidence, validKpt))
            r  = ((rx,  y_), (confidence, validKpt))
            rr = ((rrx, y_), (confidence, validKpt))

        else:
            # Set validity to False to mark the node originating from an invalid keypoint or
            # reconstructed from an invalid keypoint. Hence, identifying node as part of an invalid
            # chain. This will be used later to identify node when node it is being added to graph.
            assert (not(validKpt))
            ll = ((x, y), (0, validKpt))
            l  = ((x, y), (0, validKpt))
            pv = ((x, y), (0, validKpt))
            r  = ((x, y), (0, validKpt))
            rr = ((x, y), (0, validKpt))

        return [ll, l, pv, r, rr]


    def compute_edge_weight(self, matchScore, confidenceA, confidenceB):
        '''
        Compute the edge weight according to formula
        :param matchScore:  The correlation or match score between keypoints (nodes)
        :param confidenceA: The confidence of the keypoint (A) from hpe, innate or passed from parent
        :param confidenceB: The confidence of the keypoint (B) from hpe, innate or passed from parent
        :return:            Computed weight of the directed edge (float)
        '''
        if self.describe: # descriptor (opt -> min distance)
            weight = matchScore / (confidenceA + confidenceB)
        else: # correlation (opt -> max score)
            weight = matchScore * (confidenceA + confidenceB) # matchScore * confidenceA * confidenceB
        return weight


    def add_edge_to_graph(self, kptID, nodeA, nodeB, addInv):
        '''
        Compute edge weight and add directed edge to graph
        :param kptID:   hpe keypoint ID
        :param nodeA:   node index for origin (A) of directed edge
        :param nodeB:   node index for destination (B) of directed edge
        :param addInv:  boolean, True: include active edge (weight > 0) connecting to invalid node
        :return:        No return
        '''
        # extract adjacent 2 frames
        xA, yA, tagA, fidA = self.graphNodesData[kptID, nodeA]
        xB, yB, tagB, fidB = self.graphNodesData[kptID, nodeB]
        if (tagA < 0 or tagB < 0) and not(addInv):
            # nullify the effect of edge connecting to node stemming from an invalid keypoint
            # by setting edge weight to a constant (infinity for MinST, 0 for MaxST)
            #  hence making it least likely to be chosen during optimization
            self.gDirectedEdges[kptID, nodeA, nodeB] = self.INVALID_EDGE_WEIGHT # ie. edge nodeA -> nodeB
        else:
            confA = self.nodeConfidence[kptID, nodeA]
            confB = self.nodeConfidence[kptID, nodeB]
            if self.describe: pairScore = self.desc_pair_matching_score((xA, yA), (xB, yB), (fidA, fidB))
            else: pairScore = self.corr_pair_matching_score((xA, yA), (xB, yB), (fidA, fidB))
            edgeWeight = self.compute_edge_weight(pairScore, confA, confB)
            #print('{:>2}: {}\t{}\t{}\t{}'.format(kptID, confA, confB, pairScore, edgeWeight))
            self.gDirectedEdges[kptID, nodeA, nodeB] = edgeWeight # ie. edge nodeA -> nodeB


    def add_node_to_graph(self, kptID, frmID, ptCoord, metaData, tag, grpCands):
        '''
        Add a new node to the graph (0:x, 1:y, 2:tag, 3:group) and node's confidence then return the index
            tag of node is recomputed to represent invalid keypoints
        :param kptID:       hpe keypoint ID
        :param frmID:       TSA frame # or ID which is also the group ID
        :param ptCoord:     (x, y) pixel coordinate of the keypoint in frame
        :param metaData:    (confidence, validity) of keypoint passed from hpe used in reconstruction
        :param tag:         The tag of the node {0, ..,5}
        :param grpCands:    The number of keypoint candidates per group
        :return:            The added node index (or ID) in the graph
        '''
        confidence, valid = metaData
        nid = self.next_node_in_group(kptID, grpCands * frmID, grpCands * (frmID + 1), tag)
        nodeTag = tag if valid else -tag # negative tag if invalid kpt and not original hpe node
        assert (0 <= ptCoord[0] <= self.imgW and 0 <= ptCoord[1] <= self.imgH)
        self.graphNodesData[kptID, nid] = [ptCoord[0], ptCoord[1], nodeTag, frmID]
        self.nodeConfidence[kptID, nid] = confidence
        return nid


    def kptsrecon_graphbuild(self, xPtCoord, yPtCoord, kyPtConf, faceMasks, k, candsPerGrp, nKpts, nFrms, recInv):
        '''
        Reconstruct additional (n=5) candidate keypoints from hpe found keypoints
        :param xPtCoord:    x-coord of all 13 found keypoints in frames, shape=(13, 16), dtype=np.int32
        :param yPtCoord:    y-coord of all 13 found keypoints in frames, shape=(13, 16), dtype=np.int32
        :param kyPtConf:    confidence of all 13 keypoints in 16 frames, shape=(13, 16), dtype=np.float32
        :param faceMasks:   ndarray of ones and zeros indicating face mask region
        :param k:           number of pixel points to reconstruct per hpe keypoints
        :param candsPerGrp: number of candidates (reconstructed & original) in each group
        :param nKpts:       number of keypoints
        :param nFrms:       number of frames
        :param recInv:      whether or not to reconstruct invalid keypoints and weight edges as normal
        :return:            nothing
        '''
        kCid = (k - 1) // 2  # center of k, should be 2 if k=5
        assert (k % 2 == 1 and kCid == 2)
        exitNodeTag = self.get_tags(k, mode='exit') # [0, 4, 5]     # 0:hpe_kpt, 5:end_of_chain
        enterNodeTag = self.get_tags(k, mode='enter') # [0, 1, 2]    # 0:hpe_kpt, 1:start_of_chain
        nodesPerKpt = candsPerGrp * nFrms # keypointCandidatesPerFrame * nFrms
        self.set_scan_gradient_components(faceMasks=faceMasks)
        self.scanTrack = imgtf.cast_to_uint8(np.copy(self.scnGradmag), colored=False)
        if self.describe: self.set_image_descriptors()
        self.graphNodesData = np.full(shape=(nKpts, nodesPerKpt, 4), fill_value=-1, dtype=np.int32)
        self.nodeConfidence = np.zeros(shape=(nKpts, nodesPerKpt), dtype=np.float32)
        self.gDirectedEdges = np.full(shape=(nKpts, nodesPerKpt, nodesPerKpt), fill_value=np.inf, dtype=np.float32)

        for fid in range(nFrms):
            # instantiate reconstruction images needed. ie. 5 frames/gradient components
            leftmostFID = (fid - kCid) % nFrms
            self.setup_reconstruction_from_frame(leftmostFID, k, nFrms)

            for kpt in range(nKpts):
                # Add original hpe keypoints to graphNodes and reconstruct candidate points
                pixelCoord, conf = (xPtCoord[kpt, fid], yPtCoord[kpt, fid]), kyPtConf[kpt, fid]
                kptVal = self.validKyptsInFrames[kpt, fid]
                self.add_node_to_graph(kpt, fid, pixelCoord, (conf, kptVal), 0, candsPerGrp)
                # reconLoToT has 5 adjacent frames. shape=(5, 2, 2). 5:list, 2:tuple, 2:tuple
                reconLoToT = self.reconstruct(pixelCoord, conf, kptVal, recInv)
                # Add reconstructed candidate points to graphNodes and corresponding chain edges to graphEdges
                nodeIndex = []
                pixelCoordA, metadataA = reconLoToT[0]
                nodeIndex.append(self.add_node_to_graph(kpt, leftmostFID, pixelCoordA, metadataA, 1, candsPerGrp))

                for c in range(1, len(reconLoToT)):
                    cidA, cidB = c - 1, c
                    fidA = (leftmostFID + cidA) % nFrms
                    fidB = (leftmostFID + cidB) % nFrms
                    assert ((fidB - fidA) == 1 or -14)
                    pixelCoordB, metadataB = reconLoToT[cidB]
                    nodeIndex.append(self.add_node_to_graph(kpt, fidB, pixelCoordB, metadataB, cidB + 1, candsPerGrp))
                    nodeA, nodeB = nodeIndex[cidA], nodeIndex[cidB]
                    self.add_edge_to_graph(kpt, nodeA, nodeB, recInv)

        # By now all nodes and some edges (ie. all chain edges) have been added to graphNodes
        # so add remaining 'allowed' connecting edges. Each edge connects two nodes in adjacent groups
        # 1) add edges from root node (original hpe node) in each group to all nodes in next group
        # 2) add edges from last node in one chain to the 1st and 2nd nodes in another chain
        # 3) add edges from second-to-last node in chain to 1st and 2nd nodes in another chain
        for kpt in range(nKpts):
            for fidA in range(nFrms):
                sinA = candsPerGrp * fidA
                einA = candsPerGrp * (fidA + 1)
                for nidA in range(sinA, einA):
                    if abs(self.graphNodesData[kpt, nidA, 2]) in exitNodeTag:
                        fidB = (fidA + 1) % nFrms
                        sinB = candsPerGrp * fidB
                        einB = candsPerGrp * (fidB + 1)
                        for nidB in range(sinB, einB):
                            if abs(self.graphNodesData[kpt, nidB, 2]) in enterNodeTag:
                                self.add_edge_to_graph(kpt, nidA, nidB, recInv)


    def optimal_graph_path(self, candsPerGrp, nKpts, nFrms, displayPath):
        # Find optimal configuration for each keypoint
        optKpts = np.zeros(shape=(nKpts, nFrms, 2), dtype=np.int32)
        optConf = np.zeros(shape=(nKpts, nFrms), dtype=np.float32)
        activeGraphEdges = self.gDirectedEdges != np.inf
        self.gDirectedEdges = np.where(activeGraphEdges, self.gDirectedEdges, 0)
        for kpt in range(nKpts):
            if self.describe:
                optPath, distance = gmat.min_weight_cycle(self.gDirectedEdges[kpt], activeGraphEdges[kpt], candsPerGrp)
            else:
                optPath, distance = gmat.max_weight_cycle(self.gDirectedEdges[kpt], activeGraphEdges[kpt], candsPerGrp)
            assert (len(optPath) == nFrms)
            assert (np.sum(self.graphNodesData[kpt, optPath, 3]) == 120) # sum of 0 to 15
            for nid in optPath:
                xpt, ypt, tag, fid = self.graphNodesData[kpt, nid]
                optKpts[kpt, fid] = [xpt, ypt]
                optConf[kpt, fid] = self.nodeConfidence[kpt, nid]

            if displayPath:
                gui = gmat.visualize_graph(self.graphNodesData[kpt, :, 2], self.gDirectedEdges[kpt],
                                           activeGraphEdges[kpt], kptID=kpt)
                imgtf.displayWindow(gui, 'Graph GUI')
                optgui = gmat.visualize_opt_path(gui, optPath, self.graphNodesData[kpt, :, 2])
                imgtf.displayWindow(optgui, 'Graph GUI')
        return optKpts, optConf


    def recon_from_hpe_kpts(self, scanImgs, xPtCoord, yPtCoord, kyPtConf, faceMasks,
                            k=5, nKpts=13, nFrms=16, recInv=False, displayPath=False):
        '''
        :param scanImgs:    Array of 16 frames (in order) in a scan, shape=(16, frameH, frameW, 3), dtype=np.uint8
        :param xPtCoord:    x-coord of all 13 found keypoints in frames, shape=(13, 16), dtype=np.int32
        :param yPtCoord:    y-coord of all 13 found keypoints in frames, shape=(13, 16), dtype=np.int32
        :param kyPtConf:    confidence of all 13 keypoints in 16 frames, shape=(13, 16), dtype=np.float32
        :param faceMasks:   ndarray of ones and zeros indicating face mask region
        :param k:           number of pixel points to reconstruct per hpe keypoints
        :param nKpts:       number of keypoints
        :param nFrms:       number of frames
        :param recInv:      whether or not to reconstruct invalid keypoints and weight edges as normal
        :return:            optimal reconstructed keypoint configurations for frames in scan
        '''
        assert (xPtCoord.dtype == np.int32 and yPtCoord.dtype == np.int32)
        # Note: xPtCoord and yPtCoord should contain coordinates >= 0 and <= scanImgss' frame's wdt and hgt
        # however, the coordinates in xPtCoord and yPtCoord may be less-than min bound for x-Axis / y-Axis
        # or greater-than the max bound for x-Axis / y-Axis. In such a case the restrain methods in matcher.py
        # are called. Note that this implementation is a compromise and isn't perfect because all pixel pts
        # (regardless of the slight difference) that fail similar boundary check will always result in the
        # same descriptor or correlation measure. Hence, reconstruction preference is slightly flawed
        self.scanFrames = scanImgs
        self.scnFrmPatch = self.center_about_boundaries(scanImgs, boundary='mean')
        candsPerGrp = k + 1
        self.kptsrecon_graphbuild(xPtCoord, yPtCoord, kyPtConf, faceMasks, k, candsPerGrp, nKpts, nFrms, recInv)
        return self.optimal_graph_path(candsPerGrp, nKpts, nFrms, displayPath)


    def pivotwindow_mouseevent(self, event, x, y, flags, param):
        # if the left mouse button is clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv.EVENT_LBUTTONDOWN:
            self.project_point_to_5_views(x, y)



# STATIC METHODS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
def test(kyptObj):
    scandir = '../../Data/tsa_psc/aps_scan_samples/0a27d19c6ec397661b09f7d5998e0b14'
    cv.setMouseCallback('Pivot_I', kyptObj.pivotwindow_mouseevent) # set mouse callback function for window
    for i in range(16):
        fll, fl, fi, fr, frr = (i - 2) % 16, (i - 1) % 16, i % 16, (i + 1) % 16, (i + 2) % 16
        #print('Leftmost: {}, Left: {}, Pivot: {}, Right: {}, Rightmost: {}'.format(fll, fl, fi, fr, frr))
        llimg, llgrad, llang, lldx, lldy = kyptObj.get_images(os.path.join(scandir, str(fll) + '.png'))
        limg, lgrad, lang, ldx, ldy = kyptObj.get_images(os.path.join(scandir, str(fl) + '.png'))
        pimg, pgrad, pang, pdx, pdy = kyptObj.get_images(os.path.join(scandir, str(fi) + '.png'))
        rimg, rgrad, rang, rdx, rdy = kyptObj.get_images(os.path.join(scandir, str(fr) + '.png'))
        rrimg, rrgrad, rrang, rrdx, rrdy = kyptObj.get_images(os.path.join(scandir, str(frr) + '.png'))
        images = (llimg, limg, pimg, rimg, rrimg)
        gradmag = (llgrad, lgrad, pgrad, rgrad, rrgrad)
        angles = (llang, lang, pang, rang, rrang)
        gradx = (lldx, ldx, pdx, rdx, rrdx)
        grady = (lldy, ldy, pdy, rdy, rrdy)
        kyptObj.instantiate_recon(images, gradmag, angles, gradx, grady, leftmostFid=fll)

def chng(hpe, index):
    pointID = index
    nimg = imgs[index]
    indicies = np.unravel_index(np.argmax(nimg), nimg.shape)
    x, y = hpe.decipher_cvdnn_pt((indicies[1], indicies[0]), 32, 42, imgWdt=240, imgHgt=315)
    x, y = int(x), int(y)
    nimg = imgtf.prob_to_image(nimg)
    cimg = cv.applyColorMap(nimg.astype(np.uint8), cv.COLORMAP_HOT)
    joint = cv.addWeighted(cimg, 0.8, rimg, 0.5, 0)
    joint = cv.resize(joint, (240, 315), interpolation=cv.INTER_AREA)
    cv.circle(joint, (x, y), 8, (255, 255, 255), 1)
    cv.putText(joint, str(pointID), (x + 11, y + 5), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(joint, str(pointID), (x + 11, y + 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv.LINE_AA)
    return joint

def hpaf(tuple):
    Ix, Iy = imgs[tuple[0]], imgs[tuple[1]]
    mag, ang = cv.cartToPolar(Ix, Iy)
    mag = imgtf.prob_to_image(mag)
    cimg = cv.applyColorMap(mag.astype(np.uint8), cv.COLORMAP_HOT)
    joint = cv.addWeighted(cimg, 0.8, rimg, 0.5, 0)
    return joint

def define_windows(kyptObj, nameList):
    kyptObj.winNames = nameList
    kyptObj.winNames.sort()
    cv.destroyAllWindows()
    for i in range(len(kyptObj.winNames)):
        x_p = 3830 + (i % 7) * 275
        y_p = int(i / 7) * 345
        imgtf.create_display_window(kyptObj.winNames[i], x_p, y_p, x_size=275, y_size=310)

def coco_caffee(kyptObj):
    global imgs, rimg
    from pose_detection import coco_hpe
    hpe = coco_hpe.CocoHPE()
    scandir = '../../Data/tsa_psc/aps_scan_samples/0a27d19c6ec397661b09f7d5998e0b14'

    for i in range(16):
        isSide = True if i is 4 or 12 else False
        img = cv.imread(os.path.join(scandir, str(i) + '.png'))
        img = img[:, 5:507, :]
        rimg = cv.resize(img, (32, 42), interpolation=cv.INTER_AREA)
        oimg = imgtf.separate_foreground(img, side=isSide)
        imgs = hpe.feed_to_pose_nn(oimg)[0]
        kyptObj.imgCopy = {}

        for s in range(2):
            if s == 0:
                keys = list(hpe.BODY_PART_TO_INDEX.keys())
                for key in keys:
                    kyptObj.imgCopy[key] = chng(hpe, hpe.BODY_PART_TO_INDEX[key])
            else: # s == 1
                keys = list(hpe.COCO_HEATMAP_PAFVEC.keys())
                for key in keys:
                    kyptObj.imgCopy[key] = hpaf(hpe.COCO_HEATMAP_PAFVEC[key])

            kyptObj.imgCopy["zOriginal"] = img
            kyptObj.imgCopy["zOverlay"] = oimg
            keys.append("zOriginal")
            keys.append("zOverlay")
            define_windows(kyptObj, keys)
            imgtf.displayWindows(21, kyptObj.winNames, kyptObj.imgCopy)
            kyptObj.imgCopy.clear()
            cv.destroyAllWindows()


if __name__ == "__main__":
    obj3D = Keypoints3D(512, 660, 49, display=True)
    test(obj3D)
    #coco_caffee(obj3D)
