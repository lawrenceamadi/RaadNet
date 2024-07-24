import numpy as np
import cv2 as cv
import sys


class PointMatching:
    '''
        Super class initializes keypoint matching global variables
        and implements common methods needed in correlation and descriptor sub classes
    '''
    def __init__(self, img_w, img_h, x_step, kernel):
        self.imgW = img_w
        self.imgH = img_h
        self.xStep = x_step
        self.pMin = 0
        #self.xMinBound = x_bound
        #self.xMaxBound = img_w - x_bound
        #self.yMinBound = y_bound
        #self.yMaxBound = img_h - y_bound
        self.kernelSize = kernel
        self.kernelHalf = int(kernel / 2)

    def add_boundaries(self, img, newShape, boundary='zero'):
        if boundary == 'zero': boundValue = 0
        elif boundary == 'mean': boundValue = np.mean(img)
        shift = self.kernelHalf
        hgt, wdt = img.shape[0], img.shape[1]
        newImg = np.full(shape=newShape, fill_value=boundValue, dtype=img.dtype)
        newImg[shift: hgt + shift, shift: wdt + shift] = img
        return newImg

    def set_original_images(self, pImg, rImg, lImg, rrImg, llImg):
        self.pimg = pImg
        self.rimg = rImg
        self.limg = lImg
        self.rrimg = rrImg
        self.llimg = llImg

    def set_gradient_magnitude(self, pGrad, rGrad, lGrad, rrGrad, llGrad):
        self.pgrad = pGrad
        self.rgrad = rGrad
        self.lgrad = lGrad
        self.rrgrad = rrGrad
        self.llgrad = llGrad

    def set_gradients_dx_dy(self, pdxdy, rdxdy, ldxdy, rrdxdy, lldxdy):
        # last axis of resulting arrays contains 2 units: dx, dy
        self.pdxdy = pdxdy
        self.rdxdy = rdxdy
        self.ldxdy = ldxdy
        self.rrdxdy = rrdxdy
        self.lldxdy = lldxdy

    def set_gradient_angles(self, pAng, rAng, lAng, rrAng, llAng):
        self.pang = pAng
        self.rang = rAng
        self.lang = lAng
        self.rrang = rrAng
        self.llang = llAng

    def get_img_patches(self, y_, px_, rx_, rrx_, lx_, llx_):
        # returns original image patches
        k = self.kernelHalf
        y, px, rx, rrx, lx, llx = np.array([y_, px_, rx_, rrx_, lx_, llx_]) + k # apply boundary shift
        pPatch = self.pimg[y - k: y + k + 1, px - k: px + k + 1]
        rPatch = self.rimg[y - k: y + k + 1, rx - k: rx + k + 1]
        lPatch = self.limg[y - k: y + k + 1, lx - k: lx + k + 1]
        rrPatch = self.rrimg[y - k: y + k + 1, rrx - k: rrx + k + 1]
        llPatch = self.llimg[y - k: y + k + 1, llx - k: llx + k + 1]
        return pPatch, rPatch, lPatch, rrPatch, llPatch

    def get_grad_patches(self, y_, px_, rx_, rrx_, lx_, llx_):
        # returns gradient image patches
        k = self.kernelHalf
        y, px, rx, rrx, lx, llx = np.array([y_, px_, rx_, rrx_, lx_, llx_]) + k  # apply boundary shift
        pPatch = self.pgrad[y - k: y + k + 1, px - k: px + k + 1]
        rPatch = self.rgrad[y - k: y + k + 1, rx - k: rx + k + 1]
        lPatch = self.lgrad[y - k: y + k + 1, lx - k: lx + k + 1]
        rrPatch = self.rrgrad[y - k: y + k + 1, rrx - k: rrx + k + 1]
        llPatch = self.llgrad[y - k: y + k + 1, llx - k: llx + k + 1]
        return pPatch, rPatch, lPatch, rrPatch, llPatch

    def get_net_grad_components(self, y_, px_, rx_, rrx_, lx_, llx_):
        # use only when dxdy has not been pre convoluted
        k = self.kernelHalf
        y, px, rx, rrx, lx, llx = np.array([y_, px_, rx_, rrx_, lx_, llx_]) + k  # apply boundary shift
        pPatch = self.pdxdy[y - k: y + k + 1, px - k: px + k + 1]
        rPatch = self.rdxdy[y - k: y + k + 1, rx - k: rx + k + 1]
        lPatch = self.ldxdy[y - k: y + k + 1, lx - k: lx + k + 1]
        rrPatch = self.rrdxdy[y - k: y + k + 1, rrx - k: rrx + k + 1]
        llPatch = self.lldxdy[y - k: y + k + 1, llx - k: llx + k + 1]
        netComp = pPatch + rPatch + lPatch + rrPatch + llPatch
        return netComp[:, :, 0], netComp[:, :, 1]

    def invalidate(self, rxCands, lxCands, rrxCands, llxCands):
        rxInvalidCands = np.logical_or(rxCands < 0, rxCands > self.imgW)
        lxInvalidCands = np.logical_or(lxCands < 0, lxCands > self.imgW)
        nxInvalidCands = np.logical_or(rxInvalidCands, lxInvalidCands)
        llxInvalidCands = np.logical_or(llxCands < 0, llxCands > self.imgW)
        rrxInvalidCands = np.logical_or(rrxCands < 0, rrxCands > self.imgW)
        nnxInvalidCands = np.logical_or(rrxInvalidCands, llxInvalidCands)
        return np.logical_or(nxInvalidCands, nnxInvalidCands)

    def restrain_y_np(self, yArray):
        # faster method for ndarray containing y coordinate
        yRest = np.where(yArray > self.imgH, self.imgH, yArray)
        yRest = np.where(yRest < self.pMin, self.pMin, yRest)
        return yRest

    def restrain_x_np(self, xArray):
        # faster method for ndarray containing y coordinate
        xRest = np.where(xArray > self.imgW, self.imgW, xArray)
        xRest = np.where(xRest < self.pMin, self.pMin, xRest)
        return xRest

    def restrain_y(self, y):
        # for individual y
        if self.pMin <= y <= self.imgH:
            return y
        elif self.pMin > y:
            return self.pMin
        return self.imgH

    def restrain_x(self, x):
        # for individual x
        if self.pMin <= x <= self.imgW:
            return x
        elif self.pMin > x:
            return self.pMin
        return self.imgW

    def net_orientation_magnitude(self, netGx, netGy):
        dx, dy = np.sum(netGx), np.sum(netGy)
        mag = np.sqrt(np.square(dx) + np.square(dy))
        return mag



class Correlator(PointMatching):
    '''
        Class implements different types of correlation for comparing and matching pixel points
    '''
    def __init__(self, img_w, img_h, x_step, kernel):
        PointMatching.__init__(self, img_w, img_h, x_step, kernel)

    def calc_NCC(self, pPatch, rPatch, lPatch, rrPatch, llPatch):
        # NCC score is undefined if either patches has zero variance
        # performance degrades for noisy, low-contrast regions
        IpMean = np.mean(pPatch)
        IrMean = np.mean(rPatch)
        IlMean = np.mean(lPatch)
        IrrMean = np.mean(rrPatch)
        IllMean = np.mean(llPatch)
        IpDiff = pPatch - IpMean
        IrDiff = rPatch - IrMean
        IlDiff = lPatch - IlMean
        IrrDiff = rrPatch - IrrMean
        IllDiff = llPatch - IllMean
        IpVarr = np.sqrt(np.sum(np.square(IpDiff)))
        IrVarr = np.sqrt(np.sum(np.square(IrDiff)))
        IlVarr = np.sqrt(np.sum(np.square(IlDiff)))
        IrrVarr = np.sqrt(np.sum(np.square(IrrDiff)))
        IllVarr = np.sqrt(np.sum(np.square(IllDiff)))
        pv_n_r = np.sum(IpDiff * IrDiff) / (IpVarr * IrVarr)
        pv_n_l = np.sum(IpDiff * IlDiff) / (IpVarr * IlVarr)
        pv_n_rr = np.sum(IpDiff * IrrDiff) / (IpVarr * IrrVarr)
        pv_n_ll = np.sum(IpDiff * IllDiff) / (IpVarr * IllVarr)
        pr_n_rr = np.sum(IrDiff * IrrDiff) / (IrVarr * IrrVarr)
        pl_n_ll = np.sum(IlDiff * IllDiff) / (IlVarr * IllVarr)
        score = (pv_n_r + pv_n_l + pv_n_rr + pv_n_ll + pr_n_rr + pl_n_ll) / 6
        return -2**16 if np.isnan(score) else score

    def calc_NSSD(self, pPatch, rPatch, lPatch, rrPatch, llPatch):
        # NSSD score is undefined if the denominator is zero
        IpMean = np.mean(pPatch)
        IrMean = np.mean(rPatch)
        IlMean = np.mean(lPatch)
        IrrMean = np.mean(rrPatch)
        IllMean = np.mean(llPatch)
        IpDiff = pPatch - IpMean
        IrDiff = rPatch - IrMean
        IlDiff = lPatch - IlMean
        IrrDiff = rrPatch - IrrMean
        IllDiff = llPatch - IllMean
        IpDsqr = np.square(IpDiff)
        IrDsqr = np.square(IrDiff)
        IlDsqr = np.square(IlDiff)
        IrrDsqr = np.square(IrrDiff)
        IllDsqr = np.square(IllDiff)
        pv_n_r = 0.5 * (np.sum(np.square(IpDiff - IrDiff)) / np.sqrt(np.sum(IpDsqr + IrDsqr)))
        pv_n_l = 0.5 * (np.sum(np.square(IpDiff - IlDiff)) / np.sqrt(np.sum(IpDsqr + IlDsqr)))
        pv_n_rr = 0.5 * (np.sum(np.square(IpDiff - IrrDiff)) / np.sqrt(np.sum(IpDsqr + IrrDsqr)))
        pv_n_ll = 0.5 * (np.sum(np.square(IpDiff - IllDiff)) / np.sqrt(np.sum(IpDsqr + IllDsqr)))
        pr_n_rr = 0.5 * (np.sum(np.square(IrDiff - IrrDiff)) / np.sqrt(np.sum(IrDsqr + IrrDsqr)))
        pl_n_ll = 0.5 * (np.sum(np.square(IlDiff - IllDiff)) / np.sqrt(np.sum(IlDsqr + IllDsqr)))
        score = (pv_n_r + pv_n_l + pv_n_rr + pv_n_ll + pr_n_rr + pl_n_ll) / 6
        return 2**16 if np.isnan(score) else score

    def calc_CC(self, pPatch, rPatch, lPatch, rrPatch, llPatch):
        # computes cross correlation comparison
        pv_n_r = np.sum(pPatch * rPatch)
        pv_n_l = np.sum(pPatch * lPatch)
        pv_n_rr = np.sum(pPatch * rrPatch)
        pv_n_ll = np.sum(pPatch * llPatch)
        pr_n_rr = np.sum(rPatch * rrPatch)
        pl_n_ll = np.sum(lPatch * llPatch)
        return (pv_n_r + pv_n_l + pv_n_rr + pv_n_ll + pr_n_rr + pl_n_ll) / 6

    def calc_SSD(self, pPatch, rPatch, lPatch, rrPatch, llPatch):
        # sum squared difference comparison
        pv_n_r = np.sum(np.square(pPatch - rPatch))
        pv_n_l = np.sum(np.square(pPatch - lPatch))
        pv_n_rr = np.sum(np.square(pPatch - rrPatch))
        pv_n_ll = np.sum(np.square(pPatch - llPatch))
        pr_n_rr = np.sum(np.square(rPatch - rrPatch))
        pl_n_ll = np.sum(np.square(lPatch - llPatch))
        return (pv_n_r + pv_n_l + pv_n_rr + pv_n_ll + pr_n_rr + pl_n_ll) / 6

    def corr_match_comparison(self, xpv, yCands, rxCands, rrxCands, lxCands, llxCands):
        scoreM = np.zeros(shape=rxCands.shape, dtype=np.float32)
        for yid in range(rxCands.shape[0]):
            ypv = yCands[yid]
            for xid in range(rxCands.shape[1]):
                rx, rrx, lx, llx = rxCands[yid][xid], rrxCands[yid][xid], lxCands[yid][xid], llxCands[yid][xid]
                pImP, rImP, lImP, rrImP, llImP = self.get_grad_patches(ypv, xpv, rx, rrx, lx, llx)
                net_dx, net_dy = self.get_net_grad_components(ypv, xpv, rx, rrx, lx, llx)
                crossCorrel = self.calc_CC(pImP, rImP, lImP, rrImP, llImP)
                netOrienMag = self.net_orientation_magnitude(net_dx, net_dy)
                scoreM[yid][xid] = crossCorrel * netOrienMag # todo: undo gradient magnitude and test effect
        return scoreM

    def corr_pair_match_score(self, imgPatchA, imgPatchB, dxyPatchA, dxyPatchB):
        cCorr = np.sum(imgPatchA * imgPatchB)
        netGx = dxyPatchA[:, :, 0] + dxyPatchB[:, :, 0]
        netGy = dxyPatchA[:, :, 1] + dxyPatchB[:, :, 1]
        gaMag = self.net_orientation_magnitude(netGx, netGy)
        return cCorr * gaMag # todo: undo gradient magnitude and test effect

    def get_best_match(self, xpvPt, yCands, rxCands, rrxCands, lxCands, llxCands):
        invalidCands = self.invalidate(rxCands, lxCands, rrxCands, llxCands)
        xpv, yCs = self.restrain_x(xpvPt), self.restrain_y_np(yCands)
        rxCs, rrxCs = self.restrain_x_np(rxCands), self.restrain_x_np(rrxCands)
        lxCs, llxCs = self.restrain_x_np(lxCands), self.restrain_x_np(llxCands)

        # for correlation generated match scoring
        scoreMatch = self.corr_match_comparison(xpv, yCs, rxCs, rrxCs, lxCs, llxCs)
        badMin = np.min(scoreMatch) - 1
        validM = np.where(invalidCands, badMin, scoreMatch)
        optScr, argMaxM = np.max(validM), np.argmax(validM)
        indiciesM = np.unravel_index(argMaxM, validM.shape)
        assert (optScr == validM[indiciesM])

        yM = yCs[indiciesM[0]]
        rxM, lxM = rxCs[indiciesM], lxCs[indiciesM]
        rrxM, llxM = rrxCs[indiciesM], llxCs[indiciesM]
        return optScr, [yM, xpv, rxM, lxM, rrxM, llxM]

    def first_order_local_opt_match(self, ypvPt, xpvPt, xnPt, ngrad, nBound=9):
        #*** hard coded for correlation using gradient image
        k = self.kernelHalf
        x, y = xpvPt + k, ypvPt + k # apply boundary shift (+k)
        neighCands = self.restrain_x_np(np.arange(-nBound, nBound + 1, self.xStep) + xnPt)
        scores = np.zeros(shape=(len(neighCands)), dtype=np.float32)
        pPatch = self.pgrad[y - k: y + k + 1, x - k: x + k + 1]
        for i in range(len(neighCands)):
            xn = neighCands[i] + k # apply boundary shift (+k)
            nPatch = ngrad[y - k: y + k + 1, xn - k: xn + k + 1]
            scores[i] = np.sum(pPatch * nPatch)
        optIndex = np.argmax(scores)
        return neighCands[optIndex]

    def second_order_local_opt_match(self, ypvPt, xpvPt, xnPt, xnnPt, ngrad, nngrad, nBound=15):
        #*** hard coded for correlation using gradient image
        k = self.kernelHalf
        x, y, xn = xpvPt + k, ypvPt + k, xnPt + k # apply boundary shift (+k)
        neighCands = self.restrain_x_np(np.arange(-nBound, nBound + 1, self.xStep) + xnnPt)
        scores = np.zeros(shape=(len(neighCands)), dtype=np.float32)
        pPatch = self.pgrad[y - k: y + k + 1, x - k: x + k + 1]
        nPatch = ngrad[y - k: y + k + 1, xn - k: xn + k + 1]
        for i in range(len(neighCands)):
            xnn = neighCands[i] + k # apply boundary shift (+k)
            nnPatch = nngrad[y - k: y + k + 1, xnn - k: xnn + k + 1]
            scores[i] = (np.sum(pPatch * nnPatch) + np.sum(nPatch * nnPatch)) / 2
        optIndex = np.argmax(scores)
        return neighCands[optIndex]

    def left_1storder_local_opt_match(self, ypvPt, xpvPt, xnPt, nBound=9):
        return self.first_order_local_opt_match(ypvPt, xpvPt, xnPt, self.lgrad, nBound)

    def right_1storder_local_opt_match(self, ypvPt, xpvPt, xnPt, nBound=9):
        return self.first_order_local_opt_match(ypvPt, xpvPt, xnPt, self.rgrad, nBound)

    def left_2ndorder_local_opt_match(self, ypvPt, xpvPt, xnPt, xnnPt, nBound=15):
        return self.second_order_local_opt_match(ypvPt, xpvPt, xnPt, xnnPt, self.lgrad, self.llgrad, nBound)

    def right_2ndorder_local_opt_match(self, ypvPt, xpvPt, xnPt, xnnPt, nBound=15):
        return self.second_order_local_opt_match(ypvPt, xpvPt, xnPt, xnnPt, self.rgrad, self.rrgrad, nBound)



class Descriptor(PointMatching):
    '''
        Class implements different types of descriptors for comparing and matching pixel points
    '''

    def __init__(self, img_w, img_h, step, kernel, descType='orb', descDist=None):
        PointMatching.__init__(self, img_w, img_h, step, kernel)
        self.yStep = step
        self.descRegHgt = self.imgH - self.pMin #+ 1 # bounded image region height used for descriptor
        self.descRegWdt = self.imgW - self.pMin #+ 1 # bounded image region width used for descriptor
        assert (self.descRegHgt > 0 and self.descRegWdt > 0)
        self.descYunits = int(round(self.descRegHgt / self.yStep, 0)) + 1 # +1 needed for start point 0 (inclusive)
        self.descXunits = int(round(self.descRegWdt / self.xStep, 0)) + 1 # +1 needed for start point 0 (inclusive)
        self.yMaxMultiple = self.desc_yindex_to_ypt(int(self.descRegHgt / self.yStep))
        self.xMaxMultiple = self.desc_xindex_to_xpt(int(self.descRegWdt / self.xStep))
        #print(self.imgH, self.yMinBound, self.yMaxBound, self.descRegHgt, self.descYunits)
        #print(self.imgW, self.xMinBound, self.xMaxBound, self.descRegWdt, self.descXunits)
        #sys.exit()
        # use very large number for MAX except np.inf as this may result in empty optimal path in graph
        self.DESCPAIR_MAXNORM = 2**10 # np.finfo(np.float32).max only works for float datatype
        # reference: https://docs.opencv.org/3.4.2/d2/de8/group__core__array.html
        if descType == 'sift':
            self.dptr = cv.xfeatures2d.SIFT_create()
            self.descType = np.float32
            # distance metrics options: NORM_L2, NORM_RELATIVE_L2, NORM_L2SQR, NORM_L1, NORM_RELATIVE_L1
            self.distMetric = cv.NORM_L2 if descDist is None else descDist
        if descType == 'surf':
            self.dptr = cv.xfeatures2d.SURF_create()
            self.descType = np.float32
            # distance metrics options: NORM_L2, NORM_RELATIVE_L2, NORM_L1, NORM_RELATIVE_L1
            self.distMetric = cv.NORM_L2 if descDist is None else descDist
        if descType == 'orb':
            self.dptr = cv.ORB_create()
            self.dptr.setPatchSize(kernel)
            self.descType = np.uint8
            # distance metrics options: NORM_HAMMING, NORM_HAMMING2, NORM_L1, NORM_RELATIVE_L1
            self.distMetric = cv.NORM_HAMMING if descDist is None else descDist

    def set_descriptors(self, pdesc, rdesc, ldesc, rrdesc, lldesc):
        # last axis of resulting arrays contains 2 units: dx, dy
        self.pdesc = pdesc
        self.rdesc = rdesc
        self.rrdesc = rrdesc
        self.ldesc = ldesc
        self.lldesc = lldesc

    def y_nearest_multiple(self, y):
        # maps given y coordinate to closest yStep-multiple with an already computed descriptor
        # Note. the y coordinate may be less than self.yMinBound or greater than self.yMaxBound
        assert (0 <= y <= self.imgH)
        y_ = self.restrain_y(y)
        dscNum = y_ - self.pMin # number (mapped from pixel coordinate) used to compute descriptor
        assert (y_ >= self.yStep and dscNum >= 0)
        if dscNum % self.yStep == 0: return y_
        nearestIndexMult = self.ypt_to_desc_yindex(y_) # nearest index that is a multiple of the yStep
        return min(self.desc_yindex_to_ypt(nearestIndexMult), self.yMaxMultiple)

    def x_nearest_multiple(self, x):
        # maps given x coordinate to closest, xStep-multiple with an already computed descriptor
        # Note. the x coordinate may be less than self.xMinBound or greater than self.xMaxBound
        assert (0 <= x <= self.imgW)
        x_ = self.restrain_x(x)
        dscNum = x_ - self.pMin # number (mapped from pixel coordinate) used to compute descriptor
        assert (x_ >= self.xStep and dscNum >= 0)
        if dscNum % self.xStep == 0: return x_
        nearestIndexMult = self.xpt_to_desc_xindex(x_) # nearest index that is a multiple of the xStep
        return min(self.desc_xindex_to_xpt(nearestIndexMult), self.xMaxMultiple)

    def desc_yindex_to_ypt(self, yindex):
        assert(0 <= yindex <= self.descYunits)
        # min needed when self.descRegHgt isn't a multiple of self.yStep
        return min(yindex * self.yStep + self.pMin, self.imgH)

    def desc_xindex_to_xpt(self, xindex):
        assert(0 <= xindex <= self.descXunits)
        # min needed when self.descRegWdt isn't a multiple of self.xStep
        return min(xindex * self.xStep + self.pMin, self.imgW)

    def ypt_to_desc_yindex(self, y):
        assert (self.pMin <= y <= self.imgH)
        return int(round((y - self.pMin) / self.yStep, 0))

    def xpt_to_desc_xindex(self, x):
        assert (self.pMin <= x <= self.imgW)
        return int(round((x - self.pMin) / self.xStep, 0))

    def get_point_descriptors(self, y, px, rx, rrx, lx, llx):
        # returns already computed pixel point descriptors
        yIndx = self.ypt_to_desc_yindex(y)
        pDsrpt = self.pdesc[yIndx, self.xpt_to_desc_xindex(px)]
        rDsrpt = self.rdesc[yIndx, self.xpt_to_desc_xindex(rx)]
        lDsrpt = self.ldesc[yIndx, self.xpt_to_desc_xindex(lx)]
        rrDsrpt = self.rrdesc[yIndx, self.xpt_to_desc_xindex(rrx)]
        llDsrpt = self.lldesc[yIndx, self.xpt_to_desc_xindex(llx)]
        return pDsrpt, rDsrpt, lDsrpt, rrDsrpt, llDsrpt

    def points_to_keypoints(self, cy, cx, rx, lx, rrx, llx):
        cKpt = [cv.KeyPoint(cx, cy, self.kernelSize)]
        rKpt = [cv.KeyPoint(rx, cy, self.kernelSize)]
        lKpt = [cv.KeyPoint(lx, cy, self.kernelSize)]
        rrKpt = [cv.KeyPoint(rrx, cy, self.kernelSize)]
        llKpt = [cv.KeyPoint(llx, cy, self.kernelSize)]
        return cKpt, rKpt, lKpt, rrKpt, llKpt

    def desc_pair_match_score(self, dprA, dprB):
        # descriptor with zero vector is assumed to be invalid
        if np.all(dprA == 0) and np.all(dprB == 0): return self.DESCPAIR_MAXNORM
        return cv.norm(dprA, dprB, self.distMetric)

    def descriptor_match_score(self, cDpr, rDpr, lDpr, rrDpr, llDpr, w1=0.3, w2=0.2):
        c_r = self.desc_pair_match_score(cDpr, rDpr)
        c_l = self.desc_pair_match_score(cDpr, lDpr)
        r_rr = self.desc_pair_match_score(rDpr, rrDpr)
        l_ll = self.desc_pair_match_score(lDpr, llDpr)
        matchScore = w1 * (c_r + c_l) + w2 * (r_rr + l_ll)
        return matchScore

    def describe_image(self, image, vSize):
        knRadius = self.kernelSize
        descriptor = np.zeros(shape=(self.descYunits, self.descXunits, vSize), dtype=self.descType)
        for i in range(self.descYunits):
            y = self.desc_yindex_to_ypt(i)
            if i + 1 == self.descYunits and y > self.imgH: y = self.imgH # exclusively for last index
            assert (self.pMin <= y <= self.imgH)
            for j in range(self.descXunits):
                x = self.desc_xindex_to_xpt(j)
                if j + 1 == self.descXunits and x > self.imgW: x = self.imgW # exclusively for last index
                assert (self.pMin <= x <= self.imgW)
                kpt = [cv.KeyPoint(x, y, knRadius)]
                kpt, des = self.dptr.compute(image, kpt)
                # Note, there are cases when compute() returns an empty keypoint list and None descriptor
                # indicating that a credible descriptor could not be computed for the given keypoint
                if des is not None: descriptor[i, j] = des
        return descriptor

    def calc_Descriptor(self, y, cx, rx, rrx, lx, llx):
        #print(self.yMinBound, self.yMaxBound, y, '\t', self.xMinBound, self.xMaxBound, cx, rx, rrx, lx, llx)
        cDpr, rDpr, lDpr, rrDpr, llDpr = self.get_point_descriptors(y, cx, rx, rrx, lx, llx)
        return self.descriptor_match_score(cDpr, rDpr, lDpr, rrDpr, llDpr)

    def desc_match_comparison(self, xpv, yCands, rxCands, rrxCands, lxCands, llxCands):
        self.scoreM = np.zeros(shape=rxCands.shape, dtype=np.float32)
        for yid in range(rxCands.shape[0]):
            ypv = yCands[yid]
            for xid in range(rxCands.shape[1]):
                rx, rrx, lx, llx = rxCands[yid][xid], rrxCands[yid][xid], lxCands[yid][xid], llxCands[yid][xid]
                matchScore = self.calc_Descriptor(ypv, xpv, rx, rrx, lx, llx)
                self.scoreM[yid][xid] = matchScore
        return self.scoreM

    def get_best_match(self, xpvPt, yCands, rxCands, rrxCands, lxCands, llxCands):
        invalidCands = self.invalidate(rxCands, lxCands, rrxCands, llxCands)
        xpv, yCs = self.restrain_x(xpvPt), self.restrain_y_np(yCands)
        rxCs, rrxCs = self.restrain_x_np(rxCands), self.restrain_x_np(rrxCands)
        lxCs, llxCs = self.restrain_x_np(lxCands), self.restrain_x_np(llxCands)

        # for description generated match scoring
        scoreMatch = self.desc_match_comparison(xpv, yCs, rxCs, rrxCs, lxCs, llxCs)
        badMax = np.max(scoreMatch) + 1
        validM = np.where(invalidCands, badMax, scoreMatch)
        optScr, argMinM = np.min(validM), np.argmin(validM)
        indiciesM = np.unravel_index(argMinM, validM.shape)
        assert (optScr == validM[indiciesM])

        yM = yCs[indiciesM[0]]
        rxM, lxM = rxCs[indiciesM], lxCs[indiciesM]
        rrxM, llxM = rrxCs[indiciesM], llxCs[indiciesM]
        return optScr, [yM, xpv, rxM, lxM, rrxM, llxM]
