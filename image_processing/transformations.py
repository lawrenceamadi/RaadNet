'''
    Class implements various image transformations
'''

import cv2 as cv
import numpy as np
import sys
from tabulate import tabulate


# Display Functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def create_display_window(name, x_pos, y_pos, x_size=384, y_size=495):
    cv.namedWindow(name, cv.WINDOW_NORMAL + cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(name, x_size, y_size)
    cv.moveWindow(name, x_pos, y_pos)


def define_windows(winNames, winPerRow, rWdt, rHgt, xStart=0, yStart=0, titleHgt=35):
    #assert (isinstance(winPerRow, int))
    winNames.sort()
    cv.destroyAllWindows()
    for i in range(len(winNames)):
        x_p = xStart + (i % winPerRow) * rWdt
        y_p = yStart + int(i / winPerRow) * (rHgt + titleHgt) #345
        create_display_window(winNames[i], x_p, y_p, x_size=rWdt, y_size=rHgt)


def displayAll(iblur, igrad, tgrad, b3img, foregd, tforeg, maskc, oimg, x=1950):
    cv.namedWindow("Blur")
    cv.namedWindow("Gradient")
    cv.namedWindow("GradThresh")
    cv.namedWindow("XYZMerge")
    cv.namedWindow("Foreground")
    cv.namedWindow("ForeThresh")
    cv.namedWindow("UprLwrJoin")
    cv.namedWindow("Overlay")
    cv.moveWindow("Blur", x, 100)
    cv.moveWindow("Gradient", x + 400, 100)
    cv.moveWindow("GradThresh", x + 800, 100)
    cv.moveWindow("XYZMerge", x + 1200, 100)
    cv.moveWindow("Foreground", x, 550)
    cv.moveWindow("ForeThresh", x + 400, 550)
    cv.moveWindow("UprLwrJoin", x + 800, 550)
    cv.moveWindow("Overlay", x + 1200, 550)
    cv.imshow("Blur", iblur)
    cv.imshow("Gradient", igrad)
    cv.imshow("GradThresh", tgrad)
    cv.imshow("XYZMerge", b3img)
    cv.imshow("Foreground", foregd)
    cv.imshow("ForeThresh", tforeg)
    cv.imshow("UprLwrJoin", maskc)
    cv.imshow("Overlay", oimg)

    waiting = True
    while waiting:
        key = cv.waitKey(1)
        if key == ord("n"):
            cv.destroyWindow("Blur")
            cv.destroyWindow("Gradient")
            cv.destroyWindow("GradThresh")
            cv.destroyWindow("XYZMerge")
            cv.destroyWindow("Foreground")
            cv.destroyWindow("ForeThresh")
            cv.destroyWindow("UprLwrJoin")
            cv.destroyWindow("Overlay")
            waiting = False
        elif key == ord("q"):
            sys.exit()


def displayWindows(winNum, winNames, imgCopy, upto=True):
    sindex = 0 if upto else winNum - 1
    for i in range(sindex, winNum):
        name = winNames[i]
        cv.imshow(name, imgCopy[name])
    if cv.waitKey(0) == ord('q'):
        sys.exit()

def displayWindow(image, winName='IMAGE', x=100, y=100):
    cv.namedWindow(winName)
    cv.moveWindow(winName, x, y)
    cv.imshow(winName, image)
    if cv.waitKey(0) == ord('q'):
        sys.exit()


def cycle_through_image_formats(imgBGR):
    displayWindow(imgBGR, 'BGR')                                                    # channels
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2BGRA), 'BGRA')          # 4
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2RGB), 'RGB')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2RGBA), 'RGBA')          # 4
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HLS), 'HLS')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HLS_FULL), 'HLS_FULL')  # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HSV), 'HSV')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HSV_FULL), 'HSV_FULL')  # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2LAB), 'LAB')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2Lab), 'Lab')            # 3 same as LAB
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2LUV), 'LUV')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2Luv), 'Luv')            # 3 same as LUV
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2XYZ), 'XYZ')            # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YUV), 'YUV')            # 3
    #displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YUV_I420), 'YUV_1420') # 1
    #displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YUV_IYUV), 'YUV_IYUV') # 1
    #displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YUV_YV12), 'YUV_YV12') # 1
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YCR_CB), 'YCR_CB')      # 3
    displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YCrCb), 'YCrCb')        # 3
    #displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2BGR555), 'BGR555')     # 2
    #displayWindow(cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2BGR565), 'BGR565')     # 2
    displayWindow(imgBGR, 'back to BGR')                                            # 3


def display_image_formats(imgBGR):
    formats = {}
    formats['BGR'] = imgBGR                                                    # channels
    formats['BGRA'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2BGRA)          # 4
    formats['RGB'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2RGB)            # 3
    formats['RGBA'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2RGBA)          # 4
    formats['HLS'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HLS)            # 3
    formats['HLS_FULL'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HLS_FULL)  # 3
    formats['HSV'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HSV)            # 3
    formats['HSV_FULL'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2HSV_FULL)  # 3
    formats['LAB'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2LAB)            # 3
    formats['LUV'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2LUV)            # 3
    formats['XYZ'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2XYZ)            # 3
    formats['YUV'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YUV)            # 3
    formats['YCR_CB'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YCR_CB)      # 3
    formats['YCrCb'] = cv.cvtColor(np.copy(imgBGR), cv.COLOR_BGR2YCrCb)        # 3
    windowNames = list(formats.keys())
    define_windows(windowNames, winPerRow=7, rWdt=274, rHgt=360, xStart=-7, yStart=100)
    displayWindows(14, windowNames, formats)


def mark_point(img, pt, pointID, color=(0, 0, 255)):
    x, y = pt[0], pt[1]
    cv.line(img, (x-7, y), (x+7, y), color, thickness=2)
    cv.line(img, (x, y-7), (x, y+7), color, thickness=2)
    cv.putText(img, str(pointID), (x+10, y+5), cv.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv.LINE_AA)


def read_image(file, channel, chOrder='BGR', size=None, cropout=(0, 0)):
    '''
    Reads the image and may enhance the contrast or other transformations
    :param file:    image file path
    :param channel: channel, bgr:3 or gray:1
    :param chOrder: channel order, eg bgr, rgb
    :param size:    None or tuple=(W, H) to resize final image to, eg. (256, 336)
    :param cropout: offset tuple=(x, y) to remove from boundary(s) before resizing
    :return:        read and transformed image
    '''
    try:
        img = cv.imread(file)
        if channel == 3 and chOrder == 'RGB':
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if channel == 1:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # crop top and both sides off if necessary
        xstart, xend = cropout[0], img.shape[1] - cropout[0]
        ystart = cropout[1]
        img = img[ystart:, xstart:xend, :]

        if size:
            img = cv.resize(img, size, interpolation=cv.INTER_CUBIC)

    except: #IOError
        print('Image Read Error: filepath: {} may not exist'.format(file))
        sys.exit()

    return img


def colored_3d_plot(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    #assert (img.shape[2] == 3)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

    c1, c2, c3 = cv.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(c1.flatten(), c2.flatten(), c3.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("1st Channel")
    axis.set_ylabel("2nd Channel")
    axis.set_zlabel("3rd Channel")
    plt.show()


def foreground_seperation_bycolor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    light_green = (118, 173, 76)
    dark_green = (181, 185, 132)
    mask = cv.inRange(hsv, light_green, dark_green)
    result = cv.bitwise_and(img, img, mask=mask)
    return result


# Code from hw2
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Track bar handlers
def cvtbshandler(x):
    p = cv.getTrackbarPos('Trackbar', 'Image')
    cv.imshow('Image', blur(gimg, kernel=(p, p)))

def tbshandler(x):
    p = cv.getTrackbarPos('Trackbar', 'Image')
    cv.imshow('Image', blur(gimg, kernel=(p, p)))

# handles slider for plotting gradient vectors
def gradvechandler(x):
    skip = cv.getTrackbarPos('Trackbar', 'Image')
    cv.imshow('Image', gradvec(gimg, skip))

# handles slider changes for rotation
def rothandler(x):
     p = cv.getTrackbarPos('Trackbar', 'Image')
     cv.imshow('Image', rotate(p, gimg))

def set_image(grayImage):
    global gimg
    gimg = grayImage

# cycle through color channels of the image
def channelcycle(image,c):
    chimg = np.array(image)
    a = c%3
    b = (c+1)%3
    chimg[:,:,a] *= 0
    chimg[:,:,b] *= 0
    return chimg

# draw gradient vector at image point
def draw_grad_vector(image, x, y, dx, dy, color=[0, 0, 255]):
    mag = np.sqrt(np.square(dx) + np.square(dy))
    if mag > 0:  # implies change in pixel occurred
        angle = np.arctan2(dx, dy)
        # compute the end-point coordinates u -> v of the gradient vector with the half length k
        # and angle such that the vector crosses the pixel point half way
        k = 25 # int(np.ceil(mag) / 2)
        u = (int(x - k * np.cos(angle)), int(y - k * np.sin(angle)))
        v = (int(x + k * np.cos(angle)), int(y + k * np.sin(angle)))
        cv.arrowedLine(image, u, v, color, thickness=2)
    cv.circle(image, (x, y), 3, [255, 0, 0], thickness=-1)

# plot the gradient vector of image
def gradvec(image, N):
    if N>0:
        K = 7
        gryimg = np.array(image, dtype='float32')
        dx, dy = np.gradient(gryimg)
        row = image.shape[0]
        col = image.shape[1]
        vimg = np.copy(image)       # copy the image to avoid corrupting original image

        for x in range(0,row,N):
            for y in range(0,col,N):
                mag = np.sqrt(np.square(dx[x,y])+np.square(dy[x,y]))
                if mag > 0:         # implies change in pixel occurred
                    angle = np.arctan2(dx[x,y], dy[x,y])
                    # compute the end-point coordinates (u,v) of the gradient vector
                    # with a length K and the gradient angle
                    K = int(np.ceil(mag)) # K is a variable. Comment out this line if you want K constant
                    u = int(x + K*np.cos(angle))
                    v = int(y + K*np.sin(angle))
                    vimg = cv.arrowedLine(vimg,(y,x),(v,u),[255,0,0])
        return vimg

# rotate image by angle
def rotate(angle,image):
    #angle = angle*np.pi/180     # why multiply by np.pi/180
    rows = image.shape[0]
    cols = image.shape[1]
    M = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv.warpAffine(image,M,(cols,rows), flags=cv.WARP_INVERSE_MAP)   # affine transformation
    return dst


# Code for foreground separation for HPE
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def blur(img, kernel=(5, 5), iterations=1):
    gbimg = np.copy(img)
    for i in range(iterations):
        gbimg = cv.GaussianBlur(gbimg, kernel, 0)
    return gbimg

def merge(img1, img2):
    sum = np.where(img1 == 255, 255, img2)
    return sum

def px_threshold(img, side):
    ret, f = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if side: return f
    else:
        split = int(img.shape[1] / 2)
        l = img[:, :split]
        r = img[:, split:]
        ret_l, l = cv.threshold(l, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        ret_r, r = cv.threshold(r, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        s = np.append(l, r, axis=1)
        return merge(f, s)

def morph_closeup(img, iterations=1, k=3):
    kernel = np.ones((k, k), np.uint8)
    mcgimg = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return mcgimg

def vertical_split(img, f=0.66):
    split = int(img.shape[0] * f)
    return img[:split, :], img[split:, :]

def vertical_join(upper, lower):
    return np.vstack((upper, lower))

def get_binary_image_legacy(img, eimg, wdt, side):
    '''
    foreground extraction using adaptive thresholding
    :param img:     color:bgr image
    :return:        binary image (black:0 and white:255)
    '''
    blimg = blur(img, (5, 5))
    gbimg = cv.cvtColor(blimg, cv.COLOR_BGR2GRAY)
    tgimg = cv.adaptiveThreshold(gbimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 2)

    geimg = cv.cvtColor(eimg, cv.COLOR_BGR2GRAY)
    gb2img = blur(eimg, (3, 3), 3)
    gb2img = cv.cvtColor(gb2img, cv.COLOR_BGR2GRAY)
    bimg = merge(px_threshold(geimg, wdt, side), px_threshold(gb2img, wdt, side))

    bgimg = blur(merge(tgimg, bimg), (9, 9))
    mcgimg = morph_closeup(bgimg, 5)
    b2img = cv.GaussianBlur(mcgimg, (9, 9), 0)
    b2img = blur(b2img, (5, 5))
    g2img = px_threshold(b2img, wdt, side)
    g2img = cv.cvtColor(g2img, cv.COLOR_GRAY2BGR)

    return blimg, tgimg, bimg, bgimg, mcgimg, b2img, g2img

def aps_binary_image(img, eimg, side, vsplit):
    '''
    foreground extraction for .aps images, uses gradient magnitude
    :param img:     color:bgr image
    :param vsplit:  whether or not to split to upper & lower
    :return:        binary image (black:0 and white:255)
    '''
    iblur = blur(img, (11, 11))
    igray = cv.cvtColor(iblur, cv.COLOR_BGR2GRAY)
    mag, ang, dx, dy = sobel_gradient(igray)
    igrad = normalize_image(mag, cast=True, colored=False)
    ret, tgrad = cv.threshold(igrad, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    geimg = cv.cvtColor(eimg, cv.COLOR_BGR2GRAY)
    xyz = cv.cvtColor(img, cv.COLOR_BGR2XYZ)
    bxyz = cv.GaussianBlur(xyz, (5, 5), 0)
    xyzb1, xyzb2, xyzb3 = cv.split(bxyz)
    b1img = merge(px_threshold(geimg, side), px_threshold(xyzb2, side))
    b2img = merge(px_threshold(xyzb1, side), px_threshold(xyzb3, side))
    b3img = merge(b1img, b2img)

    foregd = merge(tgrad, b3img)
    bforeg = cv.GaussianBlur(foregd, (5, 5), 0)
    tforeg = px_threshold(bforeg, side)
    if vsplit:
        upper, lower = vertical_split(tforeg)
        upmorph = morph_closeup(upper, iterations=3)
        morphed = vertical_join(upmorph, lower)
    else:
        morphed = morph_closeup(tforeg, iterations=3)

    maskc = merge(foregd, morphed)
    return iblur, igrad, tgrad, b3img, foregd, tforeg, maskc

def a3daps_binary_image(img, side):
    '''
    foreground extraction for .a3daps images. uses gradient magnitude
    :param img:     color:bgr image
    :param vsplit:  whether or not to split to upper & lower
    :return:        binary image (black:0 and white:255)
    '''
    iblur = cv.GaussianBlur(img, (9, 9), 0)
    igray = cv.cvtColor(iblur, cv.COLOR_BGR2GRAY)
    mag, ang, dx, dy = sobel_gradient(igray)
    igrad = normalize_image(mag, cast=True, colored=False)
    ret, tgrad = cv.threshold(igrad, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret, tgray = cv.threshold(igray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    foregd = merge(tgrad, tgray)
    bforeg = cv.GaussianBlur(foregd, (9, 9), 0)
    ret, tforeg = cv.threshold(bforeg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    iterations = 1 if side else 3
    maskc = morph_closeup(tforeg, iterations=iterations)

    return iblur, igrad, tgrad, tgray, foregd, tforeg, maskc

def enhance_contrast(image, factor=2.5):
    # enhance image contrast to make more visible
    imgclone = np.copy(image)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
    lab = cv.cvtColor(imgclone, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv.merge((l2, a, b))  # merge channels
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR

def overlay_image(binary, foreground, background=200, cvtMask=True):
    mask = cv.cvtColor(binary, cv.COLOR_GRAY2BGR) if cvtMask else binary
    overlay = np.where(mask == 255, foreground, background)
    return overlay

def mask_and_overlay(rimg, side=False, vsplit=False, isaps=True, display=False):
    if isaps:
        eimg = enhance_contrast(rimg, 3)
        iblur, igrad, tgrad, b3img, foregd, tforeg, maskc = aps_binary_image(rimg, eimg, side, vsplit)
    else:
        iblur, igrad, tgrad, b3img, foregd, tforeg, maskc = a3daps_binary_image(rimg, side)
    oimg = overlay_image(maskc, rimg)
    if display:
        displayAll(iblur, igrad, tgrad, b3img, foregd, tforeg, maskc, oimg)
    return maskc, oimg

def simple_foreground_extract(img):
    iblur = blur(img, (9, 9))
    igray = cv.cvtColor(iblur, cv.COLOR_BGR2GRAY)
    ret, i_bin = cv.threshold(igray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    mag, ang, dx, dy = sobel_gradient(igray)
    igrad = normalize_image(mag, cast=True, colored=False)
    ret, g_bin = cv.threshold(igrad, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    foregd = merge(g_bin, i_bin)
    oimg = overlay_image(foregd, img)
    return oimg

def separate_foreground(img, wdt=256, hgt=336, side=False, vsplit=False, display=False):
    '''
    reads the image and may enhance the contrast or other transformations
    :param img:     original image
    :param wdt:     width to resize image to
    :param hgt:     height to resize image to
    :param side:    True if frameID is 4 or 12. False otherwise
    :param vsplit:  whether or not to split to upper & lower
    :return:        transformed image with foreground separated
    '''
    rimg = cv.resize(img, (wdt, hgt), interpolation=cv.INTER_CUBIC)
    maskc, oimg = mask_and_overlay(rimg, side=side, vsplit=vsplit, display=display)
    return oimg


# Gradient Functions from keypoint matching
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def prob_to_image(array):
    '''
    cast probability from 0/1 to 0/255. Note because of machine precision 0 & 1 may not be exactly 0 & 1
    :param array: float array containing values in interval [0, 1]
    :return: float array in the range [0, 255]
    '''
    contained = np.where(array < 0, 0, array)
    contained = np.where(contained > 1, 1, contained)
    return contained * 255

def cast_to_uint8(array, colored=True):
    cast = np.uint8(np.around(array, 0))
    if colored:
        cast = cv.cvtColor(cast, cv.COLOR_GRAY2BGR)
    return cast

def normalize_image(image, interval=[0, 255], cast=False, colored=False):
    # Not this function works best if image is all positive values (consider subtract_min)
    # consider doing absolute value first for dx or dy
    array = np.float32(image.copy())
    epsilon = np.finfo(float).eps  # to avoid divison by 0
    amax, amin = np.max(array), np.min(array)
    subtract_min = array - amin
    a_range = amax - amin
    i_range = interval[1] - interval[0]
    compressed = subtract_min / max(a_range, epsilon) * i_range
    if cast:
        return cast_to_uint8(compressed, colored=colored)
    return compressed

def laplacian_gradient(image):
    # second order derivative. First order derivative computed with sobel
    return cv.Laplacian(image, cv.CV_64F)

def sobel_gradient(image, x=1, y=1, k=-1):
    '''
    Computes and returns dx, dy, gradient magnitude and direction of all pixels in image
    :param image:   grayscale image. Image should have been passed through gaussian blur
    :param x:       order of derivative on x-axis, x=1 implies first order derivative
    :param y:       order of derivative on y-axis, y=1 implies first order derivative
    :param k:       kernel size, k = -1 implies 3x3 Scharr, k != -1 implies kxk Sobel
    :return:        pixel-wise ndarrays of dx, dy, gradient magnitude, angle (degrees)
    '''
    sobel_gx = cv.Sobel(image, ddepth=cv.CV_32F, dx=x, dy=0, ksize=k)
    sobel_gy = cv.Sobel(image, ddepth=cv.CV_32F, dx=0, dy=y, ksize=k)
    mag, ang = cv.cartToPolar(sobel_gx, sobel_gy, angleInDegrees=True)
    return mag, ang, sobel_gx, sobel_gy

def sobel_gradient_mag(image, x=1, y=1):
    # x=1, y=1 implies first order derivative
    sobel_gx = cv.Sobel(image, cv.CV_64F, x, 0)
    sobel_gy = cv.Sobel(image, cv.CV_64F, 0, y)
    mag, ang = cv.cartToPolar(sobel_gx, sobel_gy)
    return mag # mag >= 0

def numpy_gradient_mag(image):
    dy, dx = np.gradient(image)
    gradmag = np.sqrt(np.square(dx)+np.square(dy))
    return gradmag


# Neural Nets
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def channel_mean(trainX):
    '''
    Computes and returns the pixel mean of each channel (in last dimension) of image dataset
    :param trainX:  5D array of train set images where axis=-1 represents the channel(s)
    :return:        an array of size=# of channels containing mean of pixels in channel
    '''
    cMeans = np.zeros(shape=(trainX.shape[-1]), dtype=np.float32)
    for c in range(trainX.shape[-1]):
        cMeans[c] = np.mean(trainX[..., c])
    return cMeans

def mean_subtraction(imageSet, mean):
    '''
    Implements mean subtraction which is a form of input normalization that centers data about a point (0)
        if mean is a vector then per channel mean subtraction is performed
        if mean is a scalar then a fixed mean is subtracted from all channels
    :param imageSet:    5D / 4D np.float array of image dataset, assumes channel is on the last axis
    :param mean:        1D vector of size=# of channels in imageSet / scalar (np.float)
    :return:            5D / 4D array of imageSet with the mean subtracted (np.float)
    '''
    return imageSet - mean

def max_division(imageSet, max=255):
    '''
    Implements input normalization by dividing by the max pixel which is a form of scaling
        if max is a vector the per channel max division is performed
        if max is a scalar then a fixed max is divided into all channels
    :param imageSet:    5D / 4D np.float array of image dataset, assumes channel is on the last axis
    :param max:         1D vector of size=# of channels in imageSet / scalar (np.float)
    :return:            5D / 4D array of imageSet with the mean subtracted (np.float)
    '''
    return imageSet / max


# Connected Components
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def show_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv.imshow('labeled.png', labeled_img)
    if cv.waitKey(0) == ord('q'): sys.exit()


def connected_components(binaryImage, connection=8, algorithm=cv.CCL_GRANA, display=False):
    imgArea = binaryImage.shape[0] * binaryImage.shape[1]
    retval, labels, stats, centroids = cv.connectedComponentsWithStatsWithAlgorithm(binaryImage,
                                            connectivity=connection, ltype=cv.CV_16U, ccltype=algorithm)
    #print('retval:{}\ncentroids:\n{}\nstats:\n{}\n'.format(retval, centroids, stats))
    if display: show_components(labels)
    totalSegCount = retval
    segAreas = stats[:, 4]
    segAreaLess1H = np.sum(np.where(segAreas < 100, 1, 0))
    segAreaLess1K = np.sum(np.where(np.logical_and(segAreas >= 100, segAreas < 1000), 1, 0))
    segAreaLess10K = np.sum(np.where(np.logical_and(segAreas >= 1000, segAreas < 10000), 1, 0))
    segAreaGreater10K = np.sum(np.where(segAreas >= 10000, 1, 0))
    maxIndicies = (-segAreas).argsort()[:2]
    max2Percent = np.sum(segAreas[maxIndicies]) / imgArea
    max1stArea = segAreas[maxIndicies[0]]
    if len(maxIndicies) > 1:
        max2ndArea = segAreas[maxIndicies[1]]
        max2ndCenter = centroids[maxIndicies[1]] # most likely foreground if stats[:2, maxIndicies[1]] == [0, 0]
    else:
        max2ndArea, max2ndCenter = 0, [0, 0]

    return np.array([totalSegCount, segAreaLess1H, segAreaLess1K, segAreaLess10K, segAreaGreater10K,
                     max2Percent, max1stArea, max2ndArea, max2ndCenter[0], max2ndCenter[1]])


def segmented_parts(image):
    statSummary = np.zeros(shape=(4, 10), dtype=np.float32)
    statSummary[0] = connected_components(image, algorithm=cv.CCL_GRANA, connection=4)
    statSummary[1] = connected_components(image, algorithm=cv.CCL_GRANA, connection=8)
    statSummary[2] = connected_components(image, algorithm=cv.CCL_WU, connection=4)
    statSummary[3] = connected_components(image, algorithm=cv.CCL_WU, connection=8)
    return statSummary


def summarize_ccs(ccStatsSummary):
    '''
    Summarize connected components stats by computing mean metrics for 64 frames
    and average mean metrics for all frames
    :param ccStatsSummary: ndarray of shape: (#scans, 64, 4, 10)
    :return: ndarray of shape: (65, 4, 8)
    '''
    meanPerFrame = np.mean(ccStatsSummary[:, :, :, :8], axis=0)
    assert (meanPerFrame.shape == (64, 4, 8))
    avgAllFrames = np.mean(meanPerFrame, axis=0)
    assert (avgAllFrames.shape == (4, 8))
    summary = np.concatenate((meanPerFrame, np.expand_dims(avgAllFrames, axis=0)), axis=0)
    assert (summary.shape == (65, 4, 8))

    headers = ['# of seg', 'segs < 1H', 'segs < 1K', 'segs < 10K',
               'segs > 10K', 'max 2 %', 'max area', '2nd max area']
    methods = ['GRANA_4', 'GRANA_8', 'WU_4', 'WU_8']
    '''for i, algo in enumerate(methods):
        table = tabulate(summary[:, i, :], headers, tablefmt="fancy_grid")
        print('{}:\n{}\n'.format(algo, table))'''
    print(tabulate(avgAllFrames, headers, tablefmt="fancy_grid"))

