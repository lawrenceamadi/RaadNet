##print('\nKeras Logger Script Called\n')
# https://stackoverflow.com/questions/52469866/displaying-images-on-tensorboard-through-keras
import os
import io
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from skimage import transform



class TensorBoardWriter:

    def __init__(self, cfg, tb_log_dir):
        assert(os.path.isdir(tb_log_dir))
        self.tb_log_dir = tb_log_dir
        hgt, wdt, __ = cfg.MODEL.IMAGE_SIZE
        self.img_dim = (wdt, hgt)
        self.pooling_roi = cfg.MODEL.EXTRA.RES_CONV_VERSION in ['v5']
        self.image_writer = tf.summary.create_file_writer(self.tb_log_dir, max_queue=12)
        self.scalar_writer = tf.summary.create_file_writer(self.tb_log_dir)
        self.histogram_writer = tf.summary.create_file_writer(self.tb_log_dir)

    def tblog_image(self, tag, image, roi_input, roi_coord, step):
        '''
        Tensorboard log of input network-cropped-image
            after indicating roi mask and bounding-box/bounding-polygon
        :param tag: name tag of image to be logged
        :param image: input nci (image) to be logged
        :param roi_input: roi binary mask (roi-masking) or diagonal corners of roi (roi-pooling)
        :param roi_coord: x,y coordinates of roi bounding-box/polygon
        :param step: tensorboard log step
        :return: N/A
        '''
        #min_max = [-1.0, 1.0] # [np.min(image), np.max(image)]
        #image = scale_image_pixels(image, min_max)
        image = self.reverse_preprocess_func(image)

        # green-out pixels outside roi. assumes RGB
        if self.pooling_roi:
            # roi_input is relative roi bounding-box coordinates
            # roi_input[-1] == [s_x/roi_wdt, s_y/roi_hgt, e_x/roi_wdt, e_y/roi_hgt]
            s_x, e_x, s_y, e_y = roi_coord
            s_x, s_y = max(0, s_x), max(0, s_y)
            image[:s_y, :, [0, 2]] = 0
            image[e_y:, :, [0, 2]] = 0
            image[:, :s_x, [0, 2]] = 0
            image[:, e_x:, [0, 2]] = 0
        else:
            # roi_input is roi mask
            roi_input = transform.resize(roi_input, self.img_dim, preserve_range=True)
            roi_input = np.where(roi_input > 0.5, 1, 0)
            image[:, :, [0, 2]] *= roi_input

        uint8_img = image.astype(np.uint8, copy=False)
        uint8_img = draw_roi_oriented_polygon(uint8_img, roi_coord)
        uint8_img = np.expand_dims(uint8_img, axis=0)
        with self.image_writer.as_default():
            tf.summary.image(tag, uint8_img, step=step)

    def tblog_scalar(self, tag, scalar, step):
        with self.scalar_writer.as_default():
            tf.summary.scalar(tag, scalar, step=step)

    def tblog_histogram(self, tag, tensor, step):
        with self.histogram_writer.as_default():
            tf.summary.histogram(tag, tensor, step=step)

    def increment_epoch(self):
        self.batch_epoch += 1
        #print("\n\n\nself.batch_epoch: {}\n\n\n".format(self.batch_epoch))

    def increment_step(self):
        self.batch_step += 1

    def set_epoch(self, epoch):
        self.batch_epoch = epoch

    def set_step(self, step):
        self.batch_step = step

    def close(self):
        """
        To be called in the end
        """
        self.image_writer.flush()
        self.image_writer.flush()

    def reverse_preprocess_func(self, image):
        # reverse effect of tf.keras.applications.mobilenet_v2.preprocess_input
        reversed = (image + 1.0) * 127.5
        return reversed #.astype(np.uint8)


def scale_image_pixels(image, min_max, interval=(0,255), ret_uint8=False):
    # Not this function works best if image is all positive values (consider subtract_min)
    # consider doing absolute value first for dx or dy
    epsilon = np.finfo(float).eps  # to avoid division by 0
    amin, amax = min_max
    amin = min(amin, np.min(image))
    amax = max(amax, np.max(image))
    subtract_min = image.astype(np.float32) - amin
    a_range = amax - amin
    i_range = interval[1] - interval[0]
    compressed = (subtract_min / max(a_range, epsilon) * i_range) + interval[0]
    compressed = np.clip(np.around(compressed, 0), interval[0], interval[1])
    if ret_uint8:
        return compressed.astype(np.uint8)
    return compressed #.astype(np.uint8)


def draw_roi_oriented_polygon(img, roi_coord, ink=(255,255,255), t=1):
    #print(roi_coord, roi_coord.shape, img.shape, img.dtype)
    #sys.exit(0)
    # need vertices coordinates in np.int32 and (rows,1,2) shape
    pts = roi_coord.astype(np.int32, copy=False).reshape((-1, 1, 2))
    cv.polylines(img, [pts], True, ink, t)
    return img


def draw_roi_aligned_rectbbox(img, roi_coord, ink=(255,0,0), t=1):
    x1, x2, y1, y2 = roi_coord
    ink = np.asarray(ink)
    # border is drawn such that the borders lines occupy pixels outside the roiR
    img[y1 - t: y1, x1 - t: x2 + t] = ink # top horizontal line
    img[y2: y2 + t, x1 - t: x2 + t] = ink # bottom horizontal line
    img[y1 - t: y2 + t, x1 - t: x1] = ink # left vertical line
    img[y1 - t: y2 + t, x2: x2 + t] = ink # right vertical line
    return img


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.compat.v1.summary.Summary.Image(height=height, width=width,
                                              colorspace=channel,
                                              encoded_image_string=image_string)