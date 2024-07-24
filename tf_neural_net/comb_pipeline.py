'''
data loader for keras, tensorflow
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''
##print('\nComb Pipeline Called\n')
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf

from time import time
from matplotlib.path import Path
from shapely.geometry import Polygon
from numpy.random import randint, uniform

sys.path.append('../')
from tf_neural_net.build_models import pool_mask_cfg
from tf_neural_net.data_preprocess import bb_coords, bb_half_dims
from tf_neural_net.commons import map_zones_frameids_to_indexes
from tf_neural_net.commons import duplicate_frames_v2, get_subnet_output_names
from tf_neural_net.commons import BODY_ZONES_GROUP, GROUP_AUG_CONFIG, SUBNET_CONFIGS


class DataPipeline(object):
    'Generates image data for Keras'
    def __init__(self, cfg, input_data, n_samples, d_set, grp_zones_frames,
                 preload_reg_dims, minority_cnt, constants, logger, unique_ids=None):
        # read-only instance variables
        self.cfg = cfg
        self.d_set = d_set
        self.unique_smpids = unique_ids
        self.data_sample_size = n_samples
        self.body_zone_names = grp_zones_frames.keys()  # grp_zone_names

        # initialize important boolean variables that configure the behavior of pipeline
        self.is_training = d_set=='train'
        self.default_loss_branch_wgt = np.asarray(self.cfg.LOSS.DEFAULT_LOSS_BRANCH_WGT)
        self.compute_sample_wgts = cfg.LOSS.PASS_SAMPLE_WGTS and self.is_training
        self.use_threat_boxes = cfg.LABELS.USE_THREAT_BBOX_ANOTS and self.is_training
        self.gt_mismatch_wgt = cfg.LABELS.THREAT_ANOTS_MISMATCH
        self.SUBNET_OUTPUTS = cfg.MODEL.SUBNET_TYPE in SUBNET_CONFIGS
        if self.SUBNET_OUTPUTS:
            self.SUBNET_OUT_NAMES = \
                get_subnet_output_names(constants['subnet_tags'], cfg.LOSS.NET_OUTPUTS_ID[0])
            self.bdgrp_subnet = cfg.MODEL.SUBNET_TYPE=='body_groups'
        else: self.SUBNET_OUT_NAMES, self.bdgrp_subnet = None, False
        self.MULTI_OUTPUTS = len(cfg.LOSS.NET_OUTPUTS_ID)>1
        self.OUT_1_ID = cfg.LOSS.NET_OUTPUTS_ID[0]
        if self.MULTI_OUTPUTS:
            self.OUT_2_ID = cfg.LOSS.NET_OUTPUTS_ID[1]
            assert(d_set!='train' or (not self.use_threat_boxes or self.OUT_2_ID=='gt'))

        # initialize global data for all samples in subset
        self.X, self.y, self.w, self.roi_coord, self.nci_tlc, self.threat_bboxes = input_data
        assert(np.all(np.sum(self.roi_coord[:,:,:,:,0], axis=-1)>=0)), "sum of each roi xcoord >=0"
        assert(np.all(np.sum(self.roi_coord[:,:,:,:,1], axis=-1)>=0)), "sum of each roi ycoord >=0"

        if self.compute_sample_wgts:
            self.class_weights = np.asarray([cfg.LOSS.BENIGN_CLASS_WGT, cfg.LOSS.THREAT_CLASS_WGT])
            self.segconf_max_is_min = cfg.LOSS.SEG_CONFIDENCE_MIN==cfg.LOSS.SEG_CONFIDENCE_MAX
        else: self.class_weights = None

        # compute images per sample and network input shapes
        self.IMGS_PER_SAMPLE = 0
        for zone_name in self.body_zone_names:
            imgs_per_zone = len(grp_zones_frames[zone_name])
            if imgs_per_zone>self.IMGS_PER_SAMPLE: self.IMGS_PER_SAMPLE = imgs_per_zone
        self.IMGS_INSAMP_SHAPE = (self.IMGS_PER_SAMPLE, *cfg.MODEL.IMAGE_SIZE)
        self.ROI_POOLING = cfg.MODEL.EXTRA.RES_CONV_VERSION in ['v5']
        self.ROI_MASKING = not self.ROI_POOLING
        if self.ROI_POOLING:
            self.ROIS_INSAMP_SHAPE = (self.IMGS_PER_SAMPLE, 1, 4)
            self.ROIS_ACTUAL_SHAPE = self.ROIS_INSAMP_SHAPE
        else:
            self.ROIS_ACTUAL_SHAPE = self.IMGS_INSAMP_SHAPE[:-1] + (1,)
            fe_out_shape = constants['fe_out_shape'][:-1] # exclude channel
            self.ROIS_INSAMP_SHAPE = (self.IMGS_PER_SAMPLE, *fe_out_shape, 1)
            self.DS_POOL_K, self.DS_STRIDE = pool_mask_cfg(cfg.MODEL.IMAGE_SIZE, fe_out_shape)
        self.ROI_COORD_SHAPE = (self.IMGS_PER_SAMPLE, *self.roi_coord.shape[3:])  # roi bounding-box

        # ROI: Region of Interest, RTC: Region to Crop, PLW: Preloaded Region Window
        self.ZONE_ROI_DIMS = constants['zone_roi_dims']
        self.ZONE_PRW_DIMS = preload_reg_dims
        self.ZONE_RTC_DIMS = np.int32(cfg.MODEL.REGION_DIM)
        self.SCALED_FRM_DIM = constants['frm_dim'] # dimensions of image frame (may be scaled)
        # Actual cropped images fed to network
        self.set_nci_constants(cfg.MODEL.IMAGE_SIZE)
        self.BATCH_SIZE = eval('cfg.{}.BATCH_SIZE'.format(self.d_set.upper()))
        self.EOE_SHUFFLE = eval('cfg.{}.EOE_SHUFFLE'.format(self.d_set.upper()))
        self.generator_stpep = int(self.data_sample_size / self.BATCH_SIZE)
        self.MAP_ZONE_FID_TO_IDX = map_zones_frameids_to_indexes(grp_zones_frames)
        self.zone_ordered_frames = duplicate_frames_v2(grp_zones_frames, self.IMGS_PER_SAMPLE,
            cfg.MODEL.IMAGES_PER_SEQ_GRP, constants['min_gap'], constants['n_frames'], logger)
        self.ZONE_IDX_TO_TAG = constants['zone_idx_to_tag']
        self.ZONE_TAG_TO_IDX = constants['zone_tag_to_idx']
        self.ZONE_TO_GRP_IDX = constants['zone_to_grp_idx']
        self.BG_PIXELS = constants['bg_pixels']
        self.N_ZONES = len(self.body_zone_names)
        self.N_PARTS = cfg.MODEL.EXTRA.BDPART_LOGIT_UNITS
        self.M_IMG_SETS = self.IMGS_PER_SAMPLE // cfg.MODEL.IMAGES_PER_SEQ_GRP
        self.SCALE_DIM_FACTOR = constants['scale_ftr']
        self.EC_GRID_TILES = tuple(cfg.AUGMENT.GRID_TILES)
        if cfg.DATASET.COLOR_RGB:
            self.EC_COLOR_SPACE_CVT = (cv.COLOR_RGB2LAB, cv.COLOR_LAB2RGB)
        else: self.EC_COLOR_SPACE_CVT = (cv.COLOR_BGR2LAB, cv.COLOR_LAB2BGR)
        self.UNMARKED = -1 # Note! cannot be >=0 because of sample weight computation
                           # img_t,      roi_t,      rcv_t,      bbc_t,    tgt_t,
        self.OUTPUT_TYPES = [tf.float32, tf.float32, tf.float32, tf.int32, tf.int32,
                        # tol_t/pgt_t, t_wgt,      p_wgt,      idx_t,    o_idx
                             tf.int32, tf.float32, tf.float32, tf.int32, tf.int32]
        self.set_aug_params(minority_cnt, constants['max_xy_shift_aug'], logger)
        self.set_roi_and_rcv_configs()
        self.set_preprocess_func()
        self.log_meta_info(logger)

        # writeable variables
        self.cnt_on_zone_loss = 0
        self.cnt_on_iset_loss = 0
        self.cnt_neither_loss = 0
        self.cnt_gt_mismatch = 0
        self.total_time = 0
        self.total_call = 0

        # global variables that changes per tf.data call
        self.sample_images = np.empty(self.IMGS_INSAMP_SHAPE, dtype=np.float32) #* must be float !!!
        self.sample_znrois = np.zeros(self.ROIS_ACTUAL_SHAPE, dtype=np.float32) #* must be np.zeros
        self.reg_comp_vecs = np.empty((self.IMGS_PER_SAMPLE, self.N_ZONES), dtype=np.float32) #*
        self.reg_threat_lbl = np.full((self.IMGS_PER_SAMPLE), self.UNMARKED, dtype=np.int32) #*
        self.rel_crp_roi_vertices = np.empty(self.ROI_COORD_SHAPE, dtype=np.int32) #* relative to NcI
        self.iset_gt_placeholder = np.zeros((self.M_IMG_SETS, 1), dtype=np.int32) #*
        self.x_shift_default = np.zeros((self.IMGS_PER_SAMPLE), dtype=np.int32) #*
        self.y_shift_default = np.zeros((self.IMGS_PER_SAMPLE), dtype=np.int32) #*

    def set_nci_constants(self, net_img_size):
        self.NCI_WIN_HGT, self.NCI_WIN_WDT = net_img_size[:2]
        self.NCI_WIN_DIM = np.asarray([self.NCI_WIN_WDT, self.NCI_WIN_HGT])
        self.NCI_WIN_HALF_HGT = self.NCI_WIN_HGT // 2
        self.NCI_WIN_HALF_WDT = self.NCI_WIN_WDT // 2
        # assemble grid x,y point coordinates of nci
        pts_x, pts_y = np.meshgrid(np.arange(self.NCI_WIN_WDT),
                                   np.arange(self.NCI_WIN_HGT)) # make a canvas with coordinates
        pts_x, pts_y = pts_x.flatten(), pts_y.flatten()
        self.NCI_GRID_PTS = np.vstack((pts_x, pts_y)).T

    def set_aug_params(self, minority_cnt, max_xy_shift_aug, logger):
        self.MAX_AUG_X_SHIFT = max_xy_shift_aug[0]
        self.MAX_AUG_Y_SHIFT = max_xy_shift_aug[1]
        # Data augmentation initializations
        self.data_augment = eval('self.cfg.{}.AUGMENTATION'.format(self.d_set.upper()))
        if self.data_augment:
            self.augment_odd_max = int(round(self.data_sample_size / minority_cnt, 0)) # = 22
            self.augment_not_max = int(round(self.augment_odd_max * self.cfg.AUGMENT.ODDS, 0))
            assert((self.cfg.AUGMENT.ODDS>0)==(self.augment_not_max>=1))

            xshift_aug = True if self.cfg.AUGMENT.X_SHIFT>0 else False
            yshift_aug = True if self.cfg.AUGMENT.Y_SHIFT>0 else False
            rotate_aug = True if self.cfg.AUGMENT.ROTATE>0 else False
            s_zoom_aug = True if self.cfg.AUGMENT.S_ZOOM>0 else False
            h_flip_aug = self.cfg.AUGMENT.H_FLIP
            if self.cfg.AUGMENT.BRIGHTNESS>0 and self.cfg.AUGMENT.P_CONTRAST>0:
                bright_aug = True
            else: bright_aug = False
            if self.cfg.AUGMENT.N_CONTRAST>0: n_cont_aug = True
            else: n_cont_aug = False

            # Order of augmentation is important and must be maintained as in GROUP_AUG_CONFIG
            exp_aug_configs = np.array([xshift_aug, yshift_aug, rotate_aug,
                                        s_zoom_aug, h_flip_aug, bright_aug, n_cont_aug])
            self.xshift_idx, self.yshift_idx, self.rotate_idx = 0, 1, 2
            self.s_zoom_idx, self.h_flip_idx, self.bright_idx, self.n_cont_idx = 3, 4, 5, 6

            self.augment_configs = {}
            msg = '\nAdjusted Max Shift Augmentation per Body Zone:\n\t    X_Shift\tY_Shift'

            for i, zone_name in enumerate(self.body_zone_names):
                # set augmentation configurations for zone
                grp_name = BODY_ZONES_GROUP[zone_name]
                grp_aug_defaults = np.array(GROUP_AUG_CONFIG[grp_name])
                zone_aug_configs = np.logical_and(exp_aug_configs, grp_aug_defaults)
                adj_max_x_sft = self.MAX_AUG_X_SHIFT[i]
                adj_max_y_sft = self.MAX_AUG_Y_SHIFT[i]
                if adj_max_x_sft==0: zone_aug_configs[self.xshift_idx] = False
                if adj_max_y_sft==0: zone_aug_configs[self.yshift_idx] = False
                self.augment_configs[zone_name] = zone_aug_configs
                # record zone's max x-y-shift augmentation
                msg += '\n{:>9}:\t{:>2}\t{:>2}'.format(zone_name, adj_max_x_sft, adj_max_y_sft)

            if logger is not None: logger.log_msg(msg)
            print(msg)

    def set_roi_and_rcv_configs(self):
        roi_zone_wdts = self.ZONE_ROI_DIMS[:, 0]
        roi_zone_hgts = self.ZONE_ROI_DIMS[:, 1]
        roi_area_at_zone_i = roi_zone_wdts * roi_zone_hgts

        # set function for retracing ROI given x-y-shift image augmentation and others
        if self.cfg.AUGMENT.WANDERING_ROI:
            self.get_roi = moving_aligned_roi_coord
            min_expected_intercept_wdt = roi_zone_wdts - self.MAX_AUG_X_SHIFT # allow error of -2?
            min_expected_intercept_hgt = roi_zone_hgts - self.MAX_AUG_Y_SHIFT # allow error of -2?
        else:
            self.get_roi = retrace_aligned_roi_coord
            min_expected_intercept_wdt = roi_zone_wdts # allow error of -2?
            min_expected_intercept_hgt = roi_zone_hgts # allow error of -2?
        min_expected_intercept_area = min_expected_intercept_wdt * min_expected_intercept_hgt

        # compute expected values (or range of values) of RCV_i (RCV at position i).
        # but this is only relevant if using axis-aligned bounding-box
        if self.cfg.MODEL.EXTRA.RCV_IOU_OF=='nci_vs_rois_per_zone':
            self.compute_rcv = aligned_rcv_as_nci_vs_rois
            net_cropped_img_area = self.NCI_WIN_WDT * self.NCI_WIN_HGT # allow error of +2?
            max_expected_union_area = net_cropped_img_area + roi_area_at_zone_i \
                                      - min_expected_intercept_area
            if self.cfg.AUGMENT.WANDERING_ROI:
                min_expected_iou_at_i = min_expected_intercept_area / max_expected_union_area
            else:
                min_expected_iou_at_i = min_expected_intercept_area / max_expected_union_area
        elif self.cfg.MODEL.EXTRA.RCV_IOU_OF=='roi_vs_rois_per_zone':
            self.compute_rcv = aligned_rcv_as_roi_vs_rois
            max_expected_union_area = roi_area_at_zone_i + roi_area_at_zone_i \
                                      - min_expected_intercept_area
            if self.cfg.AUGMENT.WANDERING_ROI:
                min_expected_iou_at_i = min_expected_intercept_area / max_expected_union_area
            else:
                min_expected_iou_at_i = min_expected_intercept_area / max_expected_union_area # or 1
        else:
            assert(False), 'unrecognized RCV_IOU_OF:{}'.format(self.cfg.MODEL.EXTRA.RCV_IOU_OF)

        if self.cfg.MODEL.EXTRA.ROI_TYPE=='aligned':
            self.MIN_EXPECTED_IOU_AT_I = np.around(min_expected_iou_at_i, 4)
        else:
            self.MIN_EXPECTED_IOU_AT_I = np.zeros(self.N_ZONES)  # default has a null effect
            # reset get_roi function
            self.get_roi = retrace_oriented_roi_coord
            # reset compute_rcv function
            if self.cfg.MODEL.EXTRA.RCV_IOU_OF=='nci_vs_rois_per_zone':
                self.compute_rcv = oriented_rcv_as_nci_vs_rois
            else: self.compute_rcv = oriented_rcv_as_roi_vs_rois

    def set_preprocess_func(self):
        if self.cfg.DATASET.PREPROCESS:
            fe_net_id = self.cfg.MODEL.EXTRA.FE_NETWORK
            if fe_net_id=='mobilenet_v2':
                self.preprocess_func = tf.keras.applications.mobilenet_v2.preprocess_input
            elif fe_net_id=='xception':
                self.preprocess_func = tf.keras.applications.xception.preprocess_input
            elif fe_net_id=='inception_resnet_v2':
                self.preprocess_func = tf.keras.applications.inception_resnet_v2.preprocess_input
            elif fe_net_id=='resnet152_v2':
                self.preprocess_func = tf.keras.applications.resnet_v2.preprocess_input
            elif fe_net_id=='nasnet_mobile':
                self.preprocess_func = tf.keras.applications.nasnet.preprocess_input
            elif fe_net_id=='densenet_121':
                self.preprocess_func = tf.keras.applications.densenet.preprocess_input
            else:
                assert(False), 'Unrecognized FE_NETWORK:{}'.format(self.cfg.MODEL.EXTRA.FE_NETWORK)
        elif self.cfg.DATASET.NORMALIZE:
            self.preprocess_func =  custom_preprocess
        else:
            assert(False), 'Only 1 of cfg.DATASET.PREPROCESS or cfg.DATASET.NORMALIZE can be True'

    def switch_loss_func(self):
        self.default_loss_branch_wgt = np.flipud(self.default_loss_branch_wgt)

    def change_benign_class_wgt(self):
        if self.compute_sample_wgts:  # or hasattr(self, 'class_weights')
            self.class_weights[0] = self.cfg.LOSS.EXTRA.SL_BENIGN_CLASS_WGT

    def feed_sample_inputs_xm_out(self, sample_indexes):
        '''
        Compiles and return region images of a body part and index of sample_id and output
        :param sample_indexes: tensor with (sample_id_indx, zone_type_indx, scan_id_indx)
        :return: 3 parameters (described above) for a single sample
        '''
        img_t, roi_t, rcv_t, bbc_t, tgt_t, tol_t, t_wgt, g_wgt, idx_t, o_idx = \
            tf.py_function(func=self.feed_sample_inputs_all_out,
                           inp=[sample_indexes], Tout=self.OUTPUT_TYPES)
        img_t = tf.reshape(img_t, shape=self.IMGS_INSAMP_SHAPE)
        roi_t = tf.reshape(roi_t, shape=self.ROIS_ACTUAL_SHAPE)
        if self.ROI_MASKING:
            # downsample roi mask input
            roi_t = tf.keras.layers.AveragePooling2D(pool_size=self.DS_POOL_K,
                                                     strides=self.DS_STRIDE)(roi_t)
        rcv_t = tf.reshape(rcv_t, shape=(self.IMGS_PER_SAMPLE, self.N_ZONES))
        input = {'crop_reg_imgs': img_t, 'roi_msks_bbxs': roi_t, 'reg_comp_vecs': rcv_t}
        idx_t = tf.reshape(idx_t, shape=(1,))
        o_idx = tf.reshape(o_idx, shape=(1,))
        return input, idx_t, o_idx

    def feed_sample_inputs_xym_out(self, sample_indexes):
        '''
        Compiles and return region images of a body part, only the zone-level threat gt,
            roi bounding-box coordinates of the cropped images, index of the sample_id and output
        :param sample_indexes: tensor with (sample_id_indx, zone_type_indx, scan_id_indx)
        :return: 5 parameters (described above) for a single sample
        '''
        img_t, roi_t, rcv_t, bbc_t, tgt_t, tol_t, t_wgt, g_wgt, idx_t, o_idx = \
            tf.py_function(func=self.feed_sample_inputs_all_out,
                           inp=[sample_indexes], Tout=self.OUTPUT_TYPES)
        img_t = tf.reshape(img_t, shape=self.IMGS_INSAMP_SHAPE)
        roi_t = tf.reshape(roi_t, shape=self.ROIS_ACTUAL_SHAPE)
        if self.ROI_MASKING:
            # downsample roi mask input
            roi_t = tf.keras.layers.AveragePooling2D(pool_size=self.DS_POOL_K,
                                                     strides=self.DS_STRIDE)(roi_t)
        rcv_t = tf.reshape(rcv_t, shape=(self.IMGS_PER_SAMPLE, self.N_ZONES))
        bbc_t = tf.reshape(bbc_t, shape=self.ROI_COORD_SHAPE)
        input = {'crop_reg_imgs': img_t, 'roi_msks_bbxs': roi_t, 'reg_comp_vecs': rcv_t}
        gt_lb = tf.reshape(tgt_t, shape=(1,))
        idx_t = tf.reshape(idx_t, shape=(1,))
        o_idx = tf.reshape(o_idx, shape=(1,))
        return input, gt_lb, bbc_t, idx_t, o_idx

    def feed_sample_inputs_xyw_out(self, sample_indexes):
        '''
        Compiles and return region images of a body part, ground-truth labels for each output,
            and also sample weights for each output
        :param sample_indexes: tensor with (sample_id_indx, zone_type_indx, scan_id_indx)
        :return: 3 parameters (described above) for a single sample
        '''
        img_t, roi_t, rcv_t, bbc_t, tgt_t, tol_t, t_wgt, g_wgt, idx_t, o_idx = \
            tf.py_function(func=self.feed_sample_inputs_all_out,
                           inp=[sample_indexes], Tout=self.OUTPUT_TYPES)
        img_t = tf.reshape(img_t, shape=self.IMGS_INSAMP_SHAPE)
        roi_t = tf.reshape(roi_t, shape=self.ROIS_ACTUAL_SHAPE)
        if self.ROI_MASKING:
            # downsample roi mask input
            roi_t = tf.keras.layers.AveragePooling2D(pool_size=self.DS_POOL_K,
                                                     strides=self.DS_STRIDE)(roi_t)
        rcv_t = tf.reshape(rcv_t, shape=(self.IMGS_PER_SAMPLE, self.N_ZONES))
        input = {'crop_reg_imgs': img_t, 'roi_msks_bbxs': roi_t, 'reg_comp_vecs': rcv_t}
        gt_lb = tf.reshape(tgt_t, shape=(1,))
        s_wgt = tf.reshape(t_wgt, shape=(1,))

        if self.SUBNET_OUTPUTS:
            gt_lb_dict = {}  # todo: possibly reason why memory grows over time
            s_wgt_dict = {}  #      try using global dicts and clear before updating
            dummy_wgt = tf.reshape(0., shape=(1,))
            for idx, z_tag in enumerate(self.SUBNET_OUT_NAMES):
                gt_lb_dict[z_tag] = gt_lb
                s_wgt_dict[z_tag] = s_wgt if idx==o_idx else dummy_wgt
            return (input, gt_lb_dict, s_wgt_dict)
        elif self.MULTI_OUTPUTS:
            tol_t = tf.reshape(tol_t, shape=(self.M_IMG_SETS, 1))
            gt_lb_dict = {self.OUT_1_ID: gt_lb, self.OUT_2_ID: tol_t}  # ***
            ## tol_t = tf.fill((self.M_IMG_SETS, 1), tgt_t) # region threat label without threat bbox
            ## pgt_t = tf.reshape(pgt_t, shape=(self.N_PARTS,)) # body-part ground-truth tensor
            ## gt_lb_dict = {'t': gt_lb, 'p': pgt_t}
            g_wgt = tf.reshape(g_wgt, shape=(1,))
            s_wgt_dict = {self.OUT_1_ID: s_wgt, self.OUT_2_ID: g_wgt}  # ***
            ## p_wgt = tf.reshape(p_wgt, shape=(1,))
            ## s_wgt_dict = {'t': s_wgt, 'p': p_wgt}
            return input, gt_lb_dict, s_wgt_dict
        return input, gt_lb, s_wgt

    #@tf.function
    def feed_sample_inputs_xy_out(self, sample_indexes):
        '''
        Compiles and return region images of a body part and ground-truth labels for each output
        :param sample_indexes: tensor with (sample_id_indx, zone_type_indx, scan_id_indx)
        :return: 2 parameters (described above) for a single sample
        '''
        img_t, roi_t, rcv_t, bbc_t, tgt_t, tol_t, t_wgt, g_wgt, idx_t, o_idx = \
            tf.py_function(func=self.feed_sample_inputs_all_out,
                           inp=[sample_indexes], Tout=self.OUTPUT_TYPES)
        img_t = tf.reshape(img_t, shape=self.IMGS_INSAMP_SHAPE)
        roi_t = tf.reshape(roi_t, shape=self.ROIS_ACTUAL_SHAPE)
        if self.ROI_MASKING:
            # downsample roi mask input
            roi_t = tf.keras.layers.AveragePooling2D(pool_size=self.DS_POOL_K,
                                                     strides=self.DS_STRIDE)(roi_t)
        rcv_t = tf.reshape(rcv_t, shape=(self.IMGS_PER_SAMPLE, self.N_ZONES))
        input = {'crop_reg_imgs': img_t, 'roi_msks_bbxs': roi_t, 'reg_comp_vecs': rcv_t}
        gt_lb = tf.reshape(tgt_t, shape=(1,))
        if self.MULTI_OUTPUTS:
            tol_t = tf.reshape(tol_t, shape=(self.M_IMG_SETS, 1))
            gt_lb = {self.OUT_1_ID: gt_lb, self.OUT_2_ID: tol_t}
        return input, gt_lb

    def feed_sample_inputs_all_out(self, sample_indexes):
        t0 = time()

        sample_id_indx, zone_type_indx, scan_id_indx = sample_indexes.numpy()
        zone_name = self.ZONE_IDX_TO_TAG[zone_type_indx]
        #* smp_imgs, smp_rois, smp_pit_gt, smp_rcvs, smp_bbcs = \
        #*     self.load_single_zone_images(zone_name, zone_type_indx, scan_id_indx)
        self.load_single_zone_images(zone_name, zone_type_indx, scan_id_indx)

        smp_threat_gt = self.y[zone_name][scan_id_indx]
        assert (smp_threat_gt in [0, 1]), 'smp_threat_gt must be 0/1, not {}'.format(smp_threat_gt)
        # smp_bdpart_gt = \
        #     tf.keras.utils.to_categorical(zone_type_indx, dtype='int32', num_classes=self.N_PARTS)
        if self.bdgrp_subnet:
            output_indx = self.ZONE_TO_GRP_IDX[zone_type_indx]
        else: output_indx = zone_type_indx

        #* if self.use_threat_boxes:
        #*     iset_threat_gt = np.amax(np.reshape(smp_pit_gt, (self.M_IMG_SETS, -1)), axis=1)
        #* else: iset_threat_gt = np.full((self.M_IMG_SETS, 1), smp_threat_gt, dtype=np.int32)
        if self.use_threat_boxes:
            iset_threat_gt = np.amax(np.reshape(self.reg_threat_lbl, (self.M_IMG_SETS, -1)), axis=1)
        else: iset_threat_gt = self.iset_gt_placeholder*0 + smp_threat_gt

        if self.compute_sample_wgts:
            if self.use_threat_boxes:
                # select max label from per image-group-set threat-object-labels
                max_grpset_tol = np.amax(iset_threat_gt)
                # decide which loss should be computed for sample
                # based on availability of manually labeled threat bounding-box
                not_marked = max_grpset_tol==self.UNMARKED
                zn_loss_branch_wgt = 1.0 if not_marked else 0.0
                gs_loss_branch_wgt = 1.0 - zn_loss_branch_wgt # one or the other but not both
                # match weight ia 1 if tsa gt agrees with cropped rois threat labels, 0 otherwise
                if smp_threat_gt==max_grpset_tol:
                    match_wgt = 1.0
                else:
                    match_wgt = self.gt_mismatch_wgt
                    if not not_marked: self.cnt_gt_mismatch += 1
            else:
                zn_loss_branch_wgt = self.default_loss_branch_wgt[0]  # 0.0
                gs_loss_branch_wgt = self.default_loss_branch_wgt[1]  # 1.0
                match_wgt = gs_loss_branch_wgt
                not_marked = False

            # retrieve the segmentation confidence of the cropped regions (a function of hpe conf)
            seg_conf = self.w[zone_name][scan_id_indx]
            assert (0<=seg_conf<=1), "0<= seg_conf:{} <=1 ??".format(seg_conf)
            # retrieve predefined class (benign vs. threat) weight
            class_wgt = self.class_weights[smp_threat_gt]
            # compute sample weight for both network outputs
            # Note,
            # either smp_t_wgt or smp_g_wgt or both must be 0.
            # in other words, only one or neither can be 1.0
            # smp_t_wgt is expected to be 1.0 when the sample in unmarked (ie. no threat-bbox)
            # smp_g_wgt is expected to be 1.0 when the sample is marked and the labels match
            # Hence,
            # when the sample is marked but the labels do not match, the resulting weights for
            # both smp_t_wgt and smp_g_wgt is 0, hence the model doesn't learn from the sample.
            # Therefore,
            # the model suffers from this setup if there is a high case of mis-matching labels.
            smp_t_wgt = seg_conf * class_wgt * zn_loss_branch_wgt
            smp_g_wgt = seg_conf * class_wgt * gs_loss_branch_wgt * match_wgt

            assert (smp_t_wgt>=0), 'smp_t_wgt: {} must not be negative'.format(smp_t_wgt)
            assert (smp_g_wgt>=0), 'smp_g_wgt: {} must not be negative'.format(smp_g_wgt)
            assert (smp_t_wgt * smp_g_wgt==0), 'at least one must be 0'
            assert (not not_marked or (smp_g_wgt==0)), "a implies b"
            ###assert#(self.use_threat_boxes or smp_t_wgt>0), "smp_t_wgt>0 if no threat boxes"

            # update counters
            if smp_t_wgt>0: self.cnt_on_zone_loss += 1
            elif smp_g_wgt>0: self.cnt_on_iset_loss += 1
            else: self.cnt_neither_loss += 1

        else:
            # when not training, we only care about the zone prediction
            smp_t_wgt, smp_g_wgt = 1.0, 0.0  # 1.0, 1.0

        t1 = time()
        t = t1 - t0
        ##print('elapsed time:{:.3f}, [{:>5},{:>5},{:>4}]'.format(t, sample_id_indx, scan_id_indx, zone_name))
        self.total_time += t
        self.total_call += 1

        #* return smp_imgs, smp_rois, smp_rcvs, smp_bbcs, smp_threat_gt, iset_threat_gt, \
        #*        smp_t_wgt, smp_g_wgt, sample_id_indx, output_indx
        return self.preprocess_func(self.sample_images), \
               self.sample_znrois, self.reg_comp_vecs, self.rel_crp_roi_vertices, \
               smp_threat_gt, iset_threat_gt, smp_t_wgt, smp_g_wgt, sample_id_indx, output_indx

    def load_single_zone_images(self, zone_name, zone_type_indx, scan_id_indx, db=50):
        # Generates image input per scan
        smp_imgs_ndarray = self.X[zone_name][scan_id_indx]
        fids_to_idxes = self.MAP_ZONE_FID_TO_IDX[zone_name]
        #* sample_images = np.empty(self.IMGS_INSAMP_SHAPE, dtype=np.float32) # must be float !!!
        #* sample_znrois = np.zeros(self.ROIS_ACTUAL_SHAPE, dtype=np.float32) # must be np.zeros
        #* reg_comp_vecs = np.empty((self.IMGS_PER_SAMPLE, self.N_ZONES), dtype=np.float32)
        #* reg_threat_lbl = np.full((self.IMGS_PER_SAMPLE), self.UNMARKED, dtype=np.int32)
        if self.use_threat_boxes:
            # threat object locations in frames of scan
            sbj_threat_bboxes = self.threat_bboxes[scan_id_indx]
            is_marked = sbj_threat_bboxes is not None

        # retrieve and organize region crop parameters and images augmentation parameters
        zone_aug_configs = self.augment_configs[zone_name] if self.data_augment else None
        #roi_win_dim = self.ZONE_ROI_DIMS[zone_type_indx]
        prw_dim = self.ZONE_PRW_DIMS[zone_type_indx]
        # sample_augs => (x_shift, y_shift, ang_deg, zoom_fr, hr_flip, px_brgt, px_cont, en_cont)
        sample_augs = self.per_sample_aug_params(zone_aug_configs, zone_type_indx)
        ang_deg, zoom_fr, hr_flip = sample_augs[2:5]
        inv_ang_rad = np.deg2rad(-1*ang_deg)
        ang_rad = np.deg2rad(ang_deg)
        x_ctr, y_ctr = self.NCI_WIN_HALF_WDT, self.NCI_WIN_HALF_HGT  # np.mean(roi_coord, axis=0)
        alpha = np.cos(ang_rad)  # * scale
        beta = np.sin(ang_rad)  # * scale
        rot_mtx = np.asarray([[alpha,  beta, (1.-alpha)*x_ctr - beta*y_ctr],
                              [-beta, alpha, beta*x_ctr + (1.-alpha)*y_ctr]])  # dtype=np.float32
        zoom_mtx = np.asarray([[zoom_fr, 0., x_ctr-zoom_fr*x_ctr],
                               [0., zoom_fr, y_ctr-zoom_fr*y_ctr], [0., 0., 1.]])

        # bounding boxes of all zones in all frames of scan corresponding to scan_indx
        abs_frm_nci_top_left = self.nci_tlc[scan_id_indx] # absolute nci top-left corner in frame
        abs_frm_zoi_vertices = self.roi_coord[scan_id_indx] # absolute roi bbox coordinate in frm
        #* rel_crp_roi_vertices = np.empty(self.ROI_COORD_SHAPE, dtype=np.int32) # relative to NcI

        for idx, fid in enumerate(self.zone_ordered_frames[zone_name]):
            # read frame image
            frame_idx = fids_to_idxes[fid]
            reg_img = smp_imgs_ndarray[frame_idx]
            x_sft, y_sft = sample_augs[0][idx], sample_augs[1][idx]

            # augment image and crop region
            self.sample_images[idx] = \
                self.extract_region_v2(reg_img, prw_dim, idx, sample_augs, zone_aug_configs)

            # generate roi-mask or roi-pool coordinate accordingly
            #print("scan_id_indx:{}, zone_name:{}, idx:{}, fid:{}".format(scan_id_indx, zone_name, idx, fid))
            zoi_coord = abs_frm_zoi_vertices[fid]
            nci_z_tlc = abs_frm_nci_top_left[fid, zone_type_indx] # nci_tlc of zone
            rel_crp_roi = self.get_roi(zoi_coord[zone_type_indx], nci_z_tlc, self.NCI_WIN_DIM,
                                       x_sft, y_sft, rot_mtx, zoom_mtx, hr_flip, db)
            # s_x, e_x, s_y, e_y = self.get_roi(self.NCI_WIN_DIM, roi_win_dim, x_sft, y_sft)
            # assert#(0<=s_x<=self.NCI_WIN_WDT and 0<e_x<=self.NCI_WIN_WDT)
            # assert#(0<=s_y<=self.NCI_WIN_HGT and 0<e_y<=self.NCI_WIN_HGT)
            # rel_crp_roi_vertices[idx] = [s_x, e_x, s_y, e_y]

            self.rel_crp_roi_vertices[idx] = rel_crp_roi
            # if self.ROI_POOLING:
            #     sample_znrois[idx, 0] = [s_x, s_y, e_x, e_y] # ordering for roi-pooling
            # else: # ROI MASKING
            #     sample_znrois[idx, s_y: e_y, s_x: e_x, :] = 1
            self.sample_znrois[idx] = \
                generate_roi_mask(rel_crp_roi, self.NCI_GRID_PTS, self.ROIS_ACTUAL_SHAPE[1:])

            # compute region composite vector
            self.reg_comp_vecs[idx] = \
                self.compute_rcv(zone_type_indx, zoi_coord, x_sft, y_sft, inv_ang_rad, zoom_mtx,
                    self.NCI_WIN_DIM, self.MIN_EXPECTED_IOU_AT_I, self.cfg.AUGMENT.WANDERING_ROI)

            # deduce cropped image threat label from the degree of overlap with a threat object
            if self.use_threat_boxes and is_marked:
                threat_obj_coords = sbj_threat_bboxes[fid]
                self.reg_threat_lbl[idx] = \
                    anomaly_object_overlap(zoi_coord[zone_type_indx], threat_obj_coords,
                            self.SCALE_DIM_FACTOR, self.cfg.LABELS.THREAT_OVERLAP_THRESH)

        # validate computations
        #assert#(np.all(-db<=rel_crp_roi_vertices)), "{}>rel_crp_roi:\n{}".format(-db, rel_crp_roi_vertices)
        #assert#(np.all(rel_crp_roi_vertices[:,:,0]<=self.NCI_WIN_WDT+db)), "x:{}".format(rel_crp_roi_vertices[:,:,0])
        #assert#(np.all(rel_crp_roi_vertices[:,:,1]<=self.NCI_WIN_HGT+db)), "y:{}".format(rel_crp_roi_vertices[:,:,0])
        assert (np.min(self.sample_images)>=0), "sample_images min: {}".format(self.sample_images)
        if self.ROI_POOLING:
            self.sample_znrois[:, :, [0, 2]] /= self.NCI_WIN_WDT
            self.sample_znrois[:, :, [1, 3]] /= self.NCI_WIN_HGT
            assert (np.all(self.sample_znrois>=0) and np.all(self.sample_znrois<=1)), "rois bbox:\n{}".format(self.sample_znrois)

        #* return self.preprocess_func(sample_images), sample_znrois, \
        #*        reg_threat_lbl, reg_comp_vecs, rel_crp_roi_vertices

    def extract_region_v2(self, image, preload_win_dim, idx, img_aug, zone_aug_configs):
        '''
        Apply image augmentation to pre-loaded image before cropping a fixed dimension region
        :param image: pre-loaded image containing roi and additional neighboring pixels
        :param preload_win_dim: (wdt,hgt) dimension of the pre-loaded image
        :param idx: index of image (relative to the array of multi-view images)
        :param img_aug: randomly generated image augmentation configurations
        :param zone_aug_configs: whether or not to apply a specific augmentation to image
        :return: cropped image region after image augmentation
        '''
        x_max, y_max = preload_win_dim
        x_ctr, y_ctr = preload_win_dim // 2

        assert (y_max==image.shape[0] and x_max==image.shape[1])
        assert (0<=x_ctr<x_max and 0<=y_ctr<y_max)
        x_shift, y_shift, ang_deg, zoom_fr, hr_flip, px_brgt, px_cont, en_cont = img_aug

        # Image crop region augmentation
        if self.data_augment:
            # rotate augment
            # rotation about center point (x, y) must happen before shifting x and/or y
            if zone_aug_configs[self.rotate_idx]:
                image = rotate_about_pnt(image, (int(x_ctr), int(y_ctr)), ang_deg, # % 360,
                                         y_max, x_max, self.BG_PIXELS)
            # zoom/scale augment
            if zone_aug_configs[self.s_zoom_idx]:
                image = zoom_about_pnt(image, (x_ctr, y_ctr), zoom_fr,
                                       y_max, x_max, self.BG_PIXELS)
            # x-shift augment
            if zone_aug_configs[self.xshift_idx]:
                x_ctr = x_ctr + x_shift[idx]
                assert (0<x_ctr<x_max)
            # y-shift augment
            if zone_aug_configs[self.yshift_idx]:
                y_ctr = y_ctr + y_shift[idx]
                assert (0<y_ctr<y_max)

        # Crop image region
        s_x = x_ctr - self.NCI_WIN_HALF_WDT
        e_x = x_ctr + self.NCI_WIN_HALF_WDT
        s_y = y_ctr - self.NCI_WIN_HALF_HGT
        e_y = y_ctr + self.NCI_WIN_HALF_HGT
        assert (not (s_x<0 or e_x>x_max)), "preload_win_wdt should have taken into account x-shift"
        assert (not (s_y<0 or e_y>y_max)), "preload_win_hgt should have taken into account y-shift"
        assert ((e_x - s_x)==self.NCI_WIN_WDT), "{}=={} ?".format(e_x - s_x, self.NCI_WIN_WDT)
        assert ((e_y - s_y)==self.NCI_WIN_HGT), "{}=={} ?".format(e_y - s_y, self.NCI_WIN_HGT)
        crop_img = image[s_y:e_y, s_x:e_x]

        # Other image augmentation
        if self.data_augment:
            # horizontal flip augment
            if zone_aug_configs[self.h_flip_idx] and hr_flip==1:
                crop_img = np.fliplr(crop_img)
            # alter contrast augment
            if zone_aug_configs[self.n_cont_idx]:
                crop_img = enhance_contrast_v1(crop_img, en_cont,
                                               self.EC_GRID_TILES, self.EC_COLOR_SPACE_CVT)
            # alter brightness augment.
            # Must come last so that image does not have to be converted back to np.uint8
            if zone_aug_configs[self.bright_idx]:
                crop_img = alter_brightness(crop_img, px_brgt, px_cont)

        assert (crop_img.shape==tuple(self.cfg.MODEL.IMAGE_SIZE)), 'shp:{}'.format(crop_img.shape)
        return crop_img

    def per_sample_aug_params(self, zone_aug_configs, zoi_idx):
        # set default, no-augmentation values
        #* x_shift = np.zeros((self.IMGS_PER_SAMPLE), dtype=np.int32)
        #* y_shift = np.zeros((self.IMGS_PER_SAMPLE), dtype=np.int32)
        x_shift, y_shift = self.x_shift_default, self.y_shift_default
        ang_deg, zoom_fr, hr_flip, px_brgt, px_cont, en_cont = 0, 1., 0, 0, 0, 0.

        # decide whether or not to augment sample if data-augmentation is enabled
        if self.data_augment: roll_dice = randint(1, self.augment_odd_max+1)

        if self.data_augment and roll_dice>self.augment_not_max:
            # shift center left/right
            if zone_aug_configs[self.xshift_idx]:
                max_x_shift = self.MAX_AUG_X_SHIFT[zoi_idx]
                x_shift = randint(-max_x_shift, max_x_shift+1, size=self.IMGS_PER_SAMPLE)

            # shift center up/down
            if zone_aug_configs[self.yshift_idx]:
                max_y_shift = self.MAX_AUG_Y_SHIFT[zoi_idx]
                y_shift = randint(-max_y_shift, max_y_shift+1, size=self.IMGS_PER_SAMPLE)

            # rotate image
            if zone_aug_configs[self.rotate_idx]:
                ang_deg = randint(-self.cfg.AUGMENT.ROTATE, self.cfg.AUGMENT.ROTATE+1)

            # zoom in/out
            if zone_aug_configs[self.s_zoom_idx]:
                zoom_fr = uniform(1-self.cfg.AUGMENT.S_ZOOM, 1+self.cfg.AUGMENT.S_ZOOM)

            # horizontal flip
            if zone_aug_configs[self.h_flip_idx]:
                hr_flip = randint(0, 2)  # 0/1 because high is exclusive

            # change image brightness
            if zone_aug_configs[self.bright_idx]:
                px_brgt = randint(-self.cfg.AUGMENT.BRIGHTNESS, self.cfg.AUGMENT.BRIGHTNESS+1)
                px_cont = randint(-self.cfg.AUGMENT.P_CONTRAST, self.cfg.AUGMENT.P_CONTRAST+1)

            # change image contrast
            if zone_aug_configs[self.n_cont_idx]:
                en_cont = uniform(0, self.cfg.AUGMENT.N_CONTRAST)

        return (x_shift, y_shift, ang_deg, zoom_fr, hr_flip, px_brgt, px_cont, en_cont)

    def log_meta_info(self, logger):
            msg = '\n{} Set Generator:\t{} steps per epoch\n'.\
                format(self.d_set.upper(), self.generator_stpep)
            if logger is not None: logger.log_msg(msg)
            print(msg)


def retrace_oriented_roi_coord(roi_coord_wrt_frm, nci_tlc_wrt_frm, nci_dim,
                               x_shift, y_shift, rot_mtx, zoom_mtx, hr_flip, db):
    '''
    Compute the new region-of-interest bounding-polygon coordinates relative to the
        network-cropped-image after rotation, zoom, x-y-shift, and hr_flip image augmentation
    :param roi_coord_wrt_frm: (x,y) coordinates of 4 vertices of polygon in frame, shape=(4,2)
    :param nci_tlc_wrt_frm: (x,y) coordinate of the top-left corner of the nci in frame
    :param nci_dim: (wdt,hgt) of the network-cropped-image window
    :param x_shift: signed image augmentation shift magnitude on the x-axis
    :param y_shift: signed image augmentation shift magnitude on the y-axis
    :param rot_mtx: 2x3 rotation matrix derived from augmentation rotation angle about nci center
    :param zoom_mtx: 3x3 translation-scale matrix used to zoom-in/out of the roi
    :param hr_flip: 0/1 bit indicating whether image is flipped during augmentation
    :param db: maximum allowance on boundaries of computed roi bounding box coordinates
    :return: recomputed roi bounding-polygon coordinate after image augmentation
    '''
    nci_wdt, nci_hgt = nci_dim
    ##print("\n\nnci_tlc:\n{}".format(np.around(nci_tlc, 0)))
    ##print("frm roi_coord:\n{}".format(np.around(roi_coord_wrt_frm, 1)))

    # retrieve roi coordinates with-respect-to to nci window
    roi_coord_wrt_nci = roi_coord_wrt_frm - nci_tlc_wrt_frm
    ##print("rel roi_coord_wrt_nci:\n{}".format(np.around(roi_coord_wrt_nci, 1)))

    # rotate roi coordinates by rotation matrix derived from augmentation rotation angle
    roi_coord_wrt_nci = np.vstack((roi_coord_wrt_nci.T, np.ones((1,4))))  # homogenous coordinates
    roi_coord_wrt_nci = rot_mtx.dot(roi_coord_wrt_nci)  # (2x3)x(3x4) = (2x4)
    ##print("rot roi_coord_wrt_nci:\n{}".format(np.around(roi_coord_wrt_nci.T, 1)))

    # adjust roi coord
    # roi_coord_wrt_nci = np.vstack((roi_coord_wrt_nci.T, np.ones((1,4))))  # homogenous coordinates
    # roi_hormg_wrt_nci = zoom_mtx.dot(roi_coord_wrt_nci)  # (3x3)x(3x4) = (3x4)
    # roi_coord_wrt_nci = roi_hormg_wrt_nci[:2,:] / roi_hormg_wrt_nci[2,:]
    roi_coord_wrt_nci = transform_points(roi_coord_wrt_nci, zoom_mtx)
    ##print("rot roi_coord_wrt_nci:\n{}".format(np.around(roi_coord_wrt_nci.T, 1)))

    # apply nci x-y-shift augmentation to roi coordinate
    roi_coord_wrt_nci[0] -= x_shift
    roi_coord_wrt_nci[1] -= y_shift
    ##print("sft roi_coord_wrt_nci:\n{}".format(np.around(roi_coord_wrt_nci.T, 1)))

    # flip roi coordinates horizontally
    if hr_flip==1:
        roi_coord_wrt_nci[0] = nci_wdt - roi_coord_wrt_nci[0]  # horizontal flip
        ##print("flp roi_coord_wrt_nci:\n{}".format(np.around(roi_coord_wrt_nci.T, 1)))

    # reshape and validate
    roi_coord_wrt_nci = np.around(roi_coord_wrt_nci.T, 0).astype(np.int32)

    # confirm x_coordinates of ALL roi vertices are not left-side-OOB or right-side-OOB
    #assert(not np.all(roi_coord_wrt_nci[:,0]<0)), "roi_x:{}".format(roi_coord_wrt_nci[:,0])
    #assert(not np.all(roi_coord_wrt_nci[:,0]>nci_wdt)), "roi_x:{}".format(roi_coord_wrt_nci[:,0])
    # confirm y_coordinates of ALL roi vertices are not top-side-OOB or bottom-side-OOB
    #assert(not np.all(roi_coord_wrt_nci[:,1]<0)), "roi_y:{}".format(roi_coord_wrt_nci[:,1])
    #assert(not np.all(roi_coord_wrt_nci[:,1]>nci_hgt)), "roi_y:{}".format(roi_coord_wrt_nci[:,1])
    # confirm coordinates of ALL roi vertices fall within a certain db-error boundary
    #assert(np.all(-db<=roi_coord_wrt_nci)), "-db:{}>\n{}".format(-db, roi_coord_wrt_nci)
    #assert(np.all(roi_coord_wrt_nci[:,0]<=nci_wdt+db)), "x:{}".format(roi_coord_wrt_nci[:,0])
    #assert(np.all(roi_coord_wrt_nci[:,1]<=nci_hgt+db)), "y:{}".format(roi_coord_wrt_nci[:,1])
    return roi_coord_wrt_nci

def retrace_aligned_roi_coord(nci_dim, roi_dim, x_shift, y_shift):
    '''
    For Non-Wandering Region-Of-Interest.
        Compute the new region of interest bounding-box coordinates,
        RELATIVE to the cropped image after x-y-shift image augmentation
    :param nci_dim: network's cropped image dimension
    :param roi_dim: region of interest window dimension
    :param x_shift: shift on x-coordinate +/-
    :param y_shift: shift on y-coordinate +/-
    :return: roi bounding-box coordinate relative to the cropped image
    '''
    # todo: account for rotation also
    x_ctr, y_ctr = (nci_dim // 2) + (nci_dim % 2) - [x_shift, y_shift]
    x_lft, x_rgt, y_top, y_btm = bb_half_dims(roi_dim[0], roi_dim[1])

    s_x = x_ctr - x_lft
    e_x = x_ctr + x_rgt
    s_y = y_ctr - y_top
    e_y = y_ctr + y_btm

    #assert(0<=s_x<=nci_dim[0] and 0<e_x<=nci_dim[0]), "{}, {}: {}".format(s_x, e_x, nci_dim[0])
    #assert(0<=s_y<=nci_dim[1] and 0<e_y<=nci_dim[1]), "{}, {}: {}".format(s_y, e_y, nci_dim[1])
    # newly added
    #assert(e_x - s_x==roi_dim[0]), "{} - {} = {}: {}".format(e_x, s_x, e_x - s_x, roi_dim[0])
    #assert(e_y - s_y==roi_dim[1]), "{} - {} = {}: {}".format(e_y, s_y, e_y - s_y, roi_dim[1])
    return s_x, e_x, s_y, e_y

def moving_aligned_roi_coord(nci_dim, roi_dim, x_shift, y_shift):
    '''
    For Wandering Region-Of-Interest.
        Simply compute and return the fixed region of interest bounding-box coordinates,
        RELATIVE to the cropped image, regardless of x-y-shift image augmentation
    :param nci_dim: network's cropped image dimension
    :param roi_dim: region of interest window dimension
    :param x_shift: shift on x-coordinate +/-
    :param y_shift: shift on y-coordinate +/-
    :return: roi bounding-box coordinate relative to the cropped image
    '''
    # todo: account for rotation also
    x_ctr, y_ctr = nci_dim // 2
    x_lft, x_rgt, y_top, y_btm = bb_half_dims(roi_dim[0], roi_dim[1])

    s_x = x_ctr - x_lft
    e_x = x_ctr + x_rgt
    s_y = y_ctr - y_top
    e_y = y_ctr + y_btm

    #assert(0<=s_x<=nci_dim[0] and 0<e_x<=nci_dim[0]), "{}, {}: {}".format(s_x, e_x, nci_dim[0])
    #assert(0<=s_y<=nci_dim[1] and 0<e_y<=nci_dim[1]), "{}, {}: {}".format(s_y, e_y, nci_dim[1])
    # newly added
    #assert(e_x - s_x==roi_dim[0]), "{} - {} = {}: {}".format(e_x, s_x, e_x - s_x, nci_dim[0])
    #assert(e_y - s_y==roi_dim[1]), "{} - {} = {}: {}".format(e_y, s_y, e_y - s_y, nci_dim[1])
    return s_x, e_x, s_y, e_y


def anomaly_object_overlap(roi_bbox_coord, frm_tobj_coords, scale_ftr, oaf_thresh):
    # Computes the overlap between roi and threat object bounding-boxes (in numpy vectorized form)
    # Note: the roi is rigid, hence the coordinate isn't affected by shift augmentation
    # TODO: modify for bounding-polygon instead of bounding-box

    if frm_tobj_coords is None: return 0

    roi_s_x, roi_e_x = roi_bbox_coord[:2] / scale_ftr[0]
    roi_s_y, roi_e_y = roi_bbox_coord[2:] / scale_ftr[1]
    #assert(roi_e_x>=roi_s_x), "{}>{} ?".format(roi_e_x, roi_s_x)
    #assert(roi_e_y>=roi_s_y), "{}>{} ?".format(roi_e_y, roi_s_y)

    t_objs_s_x, t_objs_e_x = frm_tobj_coords[:, 0], frm_tobj_coords[:, 1]
    t_objs_s_y, t_objs_e_y = frm_tobj_coords[:, 2], frm_tobj_coords[:, 3]
    #assert(np.all(t_objs_e_x>=t_objs_s_x)), "{}>{} ?".format(t_objs_e_x, t_objs_s_x)
    #assert(np.all(t_objs_e_y>=t_objs_s_y)), "{}>{} ?".format(t_objs_e_y, t_objs_s_y)
    threatobjs_area = (t_objs_e_y - t_objs_s_y) * (t_objs_e_x - t_objs_s_x)

    intercepts_s_x = np.maximum(roi_s_x, frm_tobj_coords[:, 0]) # intercept_s_x
    intercepts_s_y = np.maximum(roi_s_y, frm_tobj_coords[:, 2]) # intercept_s_x
    intercepts_e_x = np.minimum(roi_e_x, frm_tobj_coords[:, 1]) # intercept_e_x
    intercepts_e_y = np.minimum(roi_e_y, frm_tobj_coords[:, 3])  # intercept_e_x
    intercepts_wdt = np.clip(intercepts_e_x - intercepts_s_x, 0, None)
    intercepts_hgt = np.clip(intercepts_e_y - intercepts_s_y, 0, None)
    intercepts_area = intercepts_hgt * intercepts_wdt
    #assert(np.all(intercepts_area>=0)), "intercept area: {}<0".format(intercepts_area)

    overlaps_frac = intercepts_area / threatobjs_area
    #assert(np.all(overlaps_frac>=0)), "OOB overlap fraction: {}<0".format(overlaps_frac)
    #assert(np.all(overlaps_frac<=1)), "OOB overlap fraction: {}>1".format(overlaps_frac)

    return 1 if np.amax(overlaps_frac)>oaf_thresh else 0


def generate_roi_mask(roi_polygon_vertices, nci_grid_pts, roi_mask_shape):
    '''
    Generate roi-binary-mask on nci window from roi's polygon vertices
    :param roi_polygon_vertices: (x,y) coordinates of polygon vertices, shape=(4,2)
    :param nci_grid_pts: grid (transposed) (x,y) point coordinates of nci
    :param roi_mask_shape: the shape of network-cropped-image, i.e. (hgt, wdt, 1)
    :return: generated roi-mask of dtype=np.float32
    '''
    # todo: test behavior with point coordinates <0
    pts = Path(roi_polygon_vertices)  # make a polygon
    grid = pts.contains_points(nci_grid_pts)
    mask = grid.reshape(roi_mask_shape)  # mask with points inside the polygon
    return mask.astype(np.float32)


def oriented_rcv_as_roi_vs_rois(zoi_idx, zones_poly_coord, x_shift, y_shift,
                                inv_ang_rad, zoom_mtx, nci_dim, min_expected_iou, wandering_roi):
    # Notice that rotation and zoom has no effect on roi_vs_rois rcv because
    # the constraints guarantees the roi is fully contained in all augmentations
    roi_poly_coord = zones_poly_coord[zoi_idx]  # absolute location of Zone of Interest

    # if wandering roi, use x-y-shift to adjust coordinates of roi_bbox_coord
    if wandering_roi:
        roi_poly_coord += np.tile(np.asarray([x_shift, y_shift]), (4, 1))

    # compute region-composite-vector
    n_zones = len(zones_poly_coord)
    reg_comp_vec = np.zeros(n_zones, dtype=np.float32)
    for z_idx in range(n_zones):
        reg_comp_vec[z_idx] = compute_polygons_iou(roi_poly_coord, zones_poly_coord[z_idx])
    #assert(np.all(reg_comp_vec>=0))
    #assert(np.all(np.isnan(reg_comp_vec)==False))
    reg_comp_vec = np.around(reg_comp_vec, 4)

    #iou_at_i = reg_comp_vec[zoi_idx]
    #assert(wandering_roi or np.max(reg_comp_vec)==iou_at_i) # not wandering_roi --> max rcv at i
    return reg_comp_vec

def aligned_rcv_as_roi_vs_rois(zoi_idx, zones_bbox_coord, x_shift, y_shift,
                               inv_ang_rad, zoom_mtx, nci_dim, min_expected_iou, wandering_roi):
    # Notice rotation has no effect on roi_vs_rois rcv
    roi_bbox_coord = zones_bbox_coord[zoi_idx]  # absolute location of Zone of Interest

    # if wandering roi, use x-y-shift to adjust coordinates of roi_bbox_coord
    if wandering_roi:
        roi_bbox_coord += [x_shift, x_shift, y_shift, y_shift]

    # compute region-composite-vector
    reg_comp_vec = compute_recboxes_iou_vec(roi_bbox_coord, zones_bbox_coord)
    #assert(np.all(reg_comp_vec>=0))
    #assert(np.all(np.isnan(reg_comp_vec)==False))
    reg_comp_vec = np.around(reg_comp_vec, 4)

    #iou_at_i = reg_comp_vec[zoi_idx]
    #min_iou_at_i = min_expected_iou[zoi_idx]
    #assert(iou_at_i>=min_iou_at_i), '{}: ({}>={}) ?'.format(zoi_idx, iou_at_i, min_iou_at_i)
    # below assertion is desirable but may not always hold? Especially, between neighboring limbs ?
    #assert(wandering_roi or np.max(reg_comp_vec)==iou_at_i) # not wandering_roi --> max rcv at i
    return reg_comp_vec

def oriented_rcv_as_nci_vs_rois(zoi_idx, zones_poly_coord, x_shift, y_shift,
                                inv_ang_rad, zoom_mtx, nci_dim, min_expected_iou, wandering_roi):
    '''
    Compute region-composite-vector of nci-polygon vs. each zone's roi polygon
    :param zoi_idx: body zone predefined index into ndarray
    :param zones_poly_coord: (x,y) coordinates of 4 vertices of each zone's roi polygon in frame
    :param x_shift: signed image augmentation shift magnitude on the x-axis
    :param y_shift: signed image augmentation shift magnitude on the y-axis
    :param inv_ang_rad: augmentation rotation angle but in opposite direction (radian)
    :param nci_dim: network-cropped-image window dimension (wdt,hgt)
    :param min_expected_iou: minimum expected IoU of each body zone
    :param wandering_roi: boolean indicating whether or not roi moves with shift augmentation
    :return: computed region-composite-vector
    '''
    nci_wdt, nci_hgt = nci_dim
    roi_poly_coord = zones_poly_coord[zoi_idx]  # absolute location of zoi (shape->(4,2))

    # Note, coordinates of zones_poly_coord can be OOB (ie. <0 or >frm_hgt/wdt)
    x_ctr, y_ctr = np.mean(roi_poly_coord, axis=0)

    # trace back absolute position of net-cropped-image, considering x-y-shift
    # if wandering ROI, nci coordinates isn't affected by x-y-shift aug because ROI moves with NCI
    # if ROI isn't wandering, nci coordinates is affected by x-y-shift because NCI moves about ROI
    x_lft, x_rgt, y_top, y_btm = bb_half_dims(nci_wdt, nci_hgt)
    if wandering_roi: x_shift, y_shift = 0, 0
    ns_x, ne_x, ns_y, ne_y = bb_coords(x_ctr, y_ctr, x_lft+x_shift,
                                       x_rgt+x_shift, y_top+y_shift, y_btm+y_shift) # nci dims
    #assert(ne_x - ns_x==nci_wdt)
    #assert(ne_y - ns_y==nci_hgt)
    # nci_poly_coord is a rectangular bbox with order: topLft->topRgt->btmRgt->btmLft
    nci_poly_coord = np.asarray([[ns_x, ns_y, 1], [ne_x, ns_y, 1],
                                 [ns_x, ne_y, 1], [ne_x, ne_y, 1]]).T  # homogenous
    # apply rotation augmentation by inversely rotating nci vertices coordinates
    x_ctr, y_ctr, __ = np.mean(nci_poly_coord, axis=-1)
    alpha = np.cos(inv_ang_rad)  # * scale
    beta = np.sin(inv_ang_rad)  # * scale
    inv_rot_mtx = np.asarray([[alpha,  beta, (1.-alpha)*x_ctr - beta*y_ctr],
                              [-beta, alpha, beta*x_ctr + (1.-alpha)*y_ctr]])
    nci_poly_coord = inv_rot_mtx.dot(nci_poly_coord)  # (shape->(2,4))

    # apply inverse zoom to retrieve vertices coordinates at original scale
    nci_poly_coord = transform_points(nci_poly_coord, np.linalg.inv(zoom_mtx))  # (shape->(2,4))

    # compute region-composite-vector
    n_zones = len(zones_poly_coord)
    reg_comp_vec = np.zeros(n_zones, dtype=np.float32)
    for z_idx in range(n_zones):
        reg_comp_vec[z_idx] = compute_polygons_iou(nci_poly_coord.T, zones_poly_coord[z_idx])
    #assert(np.all(reg_comp_vec>=0))
    #assert(np.all(np.isnan(reg_comp_vec)==False))
    reg_comp_vec = np.around(reg_comp_vec, 4)

    # below assertion is desirable but may not always hold. Especially, between neighboring limbs
    ###assert#(np.max(reg_comp_vec)==reg_comp_vec[zoi_idx])
    return reg_comp_vec

def aligned_rcv_as_nci_vs_rois(zoi_idx, zones_bbox_coord, x_shift, y_shift,
                               inv_ang_rad, zoom_mtx, nci_dim, min_expected_iou, wandering_roi):
    # todo: adjust for rotation
    nci_wdt, nci_hgt = nci_dim
    roi_bbox_coord = zones_bbox_coord[zoi_idx]  # absolute location of Zone of Interest

    # Note, coordinates of zones_bb_coord can be OOB (ie. <0 or >frm_hgt/wdt)
    s_x, e_x, s_y, e_y = roi_bbox_coord
    #assert(e_x>s_x and e_y>s_y)
    roi_wdt = e_x - s_x
    roi_hgt = e_y - s_y
    #assert(nci_wdt>=roi_wdt and nci_hgt>=roi_hgt)

    # trace back absolute position of net-cropped-image, considering x-y-shift
    # if wandering ROI, nci coordinates isn't affected by x-y-shift aug because ROI moves with NCI
    # if ROI isn't wandering, nci coordinates is affected by x-y-shift because NCI moves about ROI
    x_lft, x_rgt, y_top, y_btm = bb_half_dims(nci_wdt - roi_wdt, nci_hgt - roi_hgt)
    if wandering_roi: x_shift, y_shift = 0, 0
    s_x = s_x + x_shift - x_lft
    e_x = e_x + x_shift + x_rgt
    s_y = s_y + y_shift - y_top
    e_y = e_y + y_shift + y_btm
    #assert(e_x - s_x==nci_wdt)
    #assert(e_y - s_y==nci_hgt)
    nci_bb_coord = np.asarray([s_x, e_x, s_y, e_y])

    # compute region-composite-vector
    reg_comp_vec = compute_recboxes_iou_vec(nci_bb_coord, zones_bbox_coord)
    #assert(np.all(reg_comp_vec>=0))
    #assert(np.all(np.isnan(reg_comp_vec)==False))
    reg_comp_vec = np.around(reg_comp_vec, 4)

    #iou_at_i = reg_comp_vec[zoi_idx]
    #min_iou_at_i = min_expected_iou[zoi_idx]
    #assert(iou_at_i>=min_iou_at_i), '{}: ({}>={}) ?'.format(zoi_idx, iou_at_i, min_iou_at_i)
    # below assertion is desirable but may not always hold. Especially, between neighboring limbs
    ###assert#(np.max(reg_comp_vec)==iou_at_i)
    return reg_comp_vec

def compute_recboxes_iou_vec(bbox, bboxes):
    '''
    Numpy vectorized computation of Intersect-Over-Union between bb_1 and bbs_2
    :param bb_1: [s_x, e_x, s_y, e_y] absolute pixel region coordinate
    :param bbs_2: (n, [s_x, e_x, s_y, e_y]) absolute pixel coordinates of n regions
    :return: a vector of size n
    '''
    #if np.all(bb_2==0): return 0  # body part does not have a valid view in frame

    # compute intersection area
    isec_sx = np.maximum(bbox[0], bboxes[:, 0])
    isec_ex = np.minimum(bbox[1], bboxes[:, 1])
    isec_wdt = np.maximum(0, isec_ex - isec_sx)  # expect <0 if no intersection on dimension
    isec_sy = np.maximum(bbox[2], bboxes[:, 2])
    isec_ey = np.minimum(bbox[3], bboxes[:, 3])
    isec_hgt = np.maximum(0, isec_ey - isec_sy)  # expect <0 if no intersection on dimension
    isec_area = isec_wdt * isec_hgt

    # compute bb_1 and bb_2 areas
    bb_1_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    bb_2_area = (bboxes[:, 1] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 2])

    # compute and return intersection over union
    union_area = bb_1_area + bb_2_area - isec_area
    return isec_area / union_area

def compute_recboxes_iou(bb_1, bb_2):
    '''
    Compute the Intersect-Over-Union between bb_1 and bb_2
    :param bb_1: [s_x, e_x, s_y, e_y] absolute pixel region coordinate
    :param bb_2: [s_x, e_x, s_y, e_y] absolute pixel region coordinate
    :return: computed iou
    '''
    if np.all(bb_2==0): return 0  # body part does not have a valid view in frame

    # compute intersection area
    isec_sx = max(bb_1[0], bb_2[0])
    isec_ex = min(bb_1[1], bb_2[1])
    isec_wdt = max(0, isec_ex - isec_sx)  # expect <0 if no intersection on dimension
    isec_sy = max(bb_1[2], bb_2[2])
    isec_ey = min(bb_1[3], bb_2[3])
    isec_hgt = max(0, isec_ey - isec_sy)  # expect <0 if no intersection on dimension
    isec_area = isec_wdt * isec_hgt

    # compute bb_1 and bb_2 areas
    bb_1_area = (bb_1[1] - bb_1[0]) * (bb_1[3] - bb_1[2])
    bb_2_area = (bb_2[1] - bb_2[0]) * (bb_2[3] - bb_2[2])

    # compute and return intersection over union
    union_area = bb_1_area + bb_2_area - isec_area
    return isec_area / union_area

def compute_polygons_iou(poly_1, poly_2):
    '''
    Compute the Intersect-Over-Union between poly_1 and poly_2
    :param poly_1: (4:vertices, 2:[x,y]) absolute pixel region coordinate
    :param poly_2: (4:vertices, 2:[x,y]) absolute pixel region coordinate
    :return: computed iou
    '''
    #print('\n\n\npoly_1:{}, poly_2:{}\n\n\n'.format(poly_1.shape, poly_2.shape)), sys.exit()
    # Define each polygon
    polygon1_shape = Polygon(poly_1)
    polygon2_shape = Polygon(poly_2)

    # Calculate intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


### IMAGE AUGMENTATION HELPER FUNCTIONS ###

def rotate_about_pnt(image, point, angle, hgt, wdt, bg_pixels):
    '''
    Rotate image by the given angle about the given point
    :param image: image with pixels ranging [0, 255]
    :param point: (x, y) point about which image is rotated
    :param angle: signed angle to rotate image by
    :param hgt: preloaded-region-window height; same as image.shape[0]
    :param wdt: preloaded-region-window width; same as image.shape[1]
    :param bg_pixels: [r, g, b] background pixel constants
    :return: rotated uint8 image
    '''
    if angle==0: return image
    # Rotate image
    M = cv.getRotationMatrix2D(point, angle, 1)
    img = cv.warpAffine(image, M, (wdt, hgt),
                        borderMode=cv.BORDER_CONSTANT, borderValue=bg_pixels)
    return img

def zoom_about_pnt(image, point, zoom_ftr, hgt, wdt, bg_pixels):
    '''
    Zoom in/out (depending on the zoom_fr) about the given point
    :param image: image with the pixels ranging from [0, 255]
    :param point: (x, y) point about which the image is zoomed
    :param zoom_ftr: specifies the scale to scale-up/zoom-in (>1.) or scale-down/zoom-out (<1.)
    :param hgt: preloaded-region-window height; same as image.shape[0]
    :param wdt: preloaded-region-window width; same as image.shape[1]
    :param bg_pixels: [r, g, b] background pixel constants
    :return: rotated uint8 image
    '''
    #assert(0<zoom_ftr<2), 'zoom_ftr:{}'.format(zoom_ftr)
    if zoom_ftr==1: return image
    # Zoom in/out
    # translation and scale transformation matrix.
    x, y = point
    M = np.asarray([[zoom_ftr, 0., x-zoom_ftr*x], [0., zoom_ftr, y-zoom_ftr*y]])
    img = cv.warpAffine(image, M, (wdt, hgt),
                        borderMode=cv.BORDER_CONSTANT, borderValue=bg_pixels)
    return img

def enhance_contrast_v1(image, factor, grid_tiles=(5,5),
                        color_space_cvt=(cv.COLOR_RGB2LAB, cv.COLOR_LAB2RGB)):
    '''
    Enhance image contrast with adaptive histogram equalization
        on luminus channel of lab color-space
    :param image: image with pixels ranging [0, 255]
    :param factor: image contrast enhancement factor
    :return: contrast enhanced unit8 image
    '''
    if factor<1: return image # drastic change when factor<=0
    # enhance image contrast to make more visible
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=factor, tileGridSize=grid_tiles)
    lab = cv.cvtColor(image, color_space_cvt[0])  # convert from RGB/BGR to LAB color space
    #assert(lab.shape[2]==3)
    l = lab[:, :, 0]
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab[:, :, 0] = l2   # merge channels
    return cv.cvtColor(lab, color_space_cvt[1])  # convert from LAB back to RGB/BGR

def enhance_contrast_v2(image, factor, clahe,
                        color_space_cvt=(cv.COLOR_RGB2LAB, cv.COLOR_LAB2RGB)):
    '''
    Enhance image contrast with adaptive histogram equalization
    :param image: image with pixels ranging [0, 255]
    :param factor: image contrast enhancement factor
    :param clahe: pre-initialized clache object for enhancement
    :return: contrast enhanced unit8 image
    '''
    if factor<1: return image # drastic change when factor<=0
    # enhance image contrast to make more visible
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv.cvtColor(image, color_space_cvt[0])  # convert from RGB/BGR to LAB color space
    #assert(lab.shape[2]==3)
    l = lab[:, :, 0]
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab[:, :, 0] = l2   # merge channels
    return cv.cvtColor(lab, color_space_cvt[1])  # convert from LAB back to RGB/BGR

def alter_brightness(image, brightness, contrast):
    '''
    Lighten/darken image by adjusting image brightness and contrast
    :param image: image with pixels ranging [0, 255]
    :param brightness: brightness adjustment factor
    :param contrast: contrast adjustment factor
    :return: adjusted unit8 image
    '''
    #assert(-127<=brightness<=127 and -127<=contrast<=127)
    img = np.int32(image)
    img = img * (contrast/127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return img

def transform_points(pts_coord, trans_mtx):
    #assert(pts_coord.shape==(2,4)), 'pts_coord.shape:{}'.format(pts_coord.shape)
    if np.array_equal(trans_mtx, np.identity(3)): return pts_coord
    #hom_pts_coord = np.vstack((pts_coord.T, np.ones((1,4))))  # homogenous coordinates
    hom_pts_coord = np.vstack((pts_coord, np.ones((1,4))))  # homogenous coordinates
    hom_pts_coord = trans_mtx.dot(hom_pts_coord)  # (3x3)x(3x4) = (3x4)
    return hom_pts_coord[:2,:] / hom_pts_coord[2,:]

def custom_preprocess(images, relative_to_img=True):
    if relative_to_img:
        return ((images / np.max(images)) * 2) - 1
    # dependent of image and identical to mobilenet_v2 preprocessing
    return (images / 127.5) - 1.0