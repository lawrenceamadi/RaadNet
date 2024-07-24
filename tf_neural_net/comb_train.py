'''
    Train and validate model
'''
from __future__ import absolute_import, division, print_function, unicode_literals

##print('\nZone Main Script Called\n')
# Code Snippet for reproducible results
# https://github.com/keras-team/keras/issues/2743
# https://keras.io/getting-started/faq/..
# ..#how-can-i-obtain-reproducible-results-using-keras-during-development
#------------------------------------------------------------------------------
# START
import os
import gc
import sys
# Run script with command: $ PYTHONHASHSEED=0 python your_program.py
# or uncomment command below to use only CPU:
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
##os.environ['PYTHONHASHSEED'] = '0' # also needed for reproducibility

import random
import numpy as np
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.config.experimental_run_functions_eagerly(True) # deprecated
# tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Necessary for reproducibility with tf.keras
random.seed(12345) # Python generated random numbers in a well-defined state.
np.random.seed(42) # Numpy generated random numbers in a well-defined initial state.
tf.random.set_seed(57684) # tf generated random numbers in a well-defined initial state.

physical_gpus = tf.config.experimental.list_physical_devices('GPU')
n_physical_gpus = len(physical_gpus)
if n_physical_gpus>0:
    # Allow dynamic memory growth on demand
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        virtual_gpu_bsf = 1.0
        # BEGIN
        # Create 2 virtual GPUs with 1GB memory each (in dev on laptop)
        if n_physical_gpus==1:
            cmd = input('[PROMPT] {} GPU is detected. Would you like to create virtual GPUs? '
                        '\ny/yes, n/no - '.format(n_physical_gpus))
            if cmd.lower() in ['y','yes']:
                print("Creating virtual GPUs..")
                virtual_gpu_bsf = 0.5  # virtual gpu batch size per replica factor
                tf.config.experimental.set_virtual_device_configuration(
                    physical_gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        # END
        n_logical_gpus = len(tf.config.experimental.list_logical_devices('GPU'))
        print("\n{} Physical GPUs, {} Logical GPUs".format(n_physical_gpus, n_logical_gpus))
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        print('\nMemory growth must be done before GPUs are initialized\nOR')
        print('\nVirtual devices must be set before GPUs have been initialized')
        sys.exit(0)
else:
    cmd = input('[PROMPT] {} GPU is detected, continue with CPU? y/yes, n/no - '.format(n_physical_gpus))
    if cmd.lower() in ['n','no']: sys.exit(0)
    n_physical_gpus, n_logical_gpus, virtual_gpu_bsf = 1, 1, 1.

# END
#------------------------------------------------------------------------------

# turn off the AVX/FMA CPU warning
##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
#------------------------------------------------------------------------------
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
from sklearn.preprocessing import minmax_scale, scale

sys.path.append('../')
from tf_neural_net.comb_pipeline import DataPipeline
from tf_neural_net.data_preprocess import read_image
from tf_neural_net.data_preprocess import crop_aligned_rois, crop_oriented_rois  # crop_rois_func
from tf_neural_net.keras_model import fe_network
from tf_neural_net.keras_logger import TensorBoardWriter, scale_image_pixels
from tf_neural_net.build_models import grp_model_v3_td, grp_model_v4_td  # net_arch
from tf_neural_net.build_models import grp_model_v5_td, grp_model_v6_td  # net_arch
from tf_neural_net.custom_objects import get_optimizer, get_loss, CustomBCELoss
from tf_neural_net.comb_callbacks import ValidMetricCallback, OptimalStepsCallback
from tf_neural_net.comb_callbacks import StepWiseLearningRateScheduler, CosineLearningRateScheduler
from tf_neural_net.default import default_experiment_configuration
from tf_neural_net.default import runtime_args, update_config
from tf_neural_net.evaluation import run_evaluations, get_labels_and_probs
from tf_neural_net.evaluation import get_hdf5_model, get_json_model, load_weights_into_model
from tf_neural_net.commons import get_pseudo_data_naming_config
from tf_neural_net.commons import get_roi_dims, remove_files, get_clip_values, df_map_to_dict_map
from tf_neural_net.commons import frm_zone_kpts_map_type, adjust_xy_shift, map_aps_frames_2_a3daps
from tf_neural_net.commons import define_region_windows, cv_close_windows, get_grp_ordered_zones
from tf_neural_net.commons import replace_keys_with_a3daps_fids, scheduler, truncate_strings
from tf_neural_net.commons import replace_entries_with_a3daps_fids, rename_columns_with_a3daps_fids
from tf_neural_net.commons import zone_index_id_map, log_bdgrp_seg_config, anomaly_objects_location
from tf_neural_net.commons import BDGRP_ALIGNED, BDGRP_ORIENTED, BDGRP_ADJUSTMENT, BODY_ZONES_GROUP
from tf_neural_net.commons import DATA_NUM_FRAMES, ZONE_NAME_TO_ID, SUBNET_CONFIGS, TSA_PIXELS_RANGE


def single_zone_data_with_labels(zone_name, subset_df, subset_tags):
    '''
    fetches the sampleID (combination of scanID and zoneName) of a particular body zone
    and their corresponding strong TSA ground-truth labels from a subset of the dataset
    :param zone_name: name of zone to filter by or fetch samples of
    :param subset_df: a filtered dataframe containing data of the subset
    :param subset_tags: a list of dataset subset tags (train, valid, or test) that are in subset
    :return: a list of lists containing sample IDs and a dictionary (key:sampleID, value:gtLabel)
    '''
    small_portion = _cfg.DATASET.SUBSETS.SMALL_PORTION
    grouping_col = _cfg.DATASET.SUBSETS.GROUP_TYPE
    gtc_df = pd.read_csv(_cfg.LABELS.CSV.GT_ZONE)
    zone_id = ZONE_NAME_TO_ID[zone_name]
    sample_ids_list = list([[], []])
    class_gt = dict()

    for index, row in subset_df.iterrows():
        if small_portion and index>57: break  # [57, 69]
        scan_id = row['scanID']
        set_t = row[grouping_col]
        sample_id = '{}_{}'.format(scan_id, zone_name)
        df_id = '{}_{}'.format(scan_id, zone_id)
        gt_label = int(gtc_df.loc[gtc_df['Id']==df_id, 'Probability'])
        assert(set_t in subset_tags)
        sample_ids_list[gt_label].append(sample_id)
        class_gt[sample_id] = gt_label

    return sample_ids_list, class_gt


def single_zone_data_with_pseudo_labels(zone_name, sample_ids_list=None, class_gt=None, thresh=0.5):
    '''
    fetches the sampleID (combination of scanID and zoneName) of a particular body zone
    and their corresponding pseudo ground-truth labels derived from previously predicted
    threat probabilities (of the test subset)
    :param zone_name: name of zone to filter by or fetch samples of
    :param sample_ids_list: list of lists containing sample IDs OR None
    :param class_gt: a dictionary of (key:sampleID, value:gtLabel) OR None
    :param thresh: the threat probability threshold for threat class
    :return: updated sample_ids, class_gt, and threat probabilities as pseudo label confidence
    '''
    assert(_cfg.LABELS.USE_PSEUDO_LABELS)
    prob_df = pd.read_csv(_cfg.LABELS.CSV.PRED_PROB)
    zone_id = ZONE_NAME_TO_ID[zone_name]
    if sample_ids_list is None: sample_ids = list([[], []])
    if class_gt is None: class_gt = dict()
    lbl_conf = dict()

    for index, row in prob_df.iterrows():
        scan_id, row_zone_id = row['Id'].split('_')
        if row_zone_id==zone_id: # and index<170:
            t_prob = round(float(row['Probability']), 4)
            if t_prob>thresh:
                gt_label = 1
                gt_conf = t_prob
            else:
                gt_label = 0
                gt_conf = 1 - t_prob
            sample_id = '{}_{}'.format(scan_id, zone_name)
            sample_ids_list[gt_label].append(sample_id)
            class_gt[sample_id] = gt_label
            lbl_conf[sample_id] = gt_conf

    return sample_ids_list, class_gt, lbl_conf


def combine_resample(unique_smp_ids_list, unique_smp_lbs_dict,
                     zone_name, resample_threat, start_index):
    # resample by duplicating indexes into data_ids_list
    unique_smp_lbs_list = list(unique_smp_lbs_dict.values())
    n_total_ = len(unique_smp_lbs_list)
    n_minority = np.sum(unique_smp_lbs_list)  # sums 1s
    n_majority = n_total_ - n_minority
    assert(n_total_==(len(unique_smp_ids_list[0]) + len(unique_smp_ids_list[1]))) #unique entries

    n_benign_ids = len(unique_smp_ids_list[0])
    n_threat_ids = len(unique_smp_ids_list[1])
    assert(n_majority==n_benign_ids)
    # order matters in unique_smp_ids. benign samples before threat samples
    unique_smp_ids = unique_smp_ids_list[0] + unique_smp_ids_list[1]

    assert(start_index>=0)
    start_threat_index = start_index + n_benign_ids
    exend_threat_index = start_threat_index + n_threat_ids # exclusive end
    benign_ids_indexes = np.arange(start_index, start_threat_index, dtype=np.int32)
    threat_ids_indexes = np.arange(start_threat_index, exend_threat_index, dtype=np.int32)

    if resample_threat:
        resample_factor = int(round((n_total_ - n_minority) / n_minority, 0))
        n_minority_new = n_minority * resample_factor
        threat_ids_indexes = np.tile(threat_ids_indexes, resample_factor)
    else:
        n_minority_new = n_minority
        resample_factor = 1

    smp_ids_indexes = np.concatenate([benign_ids_indexes, threat_ids_indexes]) # possibly resampled

    msg = '\n{} Zone:'.format(zone_name)
    msg += '\nBefore resampling: {} total, {} benign samples, {} threat samples'.\
            format(n_total_, n_majority, n_minority)
    msg += '\n\tminority class resampled by a factor of: {}'.format(resample_factor - 1)
    msg += '\nAfter resampling: {} total, {} benign samples, {} threat samples'.\
            format(smp_ids_indexes.shape[0], n_majority, n_minority_new)

    counters = {'n_benign': n_majority, 'n_threat': n_minority_new, 'n_minority': n_minority}
    return unique_smp_ids, smp_ids_indexes, msg, counters


def collect_body_group_data(subset_df, d_set, subset_tags):
    msg = '\n{} Dataset Info.:\n-------------------------------'.format(d_set.upper())
    _logger.log_msg('\n' + msg), print(msg)
    unique_samples = 0

    # Collect data for each zone in group
    unique_ids_list = list() # list of lists
    lbs_dict = dict() # list of dicts
    lbs_conf_dict = dict() # list of dicts
    ids_indexes_list = list()
    grp_indexes = dict() # key: group name, value: list of indexes of all ids in group
    per_zone_samples = dict()
    samples_cnt, benign_cnt, threat_cnt, minority_cnt = 0, 0, 0, 0
    resample = eval('_cfg.{}.MINORITY_RESAMPLE'.format(d_set.upper()))
    start_index, expected_n_frames = 0, 0

    for z_idx in range(len(_present_zones)):
        zone_name = _present_zones[z_idx]
        derived_n_frames = len(_zone_ordered_fids[zone_name])

        if len(_present_zones)==2:
            # make assertion only when dealing with zones of a single group
            if z_idx==0: expected_n_frames = derived_n_frames
            else: assert(expected_n_frames==derived_n_frames)
        if derived_n_frames>expected_n_frames:
            expected_n_frames = derived_n_frames

        # Get training data samples and corresponding labels
        unique_ids, lbs = single_zone_data_with_labels(zone_name, subset_df, subset_tags)
        if _cfg.LABELS.USE_PSEUDO_LABELS and d_set=='train':
            unique_ids, lbs, lbs_conf = \
                single_zone_data_with_pseudo_labels(zone_name, unique_ids, lbs)
            lbs_conf_dict.update(lbs_conf)

        unique_samples += len(unique_ids[0]) + len(unique_ids[1])
        # Re-sample data sample indexes to balance classes
        unique_ids, ids_indexes, msg, count = \
            combine_resample(unique_ids, lbs, zone_name, resample, start_index)
        _logger.log_msg(msg), print(msg)
        # Update or add to sample ids' list, sample ids indexes', and sample labels' dicts
        unique_ids_list.extend(unique_ids) # must be done in order (order matters)
        lbs_dict.update(lbs)
        ids_indexes_list.append(ids_indexes)
        # register indexes of zone ids with corresponding group in dictionary
        grp_name = BODY_ZONES_GROUP[zone_name]
        grp_indexes[grp_name] = grp_indexes.get(grp_name, list()) + list(ids_indexes)

        # Update counters
        per_zone_samples[zone_name] = len(unique_ids)
        start_index += len(unique_ids)
        samples_cnt += ids_indexes.shape[0]
        benign_cnt += count['n_benign']
        threat_cnt += count['n_threat']
        minority_cnt += count['n_minority']

    ids_indexes = np.concatenate(ids_indexes_list)
    max_index = np.max(ids_indexes)
    np.random.shuffle(ids_indexes) # order is irrelevant
    assert(ids_indexes.shape[0]==samples_cnt)
    assert(unique_samples==len(unique_ids_list))
    assert(max_index==len(unique_ids_list) - 1)

    msg = '\nCollected {} Set:'.format(d_set.upper())
    msg += '\n{:,} total samples or {:,} total images.'. \
            format(samples_cnt, samples_cnt * expected_n_frames)
    msg += '\n{:,} benign samples, {:,} resampled & {:,} actual threat samples'. \
            format(benign_cnt, threat_cnt, minority_cnt)
    msg += '\n{:,} unique total samples, index ndarray dtype: {}'.\
            format(unique_samples, ids_indexes.dtype)
    _logger.log_msg(msg), print(msg)
    return ids_indexes, unique_ids_list, lbs_dict, \
           lbs_conf_dict, grp_indexes, minority_cnt, per_zone_samples


def get_region_data(sample_ids_indexes, unique_sample_ids, sample_lbs_dict, sample_lbs_conf_dict,
                    set_kpts_df, perzone_nsamples, prw_shapes, d_set, debug_display=False):
    cropped_images = dict()
    sample_gtlabel = dict()
    lbl_confidence = dict()
    seg_confidence = dict()
    byte_size = 0
    for z_name in _present_zones:
        a_shp = (perzone_nsamples[z_name], len(_zone_ordered_fids[z_name]), *prw_shapes[z_name])
        cropped_images[z_name] = np.empty(a_shp, dtype=np.uint8) # eg. (1247,?12,?87,?85,3)
        sample_gtlabel[z_name] = np.empty((perzone_nsamples[z_name]), dtype=np.uint8) # {0,1}
        lbl_confidence[z_name] = np.empty((perzone_nsamples[z_name]), dtype=np.float16)
        seg_confidence[z_name] = np.empty((perzone_nsamples[z_name]), dtype=np.float16)
        byte_size += cropped_images[z_name].nbytes

    msg = '\nReading and parsing images and majority inputs to memory\n{}-set '.format(d_set)
    msg += 'images will occupy {:,} bytes or {:.1f} GB'.format(byte_size, byte_size / (1024**3))
    _logger.log_msg(msg), print(msg)

    pzs_cnts = np.asarray(list(perzone_nsamples.values()))  # per-zone-sample counts
    assert(np.all(pzs_cnts==pzs_cnts[0])), 'all zones must have equal counts of unique cases'
    # Track body part or zone's bounding-box or bounding polygon
    all_roi_bbs = np.zeros((pzs_cnts[0], N_FRAMES, 17, *ROI_BB_SHAPE), dtype=np.int16)
    all_nci_tlc = np.zeros((pzs_cnts[0], N_FRAMES, 17, 1, 2), dtype=np.int16) # nci top-left-corner

    # list data structure for threat object locations
    use_threat_boxes = _cfg.LABELS.USE_THREAT_BBOX_ANOTS and d_set=='train'
    if use_threat_boxes:
        threat_bboxes = [list()] * pzs_cnts[0]
        # DataFrame containing manually labelled bounding-box location of threat objects in scan
        threat_obj_df = pd.read_csv(_cfg.LABELS.CSV.THREAT_OBJ_POS)
    else: threat_bboxes = None

    # collect list of scan ids for train and test set if using soft labels of test-set
    with_testset_pseudo_labels = _cfg.LABELS.USE_PSEUDO_LABELS and d_set=='train'
    if with_testset_pseudo_labels:
        train_set_scanids = os.listdir(_cfg.DATASET.IMAGE_DIR)

    # initialize some variables for cropping regions
    crop_rois_func = eval("crop_{}_rois".format(_cfg.MODEL.EXTRA.ROI_TYPE))
    rgb_img = _cfg.DATASET.COLOR_RGB  # if true the read BGR image is converted to RGB
    nci_dim = _cfg.MODEL.IMAGE_SIZE[:2]  # or _cfg.MODEL.REGION_DIM * SCALE_DIM_FACTOR
    db = _cfg.DATASET.XY_BOUNDARY_ERROR  # allowance for nci/roi coordinates to fall Out-Of-Bounds
    poly_deg = _cfg.MODEL.EXTRA.POLYNOMIAL_DEGREE # degree of polynomial to fit to spatial hist-line
    if debug_display:
        fig, ax_sp = plt.subplots(figsize=(3, 3))
        fig.tight_layout(pad=0)
    else: ax_sp = None

    n_samples = len(sample_ids_indexes)
    n_unique_samples = len(unique_sample_ids)
    ds_sample_indexes = np.empty((n_samples, 3), dtype=sample_ids_indexes.dtype)
    ds_sample_indexes_dict = dict()
    scanid_indexer_into_zone_array = dict()
    new_scan_index = 0
    unique_samples_cnt = 0

    for i, sample_id_indx in enumerate(sample_ids_indexes):
        sample_id = unique_sample_ids[sample_id_indx]
        sample_indexes = ds_sample_indexes_dict.get(sample_id, None)

        # Process each UNIQUE sample_id ONLY once
        if sample_indexes is None:
            unique_samples_cnt += 1
            scan_id, z_name = sample_id.split('_')
            ##print("\n{:>2}. {}:{}\n".format(i, sample_id_indx, sample_id))

            # find location of scan images
            if with_testset_pseudo_labels:
                if scan_id in train_set_scanids: img_dir = _cfg.DATASET.IMAGE_DIR
                else: img_dir = _cfg.DATASET.W_IMG_DIR
            else: img_dir = _cfg.DATASET.IMAGE_DIR

            scan_dir = os.path.join(img_dir, scan_id)
            ordered_fids_of_zone = _zone_ordered_fids[z_name]
            n_unique_ipz = len(ordered_fids_of_zone)
            zone_type_indx = ZONE_TAG_TO_IDX[z_name]
            scan_id_indx = scanid_indexer_into_zone_array.get(scan_id, None)

            if scan_id_indx is None:
                scan_id_indx = new_scan_index
                scanid_indexer_into_zone_array[scan_id] = scan_id_indx
                new_scan_index += 1

                if use_threat_boxes:
                    # parse threat object locations
                    threat_bboxes[scan_id_indx] = \
                        anomaly_objects_location(scan_id, threat_obj_df, N_FRAMES)

            assert(scan_id_indx<perzone_nsamples[z_name])
            sample_gtlabel[z_name][scan_id_indx] = sample_lbs_dict[sample_id]
            lbl_confidence[z_name][scan_id_indx] = sample_lbs_conf_dict.get(sample_id, 1.)
            reg_imgs, reg_conf, roi_bbs, nci_tlc = \
                crop_rois_func(scan_dir, scan_id, z_name, ordered_fids_of_zone, set_kpts_df,
                               _zfk_map[z_name], ZFK_MAP_TYPE[z_name], prw_shapes[z_name],
                               n_unique_ipz, BDGRP_CROP_CONFIG, BDGRP_ADJUSTMENT, FRM_DIM,
                               SCALE_DIM_FACTOR, FRM_CORNERS, IMG_CEF, BG_PIXELS, nci_dim,
                               poly_deg, rgb_img, ax_sp, debug_display, db)
            cropped_images[z_name][scan_id_indx] = reg_imgs
            seg_confidence[z_name][scan_id_indx] = np.mean(reg_conf)
            for idx, fid in enumerate(ordered_fids_of_zone):
                all_roi_bbs[scan_id_indx, fid, zone_type_indx] = roi_bbs[idx]
                all_nci_tlc[scan_id_indx, fid, zone_type_indx] = nci_tlc[idx]

            sample_indexes = [sample_id_indx, zone_type_indx, scan_id_indx]
            ds_sample_indexes_dict[sample_id] = sample_indexes

        ds_sample_indexes[i] = sample_indexes
        if (i+1)%1000==0 or (i+1)==n_samples:
            msg = '{:>9,} of {:,} samples passed, {:>9,} of {:,} unique samples memoized'.\
                    format(i+1, n_samples, unique_samples_cnt, n_unique_samples)
            _logger.log_msg(msg), print(msg)

    assert(np.all(np.sum(all_roi_bbs[:,:,:,:,0], axis=-1)>=0)), "sum of each roi x coord >= 0"
    assert(np.all(np.sum(all_roi_bbs[:,:,:,:,1], axis=-1)>=0)), "sum of each roi y coord >= 0"

    if _cfg.DATASET.SCALE_PIXELS:
        channels = ('red','green','blue') if _cfg.DATASET.COLOR_RGB else ('blue','green','red')
        for z_name in _present_zones:
            for idx, color_channel in enumerate(channels):
                # eg. (1247,?12,?87,?85,3)
                cropped_images[z_name][:,:,:,:,idx] = \
                    scale_image_pixels(cropped_images[z_name][:,:,:,:,idx],
                                       PIXELS_MINMAX[color_channel], ret_uint8=True)

    if _cfg.LOSS.PASS_SAMPLE_WGTS and d_set=='train':
        seg_gtlb_confs = aggregate_sample_confidence(seg_confidence, lbl_confidence)
    else: seg_gtlb_confs = None

    if debug_display:
        cv_close_windows()
        plt.close()
    return ds_sample_indexes, \
           (cropped_images, sample_gtlabel, seg_gtlb_confs, all_roi_bbs, all_nci_tlc, threat_bboxes)


def aggregate_sample_confidence(seg_confidence, lbl_confidence):
    '''
    First do min-max scaling of segmentation confidence, then aggregate scaled
    segmentation confidence and gt-label confidence by multiplying the two
    :param seg_confidence: dictionary of (key:zone_name, value:ndarray)
    :param lbl_confidence: dictionary of (key:zone_name, value:ndarray)
    :return: aggregated segmentation and gt-label confidence
    '''
    seg_lbl_conf_agg = dict()
    z_max = _cfg.LOSS.SEG_CONFIDENCE_MAX
    msg = '\nAggregating segmentation and label confidence per zone, (min, avg, max)\n'

    for z_name in _present_zones:
        z_lbl_conf = lbl_confidence[z_name]
        z_seg_conf = seg_confidence[z_name]
        z_min = _cfg.LOSS.SEG_CONFIDENCE_MIN
        if z_min==z_max:
            # reset segmentation confidence to 1.0 in order to have the effect of training
            # without seg_confidence. Then multiply by label confidence
            z_agg_conf = np.around(1. * z_lbl_conf, 4)
        else:
            z_min = max(np.min(z_seg_conf), z_min)
            assert(0<z_min<z_max)
            assert(z_seg_conf.ndim==1)
            msg += '\t{:<7}\tInitial Seg. Conf: ({:.2f}, {:.2f}, {:.2f})'.\
                    format(z_name, np.min(z_seg_conf), np.mean(z_seg_conf), np.max(z_seg_conf))

            if _cfg.LOSS.SEG_CONFIDENCE_CTR:
                z_seg_conf = scale(z_seg_conf, with_mean=True, with_std=0, copy=False)
                msg += '\tAfter Mean Centering: ({:>5.2f}, {:>5.2f}, {:>5.2f})'. \
                    format(np.min(z_seg_conf), np.mean(z_seg_conf), np.max(z_seg_conf))

            z_seg_conf = minmax_scale(z_seg_conf, feature_range=(z_min, z_max), copy=False)
            z_agg_conf = np.around(z_seg_conf * z_lbl_conf, 4)

        seg_lbl_conf_agg[z_name] = z_agg_conf # todo: reset type to float16?
        msg += '\tSegConf After Scaling: ({:.2f}, {:.2f}, {:.2f})'. \
                format(np.min(z_seg_conf), np.mean(z_seg_conf), np.max(z_seg_conf))
        msg += '\tGT. Label Conf: ({:.2f}, {:.2f}, {:.2f})'. \
                format(np.min(z_lbl_conf), np.mean(z_lbl_conf), np.max(z_lbl_conf))
        msg += '\tAggregated Conf: ({:.2f}, {:.2f}, {:.2f})\n'. \
                format(np.min(z_agg_conf), np.mean(z_agg_conf), np.max(z_agg_conf))

    _logger.log_msg(msg), print(msg)
    return seg_lbl_conf_agg


def set_wgt_shape_xyw_out(x, y, w):
    if _is_multi_outputs:
        for idx, output_id in enumerate(_cfg.LOSS.NET_OUTPUTS_ID):
            #w[output_id] = tf.reshape(w[output_id], (_cfg.TRAIN.BATCH_SIZE,))
            w[idx] = tf.reshape(w[idx], (_cfg.TRAIN.BATCH_SIZE,))
        return x, y, w
    return x, y, tf.reshape(w, (_cfg.TRAIN.BATCH_SIZE,))


def training_dataset(pipe_obj, sample_indexes, interleave=False, cache_path=False):
    if _cfg.LOSS.PASS_SAMPLE_WGTS:
        # create (image, label, weight) zip to iterate over
        map_func = pipe_obj.feed_sample_inputs_xyw_out
    else:
        # create (image, label) zip to iterate over
        map_func = pipe_obj.feed_sample_inputs_xy_out

    if _cfg.TRAIN.WORKERS<0: n_workers = tf.data.AUTOTUNE
    elif _cfg.TRAIN.WORKERS==0: n_workers = None
    else: n_workers = _cfg.TRAIN.WORKERS
    n = len(sample_indexes)
    #n_epochs = _cfg.TRAIN.END_EPOCH - _cfg.TRAIN.BEGIN_EPOCH

    # training data producer
    ds = tf.data.Dataset.from_tensor_slices(sample_indexes)
    ds = ds.shuffle(n, seed=n, reshuffle_each_iteration=_cfg.TRAIN.EOE_SHUFFLE)
    if cache_path != False: ds = ds.cache(filename=cache_path)
    ds = ds.repeat()#count=n_epochs)  # repeat AFTER shuffle, BEFORE cache, interleave and map
    ds = ds.map(map_func, num_parallel_calls=n_workers, deterministic=False)
    if interleave:
        ds = ds.interleave(lambda x,y,w: tf.data.Dataset.from_tensors((x,y,w)),
                           num_parallel_calls=n_workers, deterministic=False)
    ds = ds.batch(_cfg.TRAIN.BATCH_SIZE)
    if _cfg.TRAIN.QUEUE_SIZE>0: ds = ds.prefetch(buffer_size=_cfg.TRAIN.QUEUE_SIZE)
    #if cache_path != False: ds = ds.cache(filename=cache_path)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF #DATA
    ds = ds.with_options(options)
    return ds


def validate_dataset_v1_cbk(pipe_obj, sample_indexes, interleave=False, cache_path=False):
    ## custom validation via callback
    # create (image, label, id_index, roi_bb) zip to iterate over
    map_func = pipe_obj.feed_sample_inputs_xym_out

    if _cfg.VALID.WORKERS<0: n_workers = tf.data.AUTOTUNE
    elif _cfg.VALID.WORKERS==0: n_workers = None
    else: n_workers = _cfg.VALID.WORKERS
    n = len(sample_indexes)

    # validation data producer
    ds = tf.data.Dataset.from_tensor_slices(sample_indexes)
    ds = ds.shuffle(n, reshuffle_each_iteration=_cfg.VALID.EOE_SHUFFLE)
    ds = ds.map(map_func, num_parallel_calls=n_workers)
    if interleave:
        ds = ds.interleave(lambda x,y,i,b,o: tf.data.Dataset.from_tensors((x,y,i,b,o)),
                           num_parallel_calls=n_workers, deterministic=True) #False)
    ds = ds.batch(_cfg.VALID.BATCH_SIZE)
    if cache_path != False: ds = ds.cache(filename=cache_path)
    # Most dataset input pipelines should end with a call to prefetch
    if _cfg.VALID.QUEUE_SIZE>0: ds = ds.prefetch(buffer_size=_cfg.VALID.QUEUE_SIZE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF #DATA
    ds = ds.with_options(options)
    return ds


def validate_dataset_v2_fit(pipe_obj, sample_indexes, interleave=False, cache_path=False):
    # generic validation via model.fit

    # create (image, label, weight) zip to iterate over
    map_func = pipe_obj.feed_sample_inputs_xyw_out

    if _cfg.VALID.WORKERS<0: n_workers = tf.data.AUTOTUNE
    elif _cfg.VALID.WORKERS==0: n_workers = None
    else: n_workers = _cfg.VALID.WORKERS
    n = len(sample_indexes)

    # validation data producer
    ds = tf.data.Dataset.from_tensor_slices(sample_indexes)
    ds = ds.shuffle(n, reshuffle_each_iteration=_cfg.VALID.EOE_SHUFFLE)
    ds = ds.map(map_func, num_parallel_calls=n_workers)
    if interleave:
        ds = ds.interleave(lambda x,y,w: tf.data.Dataset.from_tensors((x,y,w)),
                           num_parallel_calls=n_workers, deterministic=True) #False)
    ds = ds.batch(_cfg.VALID.BATCH_SIZE)
    if _cfg.VALID.QUEUE_SIZE>0: ds = ds.prefetch(buffer_size=_cfg.VALID.QUEUE_SIZE)
    if cache_path != False: ds = ds.cache(filename=cache_path)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF #DATA
    ds = ds.with_options(options)
    return ds


def compile_dataset(ids_indexes, unique_ids, lbs_dict, lbs_conf_dict,
                    subset_kpts_df, perzone_smpcnt, prw_shapes, d_set, subset_tags):
    data_cfg = get_pseudo_data_naming_config(_cfg, len(unique_ids), d_set, subset_tags)
    data_dir = os.path.join(_cfg.DATASET.RETAINED_DATA_DIR, data_cfg)
    data_configs = os.listdir(_cfg.DATASET.RETAINED_DATA_DIR)

    is_ds_cached = eval('_cfg.{}.CACHE_TF_DATA'.format(d_set.upper()))
    keep_in_memory = _cfg.DATASET.RETAIN_IN_MEMORY and not is_ds_cached

    if data_cfg not in data_configs or _cfg.DATASET.FRESH_BUILD:
        # compile data
        smp_indexes, data = \
            get_region_data(ids_indexes, unique_ids, lbs_dict, lbs_conf_dict, subset_kpts_df,
                            perzone_smpcnt, prw_shapes, d_set, _cfg.DEBUG.TURN_ON_DISPLAY)

        if _cfg.DATASET.PERSIST_ON_DISK:
            # save on disk
            os.makedirs(data_dir, exist_ok=True)
            print('[INFO] Writing memoized data permanently to {}'.format(data_dir))
            np.save(os.path.join(data_dir, 'smp_indexes.npy'), smp_indexes)
            np.save(os.path.join(data_dir, 'all_roi_bbs.npy'), data[3])
            np.save(os.path.join(data_dir, 'all_nci_tlc.npy'), data[4])

            if _cfg.LABELS.USE_THREAT_BBOX_ANOTS:
                threat_bboxes_path = os.path.join(data_dir, 'threat_bboxes.pkl')
                with open(threat_bboxes_path, 'wb') as f:
                    pickle.dump(data[5], f)

            for z_name in _present_zones:
                np.save(os.path.join(data_dir, 'X_{}.npy'.format(z_name)), data[0][z_name])
                np.save(os.path.join(data_dir, 'y_{}.npy'.format(z_name)), data[1][z_name])
                if data[2] is not None:
                    np.save(os.path.join(data_dir, 'w_{}.npy'.format(z_name)), data[2][z_name])

        if keep_in_memory: return smp_indexes, data

    # memory map the stored 'data' ndarrays, in order to access indexes directly from disk
    memmap = None if keep_in_memory else 'r'
    smp_indexes = np.load(os.path.join(data_dir, 'smp_indexes.npy'))
    all_roi_bbs = np.load(os.path.join(data_dir, 'all_roi_bbs.npy'), mmap_mode=memmap)
    all_nci_tlc = np.load(os.path.join(data_dir, 'all_nci_tlc.npy'), mmap_mode=memmap)

    if _cfg.LABELS.USE_THREAT_BBOX_ANOTS:
        threat_bboxes_path = os.path.join(data_dir, 'threat_bboxes.pkl')
        with open(threat_bboxes_path, 'rb') as f:
            threat_bboxes = pickle.load(f)
    else: threat_bboxes = None

    cropped_images = dict()
    sample_gtlabel = dict()
    seg_gtlb_confs = dict() if _cfg.LOSS.PASS_SAMPLE_WGTS and d_set=='train' else None
    for z_name in _present_zones:
        cropped_images[z_name] = \
            np.load(os.path.join(data_dir, 'X_{}.npy'.format(z_name)), mmap_mode=memmap)
        sample_gtlabel[z_name] = \
            np.load(os.path.join(data_dir, 'y_{}.npy'.format(z_name)), mmap_mode=memmap)
        if _cfg.LOSS.PASS_SAMPLE_WGTS and d_set=='train':
            seg_gtlb_confs[z_name] = \
                np.load(os.path.join(data_dir, 'w_{}.npy'.format(z_name)), mmap_mode=memmap)

    return smp_indexes, \
           (cropped_images, sample_gtlabel, seg_gtlb_confs, all_roi_bbs, all_nci_tlc, threat_bboxes)


def input_setup(saved_paths, d_set, subset_tags):
    subset_df = \
        eval('_set_df[_set_df.{}.isin(subset_tags)]'.format(_cfg.DATASET.SUBSETS.GROUP_TYPE))

    # Get training and validation data samples and corresponding labels
    ids_indexes, unique_ids, lbs_dict, lbs_conf_dict, grp_indexes, min_cnt, perzone_smpcnt = \
        collect_body_group_data(subset_df, d_set, subset_tags)

    # Get predicted keypoint location and confidence dataframe
    if _cfg.LABELS.USE_PSEUDO_LABELS and d_set=='train':
        subset_kpts_df = _kpts_df
    else: # filter only needed subset
        scan_ids = subset_df.scanID.tolist()
        subset_kpts_df = _kpts_df[_kpts_df.scanID.isin(scan_ids)]

    # Compute preload window-to-crop dimensions
    img_augment = eval('_cfg.{}.AUGMENTATION'.format(d_set.upper()))
    if img_augment:
        rot_angle, zoom_ftr = _cfg.AUGMENT.ROTATE, _cfg.AUGMENT.S_ZOOM
        xy_sft_f = (_cfg.AUGMENT.X_SHIFT, _cfg.AUGMENT.Y_SHIFT)
    else:
        rot_angle, zoom_ftr = 0., 0.
        xy_sft_f = (0., 0.)
    max_xy_shift = adjust_xy_shift(_cfg.MODEL.IMAGE_SIZE[:2], xy_sft_f,
                                   _present_zones, ZONE_TAG_TO_IDX, ZONE_ROI_DIMS)
    prw_shapes, prw_dims = \
        define_region_windows(_present_zones, ZONE_TAG_TO_IDX, BDGRP_CROP_CONFIG,
                              _cfg.MODEL.REGION_DIM, _cfg.MODEL.IMAGE_SIZE[2], max_xy_shift,
                              rot_angle, zoom_ftr, SCALE_DIM_FACTOR, _cfg.MODEL.FORCE_SQUARE)
    msg = '\n{} set:  Preload-Region-Window  Region-of-Interest'.format(d_set)
    for i, zone_name in enumerate(_present_zones):
        grp_name = BODY_ZONES_GROUP[zone_name]
        roi_wdim = np.int32(BDGRP_CROP_CONFIG[grp_name][:2] * SCALE_DIM_FACTOR)
        msg += '\n{:>9}:\t\t{}\t\t{}'.format(zone_name, prw_dims[i], roi_wdim)
    _logger.log_msg(msg), print(msg)

    # Read images into memory and in order to setup data pipeline
    smp_indexes, in_data = compile_dataset(ids_indexes, unique_ids, lbs_dict, lbs_conf_dict,
                                subset_kpts_df, perzone_smpcnt, prw_shapes, d_set, subset_tags)
    # Set up tf.data input pipeline
    pipe_constants = \
        {'zone_roi_dims':ZONE_ROI_DIMS, 'zone_idx_to_tag':ZONE_IDX_TO_TAG, 'n_frames':N_FRAMES,
         'zone_tag_to_idx':ZONE_TAG_TO_IDX, 'zone_to_grp_idx':ZONE_TO_GRP_IDX, 'frm_dim':FRM_DIM,
         'scale_ftr':SCALE_DIM_FACTOR, 'bg_pixels':BG_PIXELS, 'fe_out_shape':FE_OUT_SHAPE,
         'min_gap':MIN_GAP, 'max_xy_shift_aug':max_xy_shift, 'subnet_tags':_subnet_tags}
    pipe = DataPipeline(_cfg, in_data, len(ids_indexes), d_set, _zone_ordered_fids,
                        prw_dims, min_cnt, pipe_constants, _logger, unique_ids=unique_ids)

    cache = eval('_cfg.{}.CACHE_TF_DATA'.format(d_set.upper()))

    if d_set=='train':
        if cache: cache = saved_paths['{}_cache_path'.format(d_set)]
        ds = training_dataset(pipe, smp_indexes, cache_path=cache)
    elif d_set=='valid':
        ds_v1, ds_v2 = None, None
        if _cfg.VALID.CBK_VALIDATE:
            cache_v1 = saved_paths['{}_cache_path1'.format(d_set)] if cache else False
            ds_v1 = validate_dataset_v1_cbk(pipe, smp_indexes, cache_path=cache_v1)
        if _cfg.VALID.FIT_VALIDATE:
            cache_v2 = saved_paths['{}_cache_path2'.format(d_set)] if cache else False
            ds_v2 = validate_dataset_v2_fit(pipe, smp_indexes, cache_path=cache_v2)
        ds = (ds_v1, ds_v2)
    else: assert(False), 'd_set must be "train"/"valid" not {}'.format(d_set)
    return ds, pipe, unique_ids, grp_indexes


def get_callbacks(saved_paths, pipelines, val_ds, val_ids, val_grp_indexes,
                  trn_meta, schedule_lr=True, tf_ckpt=True, whole_set=False, validate=True):
    trn_pipe, val_pipe = pipelines
    n_trn_smp, trn_spe = trn_meta
    LOG_FREQ = _cfg.VALID.VALIDATE_FREQ
    # TensorBoard logging callback
    if len(_cfg.TRAIN.TB.PROFILE_BATCH)>1:
        profile_batches = '{0[0]},{0[1]}'.format(_cfg.TRAIN.TB.PROFILE_BATCH)
    else: profile_batches = _cfg.TRAIN.TB.PROFILE_BATCH[0]
    tb_callback = \
        tf.keras.callbacks.TensorBoard(log_dir=saved_paths['tb_log_path'], update_freq='epoch',
                    profile_batch=profile_batches, histogram_freq=_cfg.TRAIN.TB.HISTOGTAM_FRQ,
                    write_graph=_cfg.TRAIN.TB.WRITE_GRAPH, write_images=_cfg.TRAIN.TB.WRITE_IMAGES)
    callbacks = [tb_callback]

    if schedule_lr:
        if _cfg.TRAIN.EXTRA.LR_TYPE=='stepwise':
            # Step-wise learning rate scheduler callback
            LR_SCHEDULE = scheduler(_cfg.TRAIN.EXTRA.LR_ITERATIONS, _cfg.TRAIN.EXTRA.LR_UPDATES)
            lr_callback = StepWiseLearningRateScheduler(LR_SCHEDULE, _cfg.TRAIN.EXTRA.LR_MODE,
                                                        trn_spe, _tb_writer, _logger)
        elif _cfg.TRAIN.EXTRA.LR_TYPE=='curve':
            # Continuous learning rate scheduler callback
            epoch_interval = [_cfg.TRAIN.BEGIN_EPOCH, _cfg.TRAIN.END_EPOCH]
            func_frac = np.asarray(_cfg.TRAIN.EXTRA.FUNC_FRAC, dtype=np.float32)
            lr_callback = CosineLearningRateScheduler(_cfg.TRAIN.EXTRA.LR_CONFIG, epoch_interval,
                                                      func_frac, _cfg.TRAIN.EXTRA.LR_MODE,
                                                      trn_spe, _tb_writer, pipelines, _logger)
        callbacks.append(lr_callback)

    if tf_ckpt:
        # checkpointing callback. Create a callback that saves the model's weights every n epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saved_paths['checkpoint_path'],
                                                         verbose=1, save_weights_only=True)
        callbacks.append(cp_callback)

    if whole_set:
        opt_paths = []
        for i, name in enumerate(_cfg.TRAIN.EXTRA.OPTIMAL_NAMES):
            tag = _cfg.TRAIN.EXTRA.OPTIMAL_STEPS[i]
            if name=='avg_f1s': opt_paths.append(saved_paths['opt1_avgf1s_wgt'].format(tag))
            elif name=='log_loss': opt_paths.append(saved_paths['opt2_loglos_wgt'].format(tag))
            else: print('unrecognized optimal metric name: {}'.format(name)), sys.exit()
        OPT_SCHEDULE = scheduler(_cfg.TRAIN.EXTRA.OPTIMAL_STEPS, opt_paths)
        opt_step_cb = OptimalStepsCallback(OPT_SCHEDULE, trn_spe, _logger)
        callbacks.append(opt_step_cb)

    if validate:
        # Custom implementation on Validation Metrics, Checkpoint, and Early Stop
        clip_minmax = get_clip_values(_cfg.VALID.N_DECIMALS)
        vm_callback = \
            ValidMetricCallback(val_pipe, val_ds, val_ids, val_grp_indexes, saved_paths, _tb_writer, _logger,
                                LOG_FREQ, _cfg.VALID.EOE_VALIDATE, _cfg.TRAIN.PATIENCE, _cfg.VALID.DELAY_CHECKPOINT,
                                _cfg.VALID.N_DECIMALS, clip_minmax, _cfg.DEBUG, n_trn_smp, trn_spe)
        callbacks.append(vm_callback)
        phrase = 'IS a' if trn_spe%LOG_FREQ==0 else 'IS NOT a'
        msg = '\nModel will be validated every {} steps of {} steps per epoch. ' \
              '\n\tvalidation freq {} factor of steps per epoch'.format(LOG_FREQ, trn_spe, phrase)
        _logger.log_msg(msg), print(msg)

    return callbacks


def custom_training_loop(trn_ds, start_epoch, end_epoch):
    trained_epochs = 0

    for epoch in range(start_epoch, end_epoch):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (s_batch_inputs, y_batch_trn, w_batch_trn) in enumerate(trn_ds):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = _model(s_batch_inputs, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = _loss_fn(y_batch_trn, logits, sample_weights=w_batch_trn)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, _model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            _optimizer.apply_gradients(zip(grads, _model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * 64))

        trained_epochs += 1
    return trained_epochs


def keras_training_loop(trn_ds, trn_steps_per_epoch, val_ds, val_steps_per_epoch,
                        start_epoch, end_epoch, callbacks):
    print('[INFO] All training data is parsed after every {} training steps'.format(trn_steps_per_epoch))
    print('[INFO] Validation data is parsed after every {} validation steps'.format(val_steps_per_epoch))
    if val_ds is None: val_steps_per_epoch = None #*
    history = _model.fit(trn_ds,
                         steps_per_epoch=trn_steps_per_epoch,
                         initial_epoch=start_epoch,
                         epochs=end_epoch,
                         verbose=_cfg.TRAIN.PRINT_FREQ,
                         # shuffle has no effect when steps_per_epoch is not None.
                         validation_data=val_ds,
                         validation_steps=val_steps_per_epoch,
                         callbacks=callbacks)
    trained_epochs = len(history.history['loss']) # print(history.history.keys())
    return trained_epochs


def train_validate(exp_name, trn_ds, trn_pipe, val_ds, val_pipe, saved_paths, callbacks):
    global _dist_strategy
    print('\nTraining Experiment: {}'.format(exp_name))
    n_zones = trn_pipe.N_ZONES
    n_images = trn_pipe.IMGS_PER_SAMPLE
    m_isets = trn_pipe.M_IMG_SETS
    last_warmup_epoch = _cfg.TRAIN.BEGIN_EPOCH + _cfg.TRAIN.WARMUP_EPOCHS

    if _cfg.TRAIN.ON_WHOLE_SET:
        max_steps = max(_cfg.TRAIN.EXTRA.OPTIMAL_STEPS)
        max_epoch = int(np.ceil(max_steps / trn_pipe.generator_stpep))
        assert(last_warmup_epoch>=max_epoch)

    training_stages = [('Warm-Up', last_warmup_epoch),
                       ('Unfrozen', _cfg.TRAIN.END_EPOCH)]

    trained_epochs = 0
    for stage in training_stages:
        stage_name, end_epoch = stage
        start_epoch = _cfg.TRAIN.BEGIN_EPOCH + trained_epochs

        if end_epoch>start_epoch:
            if _cfg.VALID.CBK_VALIDATE:
                callbacks[-1].set_start_end_epochs(start_epoch, end_epoch, stage_name)
            if _cfg.VALID.CBK_VALIDATE and _cfg.TRAIN.ON_WHOLE_SET:
                callbacks[-2].set_start_epoch(start_epoch)
            elif _cfg.TRAIN.ON_WHOLE_SET:
                callbacks[-1].set_start_epoch(start_epoch)
            # update batch epoch and step of TensorBoard custom writer
            _tb_writer.set_step(start_epoch * trn_pipe.generator_stpep)
            _tb_writer.set_epoch(start_epoch)

            # Distributed GPU training when more than 1 available gpu
            _dist_strategy = tf.distribute.MirroredStrategy()
            msg = '\nRunning distributed training with {} gpus'. \
                format(_dist_strategy.num_replicas_in_sync)
            model_build_compile(stage_name, n_images, n_zones, m_isets, saved_paths)
            msg += '\nModel compiled, fit/training is being initiated..\nModel will be ' \
                   'fitted on {} GPUs, with AUTOTUNE parallel workers\n'.format(_cfg.GPUS)
            _logger.log_msg(msg), print(msg)

            msg = '\n{} Training Started..'.format(stage_name)
            _logger.log_msg(msg), print(msg)

            if _cfg.TRAIN.CUSTOM_TRAIN_LOOP:
                trained_epochs = custom_training_loop(trn_ds, start_epoch, end_epoch)
            else:
                val_steps = val_pipe.generator_stpep #* if val_ds is not None else None
                trained_epochs = \
                    keras_training_loop(trn_ds, trn_pipe.generator_stpep, val_ds,
                                        val_steps, start_epoch, end_epoch, callbacks)

            msg = '\nFinished {} Sub Training, Saving Model..'.format(stage_name)
            _logger.log_msg(msg), print(msg)
            # Save models
            # Save the entire model to a HDF5 file.
            _model.save(saved_paths['hdf5_model_path'])
            # loss distribution
            cnt_loss_type = np.float32([trn_pipe.cnt_on_zone_loss, trn_pipe.cnt_on_iset_loss,
                                        trn_pipe.cnt_neither_loss])
            cnt_total = np.sum(cnt_loss_type)
            pct_loss_type = cnt_loss_type / cnt_total
            msg = '\nInput Samples %; on zone-loss: {:.2f}, on iset-loss: {:.2f}, with no loss:' \
                  ' {:.2f}, ground-truth mismatch rate: {:.2f}'.format(pct_loss_type[0],
                    pct_loss_type[1], pct_loss_type[2], trn_pipe.cnt_gt_mismatch / cnt_total)
            # pipeline average execution time
            msg += '\nData Pipeline average time per sample (over {} calls): {:.3f} seconds\n'.\
                    format(trn_pipe.total_call, trn_pipe.total_time / trn_pipe.total_call)
            _logger.log_msg(msg), print(msg)


def model_build_compile(stage_name, n_images, n_zones, m_isets, saved_paths):
    global _model, _loss_fn, _loss_coef, _optimizer, _metrics

    if stage_name=='Warm-Up':
        with _dist_strategy.scope():
            # create model
            _logger.log_msg('\n\n\nModel Architecture:\n-------------------------------')
            # Get model architecture/structure
            conv_dim = _cfg.MODEL.CONV_DIM
            net_arch = eval(_cfg.MODEL.NETWORK_ARCHITECTURE)
            _model = net_arch(_cfg, n_images, n_zones, _subnet_tags, conv_dim, False, _logger)
            # Load pretrained model parameters of another model, if enabled
            if _cfg.MODEL.TRANS_LEARN_PARAM_INIT:
                print('Executing transferred learning. Loading pretrained model parameters..\n')
                _model.load_weights(_cfg.MODEL.PRETRAINED, by_name=True)

            _optimizer = get_optimizer(_cfg) # create optimizer
            _loss_coef = None
            _loss_fn = get_loss(_cfg, logger=_logger)  # None
            _metrics = [tf.keras.metrics.BinaryCrossentropy(name='log'),
                       tf.keras.metrics.BinaryAccuracy(name='acc'),
                       tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='pre'),
                       tf.keras.metrics.Recall(name='rec')]

            if _is_multi_outputs:
                # output1_id::zt::zone-threat-prob, output2_id::gt::per img-group-threat-prob
                output1_id, output2_id = _cfg.LOSS.NET_OUTPUTS_ID
                _metrics = {output1_id: _metrics, output2_id: None}
                bce = CustomBCELoss(_cfg.LOSS.NET_OUTPUTS_LOSS[1],
                                    _cfg, m_isets, _logger, _tb_writer)
                _loss_fn = {output1_id: _loss_fn, output2_id: bce}
                out1_loss_coef, out2_loss_coef = _cfg.LOSS.NET_OUTPUTS_LOSS_COEF
                _loss_coef = {output1_id: out1_loss_coef, output2_id: out2_loss_coef}

        # compile model
        if not _cfg.TRAIN.CUSTOM_TRAIN_LOOP:
            _model.compile(loss=_loss_fn, optimizer=_optimizer,
                           metrics=_metrics, loss_weights=_loss_coef)
        # serialize model to JSON
        model_json = _model.to_json()
        with open(saved_paths['model_json_path'], "w") as json_file:
            json_file.write(model_json)

    # Unfreeze feature extraction layer after warm-up
    elif stage_name=='Unfrozen':
        if _cfg.MODEL.EXTRA.UNFREEZE_FE_BLOCKS<0:
            # unfreeze all layers of feature-extraction sub-network
            for layer in _model.get_layer('fe_td').layer.layers:
                layer.trainable = True
        else:
            # unfreeze layers in blocks >=
            block_numbers = range(_cfg.MODEL.EXTRA.UNFREEZE_FE_BLOCKS+1)
            for layer in _model.get_layer('fe_td').layer.layers:
                for block_num in block_numbers:
                    block_prefix = 'block_{}'.format(block_num)
                    if layer.name.find(block_prefix)>0:
                        layer.trainable = True
                        break


def reload_evaluate(tf_dataset, n_samples, saved_paths, opt_epochs,
                    last_warmup_epoch, body_grp_names, grp_ids_indexes, has_bdpart_output):
    # Reload and evaluate final model state
    clip_minmax = get_clip_values(_cfg.VALID.N_DECIMALS)
    n_decimals = _cfg.VALID.N_DECIMALS
    model = get_hdf5_model(saved_paths['hdf5_model_path'])
    t_gt, t_pb, p_gt, p_pb, grp_out_indexes = \
        get_labels_and_probs(tf_dataset, model, n_samples, body_grp_names, grp_ids_indexes,
            clip_minmax, n_decimals, has_bdpart_output, _is_multi_outputs, _is_subnet_output)
    run_evaluations(t_gt, t_pb, p_gt, p_pb, 'Model state at last epoch',
                    body_grp_names, grp_out_indexes, logger=_logger)

    title = 'Optimal {} model at epoch {} ({})'

    # Reload and evaluate optimal logloss model
    if opt_epochs.get('opt2', None) is not None:
        epoch = opt_epochs['opt2']
        stage = "Warm-Up" if epoch<=last_warmup_epoch else "Unfrozen"
        wgt_path = saved_paths['opt2_loglos_wgt'].format(epoch)
        #model = get_json_model(saved_paths['model_json_path'], wgt_path)
        model = load_weights_into_model(model, wgt_path)
        t_gt, t_pb, p_gt, p_pb, grp_out_indexes = \
            get_labels_and_probs(tf_dataset, model, n_samples, body_grp_names, grp_ids_indexes,
                clip_minmax, n_decimals, has_bdpart_output, _is_multi_outputs, _is_subnet_output)
        run_evaluations(t_gt, t_pb, p_gt, p_pb, title.format('Logloss', epoch, stage),
                        body_grp_names, grp_out_indexes, logger=_logger)

    # Reload and evaluate optimal f1-score model
    if opt_epochs.get('opt1', None) is not None:
        epoch = opt_epochs['opt1']
        stage = "Warm-Up" if epoch<=last_warmup_epoch else "Unfrozen"
        wgt_path = saved_paths['opt1_avgf1s_wgt'].format(epoch)
        #model = get_json_model(saved_paths['model_json_path'], wgt_path)
        model = load_weights_into_model(model, wgt_path)
        t_gt, t_pb, p_gt, p_pb, grp_out_indexes = \
            get_labels_and_probs(tf_dataset, model, n_samples, body_grp_names, grp_ids_indexes,
                clip_minmax, n_decimals, has_bdpart_output, _is_multi_outputs, _is_subnet_output)
        run_evaluations(t_gt, t_pb, p_gt, p_pb, title.format('F1score', epoch, stage),
                        body_grp_names, grp_out_indexes, logger=_logger)

    # Reload and evaluate optimal combined f1-score + log-loss model
    if opt_epochs.get('opt_combo', None) is not None:
        epoch = opt_epochs['opt_combo']
        stage = "Warm-Up" if epoch<=last_warmup_epoch else "Unfrozen"
        wgt_path = saved_paths['opt_log&f1s_wgt'].format(epoch)
        #model = get_json_model(saved_paths['model_json_path'], wgt_path)
        model = load_weights_into_model(model, wgt_path)
        t_gt, t_pb, p_gt, p_pb, grp_out_indexes = \
            get_labels_and_probs(tf_dataset, model, n_samples, body_grp_names, grp_ids_indexes,
                clip_minmax, n_decimals, has_bdpart_output, _is_multi_outputs, _is_subnet_output)
        run_evaluations(t_gt, t_pb, p_gt, p_pb, title.format('Log&F1s', epoch, stage),
                        body_grp_names, grp_out_indexes, logger=_logger)


def main(git_repo_sha):
    global _cfg, _map_df, _set_df, _kpts_df, _zfk_map, _zone_ordered_fids, _present_zones, \
        _logger, _subnet_tags, _is_multi_outputs, _is_subnet_output, _tb_writer, \
        SCALE_DIM_FACTOR, N_FRAMES, ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, \
        ZONE_ROI_DIMS, ZFK_MAP_TYPE, BG_PIXELS, FRM_CORNERS, MIN_GAP, FE_OUT_SHAPE, ROI_BB_SHAPE, \
        BDGRP_CROP_CONFIG, IMG_CEF, FRM_DIM, PIXELS_MINMAX

    # Compile experiment configuration and record
    args = runtime_args()
    _cfg = default_experiment_configuration()
    exp_name, _logger = update_config(_cfg, args, n_physical_gpus, n_logical_gpus, virtual_gpu_bsf)
    _logger.log_msg('git sha: {}\n'.format(git_repo_sha))
    os.makedirs(_cfg.DATASET.RETAINED_DATA_DIR, exist_ok=True)
    os.makedirs(_cfg.MODEL_DIR, exist_ok=True)
    cfg_rec_file = str(Path(_cfg.MODEL_DIR) / '{}_config.yaml'.format(exp_name))
    with open(cfg_rec_file, 'w') as f:
        with redirect_stdout(f): print(_cfg.dump())

    QUESTIONABLE_SCANS = [#'6cbda8596c5c9b1e31d4fab9b5a9e02b', '9e067ae96bb10fb62a3b4e7adf4d58ca',
                          #'ab52f3a07e8d37a5b7120acc81258254', '52c8235df3f0552e6c134529ca85d958',
                          #'7235e754185d3321c4b6883d001a35ad', 'd904d73f5e53eed05fef89ce0032fc1c',
                          #'e48b103b2d8bedb994c0ce62e15d1662', 'e4b560b0f6d2c44535610f38a787df93',
                          #'496ec724cc1f2886aac5840cf890988a', '56b9c0086836fe2fca86d773cacaf783',
                          #'cbc6f0a3be3d802fc3d2bd45c183a49d', '81aab48146e23ac8f9744f18459ecdbe',
                          #'0367394485447c1c3485359ba71f52cb', 'b07aab4469700f4937e71e617637e0b9',
                          '42181583618ce4bbfbc0c4c300108bf5']  # taken from Report 18 & 30

    # Trained model paths
    # TensorBoard log path
    tb_log_path = str(Path(_cfg.MODEL_DIR) / '{}_tb'.format(exp_name))
    os.makedirs(tb_log_path, exist_ok=True)
    # To save entire model to a HDF5 file
    hdf5_model_path = os.path.join(_cfg.MODEL_DIR, '{}.h5'.format(exp_name))
    # To save model in SavedModel format
    saved_model_path = str(Path(_cfg.MODEL_DIR) / exp_name)
    # To save only model architecture or structure
    model_json_path = os.path.join(_cfg.MODEL_DIR, '{}_structure.json'.format(exp_name))
    # To checkpoint model periodically
    checkpoint_path = str(Path(_cfg.MODEL_DIR) / 'ckpt' / 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Optimal model weights and others
    models_path = str(Path(_cfg.MODEL_DIR) / 'models')
    os.makedirs(models_path, exist_ok=True)
    opt_logf1s_file = str(Path(models_path) / 'wgts-{}-log&f1')
    opt_avgf1s_file = str(Path(models_path) / 'wgts-{}-avgf1s')
    opt_loglos_file = str(Path(models_path) / 'wgts-{}-loglos')
    meta_info_file = os.path.join(_cfg.MODEL_DIR, 'metadata.npy')
    constants_file = os.path.join(_cfg.MODEL_DIR, 'constants.pickle')
    # tf.data cache paths
    trn_cache_path = str(Path(_cfg.MODEL_DIR) / 'trn_cache')
    val_cache_path_v1 = str(Path(_cfg.MODEL_DIR) / 'val_cache_v1')
    val_cache_path_v2 = str(Path(_cfg.MODEL_DIR) / 'val_cache_v2')
    # Dictionary holding all model related file and directory paths
    saved_paths = {'hdf5_model_path': hdf5_model_path, 'saved_model_path': saved_model_path,
                   'model_json_path': model_json_path, 'meta_info_file': meta_info_file,
                   'constants_file': constants_file, 'checkpoint_path': checkpoint_path,
                   'opt1_avgf1s_wgt': opt_avgf1s_file, 'opt2_loglos_wgt': opt_loglos_file,
                   'opt_log&f1s_wgt': opt_logf1s_file, 'tb_log_path': tb_log_path, 'train_cache_path': trn_cache_path,
                   'valid_cache_path1': val_cache_path_v1, 'valid_cache_path2': val_cache_path_v2}

    # Get frame-to-zone-to-keypoint dictionary map
    map_df = pd.read_csv(_cfg.LABELS.CSV.FZK_MAP)
    _set_df = pd.read_csv(_cfg.DATASET.SUBSETS.GROUPINGS)
    bad_apple_row_indexes = list()
    for trouble_scanid in QUESTIONABLE_SCANS:
        bad_apple_row_indexes.append(_set_df[_set_df['scanID']==trouble_scanid].index[0])
    _set_df = _set_df.drop(bad_apple_row_indexes)
    _kpts_df = pd.read_csv(_cfg.LABELS.CSV.KPTS_SET)
    ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, _present_zones = \
        zone_index_id_map(_cfg.MODEL.GROUPS)
    assert(_cfg.MODEL.REGION_DIM[0]>=_cfg.MODEL.IMAGE_SIZE[1])
    assert(_cfg.MODEL.REGION_DIM[1]>=_cfg.MODEL.IMAGE_SIZE[0])

    tsa_ext = _cfg.DATASET.FORMAT
    _zone_ordered_fids = get_grp_ordered_zones(_present_zones, map_df, tsa_ext, _logger)
    if tsa_ext=='aps' and _cfg.DATASET.ROOT.find('a3daps')>=0:
        _zone_ordered_fids = map_aps_frames_2_a3daps(_zone_ordered_fids, _logger)
        _kpts_df = rename_columns_with_a3daps_fids(_kpts_df)
        map_df = replace_entries_with_a3daps_fids(map_df)
        BDGRP_ORIENTED_CFG = replace_keys_with_a3daps_fids(BDGRP_ORIENTED)
        N_FRAMES = DATA_NUM_FRAMES['a3daps']
        MIN_GAP = 4
    else:
        BDGRP_ORIENTED_CFG = BDGRP_ORIENTED
        N_FRAMES = DATA_NUM_FRAMES[tsa_ext]
        MIN_GAP = 1
    _zfk_map = df_map_to_dict_map(map_df, ZONE_NAME_TO_ID)
    PIXELS_MINMAX = TSA_PIXELS_RANGE[tsa_ext]

    if _cfg.MODEL.EXTRA.ROI_TYPE=='aligned':
        ROI_BB_SHAPE = (4,)
        BDGRP_CROP_CONFIG = BDGRP_ALIGNED
        ZFK_MAP_TYPE = dict.fromkeys(_present_zones, None)
    elif _cfg.MODEL.EXTRA.ROI_TYPE in ['bbprior', 'oriented']:
        ROI_BB_SHAPE = (4, 2)
        BDGRP_CROP_CONFIG = BDGRP_ORIENTED_CFG
        ZFK_MAP_TYPE = frm_zone_kpts_map_type(_zfk_map)
    else: assert(False), "Unrecognized ROI_TYPE: {}".format(_cfg.MODEL.EXTRA.ROI_TYPE)
    log_bdgrp_seg_config(_cfg.MODEL.GROUPS, BDGRP_CROP_CONFIG, _logger)
    SCALE_DIM_FACTOR = np.flip(np.float32(_cfg.MODEL.IMAGE_SIZE[:2])) / \
                       np.asarray(_cfg.MODEL.REGION_DIM) #  eg: np.asarray([0.5, 0.5])
    FRM_DIM = np.int32([512, 660] * SCALE_DIM_FACTOR)
    FRM_CORNERS = np.float32([[          0,          0, 1],     # top-left
                              [ FRM_DIM[0],          0, 1],     # top-right
                              [ FRM_DIM[0], FRM_DIM[1], 1],     # bottom-right
                              [          0, FRM_DIM[1], 1]]).T  # bottom-left
    ZONE_ROI_DIMS = get_roi_dims(_present_zones, ZONE_TAG_TO_IDX,
                                 BDGRP_CROP_CONFIG, SCALE_DIM_FACTOR)

    IMG_CEF = _cfg.DATASET.ENHANCE_VISIBILITY
    bgi = read_image(_cfg.DATASET.BG_IMAGE, rgb=_cfg.DATASET.COLOR_RGB, icef=IMG_CEF)
    BG_PIXELS = np.mean(bgi, axis=(0, 1)).astype(np.uint8, copy=False).tolist()
    FE_OUT_SHAPE = fe_network(_cfg.MODEL.EXTRA.FE_NETWORK, tuple(_cfg.MODEL.IMAGE_SIZE),
                              _cfg.MODEL.EXTRA.FE_OUT_LAYER, shape_only=True)

    # TensorBoard writer to be used for adhoc logging
    _tb_writer = TensorBoardWriter(_cfg, saved_paths['tb_log_path'])

    # build datasets, input pipelines, and callbacks
    _is_subnet_output = _cfg.MODEL.SUBNET_TYPE in SUBNET_CONFIGS
    if _is_subnet_output:
        if _cfg.MODEL.SUBNET_TYPE=='body_zones':
            _subnet_tags = truncate_strings(_present_zones)
        elif _cfg.MODEL.SUBNET_TYPE=='body_groups':
            _subnet_tags = truncate_strings(_cfg.MODEL.GROUPS)
    else: _subnet_tags = None
    _is_multi_outputs = len(_cfg.LOSS.NET_OUTPUTS_ID)>1
    has_bdpart_output = 'p' in _cfg.LOSS.NET_OUTPUTS_ID
    if _cfg.TRAIN.ON_WHOLE_SET:
        trn_subset = _cfg.DATASET.SUBSETS.TRAIN_SETS + _cfg.DATASET.SUBSETS.VALID_SETS
    else: trn_subset = _cfg.DATASET.SUBSETS.TRAIN_SETS
    trn_ds, trn_pipe, trn_ids, trn_grp_idx = input_setup(saved_paths, 'train', trn_subset)
    trn_meta = (trn_pipe.data_sample_size, trn_pipe.generator_stpep)
    val_pipe, val_ds_v1, val_ds_v2, val_ids, val_grp_idx = None, None, None, None, None
    if _cfg.VALID.CBK_VALIDATE or _cfg.VALID.FIT_VALIDATE:
        val_ds, val_pipe, val_ids, val_grp_idx = \
            input_setup(saved_paths, 'valid', _cfg.DATASET.SUBSETS.VALID_SETS)
        val_ds_v1, val_ds_v2 = val_ds
    pipelines = [trn_pipe, val_pipe]
    callbacks = get_callbacks(saved_paths, pipelines, val_ds_v1, val_ids, val_grp_idx, trn_meta,
                              tf_ckpt=_cfg.TRAIN.TF_CHECKPOINTING,
                              whole_set=_cfg.TRAIN.ON_WHOLE_SET, validate=_cfg.VALID.CBK_VALIDATE)

    # memoize certain important configurations to reproducibility during test/evaluation
    test_constants = {'BDGRP_CROP_CONFIG':BDGRP_CROP_CONFIG, 'ZFK_MAP_TYPE':ZFK_MAP_TYPE,
                      'BDGRP_ADJUSTMENT':BDGRP_ADJUSTMENT, 'ZONE_ROI_DIMS':ZONE_ROI_DIMS,
                      'PIXELS_MINMAX':PIXELS_MINMAX,
                      'ROI_BB_SHAPE':ROI_BB_SHAPE, 'FE_OUT_SHAPE':FE_OUT_SHAPE}
    with open(saved_paths['constants_file'], 'wb') as file_handle:
        pickle.dump(test_constants, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    del trn_ids, trn_grp_idx, val_ids, val_grp_idx, \
        map_df, _set_df, _kpts_df, _zfk_map, ZFK_MAP_TYPE
    gc.collect()

    # train and validate
    train_validate(exp_name, trn_ds, trn_pipe, val_ds_v2, val_pipe, saved_paths, callbacks)

    # reload and evaluate
    if _cfg.VALID.CBK_VALIDATE:
        opt1_ep = callbacks[-1].opt1_epoch
        opt1_sc = callbacks[-1].opt1_score
        opt2_ep = callbacks[-1].opt2_epoch
        opt2_sc = callbacks[-1].opt2_score
        optc_ep = callbacks[-1].opt_combo_epoch
        eowu_ep = callbacks[-1].last_warmup_epoch
        if opt1_ep<_cfg.VALID.DELAY_CHECKPOINT: opt1_ep = 0
        if opt2_ep<_cfg.VALID.DELAY_CHECKPOINT: opt2_ep = 0
        if optc_ep<_cfg.VALID.DELAY_CHECKPOINT: optc_ep = 0
        opt_epochs = dict() # {'opt1': opt1_ep, 'opt2': opt2_ep}
        if opt1_ep>0: opt_epochs['opt1'] = opt1_ep
        if opt2_ep>0: opt_epochs['opt2'] = opt2_ep
        if optc_ep>0: opt_epochs['opt_combo'] = optc_ep
        reload_evaluate(val_ds_v1, val_pipe.data_sample_size, saved_paths, opt_epochs, eowu_ep,
                        _cfg.MODEL.GROUPS, callbacks[-1].grp_ids_indexes, has_bdpart_output)

    # delete cached data
    remove_files(_cfg.MODEL_DIR, ['metadata', 'cache'])
    return exp_name


if __name__=='__main__':
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%B %d, %Y %H:%M:%S")  # Month dd, YY H:M:S

    # get current git repo hash
    #repo = git.Repo(search_parent_directories=True)
    sha = '' #repo.head.object.hexsha
    print('Program started execution at {}, with git hash: {}\n'.format(dt_string, sha))
    exp_name = main(sha)

    now = datetime.now()
    dt_string = now.strftime("%B %d, %Y %H:%M:%S")  # Month dd, YY H:M:S
    print('\nExperiment {} finished execution at {}. Using git hash: {}\n'
          .format(exp_name, dt_string, sha))