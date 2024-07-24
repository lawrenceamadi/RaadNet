'''
v1: Test with saved h5 model as base
'''

from __future__ import absolute_import, division, print_function, unicode_literals

#------------------------------------------------------------------------------
# START
import os
import sys
# Run script with command: $ PYTHONHASHSEED=0 python your_program.py
# or uncomment command below to use only CPU:
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
##os.environ['PYTHONHASHSEED'] = '0' # also needed for reproducibility

import random
import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()  # For easy reset of notebook state.

# Necessary for reproducibility with tf.keras
random.seed(12345) # Python generated random numbers in a well-defined state.
np.random.seed(42) # Numpy generated random numbers in a well-defined initial state.
tf.random.set_seed(57684) # tf generated random numbers in a well-defined initial state.

physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
    # Allow dynamic memory growth on demand
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = len(tf.config.experimental.list_logical_devices('GPU'))
        print("\n{} Physical GPUs, {} Logical GPUs".format(len(physical_gpus), logical_gpus))
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        print('Memory growth must be set before GPUs have been initialized\nOR')
        print('\nVirtual devices must be set before GPUs have been initialized')
        sys.exit(0)
physical_gpus = len(physical_gpus)
# END
#------------------------------------------------------------------------------

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append('../')
from tf_neural_net.comb_pipeline import DataPipeline
from tf_neural_net.data_preprocess import read_image
from tf_neural_net.keras_logger import scale_image_pixels
from tf_neural_net.data_preprocess import crop_aligned_rois, crop_oriented_rois  # crop_rois_func
from tf_neural_net.evaluation import load_weights_into_model, get_hdf5_model, get_json_model
from tf_neural_net.commons import df_map_to_dict_map, get_grp_ordered_zones, remove_files
from tf_neural_net.commons import adjust_xy_shift, map_aps_frames_2_a3daps, get_clip_values
from tf_neural_net.commons import zone_index_id_map, cv_close_windows, define_region_windows
from tf_neural_net.commons import replace_entries_with_a3daps_fids, truncate_strings
from tf_neural_net.commons import rename_columns_with_a3daps_fids, get_pseudo_data_naming_config
from tf_neural_net.commons import ZONE_NAME_TO_ID, DATA_NUM_FRAMES, SUBNET_CONFIGS
from tf_neural_net.commons import BODY_GROUP_ZONES, BODY_ZONES_GROUP
from tf_neural_net.default import default_experiment_configuration
from tf_neural_net.default import runtime_args, adapt_config


def single_zone_data(zone_name, scan_ids_list):
    sample_ids_list = list()

    for index, scan_id in enumerate(scan_ids_list):
        #if index<5:
        sample_id = '{}_{}'.format(scan_id, zone_name)
        sample_ids_list.append(sample_id)

    return sample_ids_list


def collect_body_group_data(scan_ids):
    # Collect data for each zone in group
    unique_ids_list = list() # list of lists
    per_zone_samples = dict()
    unique_samples = 0
    samples_cnt = 0
    assert(len(present_zones)==17)

    for z_idx in range(len(present_zones)):
        zone_name = present_zones[z_idx]

        # Get training and validation data samples and corresponding labels
        unique_ids = single_zone_data(zone_name, scan_ids)
        n_zone_samples = len(unique_ids)
        unique_samples += n_zone_samples

        # Update (add to) sample ids' list, sample ids indexes', and sample labels' dicts
        unique_ids_list.extend(unique_ids) # must be done in order (order matters)

        # Update counters
        per_zone_samples[zone_name] = n_zone_samples
        samples_cnt += n_zone_samples

    ids_indexes = np.arange(unique_samples)
    max_index = np.max(ids_indexes)
    np.random.shuffle(ids_indexes) # order is irrelevant
    assert(ids_indexes.shape[0]==samples_cnt)
    assert(unique_samples==len(unique_ids_list))
    assert(max_index==len(unique_ids_list) - 1)
    assert(samples_cnt%17==0)
    assert(samples_cnt/17==1388 or samples_cnt/17==5)

    msg = '\nCollected Test Set:'
    msg += '\n{:,} total samples. {:,} total images.'. \
            format(samples_cnt, samples_cnt * 12)
    msg += '\n{:,} unique total samples, index ndarray dtype: {}'.\
            format(unique_samples, ids_indexes.dtype)
    print(msg)
    return ids_indexes, unique_ids_list, per_zone_samples


def get_region_data(sample_ids_indexes, unique_sample_ids, set_kpts_df, perzone_nsamples,
                    prw_shapes, cfg, debug_display=False):
    cropped_images = dict()
    dummy_smp_lbls = dict()
    byte_size = 0
    for z_name in present_zones:
        a_shp = (perzone_nsamples[z_name], len(zone_ordered_fids[z_name]), *prw_shapes[z_name])
        cropped_images[z_name] = np.empty(a_shp, dtype=np.uint8) # eg. (1388x12x80x80x3)
        dummy_smp_lbls[z_name] = np.zeros((perzone_nsamples[z_name]), dtype=np.uint8)
        byte_size += cropped_images[z_name].nbytes

    msg = '\nReading and parsing images and other inputs to memory\n Test-set '
    msg += 'images will occupy {:,} bytes or {:.1f} GB'.format(byte_size, byte_size / (1024**3))
    print(msg)

    # Track body part or zone's bounding-box #
    pzs_cnts = np.asarray(list(perzone_nsamples.values()))
    assert(np.all(pzs_cnts==pzs_cnts[0])), 'all zones must have equal counts of unique cases'
    # Track body part or zone's bounding-box or bounding polygon
    all_roi_bbs = np.zeros((pzs_cnts[0], N_FRAMES, 17, *ROI_BB_SHAPE), dtype=np.int16)
    all_nci_tlc = np.zeros((pzs_cnts[0], N_FRAMES, 17, 1, 2), dtype=np.int16) # nci top-left-corner

    # initialize some variables for cropping regions
    crop_rois_func = eval("crop_{}_rois".format(cfg.MODEL.EXTRA.ROI_TYPE))
    rgb_img = cfg.DATASET.COLOR_RGB  # if true the read BGR image is converted to RGB
    nci_dim = cfg.MODEL.IMAGE_SIZE[:2]  # or _cfg.MODEL.REGION_DIM * SCALE_DIM_FACTOR
    db = cfg.DATASET.XY_BOUNDARY_ERROR  # allowance for nci/roi coordinates to fall Out-Of-Bounds
    poly_deg = cfg.MODEL.EXTRA.POLYNOMIAL_DEGREE # degree of polynomial to fit to spatial hist-line
    if debug_display:
        fig, ax_sp = plt.subplots(figsize=(3, 3))
        fig.tight_layout(pad=0)
    else: ax_sp = None

    n_samples = len(sample_ids_indexes)
    ds_sample_indexes = np.empty((n_samples, 3), dtype=sample_ids_indexes.dtype)
    ds_sample_indexes_dict = dict()
    scanid_indexer_into_zone_array = dict()
    new_scan_index = 0
    unique_samples_cnt = 0

    for i, sample_id_indx in enumerate(sample_ids_indexes):
        sample_id = unique_sample_ids[sample_id_indx]
        sample_indexes = ds_sample_indexes_dict.get(sample_id, None)
        assert(sample_indexes is None)
        #if sample_id != '5e1980e91c56d8d23ae25c088ab4a859_RFm': continue

        # Process each UNIQUE sample_id ONLY once
        if sample_indexes is None:
            unique_samples_cnt += 1
            scan_id, z_name = sample_id.split('_')
            #print("{:>5}. {}:{}".format(i, sample_id_indx, sample_id))

            scan_dir = os.path.join(images_dir, scan_id)
            ordered_fids_of_zone = zone_ordered_fids[z_name]
            n_unique_ipz = len(ordered_fids_of_zone)
            zone_type_indx = ZONE_TAG_TO_IDX[z_name]
            scan_id_indx = scanid_indexer_into_zone_array.get(scan_id, None)

            if scan_id_indx is None:
                scan_id_indx = new_scan_index
                scanid_indexer_into_zone_array[scan_id] = scan_id_indx
                new_scan_index += 1

            assert (scan_id_indx<perzone_nsamples[z_name])
            reg_imgs, reg_conf, roi_bbs, nci_tlc = \
                crop_rois_func(scan_dir, scan_id, z_name, ordered_fids_of_zone, set_kpts_df,
                               zfk_map[z_name], ZFK_MAP_TYPE[z_name], prw_shapes[z_name],
                               n_unique_ipz, BDGRP_CROP_CONFIG, BDGRP_ADJUSTMENT, FRM_DIM,
                               SCALE_DIM_FACTOR, FRM_CORNERS, IMG_CEF, BG_PIXELS, nci_dim,
                               poly_deg, rgb_img, ax_sp, debug_display, db)
            cropped_images[z_name][scan_id_indx] = reg_imgs
            for idx, fid in enumerate(ordered_fids_of_zone):
                all_roi_bbs[scan_id_indx, fid, zone_type_indx] = roi_bbs[idx]
                all_nci_tlc[scan_id_indx, fid, zone_type_indx] = nci_tlc[idx]

            sample_indexes = [sample_id_indx, zone_type_indx, scan_id_indx]
            ds_sample_indexes_dict[sample_id] = sample_indexes

        ds_sample_indexes[i] = sample_indexes
        if (i+1)%1000==0:
            print('{:>6,} of {:,} samples read into memory'.format(i + 1, n_samples))

    assert(np.all(np.sum(all_roi_bbs[:,:,:,:,0], axis=-1)>=0)), "sum of each roi x coord >= 0"
    assert(np.all(np.sum(all_roi_bbs[:,:,:,:,1], axis=-1)>=0)), "sum of each roi y coord >= 0"

    if cfg.DATASET.SCALE_PIXELS:
        channels = ('red','green','blue') if cfg.DATASET.COLOR_RGB else ('blue','green','red')
        for z_name in present_zones:
            for idx, color_channel in enumerate(channels):
                # eg. (1247,?12,?87,?85,3)
                cropped_images[z_name][:,:,:,:,idx] = \
                    scale_image_pixels(cropped_images[z_name][:,:,:,:,idx],
                                       PIXELS_MINMAX[color_channel], ret_uint8=True)

    print('{} unique samples read into memory\n'.format(unique_samples_cnt))
    assert(unique_samples_cnt==n_samples)
    if debug_display:
        cv_close_windows()
        plt.close()

    return ds_sample_indexes, \
           (cropped_images, dummy_smp_lbls, None, all_roi_bbs, all_nci_tlc, None)


def compile_dataset(ids_indexes, unique_ids, cfg,
                    subset_kpts_df, perzone_smpcnt, prw_shapes, d_set, subset_tags):
    data_cfg = get_pseudo_data_naming_config(cfg, len(unique_ids), d_set, subset_tags)
    data_dir = os.path.join(cfg.DATASET.RETAINED_DATA_DIR, data_cfg)
    data_configs = os.listdir(cfg.DATASET.RETAINED_DATA_DIR)

    is_ds_cached = eval('cfg.{}.CACHE_TF_DATA'.format(d_set.upper()))
    keep_in_memory = cfg.DATASET.RETAIN_IN_MEMORY and not is_ds_cached

    if data_cfg not in data_configs or cfg.DATASET.FRESH_BUILD:
        # compile data
        smp_indexes, data = \
            get_region_data(ids_indexes, unique_ids, subset_kpts_df,
                            perzone_smpcnt, prw_shapes, cfg, cfg.DEBUG.TURN_ON_DISPLAY)

        if cfg.DATASET.PERSIST_ON_DISK:
            # save on disk
            os.makedirs(data_dir, exist_ok=True)
            np.save(os.path.join(data_dir, 'smp_indexes.npy'), smp_indexes)
            np.save(os.path.join(data_dir, 'all_roi_bbs.npy'), data[3])
            np.save(os.path.join(data_dir, 'all_nci_tlc.npy'), data[4])

            for z_name in present_zones:
                np.save(os.path.join(data_dir, 'X_{}.npy'.format(z_name)), data[0][z_name])
                np.save(os.path.join(data_dir, 'y_{}.npy'.format(z_name)), data[1][z_name])

        if keep_in_memory: return smp_indexes, data

    # memory map the stored 'data' ndarrays, in order to access indexes directly from disk
    memmap = None if keep_in_memory else 'r'
    smp_indexes = np.load(os.path.join(data_dir, 'smp_indexes.npy'))
    all_roi_bbs = np.load(os.path.join(data_dir, 'all_roi_bbs.npy'), mmap_mode=memmap)
    all_nci_tlc = np.load(os.path.join(data_dir, 'all_nci_tlc.npy'), mmap_mode=memmap)

    cropped_images = dict()
    sample_gtlabel = dict()
    for z_name in present_zones:
        cropped_images[z_name] = \
            np.load(os.path.join(data_dir, 'X_{}.npy'.format(z_name)), mmap_mode=memmap)
        sample_gtlabel[z_name] = \
            np.load(os.path.join(data_dir, 'y_{}.npy'.format(z_name)), mmap_mode=memmap)

    return smp_indexes, \
           (cropped_images, sample_gtlabel, None, all_roi_bbs, all_nci_tlc, None)


def test_dataset(pipe_obj, sample_indexes, cfg, interleave=False, cache_path=False):
    ## batch testing via model.predict
    # create (image, label, weight) zip to iterate over
    map_func = pipe_obj.feed_sample_inputs_xm_out

    if cfg.TEST.WORKERS<0: n_workers = tf.data.experimental.AUTOTUNE
    elif cfg.TEST.WORKERS==0: n_workers = None
    else: n_workers = cfg.TEST.WORKERS
    n = len(sample_indexes)

    # validation data producer
    ds = tf.data.Dataset.from_tensor_slices(sample_indexes)
    ds = ds.shuffle(n, reshuffle_each_iteration=cfg.TEST.EOE_SHUFFLE)
    ds = ds.map(map_func, num_parallel_calls=n_workers)
    if interleave:
        ds = ds.interleave(lambda x,i,o: tf.data.Dataset.from_tensors((x,i,o)),
                           num_parallel_calls=n_workers, deterministic=False)
    ds = ds.batch(cfg.TEST.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=cfg.TEST.QUEUE_SIZE)
    if cache_path: ds = ds.cache(filename=cache_path)
    return ds


def input_setup(tst_cache_path, d_set, cfg):
    scan_ids = os.listdir(images_dir)
    # Get training and validation data samples and corresponding labels
    ids_indexes, unique_ids, perzone_smpcnt = collect_body_group_data(scan_ids)
    # Get predicted keypoint location and confidence dataframe
    subset_kpts_df = kpts_df[kpts_df.scanID.isin(scan_ids)]

    # Compute crop-for-keep window dimensions
    assert(d_set=='test')
    rot_angle, zoom_ftr = 0., 0.
    xy_sft_f = (0., 0.)
    max_xy_shift = adjust_xy_shift(cfg.MODEL.IMAGE_SIZE[:2], xy_sft_f,
                                   present_zones, ZONE_TAG_TO_IDX, ZONE_ROI_DIMS)
    prw_shapes, prw_dims = \
        define_region_windows(present_zones, ZONE_TAG_TO_IDX, BDGRP_CROP_CONFIG,
                              cfg.MODEL.REGION_DIM, cfg.MODEL.IMAGE_SIZE[2], max_xy_shift,
                              rot_angle, zoom_ftr, SCALE_DIM_FACTOR, cfg.MODEL.FORCE_SQUARE)
    print('{} set:  Preload-Region-Window  Region-of-Interest'.format(d_set))
    for i, zone_name in enumerate(present_zones):
        grp_name = BODY_ZONES_GROUP[zone_name]
        roi_wdim = np.int32(BDGRP_CROP_CONFIG[grp_name][:2] * SCALE_DIM_FACTOR)
        print('{:>9}:\t{}\t\t{}'.format(zone_name, prw_dims[i], roi_wdim))

    # Read images into memory and in order to setup data pipeline
    smp_indexes, in_data = \
        compile_dataset(ids_indexes, unique_ids, cfg,
                        subset_kpts_df, perzone_smpcnt, prw_shapes, d_set, None)
        #get_region_data(ids_indexes, unique_ids, subset_kpts_df, perzone_smpcnt, prw_shapes,
        #                cfg, cfg.DEBUG.TURN_ON_DISPLAY)
    # Set up tf.data input pipeline
    pipe_constants = \
        {'zone_roi_dims':ZONE_ROI_DIMS, 'zone_idx_to_tag':ZONE_IDX_TO_TAG, 'n_frames':N_FRAMES,
         'zone_tag_to_idx':ZONE_TAG_TO_IDX, 'zone_to_grp_idx':ZONE_TO_GRP_IDX, 'frm_dim':FRM_DIM,
         'scale_ftr':SCALE_DIM_FACTOR, 'bg_pixels':BG_PIXELS, 'fe_out_shape':FE_OUT_SHAPE,
         'min_gap':MIN_GAP, 'max_xy_shift_aug':max_xy_shift, 'subnet_tags':subnet_tags}
    pipe = DataPipeline(cfg, in_data, len(ids_indexes), d_set, zone_ordered_fids,
                        prw_dims, None, pipe_constants, None, unique_ids=unique_ids)

    cache = cfg.TEST.CACHE_TF_DATA
    if cache: cache = tst_cache_path
    ds = test_dataset(pipe, smp_indexes, cfg, cache_path=cache)

    return ds, pipe, unique_ids


def run_test(default_model, model_paths, threat_dfs, bdpart_dfs,
             threat_csvs, bdpart_csvs, tst_ds, tst_pipe, tst_sample_ids,
             has_bdpart_output, is_multi_outputs, is_subnet_output, n_decimals):
    print('\nRunning evaluation\n')
    n_samples = tst_pipe.data_sample_size
    n_models = len(model_paths)
    #n_models = len(models)
    assert(n_models==len(threat_dfs))
    assert(3<=n_decimals<=7), "recommended decimal precision is [3, 7] not {}".format(n_decimals)
    min_clip, max_clip = get_clip_values(n_decimals)

    for m_idx in range(n_models):
        #model = models[m_idx]
        model = load_weights_into_model(default_model, model_paths[m_idx])
        threat_df = threat_dfs[m_idx]
        threat_csv = threat_csvs[m_idx]
        if has_bdpart_output:
            bdpart_df = bdpart_dfs[m_idx]
            bdpart_csv = bdpart_csvs[m_idx]
        s_idx = 0

        tokens = threat_csv.split('_')
        print('\n{}. {}'.format(m_idx+1, tokens[-2]))

        # iterate over batches of validation subset, until the end
        for step_idx, sample_elements in enumerate(tst_ds):
            s_inputs, batch_sid_idx, subout_idx = sample_elements
            batch_size = len(batch_sid_idx)
            net_outputs = model.predict(s_inputs, verbose=0)
            if is_subnet_output:
                subout_idx = np.ravel(subout_idx.numpy())
                threat_output = np.asarray(net_outputs, dtype=np.float32)
                threat_output = threat_output[subout_idx, range(batch_size), [0]]
            elif is_multi_outputs:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0] #***
            elif has_bdpart_output:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0]
                bdpart_output = np.asarray(net_outputs[1], dtype=np.float32)
            else: threat_output = np.asarray(net_outputs, dtype=np.float32)[:, 0] #***

            threat_output = np.clip(threat_output, min_clip, max_clip)  # shape=(?,1)
            threat_prob = np.around(threat_output, n_decimals)
            threat_prob = np.clip(threat_prob, min_clip, max_clip)  # Added 03/06/2021 to ensure
            batch_sid_idx = np.ravel(batch_sid_idx.numpy())
            batch_smp_ids = [tst_sample_ids[elem] for elem in batch_sid_idx]

            for b_idx in range(batch_size):
                sample_id = batch_smp_ids[b_idx]
                scan_id, zone_name = sample_id.split('_')
                zone_id = ZONE_NAME_TO_ID[zone_name]
                smp_df_id = '{}_{}'.format(scan_id, zone_id)
                rec_indexes = threat_df.index[threat_df['Id']==smp_df_id].tolist()
                assert(len(rec_indexes)==1), "rec_indexes size:{}".format(len(rec_indexes))
                record_idx = rec_indexes[0]
                assert(threat_df.at[record_idx, 'Probability']==-1)
                threat_df.at[record_idx, 'Probability'] = threat_prob[b_idx]
                #threat_df.at[record_idx, 'Raw-Output'] = threat_output[b_idx]
                #assert#(threat_df.loc[threat_df['Id']==smp_id, 'Probability'].values[0]==-1)
                #threat_df.loc[threat_df['Id']==smp_id, 'Probability'] = threat_output[j]
                if has_bdpart_output:
                    zone_type_indx = ZONE_TAG_TO_IDX[zone_name]
                    bdpart_df.loc[bdpart_df['Id']==smp_df_id, 'Probability'] = \
                        bdpart_output[b_idx][zone_type_indx]

            s_idx += batch_size
            if (s_idx%(10*BATCH_SIZE)==0) or (s_idx==n_samples):
                print('{:>5}/{} samples passed..'.format(s_idx, threat_df.shape[0]))

        # verify validity and save predictions
        assert(s_idx==n_samples)
        threat_probabilities = threat_df.loc[:, 'Probability'].values
        assert(np.all(threat_probabilities>=0))
        assert(np.all(threat_probabilities<=1))
        threat_df.to_csv(threat_csv, encoding='utf-8', index=False, sep=',')
        if has_bdpart_output:
            bdpart_df.to_csv(bdpart_csv, encoding='utf-8', index=False, sep=',')



def main():
    global images_dir, prw_shapes, prw_dims, present_zones, zone_ordered_fids, kpts_df, zfk_map, \
        subnet_tags, ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, ZONE_ROI_DIMS, N_FRAMES, \
        SCALE_DIM_FACTOR, FRM_DIM, FRM_CORNERS, IMG_CEF, BG_PIXELS, MIN_GAP, PIXELS_MINMAX, \
        FE_OUT_SHAPE, ZFK_MAP_TYPE, ROI_BB_SHAPE, BDGRP_CROP_CONFIG, BDGRP_ADJUSTMENT, BATCH_SIZE

    # Test set inits
    sample_csv = '../Metadata/tsa_psc/stage2_sample_submission.csv'
    model_hdir = '../PSCNets/models/'

    args = runtime_args()
    cfg_tokens = args.cfg.split('/')
    tsa_dtype, sub_dir, trn_exp_id = cfg_tokens # eg ['aps','preserve','Comb10_2020-04-09-02-14']
    args.cfg = str(Path(model_hdir) / Path(args.cfg) / Path('{}_config.yaml'.format(trn_exp_id)))
    cfg = default_experiment_configuration()
    adapt_config(cfg, args, physical_gpus, logical_gpus)
    assert(tsa_dtype==cfg.DATASET.FORMAT)
    exp_subdir = '{}/{}/'.format(tsa_dtype, sub_dir)
    net_suffix = dict()
    if args.loglos:
        str_ints = args.logf1s.split(',') # [0, 36]
        net_suffix['log&f1'] = np.array(str_ints, dtype=np.int32)
    if args.loglos:
        str_ints = args.loglos.split(',') # [0, 72, 23500, 26200]
        net_suffix['loglos'] = np.array(str_ints, dtype=np.int32)
    if args.avgf1s:
        str_ints = args.avgf1s.split(',') # [0, 24000],
        net_suffix['avgf1s'] = np.array(str_ints, dtype=np.int32)
    assert(args.loglos or args.avgf1s)
    assert(cfg.TEST.AUGMENTATION==False)
    assert(cfg.AUGMENT.WANDERING_ROI==False)
    assert(cfg.DATASET.ROOT.find('aps_images')>=0)
    model_format = args.netfmt
    images_dir = os.path.join(cfg.DATASET.ROOT, 'test_set')

    kpts_df = pd.read_csv(cfg.LABELS.CSV.KPTS_SET) # kpts_t_csv
    map_df = pd.read_csv(cfg.LABELS.CSV.FZK_MAP) # zfk_map_csv
    assert(len(cfg.MODEL.GROUPS)==10)
    assert(cfg.MODEL.GROUPS==list(BODY_GROUP_ZONES.keys()))
    ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, present_zones = \
        zone_index_id_map(cfg.MODEL.GROUPS)
    assert(cfg.MODEL.REGION_DIM[0]>=cfg.MODEL.IMAGE_SIZE[1])
    assert(cfg.MODEL.REGION_DIM[1]>=cfg.MODEL.IMAGE_SIZE[0])

    # initialize constants shared by both comb_train.py and comb_test.py
    exp_dir = os.path.join(model_hdir, exp_subdir, trn_exp_id)
    tst_cache_path = str(Path(exp_dir) / 'tst_cache')
    constants_file = os.path.join(exp_dir, 'constants.pickle')
    with open(constants_file, 'rb') as file_handle:
        test_constants = pickle.load(file_handle)
    BDGRP_CROP_CONFIG = test_constants['BDGRP_CROP_CONFIG']
    BDGRP_ADJUSTMENT = test_constants['BDGRP_ADJUSTMENT']
    ZONE_ROI_DIMS = test_constants['ZONE_ROI_DIMS']
    PIXELS_MINMAX = test_constants['PIXELS_MINMAX']
    ZFK_MAP_TYPE = test_constants['ZFK_MAP_TYPE']
    ROI_BB_SHAPE = test_constants['ROI_BB_SHAPE']
    FE_OUT_SHAPE = test_constants['FE_OUT_SHAPE']

    tsa_ext = cfg.DATASET.FORMAT
    zone_ordered_fids = get_grp_ordered_zones(present_zones, map_df, tsa_ext)
    if tsa_ext=='aps' and cfg.DATASET.ROOT.find('a3daps')>=0:
        zone_ordered_fids = map_aps_frames_2_a3daps(zone_ordered_fids)
        kpts_df = rename_columns_with_a3daps_fids(kpts_df)
        map_df = replace_entries_with_a3daps_fids(map_df)
        N_FRAMES = DATA_NUM_FRAMES['a3daps']
        MIN_GAP = 4
    else:
        N_FRAMES = DATA_NUM_FRAMES[tsa_ext]
        MIN_GAP = 1
    zfk_map = df_map_to_dict_map(map_df, ZONE_NAME_TO_ID)

    IMG_CEF = cfg.DATASET.ENHANCE_VISIBILITY
    bgi = read_image(cfg.DATASET.BG_IMAGE, rgb=cfg.DATASET.COLOR_RGB, icef=IMG_CEF)
    BG_PIXELS = np.mean(bgi, axis=(0, 1)).astype(np.uint8, copy=False).tolist()
    SCALE_DIM_FACTOR = np.flip(np.float32(cfg.MODEL.IMAGE_SIZE[:2])) / \
                       np.asarray(cfg.MODEL.REGION_DIM)
    FRM_DIM = np.int32([512, 660] * SCALE_DIM_FACTOR)
    FRM_CORNERS = np.float32([[          0,          0, 1],     # top-left
                              [ FRM_DIM[0],          0, 1],     # top-right
                              [ FRM_DIM[0], FRM_DIM[1], 1],     # bottom-right
                              [          0, FRM_DIM[1], 1]]).T  # bottom-left
    BATCH_SIZE = cfg.TEST.BATCH_SIZE

    # read submission csv dataframe and reset all probabilities to -1.
    # and set Probability column to floating point
    df_template = pd.read_csv(sample_csv)
    df_template.loc[:, 'Probability'] = -1.
    #df_template['Raw-Output'] = -1.
    df_template = df_template.astype({'Probability':np.float32})#, 'Raw-Output':np.float32})

    #models = []
    threat_dfs = list()
    bdpart_dfs = list()
    threat_csvs = list()
    bdpart_csvs = list()
    model_paths = list()
    is_multi_outputs = len(cfg.LOSS.NET_OUTPUTS_ID)>1
    has_bdpart_output = 'p' in cfg.LOSS.NET_OUTPUTS_ID
    is_subnet_output = cfg.MODEL.SUBNET_TYPE in SUBNET_CONFIGS
    if is_subnet_output:
        if cfg.MODEL.SUBNET_TYPE=='body_zones':
            subnet_tags = truncate_strings(present_zones)
        elif cfg.MODEL.SUBNET_TYPE=='body_groups':
            subnet_tags = truncate_strings(cfg.MODEL.GROUPS)
    else: subnet_tags = None
    hdf5_model_path = os.path.join(exp_dir, '{}.h5'.format(trn_exp_id))
    default_model = get_hdf5_model(hdf5_model_path)
    for met_type in net_suffix.keys():
        for epoch_tag in net_suffix[met_type]:
            prefix = 'wgts-{}-{}'.format(epoch_tag, met_type)
            if model_format=='h5':
                #prefix = '{}_wgts-{}-{}'.format(trn_exp_id, epoch_tag, met_type)
                model_param_path = os.path.join(exp_dir, 'models', '{}.h5'.format(prefix))
            elif model_format=='tf':
                #prefix = 'wgts-{}-{}'.format(epoch_tag, met_type)
                model_param_path = os.path.join(exp_dir, 'models', prefix)

            #model = get_hdf5_model(hdf5_model_path)
            #models.append(load_weights_into_model(model, model_param_path))
            model_paths.append(model_param_path)
            path_prefix = os.path.join(exp_dir, 'models', '{}_{}'.format(prefix, trn_exp_id))
            threat_csvs.append('{}_t.csv'.format(path_prefix))
            threat_dfs.append(df_template.copy())
            if has_bdpart_output:
                bdpart_csvs.append('{}_p.csv'.format(path_prefix))
                bdpart_dfs.append(df_template.copy())

    # build datasets, input pipelines, and callbacks
    tst_ds, tst_pipe, tst_ids = input_setup(tst_cache_path, 'test', cfg)
    # remove (free up memory) variables no longer needed
    del map_df, kpts_df, zfk_map

    # run_test(models, threat_dfs, bdpart_dfs, threat_csvs, bdpart_csvs, tst_ds, tst_pipe,
    #          tst_ids, has_bdpart_output, is_multi_outputs, is_subnet_output, cfg.TEST.N_DECIMALS)
    run_test(default_model, model_paths, threat_dfs, bdpart_dfs,
             threat_csvs, bdpart_csvs, tst_ds, tst_pipe, tst_ids,
             has_bdpart_output, is_multi_outputs, is_subnet_output, cfg.TEST.N_DECIMALS)

    # delete cached data
    remove_files(exp_dir, ['metadata', 'cache'])


if __name__=='__main__':
    main()
    print('\nEvaluation complete')