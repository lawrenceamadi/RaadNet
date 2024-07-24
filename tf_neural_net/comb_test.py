'''
v1: Test with saved json model architecture as base
'''

from __future__ import absolute_import, division, print_function, unicode_literals

#------------------------------------------------------------------------------
# START
import os
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
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("\n{} Physical GPUs, {} Logical GPUs".format(len(physical_gpus), len(logical_gpus)))
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        print('Memory growth must be set before GPUs have been initialized\nOR')
        print('\nVirtual devices must be set before GPUs have been initialized')

# END
#------------------------------------------------------------------------------

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append('../')
from tf_neural_net.keras_model import fe_network
from tf_neural_net.evaluation import get_json_model
from tf_neural_net.comb_pipeline import DataPipeline
from tf_neural_net.data_preprocess import read_image
from tf_neural_net.keras_logger import scale_image_pixels
from tf_neural_net.data_preprocess import crop_aligned_rois, crop_oriented_rois  # crop_rois_func
from tf_neural_net.commons import df_map_to_dict_map, get_grp_ordered_zones
from tf_neural_net.commons import adjust_xy_shift, map_aps_frames_2_a3daps, get_clip_values
from tf_neural_net.commons import zone_index_id_map, cv_close_windows, define_region_windows
from tf_neural_net.commons import replace_entries_with_a3daps_fids, truncate_strings
from tf_neural_net.commons import rename_columns_with_a3daps_fids, frm_zone_kpts_map_type
from tf_neural_net.commons import ZONE_NAME_TO_ID, DATA_NUM_FRAMES, BODY_GROUP_ZONES
from tf_neural_net.commons import SUBNET_CONFIGS, TSA_PIXELS_RANGE
from tf_neural_net.default import default_experiment_configuration
from tf_neural_net.default import runtime_args, adapt_config


def dataset_single_zone_list_dict(zone_name, scan_ids_list):
    sample_ids_list = list()

    for index, scan_id in enumerate(scan_ids_list):
        #if index < 5:
        sample_id = '{}_{}'.format(scan_id, zone_name)
        sample_ids_list.append(sample_id)

    return sample_ids_list


def collect_body_group_data(scan_ids):
    # Collect data for each zone in group
    unique_ids_list = list() # list of lists
    per_zone_samples = dict()
    unique_samples = 0
    samples_cnt = 0
    assert (len(present_zones)==17)
    for z_idx in range(len(present_zones)):
        zone_name = present_zones[z_idx]

        # Get training and validation data samples and corresponding labels
        unique_ids = dataset_single_zone_list_dict(zone_name, scan_ids)
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
    assert (ids_indexes.shape[0]==samples_cnt)
    assert (unique_samples==len(unique_ids_list))
    assert (max_index==len(unique_ids_list) - 1)
    assert (samples_cnt%17==0)
    assert (samples_cnt/17==1388 or samples_cnt/17==5)

    msg = '\nCollected Test Set:'
    msg += '\n{:,} total samples. {:,} total images.'.format(samples_cnt, samples_cnt*12)
    msg += '\n{:,} unique total samples, index ndarray dtype: {}'.\
            format(unique_samples, ids_indexes.dtype)
    print(msg)
    return ids_indexes, unique_ids_list, per_zone_samples


def get_region_data(sample_ids_indexes, unique_sample_ids, set_kpts_df, perzone_nsamples,
                    prw_shapes, cfg, debug_display=False):
    cropped_images = dict()
    byte_size = 0
    for z_name in present_zones:
        a_shp = (perzone_nsamples[z_name], len(zone_ordered_fids[z_name]), *prw_shapes[z_name])
        cropped_images[z_name] = np.empty(a_shp, dtype=np.uint8) # eg. (1247 x 12 x 164 x 164 x 3)
        byte_size += cropped_images[z_name].nbytes

    msg = '\nReading and parsing images and other inputs to memory\n Test-set '
    msg += 'images will occupy {:,} bytes or {:.1f} GB'.format(byte_size, byte_size/(1024**3))
    print(msg)

    # Track body part or zone's bounding-box #
    pzs_cnts = np.asarray(list(perzone_nsamples.values()))
    assert (np.all(pzs_cnts==pzs_cnts[0])), 'all zones must have equal counts of unique cases'
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

    ds_sample_indexes = np.empty((len(sample_ids_indexes), 3), dtype=sample_ids_indexes.dtype)
    ds_sample_indexes_dict = dict()
    scanid_indexer_into_zone_array = dict()
    new_scan_index = 0
    unique_smp_cnt = 0

    for i, sample_id_indx in enumerate(sample_ids_indexes):
        sample_id = unique_sample_ids[sample_id_indx]
        sample_indexes = ds_sample_indexes_dict.get(sample_id, None)
        assert (sample_indexes is None)
        #if sample_id!='5e1980e91c56d8d23ae25c088ab4a859_RFm': continue

        # Process each UNIQUE sample_id ONLY once
        if sample_indexes is None:
            unique_smp_cnt += 1
            scan_id, z_name = sample_id.split('_')
            #print("{:>5}. {}:{}".format(i, sample_id_indx, sample_id))
            scan_dir = os.path.join(images_dir, scan_id)
            n_unique_ipz = len(zone_ordered_fids[z_name])
            zone_type_indx = ZONE_TAG_TO_IDX[z_name]
            scan_id_indx = scanid_indexer_into_zone_array.get(scan_id, None)
            if scan_id_indx is None:
                scan_id_indx = new_scan_index
                scanid_indexer_into_zone_array[scan_id] = scan_id_indx
                new_scan_index += 1
            assert (scan_id_indx < perzone_nsamples[z_name])
            ordered_fids_of_zone = zone_ordered_fids[z_name]
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
            print('{:>6,} of {:,} samples read into memory'.format(i+1, len(sample_ids_indexes)))

    assert (np.all(np.sum(all_roi_bbs[:,:,:,:,0], axis=-1)>=0)), "sum of each roi x coord >= 0"
    assert (np.all(np.sum(all_roi_bbs[:,:,:,:,1], axis=-1)>=0)), "sum of each roi y coord >= 0"

    if cfg.DATASET.SCALE_PIXELS:
        channels = ('red','green','blue') if cfg.DATASET.COLOR_RGB else ('blue','green','red')
        for z_name in present_zones:
            for idx, color_channel in enumerate(channels):
                # eg. (1247,?12,?87,?85,3)
                cropped_images[z_name][:,:,:,:,idx] = \
                    scale_image_pixels(cropped_images[z_name][:,:,:,:,idx],
                                       PIXELS_MINMAX[color_channel], ret_uint8=True)

    print('{} unique samples read into memory\n'.format(unique_smp_cnt))
    assert (unique_smp_cnt==len(sample_ids_indexes))
    if debug_display:
        cv_close_windows()
        plt.close()

    return ds_sample_indexes, (cropped_images, None, None, all_roi_bbs, all_nci_tlc, None)


def input_setup(d_set, cfg):
    scan_ids = os.listdir(images_dir)

    # Get training and validation data samples and corresponding labels
    ids_indexes, unique_ids, perzone_smpcnt = collect_body_group_data(scan_ids)

    # Get predicted keypoint location and confidence dataframe
    subset_kpts_df = kpts_df[kpts_df.scanID.isin(scan_ids)]

    # Compute crop-for-keep window dimensions
    assert (d_set=='test')
    rot_angle, zoom_ftr = 0., 0.
    xy_sft_f = (0., 0.)
    max_xy_shift = adjust_xy_shift(cfg.MODEL.IMAGE_SIZE[:2], xy_sft_f,
                                   present_zones, ZONE_TAG_TO_IDX, ZONE_ROI_DIMS)
    prw_shapes, prw_dims = \
        define_region_windows(present_zones, ZONE_TAG_TO_IDX, BDGRP_CROP_CONFIG,
                              cfg.MODEL.REGION_DIM, cfg.MODEL.IMAGE_SIZE[2], max_xy_shift,
                              rot_angle, zoom_ftr, SCALE_DIM_FACTOR, cfg.MODEL.FORCE_SQUARE)
    print('\nPreload Region-to-Crop window for {} set:\n{}\n'.format(d_set, prw_dims))

    # Read images into memory and in order to setup data pipeline
    smp_indexes, in_data = \
        get_region_data(ids_indexes, unique_ids, subset_kpts_df, perzone_smpcnt, prw_shapes,
                        cfg, cfg.DEBUG.TURN_ON_DISPLAY)
    # Set up tf.data input pipeline
    pipe_constants = \
        {'zone_roi_dims':ZONE_ROI_DIMS, 'zone_idx_to_tag':ZONE_IDX_TO_TAG, 'n_frames':N_FRAMES,
         'zone_tag_to_idx':ZONE_TAG_TO_IDX, 'zone_to_grp_idx':ZONE_TO_GRP_IDX, 'frm_dim':FRM_DIM,
         'scale_ftr':SCALE_DIM_FACTOR, 'bg_pixels':BG_PIXELS, 'fe_out_shape':FE_OUT_SHAPE,
         'min_gap':MIN_GAP, 'max_xy_shift_aug':max_xy_shift, 'subnet_tags':subnet_tags}
    pipe = DataPipeline(cfg, in_data, len(ids_indexes), d_set, zone_ordered_fids,
                        prw_dims, None, pipe_constants, None, unique_ids=unique_ids)

    return pipe, smp_indexes, unique_ids


def run_test(models, threat_dfs, bdpart_dfs, pipe, sample_indexes, sample_ids,
             has_bdpart_output, is_multi_outputs, is_subnet_output, subnet_type, n_decimals):
    print('\nRunning evaluation\n')
    assert (len(models)==len(threat_dfs))
    assert (3<=n_decimals<=7), "recommended decimal precision is [3, 7] not {}".format(n_decimals)
    min_clip, max_clip = get_clip_values(n_decimals)
    batch_images = np.empty((BATCH_SIZE, *pipe.IMGS_INSAMP_SHAPE), dtype=np.float32)
    batch_rmasks = np.empty((BATCH_SIZE, *pipe.ROIS_INSAMP_SHAPE), dtype=np.float32)
    batch_rcvecs = np.empty((BATCH_SIZE, pipe.IMGS_PER_SAMPLE, pipe.N_ZONES), dtype=np.float32)
    batch_smpids = ['']*BATCH_SIZE
    if is_subnet_output:
        batch_outidx = np.empty(BATCH_SIZE, dtype=np.int32)
        bdgrp_subnet = subnet_type=='body_groups'
    if has_bdpart_output:
        batch_partidx = np.empty((BATCH_SIZE), dtype=np.int32)

    s_idx = 0
    n_samples = sample_indexes.shape[0] # len(sample_indexes)
    while s_idx<n_samples:

        # fetch input data of samples into batch
        # and note their corresponding sample ids and subnet output index
        b_idx = 0
        while b_idx<BATCH_SIZE and (s_idx+b_idx)<n_samples:
            sample_id_indx, zone_type_indx, scan_id_indx = sample_indexes[s_idx+b_idx]
            zone_name = pipe.ZONE_IDX_TO_TAG[zone_type_indx]
            scan_id, z_name = sample_ids[sample_id_indx].split('_')
            assert (zone_name==z_name)
            zone_id = ZONE_NAME_TO_ID[zone_name]
            smp_id = '{}_{}'.format(scan_id, zone_id)
            batch_smpids[b_idx] = smp_id
            if is_subnet_output:
                if bdgrp_subnet:
                    batch_outidx[b_idx] = ZONE_TO_GRP_IDX[zone_type_indx]
                else: batch_outidx[b_idx] = zone_type_indx

            smp_imgs, smp_msks, smp_tlbl, smp_rcvs, smp_bbcs = \
                pipe.load_single_zone_images(zone_name, zone_type_indx, scan_id_indx)
            batch_images[b_idx] = smp_imgs
            if pipe.ROI_MASKING:
                # downsample roi mask input
                smp_msks = tf.keras.layers.AveragePooling2D(pool_size=pipe.DS_POOL_K,
                                                            strides=pipe.DS_STRIDE)(smp_msks)
            batch_rmasks[b_idx] = smp_msks
            batch_rcvecs[b_idx] = smp_rcvs
            if has_bdpart_output: batch_partidx[b_idx] = zone_type_indx
            b_idx += 1
        assert (b_idx%17==0), "batches must be a multiple of 17 to avoid @tf.function warning"
        s_inputs = {'crop_reg_imgs': tf.convert_to_tensor(batch_images[:b_idx], dtype=tf.float32),
                    'roi_msks_bbxs': tf.convert_to_tensor(batch_rmasks[:b_idx], dtype=tf.float32),
                    'reg_comp_vecs': tf.convert_to_tensor(batch_rcvecs[:b_idx], dtype=tf.float32)}

        # pass input data to models, get and record model predictions per sample
        for i, model in enumerate(models):
            net_outputs = model.predict(s_inputs, verbose=0)
            if is_subnet_output:
                threat_output = np.asarray(net_outputs, dtype=np.float32)
                threat_output = threat_output[batch_outidx, range(BATCH_SIZE), [0]]
            elif is_multi_outputs:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0] #***
            elif has_bdpart_output:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0]
                bdpart_output = np.asarray(net_outputs[1], dtype=np.float32)
            else: threat_output = np.asarray(net_outputs, dtype=np.float32)[:, 0] #***

            threat_output = np.clip(threat_output, min_clip, max_clip)  # shape=(?,1)
            threat_output = np.around(threat_output, n_decimals)
            assert (threat_output.shape[0]==b_idx)
            for j in range(b_idx):
                smp_id = batch_smpids[j]
                rec_indexes = threat_dfs[i].index[threat_dfs[i]['Id']==smp_id].tolist()
                assert (len(rec_indexes)==1)
                assert (threat_dfs[i].at[rec_indexes[0], 'Probability']==-1)
                threat_dfs[i].at[rec_indexes[0], 'Probability'] = threat_output[j]
                #assert (np.sum(np.int32(threat_dfs[i]['Id']==smp_id))==1)
                #assert (threat_dfs[i].loc[threat_dfs[i]['Id']==smp_id, 'Probability'].values[0]==-1)
                #threat_dfs[i].loc[threat_dfs[i]['Id']==smp_id, 'Probability'] = threat_output[j]
                if has_bdpart_output:
                    bdpart_dfs[i].loc[bdpart_dfs[i]['Id']==smp_id, 'Probability'] = \
                        bdpart_output[j][batch_partidx[j]]

        s_idx += b_idx
        if (s_idx%(10*BATCH_SIZE)==0) or (s_idx==n_samples):
            print('{:>5}/{} samples passed..'.format(s_idx, threat_dfs[0].shape[0]))

    return threat_dfs, bdpart_dfs


def main():
    global images_dir, prw_shapes, prw_dims, present_zones, zone_ordered_fids, kpts_df, \
        zfk_map, subnet_tags, ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, ZONE_ROI_DIMS, \
        SCALE_DIM_FACTOR, FRM_DIM, FRM_CORNERS, IMG_CEF, BG_PIXELS, N_FRAMES, MIN_GAP, \
        FE_OUT_SHAPE, ZFK_MAP_TYPE, ROI_BB_SHAPE, BDGRP_CROP_CONFIG, BDGRP_ADJUSTMENT, \
        BATCH_SIZE, PIXELS_MINMAX

    # Test set inits
    sample_csv = '../Metadata/tsa_psc/stage2_sample_submission.csv'
    model_hdir = '../PSCNets/models/'

    args = runtime_args()
    cfg_tokens = args.cfg.split('/')
    tsa_dtype, sub_dir, trn_exp_id = cfg_tokens # eg ['aps','preserve','Comb10_2020-04-09-02-14']
    args.cfg = str(Path(model_hdir) / Path(args.cfg) / Path('{}_config.yaml'.format(trn_exp_id)))
    cfg = default_experiment_configuration()
    adapt_config(cfg, args)
    assert (tsa_dtype==cfg.DATASET.FORMAT)
    exp_subdir = '{}/{}/'.format(tsa_dtype, sub_dir)
    net_suffix = dict()
    if args.loglos:
        str_ints = args.loglos.split(',') # [0, 72, 23500, 26200]
        net_suffix['loglos'] = np.array(str_ints, dtype=np.int32)
    if args.avgf1s:
        str_ints = args.avgf1s.split(',') # [0, 24000],
        net_suffix['avgf1s'] = np.array(str_ints, dtype=np.int32)
    assert (args.loglos or args.avgf1s)
    assert (cfg.TEST.AUGMENTATION==False)
    assert (cfg.AUGMENT.WANDERING_ROI==False)
    assert (cfg.DATASET.ROOT.find('aps_images')>=0)
    model_format = args.netfmt
    images_dir = os.path.join(cfg.DATASET.ROOT, 'test_set')

    kpts_df = pd.read_csv(cfg.LABELS.CSV.KPTS_SET) # kpts_t_csv
    map_df = pd.read_csv(cfg.LABELS.CSV.FZK_MAP) # zfk_map_csv
    assert (len(cfg.MODEL.GROUPS)==10)
    assert (cfg.MODEL.GROUPS==list(BODY_GROUP_ZONES.keys()))
    ZONE_IDX_TO_TAG, ZONE_TAG_TO_IDX, ZONE_TO_GRP_IDX, present_zones = \
        zone_index_id_map(cfg.MODEL.GROUPS)
    #assert (cfg.MODEL.IMAGE_SIZE[0]%8==0 and cfg.MODEL.IMAGE_SIZE[1]%8==0)
    assert (cfg.MODEL.REGION_DIM[0]>=cfg.MODEL.IMAGE_SIZE[1])
    assert (cfg.MODEL.REGION_DIM[1]>=cfg.MODEL.IMAGE_SIZE[0])

    # initialize constants shared by both comb_train.py and comb_test.py
    exp_dir = os.path.join(model_hdir, exp_subdir, trn_exp_id)
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
    #ZFK_MAP_TYPE = frm_zone_kpts_map_type(zfk_map)
    #ROI_BB_SHAPE = (4, 2) # test_constants['ROI_BB_SHAPE']
    #FE_OUT_SHAPE = fe_network(cfg.MODEL.EXTRA.FE_NETWORK, tuple(cfg.MODEL.IMAGE_SIZE),
    #                          cfg.MODEL.EXTRA.FE_OUT_LAYER, shape_only=True)

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
    #PIXELS_MINMAX = TSA_PIXELS_RANGE[tsa_ext]

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
    #ZONE_ROI_DIMS = get_roi_dims(present_zones, ZONE_TAG_TO_IDX,
    #                             BDGRP_CROP_CONFIG, SCALE_DIM_FACTOR)
    BATCH_SIZE = cfg.TEST.BATCH_SIZE_PER_GPU

    # read submission csv dataframe and reset all probabilities to -1
    df_template = pd.read_csv(sample_csv)
    df_template.loc[:, 'Probability'] = -1.

    models = []
    threat_csv = []
    bdpart_csv = []
    threat_dfs = []
    bdpart_dfs = []
    is_multi_outputs = len(cfg.LOSS.NET_OUTPUTS_ID)>1
    has_bdpart_output = 'p' in cfg.LOSS.NET_OUTPUTS_ID
    is_subnet_output = cfg.MODEL.SUBNET_TYPE in SUBNET_CONFIGS
    if is_subnet_output:
        if cfg.MODEL.SUBNET_TYPE=='body_zones':
            subnet_tags = truncate_strings(present_zones)
        elif cfg.MODEL.SUBNET_TYPE=='body_groups':
            subnet_tags = truncate_strings(cfg.MODEL.GROUPS)

    for met_type in net_suffix.keys():
        model_arch = os.path.join(exp_dir, '{}_structure.json'.format(trn_exp_id))

        for epoch_tag in net_suffix[met_type]:
            prefix = 'wgts-{}-{}'.format(epoch_tag, met_type)
            if model_format=='h5':
                #prefix = '{}_wgts-{}-{}'.format(trn_exp_id, epoch_tag, met_type)
                model_param_path = os.path.join(exp_dir, 'models', '{}.h5'.format(prefix))
            elif model_format=='tf':
                #prefix = 'wgts-{}-{}'.format(epoch_tag, met_type)
                model_param_path = os.path.join(exp_dir, 'models', prefix)

            models.append(get_json_model(model_arch, model_param_path))
            path_prefix = os.path.join(exp_dir, 'models', '{}_{}'.format(prefix, trn_exp_id))
            threat_csv.append('{}_t.csv'.format(path_prefix))
            threat_dfs.append(df_template.copy())
            if has_bdpart_output:
                bdpart_csv.append('{}_p.csv'.format(path_prefix))
                bdpart_dfs.append(df_template.copy())

    # build datasets, input pipelines, and callbacks
    eva_pipe, eva_indexes, eva_ids = input_setup('test', cfg)
    # remove (free up memory) variables no longer needed
    del map_df, kpts_df, zfk_map

    threat_dfs, bdpart_dfs = \
        run_test(models, threat_dfs, bdpart_dfs, eva_pipe, eva_indexes, eva_ids,
                 has_bdpart_output, is_multi_outputs, is_subnet_output,
                 cfg.MODEL.SUBNET_TYPE, cfg.TEST.N_DECIMALS)

    for i in range(len(threat_csv)):
        df = threat_dfs[i]
        threat_probabilities = df.loc[:, 'Probability'].values
        assert (np.all(threat_probabilities>=0))
        assert (np.all(threat_probabilities<=1))
        df.to_csv(threat_csv[i], encoding='utf-8', index=False, sep=',')
        if has_bdpart_output:
            bdpart_dfs[i].to_csv(bdpart_csv[i], encoding='utf-8', index=False, sep=',')


if __name__=='__main__':
    main()
    print('\nEvaluation complete')