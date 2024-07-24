'''
    Model architecture of localized threat detection
'''
##print('\nZone Model Called\n')
import tensorflow as tf
import numpy as np
import sys

sys.path.append('../')
from tf_neural_net.commons import get_subnet_output_names
from tf_neural_net.keras_model import fe_network, fully_connected_v3
from tf_neural_net.keras_model import conv_fc_network_v1, conv_fc_network_v2, conv_fc_network_v3
from tf_neural_net.keras_model import zone_grp_network_v3_td, zone_grp_network_v4_td
from tf_neural_net.keras_model import zone_grp_network_v5_td, zone_grp_network_v6_td



def pool_mask_cfg(image_shape, fe_out_shape, ret_2=True):
    # compute pool size and stride needed to downsample input masks
    image_dim = np.int32(image_shape[:2])
    fe_out_dim = np.int32(fe_out_shape[:2])
    assert (np.all(image_dim >= fe_out_dim))

    ds_stride = np.int32(np.floor(image_dim / fe_out_dim))
    ds_poolsz = image_dim - ((fe_out_dim - 1) * ds_stride)

    if ret_2:
        return tuple(ds_poolsz), tuple(ds_stride)

    return np.concatenate([ds_poolsz, ds_stride])


def grp_model_v3_td(cfg, n_images, n_zones, subnet_tags, dim, train_fe, logger):
    # This expects an additional one-hot-encode vec in y and predicts class of body part
    pair_size = cfg.MODEL.IMAGES_PER_SEQ_GRP
    version = cfg.MODEL.EXTRA.RES_CONV_VERSION
    assert (version in ['v3']), "Incompatible network version: {}".format(version)

    # Per image feature extraction model
    image_shape = tuple(cfg.MODEL.IMAGE_SIZE)
    out_layer = cfg.MODEL.EXTRA.FE_OUT_LAYER
    fe_net_id = cfg.MODEL.EXTRA.FE_NETWORK
    fe_net, fe_out_shape = fe_network(fe_net_id, image_shape, out_layer, train_fe, logger=logger)
    fet_h, fet_w, fet_c = fe_out_shape
    zmask_shape = (fet_h, fet_w, 1)  # tuple(cfg.MODEL.IMAGE_SIZE[:2]) + (1,)

    image_dim = np.int32(image_shape[:2])
    fe_out_dim = np.int32(fe_out_shape[:2])
    assert (np.all(image_dim >= fe_out_dim))

    # Sequence feature extraction per image
    imgs_input = tf.keras.Input(shape=(n_images, *image_shape), name='crop_reg_imgs')
    imgs_size = pair_size * fet_h * fet_w * fet_c # pair_size * np.prod(fe_out_shape)
    msks_input = tf.keras.Input(shape=(n_images, *zmask_shape), name='roi_msks_bbxs')
    msks_size = pair_size * fet_h * fet_w
    zcvs_input = tf.keras.Input(shape=(n_images, n_zones), name='reg_comp_vecs')
    rcvs_size = pair_size * n_zones

    iftr_shape = (pair_size, *fe_out_shape)
    mask_shape = (pair_size, *fe_out_shape[:2], 1)
    entity_bounds = [imgs_size, imgs_size + msks_size, imgs_size + msks_size + rcvs_size] # in-order
    merged_vec_input = tf.keras.Input(shape=(entity_bounds[2],), name='ftr_msk_rcv_vec')
    conv_fc_net = conv_fc_network_v1(cfg, merged_vec_input, entity_bounds, version,
                                     iftr_shape, mask_shape, logger=logger)

    # compute pool size and stride needed to downsample input masks
    ds_stride = np.int32(np.floor(image_dim / fe_out_dim))
    ds_poolsz = image_dim - ((fe_out_dim - 1) * ds_stride)
    ds_config = np.concatenate([ds_poolsz, ds_stride])

    grp_net = zone_grp_network_v3_td(cfg, n_images, imgs_input, msks_input, zcvs_input,
                                     fe_net, conv_fc_net, ds_config, logger=logger)
    return grp_net


def grp_model_v4_td(cfg, n_images, n_zones, subnet_tags, dim, train_fe, logger):
    # convolution architecture with roi-masking and element-wise multiplication
    pair_size = cfg.MODEL.IMAGES_PER_SEQ_GRP
    version = cfg.MODEL.EXTRA.RES_CONV_VERSION
    assert (version in ['v2', 'v3', 'v4']), "Incompatible network version: {}".format(version)

    # Per image feature extraction model
    image_shape = tuple(cfg.MODEL.IMAGE_SIZE)
    out_layer = cfg.MODEL.EXTRA.FE_OUT_LAYER
    fe_net_id = cfg.MODEL.EXTRA.FE_NETWORK
    fe_net, fe_out_shape = fe_network(fe_net_id, image_shape, out_layer, train_fe, logger=logger)
    fet_h, fet_w, fet_c = fe_out_shape
    zmask_shape = (fet_h, fet_w, 1) #tuple(cfg.MODEL.IMAGE_SIZE[:2]) + (1,)

    image_dim = np.int32(image_shape[:2])
    fe_out_dim = np.int32(fe_out_shape[:2])
    assert (np.all(image_dim >= fe_out_dim))

    # Sequence feature extraction per image
    imgs_input = tf.keras.Input(shape=(n_images, *image_shape), name='crop_reg_imgs')
    imgs_size = pair_size * fet_h * fet_w * fet_c
    msks_input = tf.keras.Input(shape=(n_images, *zmask_shape), name='roi_msks_bbxs')
    msks_size = pair_size * fet_h * fet_w
    zcvs_input = tf.keras.Input(shape=(n_images, n_zones), name='reg_comp_vecs')
    rcvs_size = pair_size * n_zones

    # shape for 3d convolutions
    iftr_shape = (pair_size, *fe_out_shape)
    mask_shape = (pair_size, *fe_out_shape[:2], 1)
    if dim == 2:
        iftr_shape = iftr_shape[1:]
        mask_shape = mask_shape[1:]
    entity_bounds = [imgs_size, imgs_size+msks_size, imgs_size+msks_size+rcvs_size] # in-order
    merged_vec_input = tf.keras.Input(shape=(entity_bounds[2],), name='ftr_msk_rcv_vec')
    conv_fc_net = conv_fc_network_v2(cfg, merged_vec_input, entity_bounds, version,
                                     iftr_shape, mask_shape, dim, logger=logger)

    grp_net = zone_grp_network_v4_td(cfg, n_images, imgs_input, msks_input, zcvs_input,
                                     fe_net, conv_fc_net, version, entity_bounds[2], logger=logger)
    return grp_net


def grp_model_v5_td(cfg, n_images, n_zones, subnet_tags, dim, train_fe, logger):
    # convolution architecture with roi-pooling
    rois_per_img = 1
    pair_size = cfg.MODEL.IMAGES_PER_SEQ_GRP
    version = cfg.MODEL.EXTRA.RES_CONV_VERSION
    assert (version in ['v5']), "Incompatible network version: {}".format(version)

    # Per image feature extraction model
    image_shape = tuple(cfg.MODEL.IMAGE_SIZE)
    roibb_shape = (rois_per_img, 4)
    out_layer = cfg.MODEL.EXTRA.FE_OUT_LAYER
    fe_net_id = cfg.MODEL.EXTRA.FE_NETWORK
    fe_net, fe_out_shape = fe_network(fe_net_id, image_shape, out_layer, train_fe, logger=logger)
    fet_h, fet_w, fet_c = fe_out_shape

    image_dim = np.int32(image_shape[:2])
    fe_out_dim = np.int32(fe_out_shape[:2])
    assert (np.all(image_dim>=fe_out_dim))

    # Sequence feature extraction per image
    imgs_input = tf.keras.Input(shape=(n_images, *image_shape), name='crop_reg_imgs')
    imgs_size = pair_size * fet_h * fet_w * fet_c
    rois_input = tf.keras.Input(shape=(n_images, *roibb_shape), name='roi_msks_bbxs')
    rois_size = pair_size * roibb_shape[-1]
    zcvs_input = tf.keras.Input(shape=(n_images, n_zones), name='reg_comp_vecs')
    rcvs_size = pair_size * n_zones

    # shape for 3d convolutions
    iftr_shape = (pair_size, *fe_out_shape)
    zois_shape = (pair_size, *roibb_shape)
    if dim == 2:
        iftr_shape = iftr_shape[1:]
        zois_shape = zois_shape[1:]
    entity_bounds = [imgs_size, imgs_size+rois_size, imgs_size+rois_size+rcvs_size] # in-order
    merged_vec_input = tf.keras.Input(shape=(entity_bounds[2],), name='ftr_zoi_rcv_vec')
    conv_fc_net = conv_fc_network_v2(cfg, merged_vec_input, entity_bounds, version,
                                     iftr_shape, zois_shape, dim, logger=logger)

    # compute pool size and stride needed to downsample input masks
    ds_stride = np.int32(np.floor(image_dim / fe_out_dim))
    ds_poolsz = image_dim - ((fe_out_dim - 1) * ds_stride)
    ds_config = np.concatenate([ds_poolsz, ds_stride])

    grp_net = zone_grp_network_v4_td(cfg, n_images, imgs_input, rois_input, zcvs_input,
                                     fe_net, conv_fc_net, version, ds_config, logger=logger)
    return grp_net


def grp_model_v6_td(cfg, n_images, n_zones, subnet_tags, dim, train_fe, logger):
    # convolution (subnet) architecture with roi-masking and element-wise multiplication
    pair_size = cfg.MODEL.IMAGES_PER_SEQ_GRP
    version = cfg.MODEL.EXTRA.RES_CONV_VERSION
    assert (version in ['v6','v7']), "Incompatible network version: {}".format(version)

    # Per image feature extraction model
    image_shape = tuple(cfg.MODEL.IMAGE_SIZE)
    out_layer = cfg.MODEL.EXTRA.FE_OUT_LAYER
    fe_net_id = cfg.MODEL.EXTRA.FE_NETWORK
    fe_net, fe_out_shape = fe_network(fe_net_id, image_shape, out_layer, train_fe, logger=logger)
    fet_h, fet_w, fet_c = fe_out_shape
    zmask_shape = (fet_h, fet_w, 1) #tuple(cfg.MODEL.IMAGE_SIZE[:2]) + (1,)

    image_dim = np.int32(image_shape[:2])
    fe_out_dim = np.int32(fe_out_shape[:2])
    assert (np.all(image_dim >= fe_out_dim))

    # Sequence feature extraction per image
    imgs_input = tf.keras.Input(shape=(n_images, *image_shape), name='crop_reg_imgs')
    imgs_size = pair_size * fet_h * fet_w * fet_c
    msks_input = tf.keras.Input(shape=(n_images, *zmask_shape), name='roi_msks_bbxs')
    msks_size = pair_size * fet_h * fet_w
    zcvs_input = tf.keras.Input(shape=(n_images, n_zones), name='reg_comp_vecs')
    rcvs_size = pair_size * n_zones

    # shape for 3d convolutions
    iftr_shape = (pair_size, *fe_out_shape)
    mask_shape = (pair_size, *fe_out_shape[:2], 1)
    if dim == 2:
        iftr_shape = iftr_shape[1:]
        mask_shape = mask_shape[1:]

    entity_bounds = [imgs_size, imgs_size+msks_size, imgs_size+msks_size+rcvs_size] # in-order
    subnet_out_names = get_subnet_output_names(subnet_tags, cfg.LOSS.NET_OUTPUTS_ID[0])

    if version=='v6':
        merged_vec_input = tf.keras.Input(shape=(entity_bounds[2],), name='ftr_msk_rcv_vec')
        conv_fc_net = conv_fc_network_v3(cfg, merged_vec_input, entity_bounds, version,
                                         iftr_shape, mask_shape, subnet_tags, dim, logger=logger)
        grp_net = \
            zone_grp_network_v5_td(cfg, n_images, imgs_input, msks_input, zcvs_input,
                                   fe_net, conv_fc_net, version, subnet_out_names,
                                   entity_bounds[2], logger=logger)
    elif version=='v7':
        fe_ftr_msk_input = tf.keras.Input(shape=(entity_bounds[1],), name='fe_ftr_msk_vec')
        conv_fc_net = conv_fc_network_v3(cfg, fe_ftr_msk_input, entity_bounds, version,
                                         iftr_shape, mask_shape, subnet_tags, dim, logger=logger)
        cftr_size = cfg.MODEL.EXTRA.RES_CONV_FILTERS[-1]
        tensor_bounds = [cftr_size, cftr_size+rcvs_size]
        ftr_zcv_input = tf.keras.Input(shape=(tensor_bounds[1],), name='snet_ftr_zcv_vec')
        wgt_reg = cfg.MODEL.WGT_REG_DENSE
        fc_units = cfg.MODEL.EXTRA.FC_UNITS
        rcv_unit = cfg.MODEL.EXTRA.RCV_LT_UNITS
        rcv_indx = cfg.MODEL.EXTRA.RCV_FC_INDEX
        fcb_activ = cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[0]
        logit_unit = cfg.MODEL.EXTRA.THREAT_LOGIT_UNITS
        drop_rate = cfg.MODEL.DROPOUT_FC
        fcl_net = fully_connected_v3(ftr_zcv_input, wgt_reg, drop_rate, fc_units,
                                     fcb_activ, rcv_unit, logit_unit, tensor_bounds,
                                     prefix='glob_fc_', merge_index=rcv_indx, logger=logger)
        grp_net = \
            zone_grp_network_v6_td(cfg, n_images, imgs_input, msks_input, zcvs_input,
                                   fe_net, conv_fc_net, fcl_net, version, subnet_out_names,
                                   entity_bounds[1], tensor_bounds[1], rcvs_size, logger=logger)
    return grp_net