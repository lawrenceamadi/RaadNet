from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

sys.path.append('../')
from tf_neural_net.custom_objects import metrics_from_cfm, CustomBCELoss, BatchWiseMetrics
from tf_neural_net.keras_model import Denoiser, Slicer, ExpandDims, CircularPad3D, ROIPooling

multizone_labels = np.arange(17, dtype=np.float32)
binary_labels = [0.0, 1.0]
eps = np.float64(1e-15)


def collect_data(tf_dataset, pipe_obj, use_smp_wgt):
    n_samples = pipe_obj.data_sample_size
    X_shape = (pipe_obj.data_sample_size, *pipe_obj.INPUT_SAMPLE_SHAPE)

    X = np.zeros(X_shape, dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)

    # Pass data to model batch by batch
    s_idx = 0
    for step_idx, sample_elements in enumerate(tf_dataset):
        if use_smp_wgt:
            s_images, s_label, s_weight, s_id_idx = sample_elements
        else: s_images, s_label, s_id_idx = sample_elements

        X_batch = s_images.numpy()
        y_true_batch = np.ravel(s_label.numpy())

        e_idx = s_idx + y_true_batch.shape[0]
        X[s_idx:e_idx] = X_batch
        y[s_idx:e_idx] = y_true_batch
        s_idx = e_idx

    assert (s_idx==n_samples)
    return X, y


def get_labels_and_probs(tf_dataset, model, n_samples, body_grp_names,
                         grp_ids_indexes, clip_minmax, n_decimals=7,
                         has_bdpart_output=False, is_multi_outputs=False, is_subnet_output=False):
    min_clip, max_clip = clip_minmax
    sid_idx_2_out_idx = np.zeros((n_samples), dtype=np.int32)
    y_threat_gt = np.zeros((n_samples), dtype=np.uint8)
    y_threat_pb = np.zeros((n_samples), dtype=np.float64)
    y_bdpart_gt, y_bdpart_pb = None, None
    if has_bdpart_output:
        y_bdpart_gt = np.zeros((n_samples, 17), dtype=np.uint8)
        y_bdpart_pb = np.zeros((n_samples, 17), dtype=np.float64)

    s_idx = 0
    for step_idx, sample_elements in enumerate(tf_dataset):
        s_inputs, s_gt_lbs, s_roi_bb, s_id_idx, subout_idx = sample_elements
        batch_samples_cnt = len(s_id_idx)
        net_outputs = model.predict(s_inputs, verbose=0)
        if is_subnet_output:
            threat_gt_lbl = np.ravel(s_gt_lbs.numpy())
            subout_idx = np.ravel(subout_idx.numpy())
            threat_output = np.asarray(net_outputs, dtype=np.float32)
            threat_output = threat_output[subout_idx, range(batch_samples_cnt), [0]]
        elif is_multi_outputs:
            threat_gt_lbl = np.ravel(s_gt_lbs.numpy())
            threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0] #***
        elif has_bdpart_output:
            threat_gt_lbl = np.ravel(s_gt_lbs['t'].numpy())
            threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0]
            bdpart_gt_lbl = s_gt_lbs['p'].numpy()
            bdpart_output = np.asarray(net_outputs[1], dtype=np.float32)
        else:
            threat_gt_lbl = np.ravel(s_gt_lbs.numpy())
            threat_output = np.asarray(net_outputs, dtype=np.float32)[:, 0] #***

        e_idx = s_idx + batch_samples_cnt
        y_threat_gt[s_idx:e_idx] = threat_gt_lbl
        y_threat_pb[s_idx:e_idx] = threat_output
        if has_bdpart_output:
            bdpart_gt_batch = bdpart_gt_lbl
            y_bdpart_gt[s_idx:e_idx] = bdpart_gt_batch
            y_bdpart_pb[s_idx:e_idx] = bdpart_output

        batch_sid_idx = np.ravel(s_id_idx.numpy())
        sid_idx_2_out_idx[batch_sid_idx] = s_idx + np.arange(batch_samples_cnt)
        s_idx = e_idx

    assert (s_idx==n_samples)
    grp_sout_indexes = dict()
    for grp_name in body_grp_names:
        grp_sid_indexes = grp_ids_indexes[grp_name]
        grp_sout_indexes[grp_name] = sid_idx_2_out_idx[grp_sid_indexes]

    y_threat_pb = np.around(np.clip(y_threat_pb, min_clip, max_clip), n_decimals)
    return y_threat_gt, y_threat_pb, y_bdpart_gt, y_bdpart_pb, grp_sout_indexes


def run_evaluations(threat_gt, threat_pb, part_gt, part_pb, name,
                    body_grp_names, grp_sout_indexes, thresh=0.5, logger=None):
    msg = '\n {} reloaded and evaluated'.format(name)
    logger.log_msg(msg), print(msg)
    custom_evaluation(threat_gt, threat_pb, body_grp_names, grp_sout_indexes, logger=logger)
    if part_pb is not None and part_gt is not None:
        p_log = log_loss(part_gt, part_pb, eps=eps)
        part_gt_indexes = np.argmax(part_gt, axis=-1)
        part_pb_indexes = np.argmax(part_pb, axis=-1)
        p_acc = accuracy_score(part_gt_indexes, part_pb_indexes)
        msg = '\n\tbody part logloss: {:.4f}, accuracy: {:.4f}'.format(p_log, p_acc)
        msg += '\n\tProb. of Threat and Body Part:'
        logger.log_msg(msg), print(msg)
        pb_axis_indexes = np.arange(part_pb.shape[0], dtype=np.int32)
        assert (len(part_gt_indexes)==len(pb_axis_indexes))
        each_part_pb = part_pb[pb_axis_indexes, part_gt_indexes]
        tnp_prob = threat_pb * each_part_pb
        custom_evaluation(threat_gt, tnp_prob, body_grp_names, grp_sout_indexes, logger=logger)
        msg = '\n\tCustom Prob. Function:'
        logger.log_msg(msg), print(msg)
        part_rec = np.where(threat_pb>=thresh, each_part_pb, 1 - each_part_pb)
        tnp_prob = (threat_pb + part_rec) / 2
        custom_evaluation(threat_gt, tnp_prob, body_grp_names, grp_sout_indexes, logger=logger)


def custom_evaluation(y_true, y_prob, body_grp_names, grp_sout_indexes, logger=None):
    y_pred = np.where(y_prob>=0.5, 1, 0)

    # accumulate confusion matrix, log loss
    cfm = confusion_matrix(y_true, y_pred, labels=binary_labels)
    auc = roc_auc_score(y_true, y_pred, labels=binary_labels)
    logloss = log_loss(y_true, y_prob, eps=eps, labels=binary_labels)

    # Compute evaluation metrics
    c_rot_cfm = np.rot90(np.rot90(cfm))
    acc, n_f1s, n_pre, n_rec = metrics_from_cfm(c_rot_cfm)
    acc, t_f1s, t_pre, t_rec = metrics_from_cfm(cfm)
    f1_avg = (t_f1s + n_f1s) / 2

    # display computed metrics
    msg = display_grp_cfm(cfm, y_true, y_pred, binary_labels, body_grp_names, grp_sout_indexes)
    msg += '\n\n   val_logls:{:.4f} - val_auc:{:.4f} - val_acc:{:.4f}' \
           '\n   val_avgf1:{:.4f} - val_f1s:{:.4f} - val_pre:{:.4f} - val_rec:{:.4f}'\
            .format(logloss, auc, acc, f1_avg, t_f1s, t_pre, t_rec)

    if logger: logger.log_msg(msg)
    print(msg)


def display_grp_cfm(all_cfm, y_true, y_pred, binary_labels, body_grp_names, grp_sout_indexes):
    grps_cfm_dict = dict()
    msg = '\n   Confusion Matrices:\n   |*Combined*| '
    for grp_name in body_grp_names:
        msg += '|{:-^8}|'.format(grp_name)
        idxes = grp_sout_indexes[grp_name]
        grps_cfm_dict[grp_name] = confusion_matrix(y_true[idxes], y_pred[idxes],
                                                        labels=binary_labels)
    msg += '\n   |{:>5}{:>5}| '.format(all_cfm[0][0], all_cfm[0][1])
    for grp_name in body_grp_names:
        grp_cfm = grps_cfm_dict[grp_name]
        msg += '|{:>4}{:>4}|'.format(grp_cfm[0][0], grp_cfm[0][1])
    msg += '\n   |{:>5}{:>5}| '.format(all_cfm[1][0], all_cfm[1][1])
    for grp_name in body_grp_names:
        grp_cfm = grps_cfm_dict[grp_name]
        msg += '|{:>4}{:>4}|'.format(grp_cfm[1][0], grp_cfm[1][1])
    return msg


def get_hdf5_model(model_path):
    ext = model_path[-3:]
    assert (ext=='.h5')

    # Recreate the exact same model, including its weights and the optimizer
    custom_layers = {'Denoiser':Denoiser, 'Slicer':Slicer, 'ExpandDims':ExpandDims,
                     'CircularPad3D':CircularPad3D, 'ROIPooling':ROIPooling,
                     'CustomBCELoss':CustomBCELoss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_layers)
    return model


def get_yaml_model(yaml_arc_file, model_wgt_path):
    # load YAML and create model
    yaml_file = open(yaml_arc_file, 'r')
    model_yaml = yaml_file.read()
    yaml_file.close()
    custom_layers = {'Denoiser':Denoiser, 'Slicer':Slicer, 'ExpandDims':ExpandDims,
                     'CircularPad3D':CircularPad3D, 'ROIPooling':ROIPooling}
    model = tf.keras.models.model_from_yaml(model_yaml, custom_objects=custom_layers)
    return load_weights_into_model(model, model_wgt_path)


def get_json_model(json_arc_file, model_wgt_path):
    # load json and create model
    json_file = open(json_arc_file, 'r')
    model_json = json_file.read()
    json_file.close()
    custom_layers = {'Denoiser':Denoiser, 'Slicer':Slicer, 'ExpandDims':ExpandDims,
                     'CircularPad3D':CircularPad3D, 'ROIPooling':ROIPooling}
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_layers)
    return load_weights_into_model(model, model_wgt_path)


def load_weights_into_model(model, model_wgt_path):
    # load weights into new model
    model.load_weights(model_wgt_path)
    return model