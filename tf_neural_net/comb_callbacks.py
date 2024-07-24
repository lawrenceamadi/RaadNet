##print('\nZone Callback Called\n')
import gc
import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from tabulate import tabulate
from scipy import stats

sys.path.append('../')
from tf_neural_net.custom_objects import metrics_from_cfm


class ValidMetricCallback(tf.keras.callbacks.Callback):
    '''
        Keras Metrics
        Implements overloaded keras class and functions for computing evaluation
        metrics for validation set, implement best model checkpointing, early stopping,
        as well as TensorBoard logging of computed metrics

        Why Use Custom Validation Instead of the Generic model.fit Validation?
        1. For step-wise validation. Generic validation runs once at the end of each epoch
        2. To log other metrics such as f1, avg f1, confusion matrix
        3. To log TP, FP, TN, FN input images
    '''
    def __init__(self, pipe_obj, tf_data, ids_list, grp_ids_index, wrt_paths,
                 tb_writer, logger, validate_freq, eoe_validation, patience, delay_ckpt,
                 n_decimals, clip_minmax, debug, trn_smp_size, trn_epoch_steps):
        super().__init__()
        self.tf_dataset = tf_data
        self.val_ids_list = ids_list
        self.grp_ids_indexes = grp_ids_index
        self.n_samples = pipe_obj.data_sample_size
        self.IMGS_PER_SAMPLE = pipe_obj.IMGS_PER_SAMPLE
        self.X_IMGS_SHAPE = pipe_obj.IMGS_INSAMP_SHAPE
        self.X_ROIS_SHAPE = pipe_obj.ROIS_INSAMP_SHAPE # roi encoding may be masks or coordinates
        self.X_BBCS_SHAPE = pipe_obj.ROI_COORD_SHAPE # roi bounding-polygon vertices coordinates
        self.BATCH_SIZE = pipe_obj.BATCH_SIZE
        self.zone_ordered_frames = pipe_obj.zone_ordered_frames
        self.body_zone_names = pipe_obj.body_zone_names
        self.body_group_names = pipe_obj.cfg.MODEL.GROUPS #pipe_obj.body_group_names
        self.EOE_SHUFFLE = pipe_obj.EOE_SHUFFLE
        self.wrt_paths = wrt_paths
        self.PATIENCE = patience
        self.DELAY_CKPT = delay_ckpt
        self.DELAY_IMGLOG = delay_ckpt
        self.N_DECIMALS = n_decimals
        self.MIN_CLIP, self.MAX_CLIP = clip_minmax
        self.tb_writer = tb_writer
        self.logger = logger
        self.debug_step = debug.ONE_PER_STEP
        self.debug_pred = debug.PREDICTIONS
        self.IMG_LOG_MAX = debug.IMG_LOG_MAX
        self.VALIDATE_FREQ = validate_freq
        self.EOE_VALIDATE = eoe_validation
        self.TRN_SAMPLE_SIZE = trn_smp_size
        self.TRN_STEPS_PER_EPOCH = trn_epoch_steps
        self.HAS_BODYPART_OUTPUT = 'p' in pipe_obj.cfg.LOSS.NET_OUTPUTS_ID
        self.OUTPUTS_ID = pipe_obj.cfg.LOSS.NET_OUTPUTS_ID
        self.MULTI_OUTPUTS = len(self.OUTPUTS_ID)>1
        self.SUBNET_OUTPUTS = pipe_obj.SUBNET_OUTPUTS
        self.SUBNET_OUT_NAMES = pipe_obj.SUBNET_OUT_NAMES
        self.logit_units = 1
        self.BINARY_LABELS = [0, 1]
        self.EPS = np.float64(1e-15)  # np.finfo(np.float32).eps
        self.runtime = 0 # in minutes
        # Best practice to block out memory for variables once and for all
        self.y_true = np.zeros((self.n_samples), dtype=np.uint8)
        self.y_pred = np.zeros((self.n_samples), dtype=np.uint8)
        self.y_prob = np.zeros((self.n_samples), dtype=np.float64)
        # map sample_id index (index) to corresponding model output index (element)
        self.sid_idx_2_out_idx = np.zeros((self.n_samples), dtype=np.int32)
        self.grp_sout_indexes = {}
        self.set_logs_keys()

        msg = '\n\n\nValidation Performance:\n-------------------------------\n'
        msg += 'optimal models will be saved to:\n{}\n{}\n'.format(
                 self.wrt_paths['opt1_avgf1s_wgt'], self.wrt_paths['opt2_loglos_wgt'])
        self.logger.log_msg(msg), print(msg)

    def set_logs_keys(self):
        self.METRIC_NAME = ['Trn-loss','Log-loss','AUC','Accuracy','Precision','Recall']
        metric_suffix = ['loss','log','auc','acc','pre','rec'] # same order as self.METRIC_NAME
        # todo: implement switch
        if self.SUBNET_OUTPUTS:
            # create list of lists containing aggregate loss and metric tags for each subnet
            self.BP_TABLE_HEADER = ['Metrics']+self.SUBNET_OUT_NAMES
            self.subnets_mettags = []
            for met_suf in metric_suffix:
                met_tags = []
                for subnet_tag in self.SUBNET_OUT_NAMES:
                    met_tags.append(subnet_tag+'_'+met_suf)
                self.subnets_mettags.append(met_tags)
        elif self.MULTI_OUTPUTS:
            # create list of 2 lists. 2nd list contains output1/threat metrics
            # while 1st list contains aggregate loss and loss of each output
            loss_tags = [metric_suffix[0]]
            for loss_id in self.OUTPUTS_ID: loss_tags.append('{}_loss'.format(loss_id))
            met_tags = []
            for met_suf in metric_suffix[1:]:
                met_tags.append('{}_{}'.format(self.OUTPUTS_ID[0], met_suf))  # eg. 'zt'
            self.subnets_mettags = [loss_tags, met_tags]
        else:
            self.subnets_mettags = [metric_suffix]

        if self.HAS_BODYPART_OUTPUT:
            p_prefix = '{}_'.format(self.OUTPUTS_ID[0]) # 'p'
            self.p_loss_tag = '{}loss'.format(p_prefix)
            self.p_acc_tag = '{}acc'.format(p_prefix)

    def set_start_end_epochs(self, start_epoch_idx, last_epoch, stage_name): # todo: is this necessary?
        assert(stage_name in ['Warm-Up','Unfrozen']), 'Unrecognized stage:{}'.format(stage_name)
        self.start_epoch_idx = start_epoch_idx
        self.last_epoch = last_epoch
        self.stage_name = stage_name

    def on_train_begin(self, logs=None):
        # instance variable default initializations
        meta_info = np.asarray([0, 0, 0, np.Inf, 0, 0])
        file = self.wrt_paths['meta_info_file']
        if os.path.isfile(file):
            meta_info = np.load(file)
        self.opt_combo_epoch = 0
        self.opt1_score = meta_info[0]
        self.opt1_epoch = int(meta_info[1])
        self.opt1_step = int(meta_info[2])
        self.opt2_score = meta_info[3]
        self.opt2_epoch = int(meta_info[4])
        self.opt2_step = int(meta_info[5])
        self.no_improvement_epochs = 0
        if self.start_epoch_idx==0: # todo: is this necessary?
            # needed to log initial (default) metrics, reset in on_epoch_end
            self.current_epoch = 0
            self.current_trn_step = 0
            # set poorest possible scores for evaluation metrics before training
            self.log_scalar_values(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if self.debug_pred: self.init_image_log_vars()

    def on_train_end(self, logs=None):
        if self.stage_name=='Warm-Up':
            self.last_warmup_epoch = self.current_epoch

        if self.current_epoch>self.last_epoch:
            # perhaps a second call to on_train_end
            if self.tb_writer is not None:
                self.tb_writer.close()
        else:
            meta_info = np.asarray([self.opt1_score, self.opt1_epoch, self.opt1_step,
                                    self.opt2_score, self.opt2_epoch, self.opt2_step,
                                    self.last_warmup_epoch])
            file = self.wrt_paths['meta_info_file']
            np.save(file, meta_info)

    def on_train_batch_end(self, batch, logs=None):
        self.current_trn_step = self.current_epoch_idx * self.TRN_STEPS_PER_EPOCH + batch + 1
        if self.current_trn_step % self.VALIDATE_FREQ==0:
            self.run_validation()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.current_epoch_idx = epoch
        self.current_epoch = self.current_epoch_idx + 1 # map epoch index 0 -> epoch 1 and so on
        msg = '\n\nEpoch {:<3} ({:<8}) ------------------------------------\n'.\
                format(self.current_epoch, self.stage_name)
        self.logger.log_msg(msg), print(msg)

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = (time.time() - self.start_time) / 60 # in minutes
        self.runtime += epoch_time

        if self.EOE_VALIDATE:
            self.run_validation()

        #todo: more efficient compilation of msg (consider using string buffer)
        msg = '\n\tEnd-of-Epoch {} ({:<8}), epoch runtime: {:.1f} mins, ' \
              'total runtime: {:.2f} hrs\n\tTraining metrics >> Overall-Train-Loss:{:.4f}\n' \
                .format(self.current_epoch, self.stage_name, epoch_time, self.runtime/60, logs['loss'])

        if self.SUBNET_OUTPUTS:
            table_rows = []
            for met_idx, met_tags in enumerate(self.subnets_mettags):
                row_data = [self.METRIC_NAME[met_idx]]
                for tag in met_tags:
                    row_data.append(logs[tag])
                table_rows.append(row_data)
            trn_summary_table = tabulate(table_rows, headers=self.BP_TABLE_HEADER, tablefmt='psql', floatfmt='.4f')
            msg += trn_summary_table

        if self.HAS_BODYPART_OUTPUT:
            p_loss = logs[self.p_loss_tag]
            p_acc = logs[self.p_acc_tag]
            msg += '\n\tCumulative training loss: {:.4f}'.format(logs['loss'])
            msg += '\n\tZone recognition: trn_loss:{:.4f} - trn_acc:{:.4f}'.format(p_loss, p_acc)

        self.logger.log_msg(msg), print(msg)
        gc.collect()  # force garbage collection to free-up unreferenced memory
        self.early_stop() # early stopping

    def run_validation(self):
        self.val_start_time = time.time()
        self.get_labels_and_predictions()
        self.compute_report_metrics()
        if self.debug_pred: self.reset_image_log_counters()

    def get_labels_and_predictions(self):
        s_idx = 0
        # iterate over batches of validation subset, until the end
        for step_idx, (s_inputs, batch_y_true, batch_bbcords, batch_sid_idx, batch_so_idx) \
                in enumerate(self.tf_dataset):
            batch_size = len(batch_sid_idx)
            net_outputs = self.model.predict(s_inputs, verbose=0)  # shape; (n_outputs, batch, units)
            if self.SUBNET_OUTPUTS:
                batch_so_idx = np.ravel(batch_so_idx.numpy())
                threat_output = np.asarray(net_outputs, dtype=np.float32)
                threat_output = threat_output[batch_so_idx, range(batch_size), [0]]
            elif self.MULTI_OUTPUTS:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0]
            elif self.HAS_BODYPART_OUTPUT:
                threat_output = np.asarray(net_outputs[0], dtype=np.float32)[:, 0]
                ## bdpart_output = np.asarray(net_outputs[1], dtype=np.float32)
            else: threat_output = np.asarray(net_outputs, dtype=np.float32)[:, 0]

            batch_y_true = np.ravel(batch_y_true.numpy())
            e_idx = s_idx + batch_size
            self.y_true[s_idx: e_idx] = batch_y_true
            self.y_prob[s_idx: e_idx] = threat_output
            threat_output = np.where(threat_output>=0.5, 1, 0)
            self.y_pred[s_idx: e_idx] = threat_output

            batch_sid_idx = np.ravel(batch_sid_idx.numpy())
            batch_bbcords = batch_bbcords.numpy()
            if self.current_epoch==1 or self.EOE_SHUFFLE:
                self.sid_idx_2_out_idx[batch_sid_idx] = s_idx + np.arange(batch_size)
            s_idx = e_idx

            # Log image sample in TensorBoard
            if (self.debug_step or self.debug_pred) and self.current_epoch>self.DELAY_IMGLOG:
                X_batch_imgs = s_inputs['crop_reg_imgs'].numpy()
                X_batch_rois = s_inputs['roi_msks_bbxs'].numpy()
                X_batch_sids = [self.val_ids_list[elem] for elem in batch_sid_idx]
                if self.debug_step:
                    t_idx = np.argmax(batch_y_true)  # log first index with a threat, if any
                    self.log_input_sample(X_batch_imgs[t_idx], X_batch_rois[t_idx],
                                          X_batch_sids[t_idx], batch_bbcords[t_idx], 'sI')
                if self.debug_pred:
                    self.record_predictions(X_batch_imgs, X_batch_rois, X_batch_sids,
                                            batch_bbcords, batch_y_true, threat_output, batch_size)
        #assert(s_idx==self.n_samples), 's_idx:{}, n_samples:{}'.format(s_idx, self.n_samples)
        if self.current_epoch==1 or self.EOE_SHUFFLE:
            for grp_name in self.body_group_names:
                grp_sid_indexes = self.grp_ids_indexes[grp_name]
                self.grp_sout_indexes[grp_name] = self.sid_idx_2_out_idx[grp_sid_indexes]

    def compute_report_metrics(self):
        # accumulate confusion matrix, log loss
        cfm = confusion_matrix(self.y_true, self.y_pred, labels=self.BINARY_LABELS)
        auc = roc_auc_score(self.y_true, self.y_pred, labels=self.BINARY_LABELS)
        # below transformation of y_prob is important to mimic TSA's log-loss with 10^-7 precision
        self.y_prob = np.around(np.clip(self.y_prob, self.MIN_CLIP, self.MAX_CLIP), self.N_DECIMALS)
        logloss = log_loss(self.y_true, self.y_prob, eps=self.EPS, labels=self.BINARY_LABELS)

        # Compute evaluation metrics
        c_rot_cfm = np.rot90(np.rot90(cfm))
        acc, n_f1s, n_pre, n_rec = metrics_from_cfm(c_rot_cfm)
        acc, t_f1s, t_pre, t_rec = metrics_from_cfm(cfm)
        f1_avg = (t_f1s + n_f1s) / 2
        self.log_scalar_values(logloss, acc, auc, f1_avg, t_f1s, t_pre, t_rec)

        # save model and log input images if performance on evaluation metric(s) improved
        is_new_opt, ckpt_msg = self.checkpoint(f1_avg, logloss)
        if is_new_opt and self.debug_pred and self.current_epoch>self.DELAY_IMGLOG:
            self.log_input_imgs()
        val_time = (time.time() - self.val_start_time) / 60  # in minutes

        msg =  '\n\tEpoch {:<3} Step:{:<6} Validation runtime: {:.1f} mins'.\
                format(self.current_epoch, self.current_trn_step, val_time)
        msg += self.display_grp_cfm(cfm)
        msg += '\n\tval_log:{:.4f} - val_auc:{:.4f} - val_acc:{:.4f} - val_avgf1:' \
               '{:.4f} - val_f1s:{:.4f} - val_pre:{:.4f} - val_rec:{:.4f}'.format(
                logloss, auc, acc, f1_avg, t_f1s, t_pre, t_rec)
        msg += '\n\t\tmax_avg_f1s:{:.4f} at epoch:{}, step:{}'\
                .format(self.opt1_score, self.opt1_epoch, self.opt1_step)
        msg += '\n\t\tmin_logloss:{:.4f} at epoch:{}, step:{} {}\n' \
            .format(self.opt2_score, self.opt2_epoch, self.opt2_step, ckpt_msg)
        self.logger.log_msg(msg), print(msg)

    def display_grp_cfm(self, all_cfm):
        msg = '\n\tConfusion Matrices:\n\t|*Combined*| '
        grps_cfm_dict = {}
        for grp_name in self.body_group_names:
            msg += '|{:-^8}|'.format(grp_name)
            idxes = self.grp_sout_indexes[grp_name]
            grps_cfm_dict[grp_name] = confusion_matrix(self.y_true[idxes], self.y_pred[idxes],
                                                       labels=self.BINARY_LABELS)
        msg += '\n\t|{:>5}{:>5}| '.format(all_cfm[0][0], all_cfm[0][1])
        for grp_name in self.body_group_names:
            grp_cfm = grps_cfm_dict[grp_name]
            msg += '|{:>4}{:>4}|'.format(grp_cfm[0][0], grp_cfm[0][1])
        msg += '\n\t|{:>5}{:>5}| '.format(all_cfm[1][0], all_cfm[1][1])
        for grp_name in self.body_group_names:
            grp_cfm = grps_cfm_dict[grp_name]
            msg += '|{:>4}{:>4}|'.format(grp_cfm[1][0], grp_cfm[1][1])
        return msg

    def sample_weight_meta_data(self, wgts):
        wgt_desc = stats.describe(wgts.flatten())
        msg = '\nSample weights. minmax: {}, mean: {:.4f}, var: {:.4f}, skew: {:.4f}\n'. \
            format(wgt_desc[1], wgt_desc[2], wgt_desc[3], wgt_desc[4])
        self.logger.log_msg(msg)
        print(msg)

    def checkpoint(self, pef1_score, pef2_score):
        msg = ''
        new_optimal = False
        tag = self.current_epoch if self.current_epoch>self.DELAY_CKPT else 0

        if self.opt2_score>pef2_score and self.opt1_score<pef1_score:
            self.opt_combo_epoch = self.current_epoch
            self.opt2_score = pef2_score
            #self.opt2_epoch = self.current_epoch
            #self.opt2_step = self.current_trn_step
            self.opt1_score = pef1_score
            #self.opt1_epoch = self.current_epoch
            #self.opt1_step = self.current_trn_step
            msg += '\n\t\t**Saved improved log-loss and avg-f1s model weights..'
            model_path = self.wrt_paths['opt_log&f1s_wgt'].format(tag)
            self.model.save_weights(model_path)
            self.no_improvement_epochs = 0
            new_optimal = True

        # opt2/logloss is only used to checkpoint best
        if (not new_optimal) and self.opt2_score>pef2_score:
            self.opt2_score = pef2_score
            self.opt2_epoch = self.current_epoch
            self.opt2_step = self.current_trn_step
            msg += '\n\t\t**Saved improved log-loss model weights..'
            model_path = self.wrt_paths['opt2_loglos_wgt'].format(tag)
            self.model.save_weights(model_path)
            self.no_improvement_epochs = 0
            new_optimal = True

        # opt1/f1-score is used for early-stopping and checkpointing
        elif (not new_optimal) and self.opt1_score<pef1_score:
            self.opt1_score = pef1_score
            self.opt1_epoch = self.current_epoch
            self.opt1_step = self.current_trn_step
            msg += '\n\t\t**Saved improved avg-f1s model weights...'
            model_path = self.wrt_paths['opt1_avgf1s_wgt'].format(tag)
            self.model.save_weights(model_path)
            self.no_improvement_epochs = 0
            new_optimal = True

        return new_optimal, msg


    def early_stop(self):
        if self.opt1_epoch!=self.current_epoch:
            # if no improvement at the end of the epoch
            self.no_improvement_epochs += 1
            msg = '\n\tNo improvement. {} epochs to early-stopping..\n' \
                    .format(self.PATIENCE - self.no_improvement_epochs)
        else:
            msg = '\n\tRecorded improvement on log-loss OR avg-f1s\n'

        if self.no_improvement_epochs>=self.PATIENCE:
            self.model.stop_training = True  # triggers call to self.on_train_end()
            msg += '\nEarly Stopping.. No improvements in last {} epochs'.format(self.PATIENCE)

        self.logger.log_msg(msg), print(msg)

    def log_scalar_values(self, logloss, roc_auc, accuracy, avgf1score, f1score, precision, recall):
        # TensorBoard logging
        self.tb_writer.tblog_scalar('val_log', logloss, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_auc', roc_auc, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_acc', accuracy, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_avgf1', avgf1score, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_f1s', f1score, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_pre', precision, self.current_trn_step)
        self.tb_writer.tblog_scalar('val_rec', recall, self.current_trn_step)

    def log_input_imgs(self):
        msg = '\n\tLogging validation images at step:{} epoch:{}'.\
                format(self.current_trn_step, self.current_epoch)
        self.logger.log_msg(msg), print(msg)
        for i in range(self.false_neg_cnt):
            self.log_input_sample(self.false_neg_imgs[i], self.false_neg_rois[i],
                                  self.false_neg_sids[i], self.false_neg_bbcs[i], 'fN')
        for i in range(self.true_neg_cnt):
            self.log_input_sample(self.true_neg_imgs[i], self.true_neg_rois[i],
                                  self.true_neg_sids[i], self.true_neg_bbcs[i], 'tN')
        for i in range(self.false_pos_cnt):
            self.log_input_sample(self.false_pos_imgs[i], self.false_pos_rois[i],
                                  self.false_pos_sids[i], self.false_pos_bbcs[i], 'fP')
        for i in range(self.true_pos_cnt):
            self.log_input_sample(self.true_pos_imgs[i], self.true_pos_rois[i],
                                  self.true_pos_sids[i], self.true_pos_bbcs[i], 'tP')

    def log_input_sample(self, X_scan_imgs, X_scan_rois, X_scan_id, rois_coord, tag):
        for idx in range(self.IMGS_PER_SAMPLE):
            scan_id, zone_name = X_scan_id.split('_')
            frame_tag = 'i{}-Fr{}'.format(idx, self.zone_ordered_frames[zone_name][idx])
            # eg. Val/fN/ULTh/7142e2ff6b927d5154afded4c90e2acd/i2-Fr2/X
            name_tag = "Val/{}/{}/{}/{}/X".format(tag, zone_name, scan_id, frame_tag)
            self.tb_writer.tblog_image(name_tag, X_scan_imgs[idx], X_scan_rois[idx],
                                       rois_coord[idx], self.current_epoch)

    def record_predictions(self, X_imgs, X_rois, X_sids, X_bbcs, y_true, y_pred, batch_size):
        # note prediction images
        pred_cnt = self.false_pos_cnt + self.true_pos_cnt + self.false_neg_cnt + self.true_neg_cnt
        if pred_cnt<(4*self.IMG_LOG_MAX):
            for i in range(batch_size):
                if y_true[i]==1 and y_pred[i]==0 and self.false_neg_cnt<self.IMG_LOG_MAX:
                    self.false_neg_cnt += 1
                    idx = self.false_neg_cnt - 1
                    self.false_neg_imgs[idx] = X_imgs[i]
                    self.false_neg_rois[idx] = X_rois[i]
                    self.false_neg_sids[idx] = X_sids[i]
                    self.false_neg_bbcs[idx] = X_bbcs[i]
                elif y_true[i]==0 and y_pred[i]==1 and self.false_pos_cnt<self.IMG_LOG_MAX:
                    self.false_pos_cnt += 1
                    idx = self.false_pos_cnt - 1
                    self.false_pos_imgs[idx] = X_imgs[i]
                    self.false_pos_rois[idx] = X_rois[i]
                    self.false_pos_sids[idx] = X_sids[i]
                    self.false_pos_bbcs[idx] = X_bbcs[i]
                elif y_true[i]==0 and y_pred[i]==0 and self.true_neg_cnt<self.IMG_LOG_MAX:
                    self.true_neg_cnt += 1
                    idx = self.true_neg_cnt - 1
                    self.true_neg_imgs[idx] = X_imgs[i]
                    self.true_neg_rois[idx] = X_rois[i]
                    self.true_neg_sids[idx] = X_sids[i]
                    self.true_neg_bbcs[idx] = X_bbcs[i]
                elif y_true[i]==1 and y_pred[i]==1 and self.true_pos_cnt<self.IMG_LOG_MAX:
                    self.true_pos_cnt += 1
                    idx = self.true_pos_cnt - 1
                    self.true_pos_imgs[idx] = X_imgs[i]
                    self.true_pos_rois[idx] = X_rois[i]
                    self.true_pos_sids[idx] = X_sids[i]
                    self.true_pos_bbcs[idx] = X_bbcs[i]

    def init_image_log_vars(self):
        self.reset_image_log_counters()
        self.false_neg_sids = [''] * self.IMG_LOG_MAX
        self.false_pos_sids = [''] * self.IMG_LOG_MAX
        self.true_neg_sids = [''] * self.IMG_LOG_MAX
        self.true_pos_sids = [''] * self.IMG_LOG_MAX
        self.false_neg_bbcs = np.empty((self.IMG_LOG_MAX, *self.X_BBCS_SHAPE), dtype=np.int16)
        self.false_pos_bbcs = np.empty((self.IMG_LOG_MAX, *self.X_BBCS_SHAPE), dtype=np.int16)
        self.true_neg_bbcs = np.empty((self.IMG_LOG_MAX, *self.X_BBCS_SHAPE), dtype=np.int16)
        self.true_pos_bbcs = np.empty((self.IMG_LOG_MAX, *self.X_BBCS_SHAPE), dtype=np.int16)
        self.false_neg_rois = np.empty((self.IMG_LOG_MAX, *self.X_ROIS_SHAPE), dtype=np.float16)
        self.false_pos_rois = np.empty((self.IMG_LOG_MAX, *self.X_ROIS_SHAPE), dtype=np.float16)
        self.true_neg_rois = np.empty((self.IMG_LOG_MAX, *self.X_ROIS_SHAPE), dtype=np.float16)
        self.true_pos_rois = np.empty((self.IMG_LOG_MAX, *self.X_ROIS_SHAPE), dtype=np.float16)
        self.false_neg_imgs = np.empty((self.IMG_LOG_MAX, *self.X_IMGS_SHAPE), dtype=np.float16)
        self.false_pos_imgs = np.empty((self.IMG_LOG_MAX, *self.X_IMGS_SHAPE), dtype=np.float16)
        self.true_neg_imgs = np.empty((self.IMG_LOG_MAX, *self.X_IMGS_SHAPE), dtype=np.float16)
        self.true_pos_imgs = np.empty((self.IMG_LOG_MAX, *self.X_IMGS_SHAPE), dtype=np.float16)

    def reset_image_log_counters(self):
        self.false_neg_cnt = 0
        self.false_pos_cnt = 0
        self.true_neg_cnt = 0
        self.true_pos_cnt = 0



class OptimalStepsCallback(tf.keras.callbacks.Callback):
    '''
        Simply checkpoints or saves the model at specific steps
    '''
    def __init__(self, steps_configs, trn_epoch_steps, logger):
        super().__init__()
        self.steps_configs = steps_configs
        self.logger = logger
        self.TRN_STEPS_PER_EPOCH = trn_epoch_steps
        self.runtime = 0 # in minutes

        msg = '\n\n\nTrain ONLY experiment, no validation:\n-------------------------------\n'
        msg += 'anticipated optimal models will be saved as follows (step/path):\n{}\n'\
                .format(self.steps_configs)
        self.logger.log_msg(msg), print(msg)

    def set_start_epoch(self, start_epoch_idx):
        self.start_epoch_idx = start_epoch_idx

    def on_train_begin(self, logs=None):
        # instance variable default initializations
        if self.start_epoch_idx==0:
            # needed to log initial (default) metrics, reset in on_epoch_end
            self.current_epoch = 0
            self.current_trn_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.current_trn_step = self.current_epoch_idx * self.TRN_STEPS_PER_EPOCH + batch + 1
        if self.current_trn_step in self.steps_configs.keys():
            path = self.steps_configs[self.current_trn_step]
            self.model.save_weights(path)
            msg =  '\n\t**Saved optimal "metric" model weights at step {}, epoch {}, to..' \
                   '\n\t  {}'.format(self.current_trn_step, self.current_epoch, path)
            msg += '\n\t\tTraining performace at step and epoch is:\n\t\t{}\n'.format(logs)
            self.logger.log_msg(msg), print(msg)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.current_epoch_idx = epoch
        self.current_epoch = self.current_epoch_idx + 1 # map epoch index 0 -> epoch 1 and so on



class StepWiseLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule."""

    def __init__(self, lr_schedules, iter_mode, trn_epoch_steps, tb_writer, logger):
        super(StepWiseLearningRateScheduler, self).__init__()
        self.LR_SCHEDULES = lr_schedules
        self.ITER_MODE = iter_mode # iteration mode: 0-epochs, 1-steps
        self.NOTABLE_ITERATIONS = list(lr_schedules.keys())
        self.NOTABLE_ITERATIONS.sort()
        self.TRN_STEPS_PER_EPOCH = trn_epoch_steps
        self.tb_writer = tb_writer
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch_idx = epoch
        if self.ITER_MODE==0:
            if not hasattr(self.model.optimizer, 'lr'):
              raise ValueError('Optimizer must have a "lr" attribute.')

            trn_epoch = epoch + 1
            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            # Call schedule function to get the scheduled learning rate.
            ret, scheduled_lr = self.lr_scheduler(trn_epoch, lr)
            if ret:
                # Set the value back to the optimizer before this epoch starts
                tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                print('\nEpoch %05d: Learning rate is %7.5f.' % (trn_epoch, scheduled_lr))
            # log lr
            self.tb_writer.tblog_scalar('epoch_lr', data=scheduled_lr, step=trn_epoch)

    def on_train_batch_begin(self, batch, logs=None):
        if self.ITER_MODE==1:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')

            trn_step = self.current_epoch_idx * self.TRN_STEPS_PER_EPOCH + batch + 1
            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            # Call schedule function to get the scheduled learning rate.
            ret, scheduled_lr = self.lr_scheduler(trn_step, lr)
            if ret:
                # Set the value back to the optimizer before this batch starts
                tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                print('\nStep %05d: Learning rate is %11.9f' % (trn_step, scheduled_lr))
            # log lr
            self.tb_writer.tblog_scalar('batch_lr', data=scheduled_lr, step=trn_step)

    def lr_scheduler(self, iteration, lr):
        """Helper function to retrieve the scheduled learning rate based on epoch."""
        if iteration<self.NOTABLE_ITERATIONS[0] or iteration>self.NOTABLE_ITERATIONS[-1]:
            return False, lr

        for s_iteration, new_lr in self.LR_SCHEDULES.items():
            if iteration==s_iteration: return True, new_lr
        return False, lr



class CosineLearningRateScheduler(tf.keras.callbacks.Callback):
        """Learning rate scheduler which adjust the learning rate as follows:
            1. Starting from a low lr linearly increase to high lr for a fraction of total steps
            2. Then gradually reduce to a low lr using a cosine curve
        """

        def __init__(self, lr_config, epoch_interval, func_frac,
                     iter_mode, trn_epoch_steps, tb_writer, pipelines, logger):
            super(CosineLearningRateScheduler, self).__init__()
            self.glob_start_epoch, self.glob_stop_epoch = epoch_interval
            self.lfi_index = 0
            self.line_func_intervals = np.empty(shape=len(func_frac)+1, dtype=np.int32)
            if iter_mode==0:
                self.line_func_intervals[0] = self.glob_start_epoch
                self.line_func_intervals[1:] = self.glob_stop_epoch * func_frac
            elif iter_mode==1:
                self.line_func_intervals[0] = self.glob_start_epoch * trn_epoch_steps
                self.line_func_intervals[1:] = self.glob_stop_epoch * trn_epoch_steps * func_frac

            # line_func, x_interval, y_interval = self.line_metadata[self.lfi_index]
            self.line_metadata = []
            for i in range(len(func_frac)):
                func_type, start_lr, end_lr = lr_config[i*3: (i+1)*3]
                x_iter_interval = self.line_func_intervals[i:i+2]
                y_lr_interval = [start_lr, end_lr]
                line_func = eval('self.{}_func'.format(func_type))
                self.line_metadata.append((line_func, x_iter_interval, y_lr_interval))

            self.ITER_MODE = iter_mode  # iteration mode: 0-epochs, 1-steps
            self.TRN_STEPS_PER_EPOCH = trn_epoch_steps
            self.SELECTIVE_BCE = len(pipelines[0].cfg.LOSS.NET_OUTPUTS_ID)>1 and \
                                 pipelines[0].cfg.LOSS.NET_OUTPUTS_LOSS[1]=='selective_loss'
            self.SELECTIVE_BCE_EPOCH = pipelines[0].cfg.LOSS.EXTRA.SELECTIVE_BCE_EPOCH
            self.tb_writer = tb_writer
            self.pipelines = pipelines
            self.logger = logger

        def on_epoch_begin(self, epoch, logs=None):
            self.tb_writer.increment_epoch()
            self.current_epoch_idx = epoch
            trn_epoch = epoch + 1

            if self.ITER_MODE==0:  # per epoch updat
                if not hasattr(self.model.optimizer, 'lr'):
                    raise ValueError('Optimizer must have a "lr" attribute.')
                # retrieve lr at epoch
                lr = self.get_lr_at_iteration(trn_epoch)
                # Set the value back to the optimizer before this epoch starts
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                # log lr
                self.tb_writer.tblog_scalar('epoch_lr', lr, step=trn_epoch)

            if self.SELECTIVE_BCE and trn_epoch==self.SELECTIVE_BCE_EPOCH:
                msg = '\nSwitching to selective bce loss at epoch: {} ..'.format(trn_epoch)
                msg += '\n\tdefault_loss_branch_wgts vs. class_wgts'
                #self.logger.log_msg(msg), print(msg)
                for subset_pipe in self.pipelines:
                    msg += '\n\t{}-set;\tbefore: {} vs. {}'.format(subset_pipe.d_set,
                            subset_pipe.default_loss_branch_wgt, subset_pipe.class_weights)
                    subset_pipe.switch_loss_func()
                    subset_pipe.change_benign_class_wgt()
                    msg += '\tafter: {} vs. {}'.format(subset_pipe.default_loss_branch_wgt,
                                                       subset_pipe.class_weights)
                self.logger.log_msg(msg), print(msg)

        def on_train_batch_begin(self, batch, logs=None):
            self.tb_writer.increment_step()
            if self.ITER_MODE==1:  # per step update
                if not hasattr(self.model.optimizer, 'lr'):
                    raise ValueError('Optimizer must have a "lr" attribute.')

                trn_step = self.current_epoch_idx * self.TRN_STEPS_PER_EPOCH + batch + 1
                # retrieve lr at epoch
                lr = self.get_lr_at_iteration(trn_step)
                # Set the value back to the optimizer before this batch starts
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                # log lr
                self.tb_writer.tblog_scalar('batch_lr', lr, step=trn_step)

        def get_lr_at_iteration(self, iteration):
            """Helper function to compute learning rate based on iteration."""
            #assert(iteration>=self.line_func_intervals[0]), 'iteration:{}'.format(iteration)
            if iteration>self.line_func_intervals[self.lfi_index + 1]:
                self.lfi_index += 1
            line_func, x_interval, y_interval = self.line_metadata[self.lfi_index]
            return line_func(iteration, x_interval, y_interval)

        def cosine_func(self, iter_x, x_iter_interval, y_lr_interval):
            theta = self.linear_func(iter_x,
                                     x_interval=x_iter_interval,
                                     y_interval=[0, np.pi])
            x = np.cos(theta)
            y = self.linear_func(x,
                                 x_interval=[1, -1],
                                 y_interval=y_lr_interval)
            return y

        def linear_func(self, iter_x, x_interval, y_interval):
            m = abs((y_interval[1] - y_interval[0]) / (x_interval[1] - x_interval[0]))
            y = m * (iter_x - x_interval[0]) + y_interval[0]
            return y



class TrainMetricCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.y_true_tensor = None
        self.y_pred_tensor = None
        self.EPS = np.float64(1e-15)
        self.first_call_of_batch = True

    def on_epoch_begin(self, epoch, logs=None):
        self.y_true_tensor = None
        self.y_pred_tensor = None

    def on_batch_begin(self, batch, logs=None):
        self.first_call_of_batch = True

    def accumulate(self, batch_y_true, batch_y_pred):
        if self.first_call_of_batch:
            if self.y_true_tensor is None and self.y_pred_tensor is None:
                self.y_true_tensor = batch_y_true
                self.y_pred_tensor = batch_y_pred
            else:
                self.y_true_tensor = tf.concatenate([self.y_true_tensor, batch_y_true], axis=-1)
                self.y_pred_tensor = tf.concatenate([self.y_pred_tensor, batch_y_pred], axis=-1)
            self.first_call_of_batch = False

        return self.y_true_tensor, self.y_pred_tensor

    def rec(self, batch_y_true, batch_y_pred):
        y_true, y_pred = self.accumulate(batch_y_true, batch_y_pred)
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def pre(self, batch_y_true, batch_y_pred):
        y_true, y_pred = self.accumulate(batch_y_true, batch_y_pred)
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1s(self, batch_y_true, batch_y_pred):
        precision = self.pre(batch_y_true, batch_y_pred)
        recall = self.rec(batch_y_true, batch_y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def acc(self, batch_y_true, batch_y_pred):
        y_true, y_pred = self.accumulate(batch_y_true, batch_y_pred)
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        true_negatives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        n_samples = tf.keras.backend.sum(tf.keras.backend.clip(y_true, 1, 1))
        return (true_positives + true_negatives) / n_samples

    def log64(self, batch_y_true, batch_y_pred):
        # -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        # max(eps, min(1 - eps, p))
        y_true, y_pred = self.accumulate(batch_y_true, batch_y_pred)
        y_true = tf.keras.backend.cast(y_true, dtype='float64')
        y_pred = tf.keras.backend.cast(y_pred, dtype='float64')
        y_pred = tf.keras.backend.maximum(self.EPS, tf.keras.backend.minimum(1 - self.EPS, y_pred))
        return tf.keras.backend.mean(-(y_true*tf.keras.backend.log(y_pred) +
                                    (1 - y_true)*tf.keras.backend.log(1 - y_pred)))