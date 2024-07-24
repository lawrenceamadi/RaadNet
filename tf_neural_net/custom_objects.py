##print('\nCustom Metrics Script Called\n')
import tensorflow as tf
import numpy as np
import sys
import os

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import log_loss


class Metric(tf.keras.callbacks.Callback):
    '''
        Keras Metrics
        Implements overloaded keras class and functions for computing metrics using sklearn
        Note:
            this is only suitable when dataset can all fit into memory at once.
            Not suitable for large datasets because entire data is re-read into
            memory in order to evaluate at the END of each EPOCH
    '''

    def on_train_begin(self, logs={}):
        # instance variable declarations
        self._val_precision = []
        self._val_recall = []
        self._val_f1 = []
        self._val_logloss = []
        self._f1_max = 0
        self._opt_epoch = 0


    def set_additional_params(self, params):
        # instance variable declarations
        self.data_X = params['dataSet'][0]
        self.data_y = params['dataSet'][1]
        self.modelDir = params['modelDir']
        self.displayMetrics = params['displayMertics']
        self.saveOptModel = params['saveOptModel']
        self.checkpoint = params['checkpoint']
        self.checkpointPerEpoch = params['checkpointPerEpoch']


    def sklearn_metrics(self):
        predict = np.asarray(self.model.predict(self.data_X, batch_size=1, verbose=1))
        plabel = np.where(predict > 0.5, 1, 0)
        target = self.data_y
        pre, rec, f1, occurences = \
            precision_recall_fscore_support(target, plabel, average='binary') # shape: (4)
        return pre, rec, f1


    def perzone_sklearn_metrics(self):
        # prediction shape = (numofzones, batch, 1)
        prediction = np.asarray(self.model.predict(self.data_X, batch_size=1, verbose=1),
                                dtype=np.float32)
        prediction = np.squeeze(prediction)
        predlabels = np.where(prediction > 0.5, 1, 0)
        zPr, zRc, zF1, zLg = [], [], [], []
        for z in range(17):
            target = self.data_y[:, z]
            rwpred = prediction[z]
            plabel = predlabels[z]
            print('\n\nZone:{}'.format(z))
            print('raw output: {}\n{}'.format(rwpred.dtype, rwpred))
            print('plabel: {}\t{}\n{}'.format(plabel.dtype, plabel.shape, plabel))
            print('target: {}\t{}\n{}'.format(target.dtype, target.shape, target))
            precision, recall, f1, occurences = \
                precision_recall_fscore_support(target, plabel, average='binary')
            loss = log_loss(target, rwpred)
            zPr.append(precision)
            zRc.append(recall)
            zF1.append(f1)
            zLg.append(loss)
        return zPr, zRc, zF1, zLg


    def on_epoch_end(self, epoch, logs={}):
        precision, recall, f1score, logloss = self.perzone_sklearn_metrics()
        self._val_precision.append(precision)
        self._val_recall.append(recall)
        self._val_f1.append(f1score)
        self._val_logloss.append(logloss)
        prMean = np.mean(precision)
        reMean = np.mean(recall)
        f1Mean = np.mean(f1score)
        lgMean = np.mean(logloss)

        # save model if optimal
        if self.saveOptModel and self._f1_max < f1Mean:
            self._f1_max = f1Mean
            self._opt_epoch = epoch + 1
            self.model.save(os.path.join(self.modelDir, 'optimal.model'))

        # checkpoint model every so often
        if self.checkpoint and epoch > 0:
            if epoch % self.checkpointPerEpoch == 0:
                self.model.save(os.path.join(self.modelDir, 'epoch_{}.model'.format(epoch + 1)))

        # display computed metrics
        if self.displayMetrics:
            print("\teval_loss: {}\teval_f1: {}\teval_precision: {}\teval_recall: {}".format(
                lgMean, f1Mean, prMean, reMean))
            print("\tmax_f1: {} at epoch: {}".format(self._f1_max, self._opt_epoch))

        return



class BatchWiseMetrics(object):
    # Metric Functions suitable for binary-class (n_classes = 1), large datasets because of per
    # batch computation. However, not recommended for monitoring overall performance of model during
    # training because per-batch values may be misleading to overall performance on entire dataset
    # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

    def __init__(self, n_classes=1):
        self.n_classes = n_classes

    def rec(self, y_true, y_pred):
        # per batch recall
        if self.n_classes == 2:
            y_true = y_true[:, 1]
            y_pred = y_pred[:, 1]
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def pre(self, y_true, y_pred):
        # per batch precision
        if self.n_classes == 2:
            y_true = y_true[:, 1]
            y_pred = y_pred[:, 1]
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(
                                tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    def f1s(self, y_true, y_pred):
        # per batch recall
        precision = self.pre(y_true, y_pred)
        recall = self.rec(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def log64(self, y_true, y_pred):
        # -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        if self.n_classes == 2:
            y_true = y_true[:, 1]
            y_pred = y_pred[:, 1]
        eps = np.float64(1e-15)
        y_true_64 = tf.keras.backend.cast(y_true, dtype='float64')
        y_pred_64 = tf.keras.backend.cast(y_pred, dtype='float64')
        # max(eps, min(1 - eps, p)). same as clip(p, eps, 1 - eps)
        y_pred_64 = tf.keras.backend.maximum(eps, tf.keras.backend.minimum(1. - eps, y_pred_64))
        return tf.keras.backend.mean(-(y_true_64*tf.keras.backend.log(y_pred_64) +
                                    (1 - y_true_64)*tf.keras.backend.log(1 - y_pred_64)))

    def log(self, y_true, y_pred):
        # yp = clip_ops.clip_by_value(y_pred, EPSILON, 1. - EPSILON)
        # -log P(yt|yp) = -(yt log(yp + eps) + (1 - yt) log(1 - yp + eps))
        if self.n_classes == 2:
            y_true = y_true[:, 1]
            y_pred = y_pred[:, 1]
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(),
                                       1. - tf.keras.backend.epsilon())
        logloss = -(y_true*tf.keras.backend.log(y_pred + tf.keras.backend.epsilon()) +
                  (1 - y_true)*tf.keras.backend.log(1 - y_pred + tf.keras.backend.epsilon()))
        return logloss

    def avglog(self, y_true, y_pred):
        return tf.keras.backend.mean(self.log(y_true, y_pred))



class CustomBCELoss(tf.keras.losses.Loss):
    def __init__(self, func=None, cfg=None, m_isets=None, logger=None, tb_writer=None, **kwargs):
        super(CustomBCELoss, self).__init__(**kwargs)
        self.tb_writer = tb_writer
        if func is not None:
            self.loss_func = eval('self.{}'.format(func))
            self.m_isets = m_isets
            self.y_thresh = np.float32(cfg.LOSS.EXTRA.Y_THRESHOLD)
            self.x_scale = np.float32(cfg.LOSS.EXTRA.X_SCALE)
            self.step_phase = cfg.LOSS.EXTRA.STEP_PHASE
            self.smooth_fr = cfg.LOSS.SMOOTH_LABELS
            self.epoch = -1
            n_classes = cfg.LOSS.NUM_CLASSES
            new_labels = np.asarray([0, 1]) * (1-self.smooth_fr) + self.smooth_fr/n_classes
            msg = '\nBCE Loss label smooth factor: {}\n\tsmoothed labels will be: {}\n' \
                .format(self.smooth_fr, new_labels)

            if logger is not None: logger.log_msg(msg)
            print(msg)

    def call(self, y_true, y_pred):
        return self.loss_func(y_true, y_pred)

    def selective_loss(self, y_true, y_pred):
        '''
        Selective multi-output binary crossentropy loss function for conditional back-propagation
        :param y_true: ground-truth tensor of shape=(batch, m_grpsets, 1)
        :param y_pred: corresponding predicted probability of shape=(batch, m_grpsets, 1)
        :return: sample-wise loss tensor of shape=(batch)
        '''
        bxm_bceloss = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=self.smooth_fr)

        step = self.tb_writer.batch_step
        #if step%self.step_phase==0: self.x_scale *= 10.0
        if step%self.step_phase==0: self.x_scale += self.x_scale
        #print("self.x_scale: {}".format(self.x_scale))
        assert (0<self.y_thresh<1.), "y_thresh: {}".format(self.y_thresh)
        assert (self.x_scale>=1.), "x_scale: {}".format(self.x_scale)

        bxm_ytrue = tf.cast(tf.squeeze(y_true, axis=-1), dtype='float32')
        zero_or_one = tf.logical_or(tf.equal(bxm_ytrue, 1.), tf.equal(bxm_ytrue, 0.))
        tf.Assert(tf.reduce_all(zero_or_one),
                  ['bxm_ytrue', tf.boolean_mask(bxm_ytrue, tf.logical_not(zero_or_one))])
        bxm_ypred = tf.squeeze(y_pred, axis=-1)
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_ypred, 0.),
                                               tf.less_equal(bxm_ypred, 1.))),
                  ['bxm_ypred', tf.reduce_min(bxm_ypred), tf.reduce_max(bxm_ypred)])

        # compute loss coefficient per image group-set per sample
        bxm_offset = bxm_ypred - self.y_thresh
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_offset, -self.y_thresh),
                                               tf.less_equal(bxm_offset, (1.-self.y_thresh)))),
                  ['bxm_offset', tf.reduce_min(bxm_offset), tf.reduce_max(bxm_offset)])

        bxm_scaled = bxm_offset * self.x_scale
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_scaled, -self.y_thresh*self.x_scale),
                                               tf.less_equal(bxm_scaled, (1.-self.y_thresh)*self.x_scale))),
                  ['bxm_scaled', tf.reduce_min(bxm_scaled), tf.reduce_max(bxm_scaled)])

        bxm_sigmoid = tf.sigmoid(bxm_scaled)
        #bxm_sigmoid = tf.clip_by_value(bxm_sigmoid, clip_value_min=0.0, clip_value_max=1.0)
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_sigmoid, 0.),
                                               tf.less_equal(bxm_sigmoid, 1.))),
                  ['bxm_sigmoid', tf.reduce_min(bxm_sigmoid), tf.reduce_max(bxm_sigmoid)])

        # make sure max iset prediction in threat-cases has a weight of 1
        # todo: method to make sure a single iset is set to 1. when likely threat prediction
        bc_max_pred = tf.reduce_max(bxm_ypred, axis=1, keepdims=True)
        bc_max_pred = tf.repeat(bc_max_pred, repeats=[self.m_isets], axis=1)
        bxm_iset_cf = tf.where(tf.equal(bxm_ypred, bc_max_pred), 1., bxm_sigmoid)

        bxm_threat_case = bxm_ytrue * bxm_iset_cf #bxm_sigmoid
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_threat_case, 0.),
                                               tf.less_equal(bxm_threat_case, 1.))),
                  ['bxm_threat_case', tf.reduce_min(bxm_threat_case), tf.reduce_max(bxm_threat_case)])

        bxm_benign_case = 1.0 - bxm_ytrue
        zero_or_one = tf.logical_or(tf.equal(bxm_benign_case, 1.), tf.equal(bxm_benign_case, 0.))
        tf.Assert(tf.reduce_all(zero_or_one),
                  ['bxm_benign_case', tf.boolean_mask(bxm_benign_case, tf.logical_not(zero_or_one))])

        bxm_coefficient = bxm_benign_case + bxm_threat_case
        tf.Assert(tf.reduce_all(tf.logical_and(tf.greater_equal(bxm_coefficient, 0.),
                                               tf.less_equal(bxm_coefficient, 1.))),
                  ['bxm_coefficient', tf.reduce_min(bxm_coefficient), tf.reduce_max(bxm_coefficient)])

        #todo: log coefficient of each class separately. index with 1/0
        bxm_loss = bxm_coefficient * bxm_bceloss
        b_avg_bceloss = tf.reduce_mean(bxm_loss, axis=-1)

        #if self.epoch<self.tb_writer.batch_epoch:
        #self.epoch = self.tb_writer.batch_epoch
        # epoch = self.tb_writer.batch_epoch
        # print("epoch: {}".format(epoch))
        # self.tb_writer.tblog_histogram('iset_ytrue', y_true, step=epoch)  # {0, 1}
        # self.tb_writer.tblog_histogram('iset_preds', y_pred, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_bce', bxm_bceloss, step=epoch)
        # self.tb_writer.tblog_scalar('iset_x_scale', tf.convert_to_tensor(self.x_scale), step=epoch)
        # self.tb_writer.tblog_histogram('iset_offset', bxm_offset, step=epoch)  # [-0.5, 0.5]
        # self.tb_writer.tblog_histogram('iset_scaled', bxm_scaled, step=epoch)  # [-5, 5]
        # self.tb_writer.tblog_histogram('iset_sigmoid', bxm_sigmoid, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_sigm_cf', bxm_iset_cf, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_threat', bxm_threat_case, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_benign', bxm_benign_case, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_coef', bxm_coefficient, step=epoch)  # [0, 1]
        # self.tb_writer.tblog_histogram('iset_coefbce', bxm_loss, step=epoch)

        return b_avg_bceloss

    def multi_bce_loss(self, y_true, y_pred):
        '''
        Computes binary crossentropy loss per image-group-set (ie. per output unit)
        :param y_true: ground-truth tensor of shape=(batch, m_grpsets, 1)
        :param y_pred: corresponding predicted probability of shape=(batch, m_grpsets, 1)
        :return: sample-wise loss tensor of shape=(batch)
        '''
        bxm_bceloss = tf.losses.binary_crossentropy(y_true, y_pred, label_smoothing=self.smooth_fr)
        b_avg_bceloss = tf.reduce_mean(bxm_bceloss, axis=-1)

        # step = self.tb_writer.batch_step
        # self.tb_writer.tblog_histogram('_iset_ytrue', y_true, step=step)  # {0, 1}
        # self.tb_writer.tblog_histogram('_iset_preds', y_pred, step=step)  # [0, 1]
        # self.tb_writer.tblog_histogram('_iset_bce', bxm_bceloss, step=step)

        return b_avg_bceloss




def get_optimizer(cfg):
    optimizer_str = cfg.TRAIN.OPTIMIZER.lower()
    if optimizer_str=='adam':
        return tf.keras.optimizers.Adam(lr=cfg.TRAIN.INIT_LR, amsgrad=cfg.TRAIN.AMSGRAD,
                                        beta_1=cfg.TRAIN.BETA_1, beta_2=cfg.TRAIN.BETA_2)
    return optimizer_str


def get_loss(cfg, logger):
    loss_str = cfg.LOSS.NET_OUTPUTS_LOSS[0].lower()
    if loss_str=='binary_crossentropy':
        n_classes = cfg.LOSS.NUM_CLASSES
        smooth_fr = cfg.LOSS.SMOOTH_LABELS
        new_labels = np.asarray([0, 1]) * (1-smooth_fr) + smooth_fr/n_classes
        msg = '\nBCE Loss label smooth factor: {}\n\tsmoothed labels will be: {}\n' \
            .format(smooth_fr, new_labels)
        logger.log_msg(msg), print(msg)
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=smooth_fr)
    return loss_str


def get_flops(model_h5_path, custom_layers):
    # https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path, custom_objects=custom_layers)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)
            print('opts:\n{}\n'.format(opts['output']))

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
            print('flops:\n{}\n'.format(flops))
            print('flops total float ops:\n{}\n'.format(flops.total_float_ops))

            return flops.total_float_ops


def metrics_from_cfm(cfM):
    true_negatives, false_positives = cfM[0][0], cfM[0][1]
    false_negatives, true_positives = cfM[1][0], cfM[1][1]

    predicted_positives = true_positives + false_positives
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0

    actual_positives = false_negatives + true_positives
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0

    pre_n_rec = precision + recall
    f1score = (2 * precision * recall) / pre_n_rec if pre_n_rec > 0 else 0.0

    accuracy = (true_negatives + true_positives) / np.sum(cfM)

    return accuracy, f1score, precision, recall