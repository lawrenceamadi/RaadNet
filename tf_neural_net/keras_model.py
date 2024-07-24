'''
    Model architecture of localized threat detection
'''
##print('\nGeneral Model Script Called\n')
import gc
import sys
import numpy as np
import tensorflow as tf


class ROIPooling(tf.keras.layers.Layer):
    """ Implements Region Of Interest Max Pooling
        for channel-last images and relative bounding box coordinates

        # Constructor parameters
            pooled_hgt, pooled_wdt (int) --
              specify hgt and wdt of layer outputs

        Shape of inputs
            [(batch_size, n_images, pooled_hgt, pooled_wdt, n_channels),
             (batch_size, n_images, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_hgt, pooled_wdt, n_channels)

    """

    def __init__(self, pooled_hgt, pooled_wdt, **kwargs):
        self.pooled_hgt = pooled_hgt
        self.pooled_wdt = pooled_wdt
        super(ROIPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_map_shape, rois_shape = input_shape
        self.FTR_MAP_HGT = int(feature_map_shape[2])
        self.FTR_MAP_WDT = int(feature_map_shape[3])
        #print('\n\n\n\n{}, {}, {}, {}\n\n\n\n'.
        #      format(feature_map_shape, rois_shape, self.FTR_MAP_HGT, self.FTR_MAP_WDT))

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert (feature_map_shape[0]==rois_shape[0])
        batch_size = feature_map_shape[0]
        n_imgs = rois_shape[1]
        n_rois = rois_shape[2]
        n_channels = feature_map_shape[4]
        return (batch_size, n_imgs, n_rois, self.pooled_hgt, self.pooled_wdt, n_channels)

    def call(self, x, **kwargs):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, n_images, pooled_hgt, pooled_wdt, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, n_images, num_rois, 4)
                        Each region of interest is defined by four relative
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, n_images, num_rois, pooled_hgt, pooled_wdt, n_channels)
        """

        def curried_pool_imgs_rois(x):
            return ROIPooling._pool_imgs_rois(x[0], x[1],
                                              self.FTR_MAP_HGT, self.FTR_MAP_WDT,
                                              self.pooled_hgt, self.pooled_wdt)

        pooled_areas = tf.map_fn(curried_pool_imgs_rois, x, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_imgs_rois(imgs_ftr_map, imgs_rois, ftrmap_hgt, ftrmap_wdt, pooled_hgt, pooled_wdt):
        """ Applies ROI pooling for multiple images; various ROIs per image
        """

        def curried_pool_rois(x_example):
            ftr_map, rois = x_example
            return ROIPooling._pool_rois(ftr_map, rois, ftrmap_hgt,
                                         ftrmap_wdt, pooled_hgt, pooled_wdt)

        pooled_areas = tf.map_fn(curried_pool_rois, [imgs_ftr_map, imgs_rois], dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_rois(ftr_map, rois, ftrmap_hgt, ftrmap_wdt, pooled_hgt, pooled_wdt):
        """ Applies ROI pooling for a single image and various ROIs
        """

        def curried_pool_roi(roi):
            return ROIPooling._pool_roi(ftr_map, roi, ftrmap_hgt,
                                        ftrmap_wdt, pooled_hgt, pooled_wdt)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(ftr_map, roi, ftrmap_hgt, ftrmap_wdt, pooled_hgt, pooled_wdt):
        """ Applies ROI pooling to a single image and a single region of interest
        """
        #print('\n\n\n\nftr_map:{}, roi:{}\n\n\n\n'.format(ftr_map.shape, roi.shape))
        # Compute the region of interest
        # todo: make class variables
        #ftr_map_hgt = int(ftr_map.shape[0])
        #ftr_map_wdt = int(ftr_map.shape[1])

        h_start = tf.cast(ftrmap_hgt * roi[0], 'int32')
        w_start = tf.cast(ftrmap_wdt * roi[1], 'int32')
        h_end = tf.cast(ftrmap_hgt * roi[2], 'int32')
        w_end = tf.cast(ftrmap_wdt * roi[3], 'int32')

        region = ftr_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_hgt = h_end - h_start
        region_wdt = w_end - w_start
        h_step = tf.cast(region_hgt / pooled_hgt, 'int32')
        w_step = tf.cast(region_wdt / pooled_wdt, 'int32')

        areas = [[(i * h_step,
                   j * w_step,
                   (i+1) * h_step if i+1<pooled_hgt else region_hgt,
                   (j+1) * w_step if j+1<pooled_wdt else region_wdt
                   )
                  for j in range(pooled_wdt)]
                 for i in range(pooled_hgt)]

        # take the maximum of each area and stack the result
        def pool_area(x):
            return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0, 1])

        pooled_ftrs = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_ftrs

    def get_config(self):
        config = super(ROIPooling, self).get_config()
        config.update({'pooled_hgt': self.pooled_hgt, 'pooled_wdt': self.pooled_wdt})
        return config


class Denoiser(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Denoiser, self).__init__(**kwargs)

    def build(self, input_shape):
        self.D = input_shape[1]
        self.H = input_shape[2]
        self.W = input_shape[3]
        self.C = input_shape[4]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input_tensor, **kwargs):
        dhw_x_chn = tf.keras.layers.Reshape((self.D * self.H * self.W, self.C))(input_tensor)

        dhw_x_dhw = tf.linalg.matmul(dhw_x_chn, dhw_x_chn, transpose_b=True)
        dhw_x_dhw = tf.keras.layers.Activation('softmax')(dhw_x_dhw)

        dhw_x_chn = tf.linalg.matmul(dhw_x_dhw, dhw_x_chn)
        return tf.keras.layers.Reshape((self.D, self.H, self.W, self.C))(dhw_x_chn)

    def get_config(self):
        config = super(Denoiser, self).get_config()
        return config


class Slicer(tf.keras.layers.Layer):
    def __init__(self, dim, sin, ein, squeeze=True, **kwargs):
        '''
        Custom function for slicing tensors.
        Necessary to encapsulate as a keras Layer in order to be called with TimeDistributed
        :param dim: axis to slice
        :param sin: start of slice index (inclusive)
        :param ein: end of slice index (exclusive)
        :param squeeze: whether or not to remove axis after slice, if 1
        :param kwargs: other arguments for keras Layer
        '''
        super(Slicer, self).__init__(**kwargs)
        assert (dim>=0), "dim:{} cannot be negative".format(dim)
        self.dim = dim
        self.sin = sin
        self.ein = ein
        self.squeeze = squeeze

    def build(self, input_shape):
        if self.sin<0: self.sin = input_shape[self.dim] + 1 + self.sin
        if self.ein<0: self.ein = input_shape[self.dim] + 1 + self.ein
        self.squeeze = self.squeeze and (self.ein - self.sin)==1

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        if self.squeeze and (self.ein - self.sin)==1:
            out_shape.pop(self.dim)
        else:
            out_shape[self.dim] = self.ein - self.sin
        return tuple(out_shape)

    def call(self, input_tensor, **kwargs):
        if self.dim==0:
            out_tensor = input_tensor[self.sin: self.ein]
        elif self.dim==1:
            out_tensor = input_tensor[:, self.sin: self.ein]
        elif self.dim==2:
            out_tensor = input_tensor[:, :, self.sin: self.ein]
        elif self.dim==3:
            out_tensor = input_tensor[:, :, :, self.sin: self.ein]
        elif self.dim==4:
            out_tensor = input_tensor[:, :, :, :, self.sin: self.ein]

        if self.squeeze:
            out_tensor = tf.keras.backend.squeeze(out_tensor, axis=self.dim)
        return out_tensor

    def get_config(self):
        config = super(Slicer, self).get_config()
        config.update({'dim':self.dim, 'sin':self.sin, 'ein':self.ein, 'squeeze':self.squeeze})
        return config


class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        # out_shape = list(input_shape[:self.axis])
        # out_shape.append(1)
        # for dim in input_shape[self.axis:]:
        #     out_shape.append(dim)
        # out_shape = tuple(out_shape)
        out_shape = input_shape[self.axis:] + (1,) + input_shape[:self.axis]
        return out_shape

    def call(self, input_tensor, **kwargs):
        return tf.keras.backend.expand_dims(input_tensor, axis=self.axis)

    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update({'axis': self.axis})
        return config


class CircularPad3D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CircularPad3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        out_shape = list(input_shape)
        out_shape[1] += 2
        return tuple(out_shape)

    def call(self, input_tensor, **kwargs):
        lft_slice = input_tensor[:, 0: 1]
        rgt_slice = input_tensor[:, -2: -1]
        out_tensor = tf.keras.layers.concatenate([rgt_slice, input_tensor, lft_slice], axis=1)
        return out_tensor

    def get_config(self):
        config = super(CircularPad3D, self).get_config()
        return config


def non_local_max_denoiser(in_tensor, dim, name):
    n_filters = tf.keras.backend.int_shape(in_tensor)[-1]
    conv_name = '{}_conv'.format(name)
    add_name = '{}_add'.format(name)
    non_local_max = Denoiser(name=name)(in_tensor)
    if dim==3:
        non_local_max = tf.keras.layers.Conv3D(n_filters, 1, name=conv_name)(non_local_max)
    elif dim==2:
        non_local_max = tf.keras.layers.Conv2D(n_filters, 1, name=conv_name)(non_local_max)
    in_tensor = tf.keras.layers.Add(name=add_name)([non_local_max, in_tensor])
    return in_tensor


def conv_lstm_block(sequence_input, wgt_reg, filters, kernel=3, pad='same', act='relu'):
    # Block 1. in: (?, f, h, w, c), out: (?, f, h/2, w/2, 128)
    block1 = tf.keras.layers.ConvLSTM2D(filters[0], kernel, activation=act, padding=pad,
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(wgt_reg),
                        return_sequences=True, name='conv_lstm_1')(sequence_input)
    block1 = tf.keras.layers.ConvLSTM2D(filters[1], kernel, activation=act, padding=pad,
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(wgt_reg),
                        return_sequences=True, name='conv_lstm_2')(block1)

    # Block 2. in: (?, f, h/2, w/2, 128), out: (?, h/4, w/4, 256)
    block2 = tf.keras.layers.ConvLSTM2D(filters[2], kernel, activation=act, padding=pad,
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(wgt_reg),
                        return_sequences=True, name='conv_lstm_3')(block1)
    block2 = tf.keras.layers.ConvLSTM2D(filters[3], kernel, activation=act, padding=pad,
                        kernel_initializer=tf.keras.initializers.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2(wgt_reg),
                        return_sequences=False, name='conv_lstm_4')(block2)
    #block2 = Dropout(0.2)(block2)
    return block2


def conv_2d(in_tensor, n_filters, conv_id, batch_norm, drop_rate, wgt_reg,
              kernel=(3, 3), pad='same', act='relu', init_kernel='he_normal'):

    if init_kernel=='he_normal':
        k_init = tf.keras.initializers.HeNormal()
    elif init_kernel=='ones':
        k_init = tf.keras.initializers.Ones()
    else: k_init = tf.keras.initializers.GlorotUniform()
    k_reg = tf.keras.regularizers.l2(wgt_reg) if wgt_reg>0 else None

    if batch_norm:
        conv = tf.keras.layers.Conv2D(n_filters, kernel, padding=pad, name=conv_id,
                                      activation=None, kernel_initializer=k_init,
                                      kernel_regularizer=k_reg)(in_tensor)
        #conv = tf.keras.layers.BatchNormalization(name='{}_bn'.format(conv_id))(conv)
        l_name = '{}_bn'.format(conv_id)
        conv = tf.keras.layers.experimental.SyncBatchNormalization(name=l_name)(conv)
        conv = tf.keras.layers.Activation(act, name='{}_actv'.format(conv_id))(conv)
    else:
        conv = tf.keras.layers.Conv2D(n_filters, kernel, padding=pad, name=conv_id,
                                      activation=act, kernel_initializer=k_init,
                                      kernel_regularizer=k_reg)(in_tensor)
    if drop_rate>0:
        layer_name = '{}_spatial_dp'.format(conv_id)
        conv = tf.keras.layers.SpatialDropout2D(drop_rate, name=layer_name)(conv)

    return conv


def conv_3d(in_tensor, n_filters, conv_id, batch_norm, drop_rate, wgt_reg,
            kernel=(3,3,3), pad='same', act='relu', init_kernel='he_normal'):

    if init_kernel=='he_normal':
        k_init = tf.keras.initializers.HeNormal()
    elif init_kernel=='ones':
        k_init = tf.keras.initializers.Ones()
    else: k_init = tf.keras.initializers.GlorotUniform()
    k_reg = tf.keras.regularizers.l2(wgt_reg) if wgt_reg>0 else None

    if batch_norm:
        conv = tf.keras.layers.Conv3D(n_filters, kernel, padding=pad, name=conv_id,
                                      activation=None, kernel_initializer=k_init,
                                      kernel_regularizer=k_reg)(in_tensor)
        #conv = tf.keras.layers.BatchNormalization(name='{}_bn'.format(conv_id))(conv)
        l_name = '{}_bn'.format(conv_id)
        conv = tf.keras.layers.experimental.SyncBatchNormalization(name=l_name)(conv)
        conv = tf.keras.layers.Activation(act, name='{}_actv'.format(conv_id))(conv)
    else:
        conv = tf.keras.layers.Conv3D(n_filters, kernel, padding=pad, name=conv_id,
                                      activation=act, kernel_initializer=k_init,
                                      kernel_regularizer=k_reg)(in_tensor)
    if drop_rate>0:
        layer_name = '{}_spatial_dp'.format(conv_id)
        conv = tf.keras.layers.SpatialDropout3D(drop_rate, name=layer_name)(conv)

    return conv


def glob_avg_pool(in_tensor, dim, name, *argv, **kwargs):
    if dim==3:
        out_tensor = tf.keras.layers.GlobalAveragePooling3D(name=name)(in_tensor)
    elif dim==2:
        out_tensor = tf.keras.layers.GlobalAveragePooling2D(name=name)(in_tensor)
    return out_tensor


def glob_max_pool(in_tensor, dim, name, *argv, **kwargs):
    if dim==3:
        out_tensor = tf.keras.layers.GlobalMaxPooling3D(name=name)(in_tensor)
    elif dim==2:
        out_tensor = tf.keras.layers.GlobalMaxPooling2D(name=name)(in_tensor)
    return out_tensor


def max_pool(in_tensor, dim, name, pool_size, stride, pad='same', flatten=False):
    if dim==3:
        out_tensor = tf.keras.layers.MaxPool3D(pool_size, stride, pad, name=name)(in_tensor)
    elif dim==2:
        out_tensor = tf.keras.layers.MaxPool2D(pool_size, stride, pad, name=name)(in_tensor)
    if flatten:
        out_tensor = tf.keras.layers.Flatten(name='{}_flat'.format(name))(out_tensor)
    return out_tensor


def fe_network(fe_net_id, image_shape, out_layer,
               is_trainable=False, shape_only=False, logger=None):
    '''
    Feature extraction base for each image
    :param image_shape: (h, w, c)
    :param is_trainable: boolean True/False
    :return:
    '''
    assert (is_trainable==False)
    param_args = {'input_shape': image_shape, 'weights': 'imagenet',
                  'include_top': False}#, 'layers': tf.keras.layers}

    if fe_net_id=='mobilenet_v2':
        # 'block_5_add': (?, 10, 10, 32)
        #base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(**param_args)
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                            input_shape=image_shape, weights='imagenet')
    elif fe_net_id=='xception':
        # 'block1_conv2_act': (?, 37, 37, 64)
        base_model = tf.keras.applications.xception.Xception(**param_args)
    elif fe_net_id=='inception_resnet_v2':
        # 'activation_2': (?, 37, 37, 64)
        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(**param_args)
    elif fe_net_id=='resnet152_v2':
        # 'conv2_block3_2_relu': (?, 10, 10, 64)
        base_model = tf.keras.applications.resnet_v2.ResNet152V2(**param_args)
    elif fe_net_id=='nasnet_mobile':
        # 'reduction_add4_stem_1': (?, ...)
        base_model = tf.keras.applications.nasnet.NASNetMobile(**param_args)
    elif fe_net_id=='densenet_121':
        # 'conv3_block12_1_relu': (?, ...)
        base_model = tf.keras.applications.densenet.DenseNet121(**param_args)

    # Extract features from an arbitrary intermediate layer
    arb_out_tensor = base_model.get_layer(out_layer).output
    t_shape = arb_out_tensor.shape[1:]  # shape excluding batch dimension
    if shape_only: return t_shape

    if not is_trainable:
        # Freeze training of feature extraction model
        for layer in base_model.layers:
            layer.trainable = is_trainable

    fe_model = tf.keras.Model(inputs=base_model.input, outputs=arb_out_tensor, name='fe_net')
    if logger is not None: fe_model.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return fe_model, t_shape


def fully_connected_v1(in_conv_ftrs, in_zcv_t, wgt_reg, drop_rate, fc_units,
                       fcb_activ, rcv_units, logit_units, prefix='fc_', merge_index=2):
    ## Convoluted features vector and rcv linear transformed vector
    ## are added together (in arrears at merge_index)
    for idx in range(1, len(fc_units)):
        assert (fc_units[idx-1]>=fc_units[idx])
    assert (51<=fc_units[merge_index]<=64)

    if in_zcv_t is not None:
        zcv_t = tf.keras.layers.Dense(rcv_units, name='{}zcv_trans'.format(prefix),
                                kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(in_zcv_t)
    dense = in_conv_ftrs
    for idx in range(len(fc_units)):
        if drop_rate>0 and idx%2==0: dense = tf.keras.layers.Dropout(drop_rate)(dense)
        dense = tf.keras.layers.Dense(fc_units[idx], name='{}layer_{}'.format(prefix, idx+1),
                                      kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(dense)
        if idx==merge_index and in_zcv_t is not None:
            dense = tf.keras.layers.Add(name='{}merge_fcv_zcv'.format(prefix))([dense, zcv_t])

    dense = tf.keras.layers.Dense(logit_units, name='{}logit'.format(prefix),
                                  kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(dense)
    if fcb_activ!='linear':
        dense = tf.keras.layers.Activation(fcb_activ, name='{}act_logit'.format(prefix))(dense)

    return dense


def fully_connected_v2(in_conv_ftrs, in_zcv_t, wgt_reg, drop_rate, fc_units,
                       fcb_activ, rcv_units, logit_units, prefix='fc_', merge_index=2):
    ## Convoluted features vector and rcv linear transformed vector
    ## are concatenated together (in advance at merge_index)
    for idx in range(1, len(fc_units)):
        assert (fc_units[idx-1]>=fc_units[idx])

    if in_zcv_t is not None:
        zcv_t = tf.keras.layers.Dense(rcv_units, name='{}zcv_trans'.format(prefix),
                                kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(in_zcv_t)
    dense = in_conv_ftrs
    for idx in range(len(fc_units)):
        if idx==merge_index and in_zcv_t is not None:
            layer_name = '{}merge_fcv_zcv'.format(prefix)
            dense = tf.keras.layers.Concatenate(name=layer_name)([dense, zcv_t])
        if drop_rate>0 and idx%2==0: dense = tf.keras.layers.Dropout(drop_rate)(dense)
        dense = tf.keras.layers.Dense(fc_units[idx], name='{}layer_{}'.format(prefix, idx+1),
                                  kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(dense)

    dense = tf.keras.layers.Dense(logit_units, name='{}logit'.format(prefix),
                                  kernel_regularizer=tf.keras.regularizers.l2(wgt_reg))(dense)
    if fcb_activ!='linear':
        dense = tf.keras.layers.Activation(fcb_activ, name='{}act_logit'.format(prefix))(dense)

    return dense


def fully_connected_v3(conv_ftrs_zcv_input, wgt_reg, drop_rate, fc_units, fcb_activ, rcv_units,
                       logit_units, entity_bounds, prefix='fc_', merge_index=2, logger=None):
    # Separate merged convolution block features and zcv tensors
    eoe_1, eoe_2 = entity_bounds
    in_conv_ftrs = Slicer(1, 0, eoe_1, name='conv_blk_ftrs')(conv_ftrs_zcv_input)
    in_zcv_t = Slicer(1, eoe_1, eoe_2, name='iset_glob_zcv')(conv_ftrs_zcv_input)

    logit = fully_connected_v2(in_conv_ftrs, in_zcv_t, wgt_reg, drop_rate, fc_units, fcb_activ,
                               rcv_units, logit_units, prefix=prefix, merge_index=merge_index)

    fcl_net = tf.keras.Model(inputs=conv_ftrs_zcv_input, outputs=logit, name='fc_blk_net')
    if logger is not None: fcl_net.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return fcl_net


def dbroi_block_v1(global_reg_t, zoi_bbox_coords, n_filters, wgt_reg, batch_norm,
                  drop_rate, block_id, conv_per_block, roi_hgt, roi_wdt, dim=3):
    # Dual Branch ROI block, with roi pooling
    assert (block_id>=1)
    assert (dim==3 or dim==2)
    conv_d = eval('conv_{}d'.format(dim))
    assert (np.all(conv_per_block>=1))
    conv_start_id = np.max(conv_per_block) * (block_id - 1) + 1


    # Global region residual convolution sub-block. Convolve on global region feature-maps
    global_reg_out = global_reg_t
    for i in range(conv_per_block[0]):
        l_name = 'reg_conv_{}'.format(conv_start_id + i)
        global_reg_out = conv_d(global_reg_out, n_filters, l_name,
                                batch_norm, drop_rate, wgt_reg)
    # Skip connection of global region residual convolution sub-block
    l_name = 'reg_skip_{}'.format(block_id)
    global_reg_out = tf.keras.layers.Add(name=l_name)([global_reg_out, global_reg_t])

    # Pool ROI on convoluted global region features
    l_name = 'pool_zoi_{}'.format(block_id)
    pooled_zoi = ROIPooling(roi_hgt, roi_wdt, name=l_name)([global_reg_out, zoi_bbox_coords])

    # Pooled RoI residual convolution sub-block. Convolve on zone's region of interest
    pooled_zoi_out = pooled_zoi
    for i in range(conv_per_block[1]):
        l_name = 'zoi_conv_{}'.format(conv_start_id + i)
        pooled_zoi_out = conv_d(pooled_zoi_out, n_filters, l_name,
                                batch_norm, drop_rate, wgt_reg)
    # Skip connection of pooled roi residual convolution sub-block
    l_name = 'zoi_skip_{}'.format(block_id)
    pooled_zoi_out = tf.keras.layers.Add(name=l_name)([pooled_zoi_out, pooled_zoi])

    return global_reg_out, pooled_zoi_out


def dcmr_block_v1(global_reg_t, masked_zoi_t, tran_zoi_masks, branch_filters,
                  wgt_reg, batch_norm, drop_rate, block_id, conv_per_block,
                  dim, join_reg_zoi, glob_tag='',  final_block=False):
    # Dual Chain with Multi-Region block, with element-wise Mask multiplication
    assert (block_id>=1)
    assert (dim==3 or dim==2)
    assert (join_reg_zoi in ['add','concat','no-join']), "Unknown join: {}".format(join_reg_zoi)
    conv_d = eval('conv_{}d'.format(dim))
    assert (np.all(conv_per_block>=1))
    conv_start_id = np.max(conv_per_block) * (block_id - 1) + 1
    reg_filters, zoi_filters = branch_filters

    # Global convolution sub-block. Convolve on global region features
    global_reg_out = global_reg_t
    for i in range(conv_per_block[0]):
        l_name = '{}reg_conv_{}'.format(glob_tag, conv_start_id+i)
        global_reg_out = conv_d(global_reg_out, reg_filters, l_name,
                                batch_norm, drop_rate, wgt_reg)
    # Skip connection of global convolution sub-block
    l_name = '{}reg_skip_{}'.format(glob_tag, block_id)
    global_reg_out = tf.keras.layers.Add(name=l_name)([global_reg_out, global_reg_t])

    # Element-wise multiplication of roi mask and convoluted global region feature-maps
    l_name = 'masked_reg_{}'.format(block_id)
    masked_reg_t = tf.keras.layers.Multiply(name=l_name)([global_reg_out, tran_zoi_masks])
    # Merge masked region feature-maps and masked roi (if given)
    if masked_zoi_t is not None: # None at the first block of network
        l_name = '{}_reg_zoi_{}'.format(join_reg_zoi, block_id)
        if join_reg_zoi=='add':
            merged_reg_zoi_t = tf.keras.layers.Add(name=l_name)([masked_reg_t, masked_zoi_t])
        elif join_reg_zoi=='concat':
            merged_reg_zoi_t = tf.keras.layers.Concatenate(name=l_name)([masked_reg_t, masked_zoi_t])
        elif join_reg_zoi=='no-join':
            merged_reg_zoi_t = masked_reg_t
    else: merged_reg_zoi_t = masked_reg_t

    # Masked RoI convolution sub-blocks. Convolve on merged, masked region on interest features
    masked_zoi_out = merged_reg_zoi_t
    for i in range(conv_per_block[1]):
        l_name = 'zoi_conv_{}'.format(conv_start_id + i)
        masked_zoi_out = conv_d(masked_zoi_out, zoi_filters, l_name,
                                batch_norm, drop_rate, wgt_reg)
    # Skip connection of merged, masked region of interest convolution sub-block
    l_name = 'zoi_skip_{}'.format(block_id)
    masked_zoi_out = tf.keras.layers.Add(name=l_name)([masked_zoi_out, merged_reg_zoi_t])

    if final_block: return masked_reg_t, masked_zoi_out
    return global_reg_out, masked_zoi_out


def dcmr_block_v2(global_reg_t, zoi_tn_list, tran_zoi_masks, branch_filters,
                  wgt_reg, batch_norm, drop_rate, block_id, conv_per_block,
                  dim, subnet_names, join_reg_zoi, glob_tag='glob_', final_block=False):
    # Dual Chain with Multi-Bodyzone-Subnetwork blocks, with element-wise Mask multiplication
    assert (block_id>=1)
    assert (dim==3 or dim==2)
    assert (join_reg_zoi in ['add','concat','no-join']), "Unknown join: {}".format(join_reg_zoi)
    conv_d = eval('conv_{}d'.format(dim))
    assert (np.all(conv_per_block>=1))
    conv_start_id = np.max(conv_per_block) * (block_id-1) + 1
    reg_filters, zoi_filters = branch_filters

    # Global convolution sub-block. Convolve on global region features
    global_reg_out = global_reg_t
    for i in range(conv_per_block[0]):
        l_name = '{}reg_conv_{}'.format(glob_tag, conv_start_id+i)
        global_reg_out = conv_d(global_reg_out, reg_filters, l_name,
                                batch_norm, drop_rate, wgt_reg)
    # Skip connection of global convolution sub-block
    l_name = '{}reg_skip_{}'.format(glob_tag, block_id)
    global_reg_out = tf.keras.layers.Add(name=l_name)([global_reg_out, global_reg_t])

    # Element-wise multiplication of roi mask and convoluted global region feature-maps
    l_name = 'masked_reg_{}'.format(block_id)
    masked_reg_t = tf.keras.layers.Multiply(name=l_name)([global_reg_out, tran_zoi_masks])
    # Merge masked region feature-maps and masked rois of each body-zone subnet (if given)
    merged_reg_zoi_tn_list = [None]*len(zoi_tn_list)
    for idx, masked_zoi_t in enumerate(zoi_tn_list):
        if masked_zoi_t is not None and join_reg_zoi in ['add','concat']:  # None at 1st block
            l_name = '{}_{}_reg_zoi_{}'.format(subnet_names[idx], join_reg_zoi, block_id)
            if join_reg_zoi=='add':
                merged_reg_zoi_tn_list[idx] = \
                    tf.keras.layers.Add(name=l_name)([masked_reg_t, masked_zoi_t])
            elif join_reg_zoi=='concat':
                merged_reg_zoi_tn_list[idx] = \
                    tf.keras.layers.Concatenate(name=l_name)([masked_reg_t, masked_zoi_t])
            elif join_reg_zoi=='no-join':
                merged_reg_zoi_tn_list[idx] = masked_reg_t
        else: merged_reg_zoi_tn_list[idx] = masked_reg_t

    # Masked RoI convolution sub-blocks. Convolve on merged, masked region on interest features
    masked_zoi_list_out = merged_reg_zoi_tn_list
    for idx, masked_zoi_t in enumerate(zoi_tn_list):
        for i in range(conv_per_block[1]):
            l_name = '{}_zoi_conv_{}'.format(subnet_names[idx], conv_start_id+i)
            masked_zoi_list_out[idx] = conv_d(masked_zoi_list_out[idx], zoi_filters, l_name,
                                              batch_norm, drop_rate, wgt_reg)
        # Skip connection of merged, masked region of interest convolution sub-block
        l_name = '{}_zoi_skip_{}'.format(subnet_names[idx], block_id)
        masked_zoi_list_out[idx] = \
            tf.keras.layers.Add(name=l_name)([masked_zoi_list_out[idx], merged_reg_zoi_tn_list[idx]])

    if final_block: return masked_reg_t, masked_zoi_list_out
    return global_reg_out, masked_zoi_list_out


def residual_conv_v2(ftr_ext_tensor, masks_tensor, wgt_reg, filters,
                       batch_norm, drop_rate, dim=3, denoise=False):
    # Denoiser for each residual block, after add. With output bit-wise multiplication with mask
    assert (dim==3 or dim==2)
    conv_d = eval('conv_{}d'.format(dim))

    # Residual Block 1. in: (?, D, H, W, C), out: (?, D, H, W, C)
    block1 = conv_d(ftr_ext_tensor, filters[0], 'conv_1', batch_norm, drop_rate, wgt_reg)
    block1 = conv_d(block1, filters[1], 'conv_2', batch_norm, drop_rate, wgt_reg)

    # Skip connection
    skip_1 = conv_d(ftr_ext_tensor, filters[2], 'scon_1', batch_norm, 0., 0., kernel=1)
    skip_1 = tf.keras.layers.Add( name='sadd_1')([block1, skip_1])
    if denoise: skip_1 = non_local_max_denoiser(skip_1, dim, 'nlmd_1')

    # Residual Block 2. in: (?, D, H, W, C), out: (?, D, H, W, C)
    block2 = conv_d(skip_1, filters[3], 'conv_3', batch_norm, drop_rate, wgt_reg)
    block2 = conv_d(block2, filters[4], 'conv_4', batch_norm, drop_rate, wgt_reg)

    # Skip connection
    skip_2 = tf.keras.layers.Add(name='sadd_2')([block2, skip_1])
    if denoise: skip_2 = non_local_max_denoiser(skip_2, dim, 'nlmd_2')

    # Residual Block 3. in: (?, D, H, W, C), out: (?, D, H, W, C)
    block3 = conv_d(skip_2, filters[5], 'conv_5', batch_norm, drop_rate, wgt_reg)
    block3 = conv_d(block3, filters[6], 'conv_6', batch_norm, drop_rate, wgt_reg)

    # Skip connection
    skip_3 = conv_d(skip_2, filters[7], 'scon_3', batch_norm, 0., 0., kernel=1)
    skip_3 = tf.keras.layers.Add(name='sadd_3')([block3, skip_3])
    if denoise: skip_3 = non_local_max_denoiser(skip_3, dim, 'nlmd_3')

    # Residual Block 4. in: (?, D, H, W, C), out: (?, D, H, W, C)
    block4 = conv_d(skip_3, filters[8], 'conv_7', batch_norm, drop_rate, wgt_reg)
    block4 = conv_d(block4, filters[9], 'conv_8', batch_norm, drop_rate, wgt_reg)

    # Skip connection
    skip_4 = tf.keras.layers.Add(name='sadd_4')([block4, skip_3])
    if denoise: skip_4 = non_local_max_denoiser(skip_4, dim, 'nlmd_4')

    # Bit wise multiple with mask. in (?, D, H, W, C), out (?, D, H, W, C)
    mask_1 = conv_d(masks_tensor, filters[9], 'cmsk_1', False, 0., 0., kernel=1)
    masked = tf.keras.layers.Multiply(name='bmul_1')([skip_4, mask_1])

    # Global Pooling: in: (?, D, H, W, C), out: (?, C)
    out_tensor = glob_avg_pool(masked, dim, 'ftrs_glob_pool')
    return out_tensor


def residual_conv_v3(ftr_ext_tensor, masks_tensor, wgt_reg, filters, batch_norm, drop_rate,
                     n_super_blocks, n_cpb, dim=3, pool_func='glob_avg', pool_k=None, stride=None,
                     denoise=False, dual_branch_output=False):
    # Attentive RoI-Masking convolution sub-network.
    # In lower ROI-Branch, output from residual block of previous stage
    # is ADDED to input to residual block in current stage
    # Transitional convolution layers are used in-between stages to increase feature-map size
    zoi_t_block = None
    reg_t_block = ftr_ext_tensor
    n_phases = len(filters)
    assert (dim==3 or dim==2)
    conv_d = eval('conv_{}d'.format(dim))
    assert (pool_func in ['glob_avg', 'glob_max', 'max'])
    pool_f = eval('{}_pool'.format(pool_func))
    tran_k = (1, 3, 3) if dim==3 else (3, 3)

    for phase_index in range(n_phases):
        phase_id = phase_index + 1
        n_filters = filters[phase_index]
        brn_filters = [n_filters, n_filters]

        if phase_index==0 or n_filters!=filters[phase_index-1]:
            # Mandatory convolution. Transform/increase to new channel size
            layer_name = 'reg_trans_conv_{}'.format(phase_id)
            reg_t_block = conv_d(reg_t_block, n_filters, layer_name,
                                 batch_norm, 0., 0., kernel=tran_k)
            if zoi_t_block is not None:
                layer_name = 'zoi_trans_conv_{}'.format(phase_id)
                zoi_t_block = conv_d(zoi_t_block, n_filters, layer_name,
                                     batch_norm, 0., 0., kernel=tran_k)
            # Non-linear transformation of region of interest mask
            layer_name = 'msk_trans_conv_{}'.format(phase_id)
            tran_zoi_masks = conv_d(masks_tensor, n_filters, layer_name, False, 0., 0.,
                                    init_kernel='ones', kernel=tran_k)

        sup_start_index = n_super_blocks * phase_index
        for sup_index in range(n_super_blocks):
            sup_id = sup_start_index + sup_index + 1
            is_last_block = (phase_index+1)==n_phases and (sup_index+1)==n_super_blocks
            # in: (?, P, H, W, C), out: (?, P, H, W, C)
            reg_t_block, zoi_t_block = \
                dcmr_block_v1(reg_t_block, zoi_t_block, tran_zoi_masks,
                              brn_filters, wgt_reg, batch_norm, drop_rate, sup_id, n_cpb,
                              dim, join_reg_zoi='add', final_block=is_last_block)
            if denoise:
                suffix = 'nlm_denoise_{}'.format(sup_id)
                reg_t_block = non_local_max_denoiser(reg_t_block, dim, 'reg_{}'.format(suffix))
                zoi_t_block = non_local_max_denoiser(zoi_t_block, dim, 'zoi_{}'.format(suffix))

    # Global Pooling: in: (?, D, H, W, C), out: (?, C)
    zoi_t_block = pool_f(zoi_t_block, dim, 'zoiftrs_pool', pool_k, stride, flatten=True)
    if dual_branch_output:
        reg_t_block = pool_f(reg_t_block, dim, 'regftrs_globpool', pool_k, stride, flatten=True)
        return zoi_t_block, reg_t_block
    return zoi_t_block


def residual_conv_v4(ftr_ext_tensor, masks_tensor, wgt_reg, filters, batch_norm, drop_rate,
                     n_super_blocks, n_cpb, dim=3, join_reg_zoi='add', concat_zois=False,
                     pool_func='glob_max', pool_k=(3,3,3), stride=(3,3,3),
                     denoise=False, dual_branch_output=False):
    # Attentive RoI-Masking convolution sub-network.
    # output of zoi blocks from all stages are CONCATENATED, pooled and passed to FCL
    # An initial convolution layer is used to increase 1st feature-map size if necessary
    zoi_t_block = None
    reg_t_block = ftr_ext_tensor
    n_phases = len(filters)
    assert (dim==3 or dim==2)
    conv_d = eval('conv_{}d'.format(dim))
    assert (pool_func in ['glob_avg', 'glob_max', 'max'])
    pool_f = eval('{}_pool'.format(pool_func))
    tran_k = (1, 3, 3) if dim==3 else (3, 3)
    zoi_t_block_s = list()

    for phase_index in range(n_phases):
        phase_id = phase_index + 1
        n_filters = filters[phase_index]
        #brn_filters = [filters[0], n_filters]
        brn_filters = [n_filters, n_filters]

        # if phase_index==0 and brn_filters[0]!=ftr_ext_tensor.shape[-1]:
        #     # start of initial phase but features is less than required
        #     reg_t_block = conv_d(reg_t_block, n_filters, 'init_reg_conv_0',
        #                          batch_norm, drop_rate, wgt_reg)

        #if phase_index==0 or brn_filters[0]!=tran_zoi_masks.shape[-1]:
        if phase_index==0 or n_filters!=filters[phase_index-1]:
            # start of initial phase but features is less than required
            layer_name = 'reg_trans_conv_{}'.format(phase_id)
            reg_t_block = conv_d(reg_t_block, n_filters, layer_name,
                                 batch_norm, drop_rate, wgt_reg)
            # Non-linear transformation of RoI mask to increase feature-map size
            layer_name = 'msk_trans_conv_{}'.format(phase_id)
            tran_zoi_masks = conv_d(masks_tensor, n_filters, layer_name, False, 0., 0.,#wgt_reg,
                                    init_kernel='ones', kernel=tran_k)

        sup_start_index = n_super_blocks * phase_index
        for sup_index in range(n_super_blocks):
            sup_id = sup_start_index + sup_index + 1
            is_last_block = (phase_index+1)==n_phases and (sup_index+1)==n_super_blocks
            # in: (?, P, H, W, C), out: (?, P, H, W, C)
            reg_t_block, zoi_t_block = \
                dcmr_block_v1(reg_t_block, zoi_t_block, tran_zoi_masks,
                              brn_filters, wgt_reg, batch_norm, drop_rate, sup_id, n_cpb,
                              dim, join_reg_zoi, final_block=is_last_block)
            if denoise:
                suffix = 'nlm_denoise_{}'.format(sup_id)
                reg_t_block = non_local_max_denoiser(reg_t_block, dim, 'reg_{}'.format(suffix))
                zoi_t_block = non_local_max_denoiser(zoi_t_block, dim, 'zoi_{}'.format(suffix))
            # append zoi_t_block
            if concat_zois: zoi_t_block_s.append(zoi_t_block)

    if concat_zois:
        zoi_t_block_s = tf.keras.layers.Concatenate(name='concat_zoiftrs')(zoi_t_block_s)
    else: zoi_t_block_s = zoi_t_block
    #print('\n\n\n\n{} ->\n\n\n\n'.format(zoi_t_block_s.shape))
    zoi_t_block_s = pool_f(zoi_t_block_s, dim, 'zoiftrs_pool', pool_k, stride, flatten=True)
    #print('\n\n\n\n-> {}\n\n\n\n'.format(zoi_t_block_s.shape))
    if dual_branch_output:
        reg_t_block = pool_f(zoi_t_block_s, dim, 'regftrs_pool', pool_k, stride, flatten=True)
        return zoi_t_block_s, reg_t_block
    return zoi_t_block_s


def residual_conv_v5(ftr_ext_tensor, zoibox_tensor, wgt_reg, filters, batch_norm, drop_rate,
                     roi_pool_dim, n_super_blocks, n_cpb, dim=3, pool_func='max',
                     pool_k=(3,3,3), stride=(3,2,2), denoise=False):
    # Attentive RoI-Pooling convolution sub-network
    assert (dim==3 or dim==2)
    n_phases = len(filters)
    conv_d = eval('conv_{}d'.format(dim))
    tran_k = (1, 3, 3) if dim==3 else (3, 3)
    assert (pool_func in ['glob_avg','glob_max','max'])
    pool = eval('{}_pool'.format(pool_func))
    roi_wdt, roi_hgt = roi_pool_dim
    zoi_t_block_s = list()

    # TODO: do not make mandatory
    # Mandatory convolution. Transform/increase to new channel size
    reg_t_block = conv_d(ftr_ext_tensor, filters[0], 'iftrs_conv_0',
                         batch_norm, 0., wgt_reg, kernel=tran_k)

    for phase_index in range(n_phases):
        n_filters = filters[phase_index]

        sup_start_index = n_super_blocks * phase_index
        for sup_index in range(n_super_blocks):
            sup_id = sup_start_index + sup_index + 1
            # in: (?, P, H, W, C), out: (?, P, H, W, C)
            reg_t_block, zoi_t_block = \
                dbroi_block_v1(reg_t_block, zoibox_tensor, n_filters, wgt_reg, batch_norm,
                               drop_rate, sup_id, n_cpb, roi_hgt, roi_wdt, dim)
            if denoise:
                suffix = 'nlm_denoise_{}'.format(sup_id)
                reg_t_block = non_local_max_denoiser(reg_t_block, dim, 'reg_{}'.format(suffix))
                zoi_t_block = non_local_max_denoiser(zoi_t_block, dim, 'zoi_{}'.format(suffix))
            # append zoi_t_block to list
            zoi_t_block_s.append(zoi_t_block)

    # Pooling: in: (?, P, R, H, W, C), out: (?, f*C), where f>=1
    zoi_t_block_s = tf.keras.layers.Concatenate(name='concat_zoiftrs')(zoi_t_block_s)
    zoi_t_block_s = Slicer(2, 0, 1, True, name='zoi_ftr_map')(zoi_t_block_s)
    zoi_t_block_s = pool(zoi_t_block_s, dim, 'zoiftrs_pool', pool_k, stride, flatten=True)
    return zoi_t_block_s


def residual_conv_v6(ftr_ext_tensor, masks_tensor, wgt_reg, filters, batch_norm, drop_rate,
                     n_super_blocks, n_cpb, subnet_names, dim=3, join_reg_zoi='add',
                     pool_btw_phases=False, pool_func='glob_avg', pool_k=None, stride=None,
                     denoise=False, dual_branch_output=False):
    # Attentive RoI-Masking convolution and per-body-zone sub-networks.
    # In lower ROI-Branch, output from residual block of previous stage
    # is ADDED to input to residual block in current stage
    # Transitional convolution layers are used in-between stages to increase feature-map size
    zoi_tn_list = [None]*len(subnet_names)
    reg_t_block = ftr_ext_tensor
    n_phases = len(filters)
    assert (dim==3 or dim==2)
    conv_d = eval('conv_{}d'.format(dim))
    assert (join_reg_zoi in ['add','concat'])
    assert (pool_func in ['glob_avg','glob_max','max'])
    pool_f = eval('{}_pool'.format(pool_func))
    tran_k = (1, 3, 3) if dim==3 else (3, 3)

    for phase_index in range(n_phases):
        phase_id = phase_index + 1
        n_filters = filters[phase_index]
        brn_filters = [n_filters, n_filters]

        if phase_index==0 or n_filters!=filters[phase_index-1]:
            # Must convolve to increase feature depth at each stage
            if phase_index!=0 or n_filters!=ftr_ext_tensor.shape[-1]:  # p->q
                if pool_btw_phases and phase_index>0:
                    l_name = 'glob_reg_trans_pool_{}'.format(phase_id)
                    reg_t_block = tf.keras.layers.MaxPool3D((1,2,2), name=l_name)(reg_t_block)
                l_name = 'glob_reg_trans_conv_{}'.format(phase_id)
                reg_t_block = conv_d(reg_t_block, n_filters, l_name,
                                     batch_norm, 0., 0., kernel=tran_k)
            for idx, zoi_t_block in enumerate(zoi_tn_list):
                if zoi_t_block is not None:
                    if pool_btw_phases and phase_index>0:
                        l_name = '{}_zoi_trans_pool_{}'.format(subnet_names[idx], phase_id)
                        zoi_t_block = tf.keras.layers.MaxPool3D((1,2,2), name=l_name)(zoi_t_block)
                    l_name = '{}_zoi_trans_conv_{}'.format(subnet_names[idx], phase_id)
                    zoi_tn_list[idx] = conv_d(zoi_t_block, n_filters, l_name,
                                              batch_norm, 0., 0., kernel=tran_k)
            # Non-linear transformation of region of interest mask
            if pool_btw_phases and phase_index>0:
                l_name = 'msk_trans_pool_{}'.format(phase_id)
                masks_tensor = tf.keras.layers.AvgPool3D((1,2,2), name=l_name)(masks_tensor)
            l_name = 'msk_trans_conv_{}'.format(phase_id)
            tran_zoi_masks = conv_d(masks_tensor, n_filters, l_name, False, 0., 0.,
                                    init_kernel='ones', kernel=tran_k)

        sup_start_index = n_super_blocks * phase_index
        for sup_index in range(n_super_blocks):
            sup_id = sup_start_index + sup_index + 1
            is_last_block = (phase_index+1)==n_phases and (sup_index+1)==n_super_blocks
            # in: (?, P, H, W, C), out: (?, P, H, W, C)
            reg_t_block, zoi_tn_list = \
                dcmr_block_v2(reg_t_block, zoi_tn_list, tran_zoi_masks, brn_filters,
                              wgt_reg, batch_norm, drop_rate, sup_id, n_cpb,
                              dim, subnet_names, join_reg_zoi, final_block=is_last_block)
            if denoise:
                suffix = 'nlm_denoise_{}'.format(sup_id)
                l_name = 'glob_reg_{}'.format(suffix)
                reg_t_block = non_local_max_denoiser(reg_t_block, dim, l_name)
                for idx, zoi_t_block in enumerate(zoi_tn_list):
                    l_name = '{}_zoi_{}'.format(subnet_names[idx], suffix)
                    zoi_tn_list[idx] = non_local_max_denoiser(zoi_t_block, dim, l_name)

    # Global Pooling: in: (?, D, H, W, C), out: (?, C)
    for idx, zoi_t_block in enumerate(zoi_tn_list):
        l_name = '{}_zoiftrs_pool'.format(subnet_names[idx])
        zoi_tn_list[idx] = pool_f(zoi_t_block, dim, l_name, pool_k, stride, flatten=True)
    if dual_branch_output:
        reg_t_block = pool_f(reg_t_block, dim, 'regftrs_globpool', pool_k, stride, flatten=True)
        return zoi_tn_list, reg_t_block
    return zoi_tn_list


def conv_fc_network_v1(cfg, merged_tensors_input, entity_bounds, version,
                       iftr_shape, mask_shape, dim=3, logger=None):
    '''This has a sub-network branch of fully-connected-layers for predicting body-part/zone
    '''
    assert (version in ['v3']), "Incompatible version: {}".format(version)
    eoe_1, eoe_2, eoe_3 = entity_bounds
    # Separate merged tensors into two and reshape as necessary
    ftr_extract_t = Slicer(1, 0, eoe_1, name='flat_ftrs')(merged_tensors_input) # 1st entity
    ftr_extract_t = tf.keras.layers.Reshape(iftr_shape, name='iset_ftrs')(ftr_extract_t)
    downsmp_masks = Slicer(1, eoe_1, eoe_2, name='flat_msks')(merged_tensors_input) # 2nd entity
    downsmp_masks = tf.keras.layers.Reshape(mask_shape, name='iset_msks')(downsmp_masks)
    regn_type_zcv = Slicer(1, eoe_2, eoe_3, name='flat_zcvs')(merged_tensors_input) # 3rd entity

    # Residual Convolution block
    wgt_reg = cfg.MODEL.WGT_REG_RESC
    drop_rate = cfg.MODEL.DROPOUT_SP
    filters = cfg.MODEL.EXTRA.RES_CONV_FILTERS
    batch_normalize = cfg.MODEL.BATCH_NORMALIZE
    enable_denoiser = cfg.MODEL.ENABLE_DENOISER
    if version=='v3':
        n_cpb = np.asarray(cfg.MODEL.EXTRA.N_CONV_PER_BLOCK)
        n_super_blocks = cfg.MODEL.EXTRA.N_SUPER_BLOCKS
        zoi_t, reg_t = residual_conv_v3(ftr_extract_t, downsmp_masks, wgt_reg, filters,
                                        batch_normalize, drop_rate, n_super_blocks, n_cpb,
                                        dim=dim, denoise=enable_denoiser, dual_branch_output=True)
    drop_rate = cfg.MODEL.DROPOUT_FC
    wgt_reg = cfg.MODEL.WGT_REG_DENSE

    # Threat Detector Fully Connected block
    fc_units = cfg.MODEL.EXTRA.FC_UNITS
    fcb_activ = cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[0]
    logit_units = cfg.MODEL.EXTRA.THREAT_LOGIT_UNITS
    threat_tens = fully_connected_v2(zoi_t, regn_type_zcv, wgt_reg, drop_rate, fc_units,
                                     fcb_activ, logit_units, prefix='threat_fc_', merge_index=0)

    # Body Zone Recognizer Fully Connected block
    fc_units = cfg.MODEL.EXTRA.FC_UNITS[0: -1]
    fcb_activ = cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[1]
    logit_units = cfg.MODEL.EXTRA.BDPART_LOGIT_UNITS
    bdpart_tens = fully_connected_v2(reg_t, None, wgt_reg, drop_rate, fc_units,
                                     fcb_activ, logit_units, prefix='bdpart_fc_')

    outputs = tf.keras.layers.Concatenate(axis=1, name='join_tensors')([threat_tens, bdpart_tens])
    conv_fc_net = tf.keras.Model(inputs=merged_tensors_input,
                                 outputs=outputs, name='conv_fc_net')
    if logger is not None: conv_fc_net.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return conv_fc_net


def conv_fc_network_v2(cfg, merged_tensors_input, entity_bounds, version,
                       iftr_shape, iroi_shape, dim=3, logger=None):
    eoe_1, eoe_2, eoe_3 = entity_bounds
    assert (version in ['v2','v3','v4','v5']), "Incompatible version: {}".format(version)

    # Separate merged tensors into two and reshape as necessary
    circular = cfg.MODEL.EXTRA.CIRCULAR_3D_PADDING
    ftr_extract_t = Slicer(1, 0, eoe_1, name='flat_ftrs')(merged_tensors_input) # 1st entity
    ftr_extract_t = tf.keras.layers.Reshape(iftr_shape, name='iset_ftrs')(ftr_extract_t)
    if circular: ftr_extract_t = CircularPad3D(name='circ_ftrs')(ftr_extract_t)
    if version in ['v2', 'v3', 'v4']: # 2nd entity
        downsmp_masks = Slicer(1, eoe_1, eoe_2, name='flat_msks')(merged_tensors_input)
        downsmp_masks = tf.keras.layers.Reshape(iroi_shape, name='iset_msks')(downsmp_masks)
        if circular: downsmp_masks = CircularPad3D(name='circ_msks')(downsmp_masks)
    elif version=='v5': # 2nd entity
        zoi_bboxes_t = Slicer(1, eoe_1, eoe_2, name='flat_zois')(merged_tensors_input)
        zoi_bboxes_t = tf.keras.layers.Reshape(iroi_shape, name='iset_zois')(zoi_bboxes_t)
        if circular: zoi_bboxes_t = CircularPad3D(name='circ_zois')(zoi_bboxes_t)
    regn_type_zcv = Slicer(1, eoe_2, eoe_3, name='flat_zcvs')(merged_tensors_input) # 3rd entity

    # all version parameters
    wgt_reg = cfg.MODEL.WGT_REG_RESC
    drop_rate = cfg.MODEL.DROPOUT_SP
    filters = cfg.MODEL.EXTRA.RES_CONV_FILTERS
    batch_normalize = cfg.MODEL.BATCH_NORMALIZE
    enable_denoiser = cfg.MODEL.ENABLE_DENOISER
    # versions>=v3
    n_cpb = np.asarray(cfg.MODEL.EXTRA.N_CONV_PER_BLOCK)
    n_super_blocks = cfg.MODEL.EXTRA.N_SUPER_BLOCKS
    pool_func = cfg.MODEL.EXTRA.POOL_FUNC
    pool_k = cfg.MODEL.EXTRA.MAX_POOL_SIZE
    stride = cfg.MODEL.EXTRA.MAX_POOL_STRIDE
    # version==v4, v6
    concat_zois = cfg.MODEL.EXTRA.CONCAT_ZOI_STG_BLOCKS
    join_reg_zoi = cfg.MODEL.EXTRA.JOIN_GLOBREG_N_PREVZOI
    # version==v5
    roi_dim = cfg.MODEL.ROI_POOL_DIM

    # Residual Convolution Sub-Network
    if version=='v2':
        conv_tensor = residual_conv_v2(ftr_extract_t, downsmp_masks, wgt_reg, filters,
                                       batch_normalize, drop_rate, dim, denoise=enable_denoiser)
    elif version=='v3':
        conv_tensor = residual_conv_v3(ftr_extract_t, downsmp_masks, wgt_reg, filters,
                                       batch_normalize, drop_rate, n_super_blocks, n_cpb,
                                       dim, pool_func, pool_k, stride, denoise=enable_denoiser)
    elif version=='v4':
        conv_tensor = residual_conv_v4(ftr_extract_t, downsmp_masks, wgt_reg, filters,
                                       batch_normalize, drop_rate, n_super_blocks, n_cpb,
                                       dim, join_reg_zoi, concat_zois,
                                       pool_func, pool_k, stride, denoise=enable_denoiser)
    elif version=='v5':
        conv_tensor = residual_conv_v5(ftr_extract_t, zoi_bboxes_t, wgt_reg, filters,
                                       batch_normalize, drop_rate, roi_dim, n_super_blocks, n_cpb,
                                       dim, pool_func, pool_k, stride, denoise=enable_denoiser)

    # Fully Connected block
    wgt_reg = cfg.MODEL.WGT_REG_DENSE
    fc_units = cfg.MODEL.EXTRA.FC_UNITS
    rcv_unit = cfg.MODEL.EXTRA.RCV_LT_UNITS
    rcv_indx = cfg.MODEL.EXTRA.RCV_FC_INDEX
    fcb_activ = cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[0]
    logit_unit = cfg.MODEL.EXTRA.THREAT_LOGIT_UNITS
    drop_rate = cfg.MODEL.DROPOUT_FC
    logit_tensor = fully_connected_v2(conv_tensor, regn_type_zcv, wgt_reg, drop_rate, fc_units,
                                      fcb_activ, rcv_unit, logit_unit, merge_index=rcv_indx)
    #print("\n\n\nin:{} -> out:{}\n\n\n\n".format(merged_tensors_input.shape, logit_tensor.shape))
    conv_fc_net = tf.keras.Model(inputs=merged_tensors_input,
                                 outputs=logit_tensor, name='conv_fc_net')
    if logger is not None: conv_fc_net.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return conv_fc_net


def conv_fc_network_v3(cfg, merged_tensors_input, entity_bounds, version,
                       iftr_shape, iroi_shape, subnet_names, dim=3, logger=None):
    eoe_1, eoe_2, eoe_3 = entity_bounds
    assert (version in ['v6','v7']), "Incompatible version: {}".format(version)

    # Separate merged tensors into two and reshape as necessary
    circular = cfg.MODEL.EXTRA.CIRCULAR_3D_PADDING
    ftr_extract_t = Slicer(1, 0, eoe_1, name='flat_ftrs')(merged_tensors_input) # 1st entity
    ftr_extract_t = tf.keras.layers.Reshape(iftr_shape, name='iset_ftrs')(ftr_extract_t)
    if circular: ftr_extract_t = CircularPad3D(name='circ_ftrs')(ftr_extract_t)
    downsmp_masks = Slicer(1, eoe_1, eoe_2, name='flat_msks')(merged_tensors_input) # 2nd entity
    downsmp_masks = tf.keras.layers.Reshape(iroi_shape, name='iset_msks')(downsmp_masks)
    if circular: downsmp_masks = CircularPad3D(name='circ_msks')(downsmp_masks)

    # all version parameters
    wgt_reg = cfg.MODEL.WGT_REG_RESC
    drop_rate = cfg.MODEL.DROPOUT_SP
    filters = cfg.MODEL.EXTRA.RES_CONV_FILTERS
    batch_normalize = cfg.MODEL.BATCH_NORMALIZE
    pool_btw_phases = cfg.MODEL.EXTRA.POOL_BTWN_PHASES
    enable_denoiser = cfg.MODEL.ENABLE_DENOISER
    # versions>=v3
    n_cpb = np.asarray(cfg.MODEL.EXTRA.N_CONV_PER_BLOCK)
    n_super_blocks = cfg.MODEL.EXTRA.N_SUPER_BLOCKS
    pool_func = cfg.MODEL.EXTRA.POOL_FUNC
    pool_k = cfg.MODEL.EXTRA.MAX_POOL_SIZE
    stride = cfg.MODEL.EXTRA.MAX_POOL_STRIDE
    # version==v4, v6, v7
    join_reg_zoi = cfg.MODEL.EXTRA.JOIN_GLOBREG_N_PREVZOI

    # Residual Convolution Sub-Network
    conv_tensor_list = \
        residual_conv_v6(ftr_extract_t, downsmp_masks, wgt_reg, filters, batch_normalize,
                         drop_rate, n_super_blocks, n_cpb, subnet_names, dim, join_reg_zoi,
                         pool_btw_phases, pool_func, pool_k, stride, denoise=enable_denoiser)
    # Fully Connected block
    wgt_reg = cfg.MODEL.WGT_REG_DENSE
    fc_units = cfg.MODEL.EXTRA.FC_UNITS
    rcv_unit = cfg.MODEL.EXTRA.RCV_LT_UNITS
    rcv_indx = cfg.MODEL.EXTRA.RCV_FC_INDEX
    fcb_activ = cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[0]
    logit_unit = cfg.MODEL.EXTRA.THREAT_LOGIT_UNITS
    drop_rate = cfg.MODEL.DROPOUT_FC
    n_subnets = len(subnet_names)
    logit_tensor_list = [None]*n_subnets

    if version=='v6':
        # Each subnet has its own fully-connected-block
        regn_type_zcv = Slicer(1, eoe_2, eoe_3, name='flat_zcvs')(merged_tensors_input) # 3rd entity
        for idx, conv_tensor in enumerate(conv_tensor_list):
            prefix = '{}_fc_'.format(subnet_names[idx])
            logit_tensor_list[idx] = \
                fully_connected_v2(conv_tensor, regn_type_zcv, wgt_reg, drop_rate, fc_units,
                                   fcb_activ, rcv_unit, logit_unit, prefix, merge_index=rcv_indx)
        # concatenate logit of each subnet and return
        outputs = tf.keras.layers.Concatenate(name='snet_logit_stack')(logit_tensor_list)
    elif version=='v7':
        # concatenate convolution features of each subnet and return
        # ftr_stack_shape = (n_subnets, filters[-1])
        # for idx, conv_tensor in enumerate(conv_tensor_list):
        #     l_name = '{}_conv_ftrs'.format(subnet_names[idx])
        #     conv_tensor_list[idx] = ExpandDims(axis=1, name=l_name)(conv_tensor)
        # # concatenate features of each subnet and return
        # outputs = tf.keras.layers.Concatenate(axis=1, name='snet_ftrs_stack')(conv_tensor_list)
        # outputs = tf.keras.layers.Reshape(ftr_stack_shape, name='conv_snet_ftrs_stack')(outputs)
        outputs = tf.keras.layers.Concatenate(name='snet_ftrs_stack')(conv_tensor_list)

    conv_fc_net = tf.keras.Model(inputs=merged_tensors_input, outputs=outputs, name='conv_fc_net')
    if logger is not None: conv_fc_net.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return conv_fc_net


def zone_grp_network_v2_td(cfg, n_images, reg_imgs_in, reg_rcvs_in,
                           fe_net, conv_fc_net, logger=None):
    # v2: (variant of v1) Fully connected layers are shared by all zone groups
    sequence = tf.keras.layers.TimeDistributed(fe_net, name='fe_td')(reg_imgs_in)

    # Group sequence into n sets of size m
    n_sets, set_size = sequence_grouping(n_images, cfg.MODEL.IMAGES_PER_SEQ_GRP)
    seq_set_ftrs_list, seq_set_zcvs_list = list(), list()
    for i in range(n_sets):
        sin, ein = i * set_size, (i+1) * set_size
        # slice and organize images for sequence set
        seq_set_ftrs = Slicer(1, sin, ein, name='ftr_slice_{}'.format(i+1))(sequence)
        layer_name = 'seq_set_ftrs_{}'.format(i+1)
        seq_set_ftrs = ExpandDims(axis=1, name=layer_name)(seq_set_ftrs)
        seq_set_ftrs_list.append(seq_set_ftrs)
        # slice and organize zone/region composite vectors for sequence set
        seq_set_zcvs = Slicer(1, sin, ein, name='zcv_slice_{}'.format(i+1))(reg_rcvs_in)
        layer_name = 'seq_set_zcvs_{}'.format(i+1)
        seq_set_zcvs = ExpandDims(axis=1, name=layer_name)(seq_set_zcvs)
        seq_set_zcvs_list.append(seq_set_zcvs)

    if n_sets>1:
        all_set_ftrs = tf.keras.layers.Concatenate(axis=1, name='join_ftrs_sets')(seq_set_ftrs_list)
        all_set_zcvs = tf.keras.layers.Concatenate(axis=1, name='join_zcvs_sets')(seq_set_zcvs_list)
    else:
        all_set_ftrs = seq_set_ftrs_list[0]
        all_set_zcvs = seq_set_zcvs_list[0]

    # Reshape and concatenate to a single tensor
    all_set_ftrs = tf.keras.layers.Reshape((n_sets, -1))(all_set_ftrs)
    all_set_zcvs = tf.keras.layers.Reshape((n_sets, -1))(all_set_zcvs)
    merged_seq_t = tf.keras.layers.Concatenate()([all_set_zcvs, all_set_ftrs])

    all_set_logits = tf.keras.layers.TimeDistributed(conv_fc_net, name='seq_set_td')(merged_seq_t)
    prefix = 'logit' if cfg.LOSS.NET_OUTPUTS_FCBLOCK_ACT[0]=='linear' else 'act_logit'

    max_logit = tf.keras.layers.GlobalMaxPooling1D(name='max_{}'.format(prefix))(all_set_logits)
    if cfg.LOSS.NET_OUTPUTS_ACT[0]!='linear':
        layer_name = 'act_max_{}'.format(prefix)
        layer_actv = cfg.LOSS.NET_OUTPUTS_ACT[0]
        max_logit = tf.keras.layers.Activation(layer_actv, name=layer_name)(max_logit)
    grp_network = tf.keras.Model(inputs=[reg_imgs_in, reg_rcvs_in],
                                 outputs=max_logit, name='region_net')
    if logger is not None: grp_network.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return grp_network


def zone_grp_network_v3_td(cfg, n_images, reg_imgs_in, zone_roi_in, reg_rcvs_in,
                           fe_net, conv_fc_net, ds_cfg, logger=None):
    '''This has a sub-network branch for threat detection and predicting body-part/zone
    '''
    sequence = tf.keras.layers.TimeDistributed(fe_net, name='fe_td')(reg_imgs_in)
    ds_masks = tf.keras.layers.AveragePooling3D(pool_size=(1, ds_cfg[0], ds_cfg[1]),
                                                strides=(1, ds_cfg[2], ds_cfg[3]))(zone_roi_in)
    # Group sequence into n sets of size m
    n_sets, set_size = sequence_grouping(n_images, cfg.MODEL.IMAGES_PER_SEQ_GRP)
    seq_set_ftrs_list, seq_set_rois_list, seq_set_zcvs_list = list(), list(), list()
    for i in range(n_sets):
        sin, ein = i * set_size, (i+1) * set_size
        # slice and organize images for sequence set
        seq_set_ftrs = Slicer(1, sin, ein, name='ftr_slice_{}'.format(i+1))(sequence)
        seq_set_ftrs = ExpandDims(axis=1, name='iset_ftrs_{}'.format(i+1))(seq_set_ftrs)
        seq_set_ftrs_list.append(seq_set_ftrs)
        # slice and organize masks for sequence set
        seq_set_rois = Slicer(1, sin, ein, name='roi_slice_{}'.format(i+1))(ds_masks)
        seq_set_rois = ExpandDims(axis=1, name='iset_rois_{}'.format(i+1))(seq_set_rois)
        seq_set_rois_list.append(seq_set_rois)
        # slice and organize zone/region composite vectors for sequence set
        seq_set_zcvs = Slicer(1, sin, ein, name='zcv_slice_{}'.format(i+1))(reg_rcvs_in)
        seq_set_zcvs = ExpandDims(axis=1, name='iset_zcvs_{}'.format(i+1))(seq_set_zcvs)
        seq_set_zcvs_list.append(seq_set_zcvs)

    if n_sets>1:
        all_set_ftrs = tf.keras.layers.Concatenate(axis=1, name='join_ftrs_sets')(seq_set_ftrs_list)
        all_set_rois = tf.keras.layers.Concatenate(axis=1, name='join_rois_sets')(seq_set_rois_list)
        all_set_zcvs = tf.keras.layers.Concatenate(axis=1, name='join_zcvs_sets')(seq_set_zcvs_list)
    else:
        all_set_ftrs = seq_set_ftrs_list[0]
        all_set_rois = seq_set_rois_list[0]
        all_set_zcvs = seq_set_zcvs_list[0]

    # Reshape and concatenate to a single tensor
    all_set_ftrs = tf.keras.layers.Reshape((n_sets, -1), name='ftrs_vec_sets')(all_set_ftrs)
    all_set_rois = tf.keras.layers.Reshape((n_sets, -1), name='rois_vec_sets')(all_set_rois)
    all_set_zcvs = tf.keras.layers.Reshape((n_sets, -1), name='rcvs_vec_sets')(all_set_zcvs)
    merged_seq_t = tf.keras.layers.Concatenate(
                    name='join_ftr_roi_rcv')([all_set_ftrs, all_set_rois, all_set_zcvs])
    joined_tensors = tf.keras.layers.TimeDistributed(conv_fc_net, name='seq_set_td')(merged_seq_t)

    # threat probability output (probability that a threat is detected in one of the images)
    all_set_threat_t = Slicer(2, 0, 1, False, name='threat_tensors')(joined_tensors)
    output1_id, output2_id = cfg.LOSS.NET_OUTPUTS_ID # ['t', 'p']
    l_name = output1_id if cfg.LOSS.NET_OUTPUTS_ACT[0]=='linear' else 'max_act_logit'
    t_prob = tf.keras.layers.GlobalMaxPooling1D(name=l_name)(all_set_threat_t)
    if cfg.LOSS.NET_OUTPUTS_ACT[0]!='linear':
        layer_actv = cfg.LOSS.NET_OUTPUTS_ACT[0]
        t_prob = tf.keras.layers.Activation(layer_actv, name=output1_id)(t_prob)

    # zone probability output (probability that images are of a given body zone)
    all_set_bdpart_t = Slicer(2, 1, -1, False, name='zone_tensors')(joined_tensors)
    all_set_bdpart_t_list = list()
    for i in range(n_sets):
        set_bdpart_t = Slicer(1, i, i+1, name='zp_slice_{}'.format(i+1))(all_set_bdpart_t)
        all_set_bdpart_t_list.append(set_bdpart_t)
    if n_sets>1:
        all_set_bdpart_t = tf.keras.layers.Add(name='merge_zp_slices')(all_set_bdpart_t_list)
    else: all_set_bdpart_t = all_set_bdpart_t_list[0]

    layer_actv = cfg.LOSS.NET_OUTPUTS_ACT[1]
    z_probs = tf.keras.layers.Activation(layer_actv, name=output2_id)(all_set_bdpart_t)

    grp_network = tf.keras.Model(inputs=[reg_imgs_in, zone_roi_in, reg_rcvs_in],
                                 outputs=[t_prob, z_probs], name='region_net')
    if logger is not None: grp_network.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return grp_network


def zone_grp_network_v4_td(cfg, n_images, reg_imgs_in, zone_roi_in, reg_rcvs_in,
                           fe_net, conv_fc_net, version, td_vec_size, logger=None):
    '''
    Network architecture designed for single and/or multi-output threat object detection
    :param cfg: experiment configurations
    :param n_images: number of images per network input example
    :param reg_imgs_in: network input tensor of cropped regions
    :param zone_roi_in: network input tensor of corresponding roi masks OR bounding-box coordinates
    :param reg_rcvs_in: network input tensor of corresponding region composite vector
    :param fe_net: feature extraction sub-network
    :param conv_fc_net: convolutional sub-network
    :param version: network architecture type or version (eg. v1, ..., v4)
    :param ds_cfg: roi masks down-sampling pool-size and stride configuration
    :param logger: message logger object
    :return: tf.keras Model that encapsulates the global neural network architecture
    '''
    assert (version in ['v2','v3','v4','v5']), "Incompatible version: {}".format(version)
    sequence = tf.keras.layers.TimeDistributed(fe_net, name='fe_td')(reg_imgs_in)
    # if version in ['v2', 'v3', 'v4']:
    #     encoded_roi = tf.keras.layers.AveragePooling3D(pool_size=(1, ds_cfg[0], ds_cfg[1]),
    #                                         strides=(1, ds_cfg[2], ds_cfg[3]))(zone_roi_in)
    # else: encoded_roi = zone_roi_in
    # encoded_roi = zone_roi_in

    # Group sequence into n sets of size m
    n_sets, set_size = sequence_grouping(n_images, cfg.MODEL.IMAGES_PER_SEQ_GRP)
    all_set_ftrs, all_set_rois, all_set_zcvs = list(), list(), list()
    for i in range(n_sets):
        sin, ein = i*set_size, (i+1)*set_size
        # slice and organize images for sequence set
        seq_set_ftrs = Slicer(1, sin, ein, name='ftr_slice_{}'.format(i+1))(sequence)
        seq_set_ftrs = ExpandDims(axis=1, name='iset_ftrs_{}'.format(i+1))(seq_set_ftrs)
        all_set_ftrs.append(seq_set_ftrs)
        # slice and organize masks for sequence set
        seq_set_rois = Slicer(1, sin, ein, name='roi_slice_{}'.format(i+1))(zone_roi_in)
        seq_set_rois = ExpandDims(axis=1, name='iset_rois_{}'.format(i+1))(seq_set_rois)
        all_set_rois.append(seq_set_rois)
        # slice and organize zone/region composite vectors for sequence set
        seq_set_zcvs = Slicer(1, sin, ein, name='zcv_slice_{}'.format(i+1))(reg_rcvs_in)
        seq_set_zcvs = ExpandDims(axis=1, name='iset_zcvs_{}'.format(i+1))(seq_set_zcvs)
        all_set_zcvs.append(seq_set_zcvs)

    if n_sets>1:
        all_set_ftrs = tf.keras.layers.Concatenate(axis=1, name='join_ftrs_sets')(all_set_ftrs)
        all_set_rois = tf.keras.layers.Concatenate(axis=1, name='join_rois_sets')(all_set_rois)
        all_set_zcvs = tf.keras.layers.Concatenate(axis=1, name='join_zcvs_sets')(all_set_zcvs)
    else:
        all_set_ftrs = all_set_ftrs[0]
        all_set_rois = all_set_rois[0]
        all_set_zcvs = all_set_zcvs[0]

    # Reshape and concatenate to a single tensor
    all_set_ftrs = tf.keras.layers.Reshape((n_sets, -1), name='ftrs_vec_sets')(all_set_ftrs)
    all_set_rois = tf.keras.layers.Reshape((n_sets, -1), name='rois_vec_sets')(all_set_rois)
    all_set_zcvs = tf.keras.layers.Reshape((n_sets, -1), name='rcvs_vec_sets')(all_set_zcvs)
    merged_seq_t = tf.keras.layers.Concatenate(name='join_ftrs_rois_rcvs'
                                               )([all_set_ftrs, all_set_rois, all_set_zcvs])

    multi_outputs =  len(cfg.LOSS.NET_OUTPUTS_ID)>1
    if multi_outputs:
        output1_id, output2_id = cfg.LOSS.NET_OUTPUTS_ID  # ['zt', 'st'])
    else:
        output1_id = cfg.LOSS.NET_OUTPUTS_ID[0]
        output2_id = 'pred_per_iset'

    assert (merged_seq_t.shape[-1]==td_vec_size), "Incompatible input shapes"
    all_set_pred = tf.keras.layers.TimeDistributed(conv_fc_net, name=output2_id)(merged_seq_t)
    max_set_pred = tf.keras.layers.GlobalMaxPooling1D(name=output1_id)(all_set_pred)
    if multi_outputs: net_outputs = [max_set_pred, all_set_pred]
    else: net_outputs = max_set_pred

    grp_network = tf.keras.Model(inputs=[reg_imgs_in, zone_roi_in, reg_rcvs_in],
                                 outputs=net_outputs, name='region_net')
    if logger is not None: grp_network.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return grp_network


def zone_grp_network_v5_td(cfg, n_images, reg_imgs_in, zone_roi_in, reg_rcvs_in, fe_net,
                           conv_fc_net, version, subnet_out_names, td_vec_size, logger=None):
    '''
    Network architecture designed for single and/or multi-output threat object detection
        With specialized convolution block subnetworks for each body zone
        and hence multi-output (one for each body zone)
    :param cfg: experiment configurations
    :param n_images: number of images per network input example
    :param reg_imgs_in: network input tensor of cropped regions
    :param zone_roi_in: network input tensor of corresponding roi masks OR bounding-box coordinates
    :param reg_rcvs_in: network input tensor of corresponding region composite vector
    :param fe_net: feature extraction sub-network
    :param conv_fc_net: convolutional sub-network
    :param version: network architecture type or version (eg. v1, ..., v6)
    :param subnet_out_names: the prefix tags of each sub-network (eg. RBp_zt)
    :param td_vec_size: size of TimeDistributed input tensor
    :param logger: message logger object
    :return: tf.keras Model that encapsulates the global neural network architecture
    '''
    assert (version in ['v6']), "Incompatible version: {}".format(version)
    n_subnets = len(subnet_out_names)
    sequence = tf.keras.layers.TimeDistributed(fe_net, name='fe_td')(reg_imgs_in)

    # Group sequence into n sets of size m
    n_sets, set_size = sequence_grouping(n_images, cfg.MODEL.IMAGES_PER_SEQ_GRP)
    all_set_ftrs, all_set_rois, all_set_zcvs = [None]*n_sets, [None]*n_sets, [None]*n_sets
    for i in range(n_sets):
        sin, ein = i * set_size, (i+1) * set_size
        # slice and organize images for sequence set
        seq_set_ftrs = Slicer(1, sin, ein, name='ftr_slice_{}'.format(i+1))(sequence)
        seq_set_ftrs = ExpandDims(axis=1, name='iset_ftrs_{}'.format(i+1))(seq_set_ftrs)
        all_set_ftrs[i] = seq_set_ftrs
        # slice and organize masks for sequence set
        seq_set_rois = Slicer(1, sin, ein, name='roi_slice_{}'.format(i+1))(zone_roi_in)
        seq_set_rois = ExpandDims(axis=1, name='iset_rois_{}'.format(i+1))(seq_set_rois)
        all_set_rois[i] = seq_set_rois
        # slice and organize zone/region composite vectors for sequence set
        seq_set_zcvs = Slicer(1, sin, ein, name='zcv_slice_{}'.format(i+1))(reg_rcvs_in)
        seq_set_zcvs = ExpandDims(axis=1, name='iset_zcvs_{}'.format(i+1))(seq_set_zcvs)
        all_set_zcvs[i] = seq_set_zcvs

    if n_sets>1:
        all_set_ftrs = tf.keras.layers.Concatenate(axis=1, name='join_ftrs_sets')(all_set_ftrs)
        all_set_rois = tf.keras.layers.Concatenate(axis=1, name='join_rois_sets')(all_set_rois)
        all_set_zcvs = tf.keras.layers.Concatenate(axis=1, name='join_zcvs_sets')(all_set_zcvs)
    else:
        all_set_ftrs = all_set_ftrs[0]
        all_set_rois = all_set_rois[0]
        all_set_zcvs = all_set_zcvs[0]

    # Reshape and concatenate to a single tensor
    all_set_ftrs = tf.keras.layers.Reshape((n_sets, -1), name='ftrs_vec_sets')(all_set_ftrs)
    all_set_rois = tf.keras.layers.Reshape((n_sets, -1), name='rois_vec_sets')(all_set_rois)
    all_set_zcvs = tf.keras.layers.Reshape((n_sets, -1), name='rcvs_vec_sets')(all_set_zcvs)
    merged_seq_t = tf.keras.layers.Concatenate(name='join_ftrs_rois_rcvs'
                                               )([all_set_ftrs, all_set_rois, all_set_zcvs])
    l_name = 'per_iset_snet_pred'
    assert (merged_seq_t.shape[-1]==td_vec_size), "Incompatible input shapes"
    iset_snet_preds = tf.keras.layers.TimeDistributed(conv_fc_net, name=l_name)(merged_seq_t)

    net_outputs = [None]*n_subnets
    for idx, output_id in enumerate(subnet_out_names):
        l_name = '{}_isets'.format(output_id)
        all_set_pred = Slicer(2, idx, idx+1, squeeze=False, name=l_name)(iset_snet_preds)
        net_outputs[idx] = tf.keras.layers.GlobalMaxPooling1D(name=output_id)(all_set_pred)

    grp_network = tf.keras.Model(inputs=[reg_imgs_in, zone_roi_in, reg_rcvs_in],
                                 outputs=net_outputs, name='region_net')
    if logger is not None: grp_network.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return grp_network


def zone_grp_network_v6_td(cfg, n_images, reg_imgs_in, zone_roi_in, reg_rcvs_in,
                           fe_net, conv_net, fcl_net, version, subnet_out_names,
                           conv_td_vec_size, fcl_td_vec_size, rcvs_size, logger=None):
    '''
    Network architecture designed for single and/or multi-output threat object detection
        With specialized convolution block subnetworks for each body zone
        and a single fully-connected-layers block for shared by all subnetworks
    :param cfg: experiment configurations
    :param n_images: number of images per network input example
    :param reg_imgs_in: network input tensor of cropped regions
    :param zone_roi_in: network input tensor of corresponding roi masks OR bounding-box coordinates
    :param reg_rcvs_in: network input tensor of corresponding region composite vector
    :param fe_net: feature extraction sub-network
    :param conv_net: residual convolutional sub-network
    :param fcl_net: fully-connected-layers sub-network
    :param version: network architecture type or version (eg. v1, ..., v6)
    :param subnet_out_names: the prefix tags of each sub-network (eg. RBp_zt)
    :param td_vec_size: size of TimeDistributed input tensor
    :param logger: message logger object
    :return: tf.keras Model that encapsulates the global neural network architecture
    '''
    assert (version in ['v7']), "Incompatible version: {}".format(version)
    n_subnets = len(subnet_out_names)
    sequence = tf.keras.layers.TimeDistributed(fe_net, name='fe_td')(reg_imgs_in)

    # Group sequence into n sets of size m
    n_sets, set_size = sequence_grouping(n_images, cfg.MODEL.IMAGES_PER_SEQ_GRP)
    all_set_ftrs, all_set_rois, iset_snet_zcvs = [None]*n_sets, [None]*n_sets, [None]*n_sets
    for i in range(n_sets):
        sin, ein = i * set_size, (i+1) * set_size
        # slice and organize images for sequence set
        seq_set_ftrs = Slicer(1, sin, ein, name='ftr_slice_{}'.format(i+1))(sequence)
        seq_set_ftrs = ExpandDims(axis=1, name='iset_ftrs_{}'.format(i+1))(seq_set_ftrs)
        all_set_ftrs[i] = seq_set_ftrs
        # slice and organize masks for sequence set
        seq_set_rois = Slicer(1, sin, ein, name='roi_slice_{}'.format(i+1))(zone_roi_in)
        seq_set_rois = ExpandDims(axis=1, name='iset_rois_{}'.format(i+1))(seq_set_rois)
        all_set_rois[i] = seq_set_rois
        # slice and organize zone/region composite vectors for sequence set
        seq_set_zcvs = Slicer(1, sin, ein, name='zcv_slice_{}'.format(i+1))(reg_rcvs_in)
        l1_name, l2_name = 'iset_zcvs_{}'.format(i+1), 'iset_zcvs_{}_repv'.format(i+1)
        seq_set_zcvs = tf.keras.layers.Reshape((rcvs_size,), name=l1_name)(seq_set_zcvs)
        seq_set_zcvs = tf.keras.layers.RepeatVector(n_subnets, name=l2_name)(seq_set_zcvs) #***todo!
        seq_set_zcvs = ExpandDims(axis=1, name='iset_snet_zcvs_{}'.format(i+1))(seq_set_zcvs)
        iset_snet_zcvs[i] = seq_set_zcvs

    if n_sets>1:
        all_set_ftrs = tf.keras.layers.Concatenate(axis=-1, name='join_ftrs_sets')(all_set_ftrs)
        all_set_rois = tf.keras.layers.Concatenate(axis=-1, name='join_rois_sets')(all_set_rois)
        iset_snet_zcvs = tf.keras.layers.Concatenate(axis=-1, name='join_zcvs_sets')(iset_snet_zcvs)
    else:
        all_set_ftrs = all_set_ftrs[0]
        all_set_rois = all_set_rois[0]
        iset_snet_zcvs = iset_snet_zcvs[0]

    # Reshape and concatenate to a single tensor
    all_set_ftrs = tf.keras.layers.Reshape((n_sets, -1), name='ftrs_vec_sets')(all_set_ftrs)
    all_set_rois = tf.keras.layers.Reshape((n_sets, -1), name='rois_vec_sets')(all_set_rois)
    merged_seq_t = \
        tf.keras.layers.Concatenate(name='join_ftrs_rois')([all_set_ftrs, all_set_rois])
    l_name = 'per_iset_snet_ftrs'
    assert (merged_seq_t.shape[-1]==conv_td_vec_size), "Incompatible input shapes"
    iset_snet_ftrs = tf.keras.layers.TimeDistributed(conv_net, name=l_name)(merged_seq_t)

    ftr_stack_shape = (n_sets*n_subnets, cfg.MODEL.EXTRA.RES_CONV_FILTERS[-1])
    zcv_stack_shape = (n_sets*n_subnets, rcvs_size)
    iset_snet_ftrs = \
        tf.keras.layers.Reshape(ftr_stack_shape, name='iset_snet_ftrs_stack')(iset_snet_ftrs)
    iset_snet_zcvs = \
        tf.keras.layers.Reshape(zcv_stack_shape, name='iset_snet_zcvs_stack')(iset_snet_zcvs)
    merged_seq_t = \
        tf.keras.layers.Concatenate(name='join_ftrs_rcvs')([iset_snet_ftrs, iset_snet_zcvs])
    l_name = 'per_iset_snet_pred_stack'
    assert (merged_seq_t.shape[-1]==fcl_td_vec_size), "Incompatible input shapes"
    iset_snet_preds = tf.keras.layers.TimeDistributed(fcl_net, name=l_name)(merged_seq_t)
    iset_snet_preds = \
        tf.keras.layers.Reshape((n_sets, n_subnets), name='per_iset_snet_pred')(iset_snet_preds)

    net_outputs = [None]*n_subnets
    for idx, output_id in enumerate(subnet_out_names):
        l_name = '{}_isets'.format(output_id)
        all_set_pred = Slicer(2, idx, idx+1, squeeze=False, name=l_name)(iset_snet_preds)
        net_outputs[idx] = tf.keras.layers.GlobalMaxPooling1D(name=output_id)(all_set_pred)

    grp_network = tf.keras.Model(inputs=[reg_imgs_in, zone_roi_in, reg_rcvs_in],
                                 outputs=net_outputs, name='region_net')
    if logger is not None: grp_network.summary(line_length=128, print_fn=logger.log_msg)
    gc.collect()
    return grp_network


def sequence_grouping(n_images, set_size):
    if n_images%set_size==0:
        return n_images//set_size, set_size
    else:
        print('Logical Error!, sequence group size is not a factor of sequence size')
        print('\tIn keras_model.py, n_images: {}, grp_size: {}'.format(n_images, set_size))
        sys.exit()