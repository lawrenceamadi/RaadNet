'''
Read and pre-process scan images by segmenting body zones and organizing them
'''
##print('\nData Preprocess Called\n')
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from sklearn.preprocessing import minmax_scale

sys.path.append('../')
from tf_neural_net.commons import cv_display_image
from pose_detection.grid_zoning_norm import frame_proir_zone_coord
from tf_neural_net.commons import BODY_ZONES_GROUP, BDGRP_ADJUSTMENT



def crop_bbprior_rois(scan_dir, scan_id, zone_name, ordered_fids_of_zone, kpts_df, fk_map,
                      fk_map_type, prw_crop_shape, n_ipz, bdgrp_crop_config, bdgrp_adjustment,
                      scaled_frm_dim, scale_dim_f, frm_corners, icef, bg_pixels, nci_dim,
                      poly_degree=4, rgb_img=True, ax_sp=None, display=False, db=0):
    '''
    Crop predefined fixed bounding-box priors of body zone in frame including extra
        (neighboring) pixels. Cropped region is referred to as preload-window.
        Also returns a region crop confidence of 1
    :param scan_dir: file system folder containing image files of scan/subject
    :param scan_id: the unique id of a specific scan/subject
    :param zone_name: a string/name identifying the zone to be cropped from scan images
    :param ordered_fids_of_zone: an ordered list of frame ids from which regions are cropped
    :param kpts_df: a DataFrame containing HPE keypoints of subject per frame per scan
    :param fk_map: dictionary indicating keypoints to use to crop a given zone from a given frame
    :param prw_crop_shape: 3-dim shape (hgt, wdt, channel) of the pre-loaded regions
    :param n_ipz: the number of unique images for the given zone
    :param bdgrp_crop_config: cropping configuration for segmenting body group using keypoints
    :param bdgrp_adjustment: configuration for adjusting position of estimated keypoint
    :param scaled_frm_dim: (wdt, hgt) dimension. The original frame is resized to this dimension
    :param scale_dim_f: the wdt & hgt scale factor (from original to scaled frame dimension)
    :param frm_corners: homogenous coordinate of frame corners top-lft->top-rgt->btm-rgt->btm-lft
    :param icef: image contrast enhancement factor
    :param nci_dim: scaled network-cropped-image (fixed sized image passed to network) dimension
    :param bg_pixels: filler background pixel values (for each channel)
    :param poly_degree: degree of polynomial to fit to spatial pixel histogram line
    :param rgb_img: if true, the read BGR image is converted to RGB before other transformations
    :param ax_sp: plt.subplots object for plotting spatial pixel intensity histogram
    :param display: whether or not to display cropped image
    :param db: maximum allowance on boundaries of computed roi bounding box coordinates
    :return: cropped region images, region crop confidence, roi bounding-box, prw top-left coord.
    '''
    frm_wdt, frm_hgt = scaled_frm_dim
    wdt_sf, hgt_sf = scale_dim_f
    resize_image = wdt_sf != 1 or hgt_sf != 1
    nci_wdt, nci_hgt = nci_dim

    prw_hgt, prw_wdt = prw_crop_shape[:2]
    assert (prw_wdt>=nci_wdt), 'prw_wdt:{}, nci_wdt:{}'.format(prw_wdt, nci_wdt)
    assert (prw_hgt>=nci_hgt), 'prw_hgt:{}, nci_hgt:{}'.format(prw_hgt, nci_hgt)
    x_nci_l, x_nci_r, y_nci_t, y_nci_b = bb_half_dims(nci_wdt, nci_hgt)  # nci corners
    x_prw_l, x_prw_r, y_prw_t, y_prw_b = bb_half_dims(prw_wdt, prw_hgt)  # prw corners

    zone_prw_imgs = np.empty((n_ipz, *prw_crop_shape), dtype=np.uint8)
    zone_seg_conf = np.ones((n_ipz), dtype=np.float16)
    roi_bb_coords = np.empty((n_ipz, 4, 2), dtype=np.int16) # top-left --> bottom-left
    nci_xy_toplft = np.empty((n_ipz, 1, 2), dtype=np.int16) # top-left corner of nci window
    assert (n_ipz==len(ordered_fids_of_zone)), '{} vs. {}'.format(n_ipz, len(ordered_fids_of_zone))

    for idx, fid in enumerate(ordered_fids_of_zone):
        # get dictionary of keypoints needed per valid zone in given frame
        zone_frm_kpts = fk_map[fid]
        assert (zone_frm_kpts is not None), 'zone_frm_kpts:{}'.format(zone_frm_kpts)

        # read frame image and crop region
        frm_img = read_image(os.path.join(scan_dir, '{}.png'.format(fid)), rgb=rgb_img, icef=icef)
        if resize_image:
            frm_img = cv.resize(frm_img, tuple(scaled_frm_dim), interpolation=cv.INTER_CUBIC)

        # get body zone region-of-interest from prior
        coord_prior = frame_proir_zone_coord(fid, zone_name)
        assert (coord_prior is not None), "{}-frm{}: {}".format(zone_name, fid, coord_prior)
        lft_x, top_y, roi_wdt, roi_hgt = coord_prior
        lft_x, roi_wdt = lft_x*wdt_sf, roi_wdt*wdt_sf
        top_y, roi_hgt = top_y*hgt_sf, roi_hgt*hgt_sf
        roi_coord_wrt_frm = np.asarray([[lft_x,         top_y        ],  # top-left
                                        [lft_x+roi_wdt, top_y        ],  # top-right
                                        [lft_x+roi_wdt, top_y+roi_hgt],  # bottom-right
                                        [lft_x,         top_y+roi_hgt]]) # bottom-left
        x, y = np.mean(roi_coord_wrt_frm, axis=0).astype(np.int32)
        roi_coord_wrt_frm = roi_coord_wrt_frm.astype(np.int32)
        # confirm x_coordinates of ALL roi vertices are not left-side-OOB or right-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,0]<0)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        assert (not np.all(roi_coord_wrt_frm[:,0]>frm_wdt)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        # confirm y_coordinates of ALL roi vertices are not top-side-OOB or bottom-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,1]<0)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        assert (not np.all(roi_coord_wrt_frm[:,1]>frm_hgt)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        roi_bb_coords[idx] = roi_coord_wrt_frm

        # crop preloaded-region-window (prw) image around roi
        ps_x, pe_x, ps_y, pe_y = bb_coords(x, y, x_prw_l, x_prw_r, y_prw_t, y_prw_b) # prw dims
        prw_top_left = [ps_x, ps_y]
        if ps_x<0 or ps_y<0 or pe_x>frm_wdt or pe_y>frm_hgt:
            frm_img, ps_x, pe_x, ps_y, pe_y = \
                pad_image_boundary(frm_img, ps_x, pe_x, ps_y, pe_y, bg_pixels, scaled_frm_dim)
        prw_img = frm_img[ps_y: pe_y, ps_x: pe_x]
        assert (prw_img.shape==prw_crop_shape), '{} vs. {}'.format(prw_img.shape, prw_crop_shape)
        zone_prw_imgs[idx] = prw_img

        # compute top-left corner coordinate of network-cropped-image (nci) within prw
        nci_tl = [x - x_nci_l, y - y_nci_t]
        nci_xy_toplft[idx, 0] = nci_tl
        assert (-db<=nci_tl[0]<=frm_wdt+db), "nci top-left x:{}, db:{}".format(nci_tl[0], db)
        assert (-db<=nci_tl[1]<=frm_hgt+db), "nci top-left y:{}, db:{}".format(nci_tl[1], db)

        if display:
            # display re-sized frame image with roi bbox
            frm_roi_img = frm_img.copy()
            top_left_frm_padding = [min(0, prw_top_left[0]), min(0, prw_top_left[1])]
            roi_coord_wrt_pad_frm = roi_coord_wrt_frm - top_left_frm_padding
            pts = roi_coord_wrt_pad_frm.reshape((-1, 1, 2))  # need vertices in (rows,1,2)
            cv.polylines(frm_roi_img, [pts], True, (150, 200, 0), 2)
            cv_display_image('image with roi bbox', frm_roi_img, 1)
            # display preloaded-region window image with roi and nci boundary centered on image
            prw_nci_img = prw_img.copy()
            pts = roi_coord_wrt_frm.reshape((-1, 1, 2))  # need vertices in (rows,1,2)
            pts -= prw_top_left  # roi coordinates relative to the prw image
            cv.polylines(prw_nci_img, [pts], True, (150, 200, 0), 1)  # relative roi
            x_ctr, y_ctr = np.asarray([[pe_x-ps_x], [pe_y-ps_y]]) // 2
            ns_x, ne_x, ns_y, ne_y = bb_coords(x_ctr, y_ctr, x_nci_l, x_nci_r, y_nci_t, y_nci_b)
            cv.rectangle(prw_nci_img, (ns_x, ns_y), (ne_x, ne_y), (0, 0, 0), 2)  # nci
            cv_display_image('prw cropped image', prw_nci_img, 0)

    return zone_prw_imgs, zone_seg_conf, roi_bb_coords, nci_xy_toplft


def crop_aligned_rois(scan_dir, scan_id, zone_name, ordered_fids_of_zone, kpts_df, fk_map,
                      fk_map_type, prw_crop_shape, n_ipz, bdgrp_crop_config, bdgrp_adjustment,
                      scaled_frm_dim, scale_dim_f, frm_corners, icef, bg_pixels, nci_dim,
                      poly_degree=4, rgb_img=True, ax_sp=None, display=False, db=0):
    '''
    Crop axis-aligned region of interests including extra (neighboring) pixels. Cropped region is
        referred to as preload-window. This is a dynamic cropping for mask-enabled training.
        Also computes the region crop confidence as the mean confidence of hpe keypoints
    :param scan_dir: file system folder containing image files of scan/subject
    :param scan_id: the unique id of a specific scan/subject
    :param zone_name: a string/name identifying the zone to be cropped from scan images
    :param ordered_fids_of_zone: an ordered list of frame ids from which regions are cropped
    :param kpts_df: a DataFrame containing HPE keypoints of subject per frame per scan
    :param fk_map: dictionary indicating keypoints to use to crop a given zone from a given frame
    :param prw_crop_shape: 3-dim shape (hgt, wdt, channel) of the pre-loaded regions
    :param n_ipz: the number of unique images for the given zone
    :param bdgrp_crop_config: cropping configuration for segmenting body group using keypoints
    :param bdgrp_adjustment: configuration for adjusting position of estimated keypoint
    :param scaled_frm_dim: (wdt, hgt) dimension. The original frame is resized to this dimension
    :param scale_dim_f: the wdt & hgt scale factor (from original to scaled frame dimension)
    :param frm_corners: homogenous coordinate of frame corners top-lft->top-rgt->btm-rgt->btm-lft
    :param icef: image contrast enhancement factor
    :param nci_dim: scaled network-cropped-image (fixed sized image passed to network) dimension
    :param bg_pixels: filler background pixel values (for each channel)
    :param poly_degree: degree of polynomial to fit to spatial pixel histogram line
    :param rgb_img: if true, the read BGR image is converted to RGB before other transformations
    :param ax_sp: plt.subplots object for plotting spatial pixel intensity histogram
    :param display: whether or not to display cropped image
    :param db: maximum allowance on boundaries of computed roi bounding box coordinates
    :return: cropped region images, region crop confidence, roi bounding-box, prw top-left coord.
    '''
    frm_wdt, frm_hgt = scaled_frm_dim
    wdt_sf, hgt_sf = scale_dim_f
    resize_image = wdt_sf != 1 or hgt_sf != 1

    grp_name = BODY_ZONES_GROUP[zone_name]
    prw_hgt, prw_wdt = prw_crop_shape[:2]
    region_meta = bdgrp_crop_config[grp_name]
    roi_wdt, roi_hgt = np.int32(region_meta[:2] * scale_dim_f)
    move_center_x = int(roi_wdt * region_meta[2])  # must be computed before padding
    move_center_y = int(roi_hgt * region_meta[3])  # must be computed before padding
    assert (prw_wdt>=roi_wdt), 'prw_wdt:{}, roi_wdt:{}'.format(prw_wdt, roi_wdt)
    assert (prw_hgt>=roi_hgt), 'prw_hgt:{}, roi_hgt:{}'.format(prw_hgt, roi_hgt)
    x_roi_l, x_roi_r, y_roi_t, y_roi_b = bb_half_dims(roi_wdt, roi_hgt) # roi dimensions
    x_prw_l, x_prw_r, y_prw_t, y_prw_b = bb_half_dims(prw_wdt, prw_hgt) # prw dimensions

    zone_images = np.empty((n_ipz, *prw_crop_shape), dtype=np.uint8)
    zone_seg_conf = np.empty((n_ipz), dtype=np.float16)
    roi_bb_coords = np.empty((n_ipz, 4), dtype=np.int16)
    assert (n_ipz==len(ordered_fids_of_zone)), '{} vs. {}'.format(n_ipz, len(ordered_fids_of_zone))
    zone_kpts_y = fk_map['y'][zone_name]

    for idx, fid in enumerate(ordered_fids_of_zone):
        # get dictionary of keypoints needed per valid zone in given frame
        frm_zone_kpts = fk_map[fid]
        zone_kpts_x = frm_zone_kpts.get(zone_name)

        # Get keypoints' coordinates and confidence available for scan's frame
        column_name = 'Frame{}'.format(fid)
        cell = kpts_df.loc[kpts_df['scanID'] == scan_id, column_name]
        frm_kpts_meta = eval(cell.values[0])  # eval() or ast.literal_eval()

        # get region center coordinate and meta data
        kpts_meta = keypoints_meta(zone_kpts_x, frm_kpts_meta)
        x, y, conf = np.mean(kpts_meta, axis=0)  # shape==(?, 3) --> shape==(3,)
        if zone_kpts_y != zone_kpts_x:
            y_kpts_meta = keypoints_meta(zone_kpts_y, frm_kpts_meta)
            __, y, ____ = np.mean(y_kpts_meta, axis=0)
        zone_seg_conf[idx] = conf
        x = int(x * wdt_sf + move_center_x)
        y = int(y * hgt_sf + move_center_y)

        # read frame image and crop region
        frm_img = read_image(os.path.join(scan_dir, '{}.png'.format(fid)), rgb=rgb_img, icef=icef)
        if resize_image:
            frm_img = cv.resize(frm_img, tuple(scaled_frm_dim), interpolation=cv.INTER_CUBIC)

        ps_x, pe_x, ps_y, pe_y = bb_coords(x, y, x_prw_l, x_prw_r, y_prw_t, y_prw_b) # prw dims
        if ps_x<0 or ps_y<0 or pe_x>frm_wdt or pe_y>frm_hgt:
            frm_img, ps_x, pe_x, ps_y, pe_y = \
                pad_image_boundary(frm_img, ps_x, pe_x, ps_y, pe_y, bg_pixels, scaled_frm_dim)
        reg_img = frm_img[ps_y: pe_y, ps_x: pe_x]
        assert (reg_img.shape==prw_crop_shape), '{} vs. {}'.format(reg_img.shape, prw_crop_shape)
        if display: cv_display_image('cropped image', reg_img, 1)
        zone_images[idx] = reg_img

        # Record original bounding-box coordinates of the region of interest.
        # Note, coordinates of roi_bb_coords can be OOB (ie. <0 or >frm_hgt/wdt)
        # Do not clip because pad_image_boundary grows image
        os_x, oe_x, os_y, oe_y = bb_coords(x, y, x_roi_l, x_roi_r, y_roi_t, y_roi_b) # roi dims
        assert (oe_x>os_x and oe_y>os_y), 'x: {} {}, y: {} {}'.format(oe_x, os_x, oe_y, os_y)
        assert (oe_x-os_x==roi_wdt and oe_y-os_y==roi_hgt), '{} {}'.format(oe_x-os_x, oe_y-os_y)
        roi_bb_coords[idx] = [os_x, oe_x, os_y, oe_y]

    return zone_images, zone_seg_conf, roi_bb_coords


def crop_oriented_rois(scan_dir, scan_id, zone_name, ordered_fids_of_zone, kpts_df, fk_map,
                       fk_map_type, prw_crop_shape, n_ipz, bdgrp_crop_config, bdgrp_adjustment,
                       scaled_frm_dim, scale_dim_f, frm_corners, icef, bg_pixels, nci_dim,
                       poly_degree=4, rgb_img=True, ax_sp=None, display=False, db=20):
    '''
    Crop oriented region of interests including extra (neighboring) pixels. Cropped region is
        referred to as preload-window. This (Version 2) is a dynamic cropping for mask-enabled
        training. Also computes the region crop confidence as the mean confidence of hpe keypoints
    :param scan_dir: file system folder containing image files of scan/subject
    :param scan_id: the unique id of a specific scan/subject
    :param zone_name: a string/name identifying the zone to be cropped from scan images
    :param ordered_fids_of_zone: an ordered list of frame ids from which regions are cropped
    :param kpts_df: a DataFrame containing HPE keypoints of subject per frame per scan
    :param fk_map: dictionary listing keypoints used to crop zone body part in each frame
    :param fk_map_type: dict of zone segm. type per frm (True:anchor, False:pillar, None:invisible)
    :param prw_crop_shape: 3-dim shape (hgt, wdt, channel) of the pre-loaded regions
    :param n_ipz: the number of unique images for the given zone
    :param bdgrp_crop_config: cropping configuration for segmenting body group using keypoints
    :param bdgrp_adjustment: configuration for adjusting position of estimated keypoint
    :param scaled_frm_dim: (wdt, hgt) dimension. The original frame is resized to this dimension
    :param scale_dim_f: the wdt & hgt scale factor (from original to scaled frame dimension)
    :param frm_corners: homogenous coordinate of frame corners top-lft->top-rgt->btm-rgt->btm-lft
    :param icef: image contrast enhancement factor
    :param bg_pixels: filler background pixel values (for each channel)
    :param nci_dim: scaled network-cropped-image (fixed sized image passed to network) dimension
    :param poly_degree: degree of polynomial to fit to spatial pixel histogram line
    :param rgb_img: if true, the read BGR image is converted to RGB before other transformations
    :param ax_sp: plt.subplots object for plotting spatial pixel intensity histogram
    :param display: whether or not to display cropped image
    :param db: maximum allowance on boundaries of computed roi bounding box coordinates
    :return: cropped region images, region crop confidence, roi bounding-box, prw top-left coord.
    '''
    frm_wdt, frm_hgt = scaled_frm_dim
    wdt_sf, hgt_sf = scale_dim_f
    resize_image = wdt_sf != 1 or hgt_sf != 1
    nci_wdt, nci_hgt = nci_dim

    grp_name = BODY_ZONES_GROUP[zone_name]
    prw_hgt, prw_wdt = prw_crop_shape[:2]
    bdpart_adjust = bdgrp_adjustment[grp_name]
    region_meta = bdgrp_crop_config[grp_name]
    roi_wdt, roi_hgt = np.int32(region_meta[:2] * scale_dim_f)
    half_roi_wdt = roi_wdt//2
    assert (prw_wdt>=nci_wdt>=roi_wdt), '{} vs. {} vs. {}'.format(prw_wdt, nci_wdt, roi_wdt)
    assert (prw_hgt>=nci_hgt>=roi_hgt), '{} vs. {} vs. {}'.format(prw_hgt, nci_hgt, roi_hgt)
    x_nci_l, x_nci_r, y_nci_t, y_nci_b = bb_half_dims(nci_wdt, nci_hgt)  # nci corners
    x_prw_l, x_prw_r, y_prw_t, y_prw_b = bb_half_dims(prw_wdt, prw_hgt)  # prw corners

    zone_prw_imgs = np.empty((n_ipz, *prw_crop_shape), dtype=np.uint8)
    zone_seg_conf = np.empty((n_ipz), dtype=np.float16)
    roi_bb_coords = np.empty((n_ipz, 4, 2), dtype=np.int16) # top-left --> bottom-left
    nci_xy_toplft = np.empty((n_ipz, 1, 2), dtype=np.int16) # top-left corner of nci window
    assert (n_ipz==len(ordered_fids_of_zone)), '{} vs. {}'.format(n_ipz, len(ordered_fids_of_zone))

    for idx, fid in enumerate(ordered_fids_of_zone):
        # get dictionary of keypoints needed per valid zone in given frame
        zone_frm_kpts = fk_map[fid]
        assert (zone_frm_kpts is not None), 'zone_frm_kpts:{}'.format(zone_frm_kpts)

        # read frame image and crop region
        frm_img = read_image(os.path.join(scan_dir, '{}.png'.format(fid)), rgb=rgb_img, icef=icef)
        if resize_image:
            frm_img = cv.resize(frm_img, tuple(scaled_frm_dim), interpolation=cv.INTER_CUBIC)

        # Get keypoints' coordinates and confidence available for scan's frame
        column_name = 'Frame{}'.format(fid)
        cell = kpts_df.loc[kpts_df['scanID'] == scan_id, column_name]
        frm_kpts_meta = eval(cell.values[0])  # eval() or ast.literal_eval()

        # get body zone metadata and compute it's confidence
        kpts_meta = keypoints_meta(zone_frm_kpts, frm_kpts_meta) # kpts_meta shape=(?, 3)
        zone_seg_conf[idx] = np.mean(kpts_meta[:, 2])
        kpts_coord = kpts_meta[:, :2] * scale_dim_f # scale anchor keypoints of body part

        # derive region-of-interest (roi) bounding polygon coordinates and center
        if fk_map_type[fid]:  # pair pillar keypoints
            # oriented bounding-box crop using anchor keypoints pair
            assert (len(zone_frm_kpts) == 2), "{}: {} not (2,)".format(zone_name, zone_frm_kpts)
            (x, y), roi_coord_wrt_frm = \
                limb_poly_crop(frm_img, kpts_coord, bdpart_adjust, half_roi_wdt, poly_degree,
                               scaled_frm_dim, frm_corners, bg_pixels, ax_sp, display, db=db)
        else:  # pair-of-pair anchor keypoints
            # corner keypoint region crop using 4 pillar keypoints
            assert (len(zone_frm_kpts[1]) == 2), "{}:{} not (2,2)".format(zone_name, zone_frm_kpts)
            (x, y), roi_coord_wrt_frm = \
                torso_poly_crop(frm_img, kpts_coord, region_meta[2:], bdpart_adjust,
                                scaled_frm_dim, fid, display)
        # confirm x_coordinates of ALL roi vertices are not left-side-OOB or right-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,0]<0)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        assert (not np.all(roi_coord_wrt_frm[:,0]>frm_wdt)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        # confirm y_coordinates of ALL roi vertices are not top-side-OOB or bottom-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,1]<0)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        assert (not np.all(roi_coord_wrt_frm[:,1]>frm_hgt)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        roi_bb_coords[idx] = roi_coord_wrt_frm
        ##print ('(x:{:.1f}, y:{:.1f})\n'.format(x, y))

        # crop preloaded-region-window (prw) image around roi
        ps_x, pe_x, ps_y, pe_y = bb_coords(x, y, x_prw_l, x_prw_r, y_prw_t, y_prw_b) # prw dims
        prw_top_left = [ps_x, ps_y]
        if ps_x<0 or ps_y<0 or pe_x>frm_wdt or pe_y>frm_hgt:
            frm_img, ps_x, pe_x, ps_y, pe_y = \
                pad_image_boundary(frm_img, ps_x, pe_x, ps_y, pe_y, bg_pixels, scaled_frm_dim)
        prw_img = frm_img[ps_y: pe_y, ps_x: pe_x]
        assert (prw_img.shape==prw_crop_shape), '{} vs. {}'.format(prw_img.shape, prw_crop_shape)
        zone_prw_imgs[idx] = prw_img

        # compute top-left corner coordinate of network-cropped-image (nci) within prw
        nci_tl = [x - x_nci_l, y - y_nci_t]
        nci_xy_toplft[idx, 0] = nci_tl
        assert (-db<=nci_tl[0]<=frm_wdt+db), "nci top-left x:{}, db:{}".format(nci_tl[0], db)
        assert (-db<=nci_tl[1]<=frm_hgt+db), "nci top-left y:{}, db:{}".format(nci_tl[1], db)

        if display:
            # display re-sized frame image with roi bbox
            frm_roi_img = frm_img.copy()
            top_left_frm_padding = [min(0, prw_top_left[0]), min(0, prw_top_left[1])]
            roi_coord_wrt_pad_frm = roi_coord_wrt_frm - top_left_frm_padding
            pts = roi_coord_wrt_pad_frm.reshape((-1, 1, 2))  # need vertices in (rows,1,2)
            cv.polylines(frm_roi_img, [pts], True, (150, 200, 0), 2)
            cv_display_image('image with roi bbox', frm_roi_img, 1)
            # display preloaded-region window image with roi and nci boundary centered on image
            prw_nci_img = prw_img.copy()
            pts = roi_coord_wrt_frm.reshape((-1, 1, 2))  # need vertices in (rows,1,2)
            pts -= prw_top_left  # roi coordinates relative to the prw image
            cv.polylines(prw_nci_img, [pts], True, (150, 200, 0), 1)  # relative roi
            x_ctr, y_ctr = np.asarray([[pe_x-ps_x], [pe_y-ps_y]]) // 2
            ns_x, ne_x, ns_y, ne_y = bb_coords(x_ctr, y_ctr, x_nci_l, x_nci_r, y_nci_t, y_nci_b)
            cv.rectangle(prw_nci_img, (ns_x, ns_y), (ne_x, ne_y), (0, 0, 0), 2)  # nci
            cv_display_image('prw cropped image', prw_nci_img, 0)

    return zone_prw_imgs, zone_seg_conf, roi_bb_coords, nci_xy_toplft


def limb_poly_crop(frm_img, anchor_kpts, limb_adjust, roi_half_wdt, poly_degree, frm_dim,
                   frm_corners, bg_pixels, ax_sp, display=False, gaus_filter=(5,5), db=20):
    '''
    Used to crop limb region body zones with 2 anchor keypoints
    :param frm_img: resized frame image
    :param anchor_kpts: pair of keypoint coordinates
    :param limb_adjust: pre-defined configuration for adjusting keypoints to proper position
    :param roi_half_wdt: pre-defined half width of the body zone's region-of-interest
    :param poly_degree: degree of polynomial to fit to spatial pixel histogram line
    :param frm_dim: resized frame image dimension
    :param frm_corners: homogenous coordinate of frame corners top-lft->top-rgt->btm-rgt->btm-lft
    :param bg_pixels: filler background pixel values (for each channel)
    :param ax_sp: plt.subplots object for plotting spatial pixel intensity histogram
    :param display: whether or not to display intermediate images for debugging
    :param gaus_filter: gaussian filter kernel window
    :param db: maximum allowance on boundaries of computed roi bounding box coordinates
    :return: roi bounding-polygon coordinates and it's center in the wrt the frame CS
    '''
    assert (len(anchor_kpts)==2), '{} == 2 ?'.format(len(anchor_kpts))  # anchor keypoint pair
    frm_wdt, frm_hgt = frm_dim
    # identify topmost and lower keypoints
    min_y_idx = np.argmin(anchor_kpts[:, 1])
    top_kpt = anchor_kpts[min_y_idx]
    top_kpt_x, top_kpt_y = top_kpt
    btm_kpt = anchor_kpts[(min_y_idx + 1) % 2]
    btm_kpt_x, btm_kpt_y = btm_kpt
    ##print ("top_kpt:{}, btm_kpt:{}".format(np.around(top_kpt, 1), np.around(btm_kpt, 1)))
    # compute rotation angle based on limb orientation
    limb_len = np.linalg.norm(top_kpt - btm_kpt)  # euclidean distance or math.dist
    assert (limb_len>0), "limb_len:{} > 0 ?".format(limb_len)
    y_len = abs(top_kpt_y - btm_kpt_y)
    ang_rad = np.arccos(y_len / limb_len)  # arccos(adjacent/hypotenuse)
    ang_drc = 1 if btm_kpt_x < top_kpt_x else -1 # +/-: counter-clockwise/clockwise rotation
    ang_deg = int(round(np.rad2deg(ang_rad), 0)) * ang_drc  # round to the nearest integer
    ##print ("limb_len:{:.1f}, y_len:{}, ang_deg:{}".format(limb_len, y_len, ang_deg))
    assert (-90<=ang_deg<=90), "ang_deg:{}".format(ang_deg)
    rot_mtx = cv.getRotationMatrix2D(tuple(btm_kpt), ang_deg, 1)
    corners = rot_mtx.dot(frm_corners)  # (2x3)x(3x4) = (2x4) or ([x,y]x[4 corners])
    ##print ("corners:\n{}".format(np.around(corners, 1)))
    lftmost_x = np.min(corners[0])
    topmost_y = np.min(corners[1])
    ##print ("leftmost_x:{:.1f}, topmost_y:{:.1f}".format(lftmost_x, topmost_y))
    # grow/shrink limb vertically according to limb_adjust config
    sft_top = limb_len * limb_adjust[0]
    sft_btm = limb_len * limb_adjust[1]
    ##print ("sft_top:{:.1f}, sft_btm:{:.1f}".format(sft_top, sft_btm))
    adj_btm_kpt = btm_kpt + sft_btm
    adj_limb_len = limb_len - sft_top + sft_btm
    assert (adj_limb_len>0), "adj_limb_len:{} > 0 ??".format(limb_len)
    # get new position of anchor keypoints
    rot_btm_kpt = adj_btm_kpt - [lftmost_x, topmost_y]
    assert (np.all(rot_btm_kpt.astype(np.int32)>=0)), "rot_btm_kpt:{}".format(rot_btm_kpt)
    rot_btm_kpt_x, rot_btm_kpt_y = rot_btm_kpt
    rot_top_kpt_y = max(0, rot_btm_kpt_y - adj_limb_len)  # <0 causes unintended np slicing behavior
    ##print ("rot_top_kpt_y:{:.1f}, rot_btm_kpt:{}".format(rot_top_kpt_y, np.around(rot_btm_kpt, 1)))
    y_top, y_btm = rot_top_kpt_y, rot_btm_kpt_y
    y_top += limb_adjust[2] * adj_limb_len
    y_btm += limb_adjust[3] * adj_limb_len
    # rotate image to position limb vertically
    abs_cos = abs(rot_mtx[0, 0])  # or abs(np.cos(ang_rad))
    abs_sin = abs(rot_mtx[0, 1])  # or abs(np.sin(ang_rad))
    rot_wdt = int(frm_hgt * abs_sin + frm_wdt * abs_cos)
    rot_hgt = int(frm_hgt * abs_cos + frm_wdt * abs_sin)
    rot_mtx += np.asarray([[0, 0, -lftmost_x],  # translation
                           [0, 0, -topmost_y]])
    rot_img = cv.warpAffine(frm_img, rot_mtx, (rot_wdt, rot_hgt),
                            borderMode=cv.BORDER_CONSTANT, borderValue=bg_pixels)
    # deduce width of limb using histogram of pixels
    #rot_img = cv.bilateralFilter(rot_img, 9, 25, 10)  # (..,9,25,25) (..,9,25,10)
    gray_img = cv.cvtColor(rot_img, cv.COLOR_BGR2GRAY)
    gray_img = cv.GaussianBlur(gray_img, gaus_filter, 0)  # cv.bilateralFilter()
    #gray_img = cv.bilateralFilter(gray_img, 9, 5, 5)  # (..,9,5,5) (..,9,7,7)
    sx_prw = int(rot_btm_kpt_x - roi_half_wdt)  # <0 causes unintended np slicing behavior
    ex_prw = int(rot_btm_kpt_x + roi_half_wdt)
    sx_prw, ex_prw = move_off_boundary(sx_prw, ex_prw, rot_wdt)
    x_range = np.arange(sx_prw, ex_prw) #+1
    ##print ("sx_prw:{}, ex_prw:{}".format(sx_prw, ex_prw))
    assert (-db<=sx_prw and ex_prw<=rot_wdt+db), '{} {} {} {}'.format(sx_prw, ex_prw, rot_wdt, db)
    assert (-db<=rot_top_kpt_y), 'rot_top_kpt_y:{:.1f}, db:{}'.format(rot_top_kpt_y, db)
    assert (rot_btm_kpt_y<=rot_hgt+db), "{:.1f} <= {} ?".format(rot_btm_kpt_y, rot_hgt+db)
    #*limb_reg = gray_img[int(rot_top_kpt_y): int(rot_btm_kpt_y), sx_prw: ex_prw] #+1
    limb_reg = gray_img[int(y_top): int(y_btm), sx_prw: ex_prw]  # *new* better for Abs
    pixs_agg = np.sum(limb_reg, axis=0)
    assert (len(pixs_agg)>0), "pixs_agg:{}\nlimb_reg:\n{}".format(pixs_agg, limb_reg)
    assert (len(x_range)==len(pixs_agg)), "is {} == {} ?".format(len(x_range), len(pixs_agg))
    poly_coef = np.polyfit(x_range, pixs_agg, poly_degree) # polynomial line coefficients
    poly_func = np.poly1d(poly_coef)
    x_mid = optimize.fminbound(-poly_func, sx_prw, ex_prw)  # x position of global maxima
    ##print ("x_mid:{:.1f}".format(x_mid))
    x_lft = optimize.fminbound(poly_func, sx_prw, x_mid+2)  # x position of left local minima
    x_rgt = optimize.fminbound(poly_func, x_mid-2, ex_prw)  # x position of right local minima
    # retrieve original coordinate of limb region boundaries
    rev_rot_mtx = cv.getRotationMatrix2D(tuple(rot_btm_kpt), -ang_deg, 1) # reverse rotation matrix
    frm_roi_coord = np.float32([[x_lft, y_top, 1],  # top-left
                                [x_rgt, y_top, 1],  # top-right
                                [x_rgt, y_btm, 1],  # bottom-right
                                [x_lft, y_btm, 1]]) # bottom-left
    frm_roi_coord = rev_rot_mtx.dot(frm_roi_coord.T)  # (2x3)x(3x4)=(2x4) or ([x,y]x[4 roi coord])
    frm_roi_coord = frm_roi_coord.T + [lftmost_x, topmost_y] # subtract padding for OOB pixels
    ##print ("frm_roi_coord:\n{}\n".format(np.around(frm_roi_coord, 1)))
    assert (np.all(frm_roi_coord>=-db)), "db:{}\nfrm_roi_coord:\n{}".format(db, frm_roi_coord)
    assert (np.all(frm_roi_coord[:,0]<=frm_wdt+db)), "frm_roi x: {}".format(frm_roi_coord[:,0])
    assert (np.all(frm_roi_coord[:,1]<=frm_hgt+db)), "frm_roi y: {}".format(frm_roi_coord[:,1])

    # display images for debugging
    if display:
        # display frame image with keypoint pair linked by a line
        frm_kpts_img = frm_img.copy()
        cv.line(frm_kpts_img, tuple(np.int32(top_kpt)),
                tuple(np.int32(btm_kpt)), (0, 0, 0), 2)  # original limb
        cv_display_image('image with keypoints', frm_kpts_img, 1)  # img with anchor kpts
        # display rotated image with limb bounding boxes (grown/shrunk limb)
        rot_limb_img = rot_img.copy()
        diag = np.int32(np.around(np.sqrt(frm_wdt**2 + frm_hgt**2), 1))  # frame diagonal
        #*cv.rectangle(rot_limb_img, (sx_prw, int(rot_top_kpt_y)),
        #              (ex_prw, int(rot_btm_kpt_y)), (0, 0, 0), 2)  # histogram region: black
        # cv.rectangle(rot_limb_img, (int(x_lft), int(rot_top_kpt_y)),
        #              (int(x_rgt), int(rot_btm_kpt_y)), (200, 0, 0), 2)  # limb region: blue
        cv.rectangle(rot_limb_img, (sx_prw, int(rot_top_kpt_y)),
                     (ex_prw, int(rot_btm_kpt_y)), (0, 0, 0), 2)  # original limb region: black
        cv.rectangle(rot_limb_img, (int(x_lft), int(y_top)),
                     (int(x_rgt), int(y_btm)), (200, 0, 0), 2)  # body zone region: black
        cv_display_image('image with part section', rot_limb_img, 1, diag)  # rot-img with limb bbox
        # display pixel intensity spatial histogram
        #*hist_reg_img = rot_img[int(rot_top_kpt_y): int(rot_btm_kpt_y), sx_prw: ex_prw] #+1
        hist_reg_img = rot_img[int(y_top): int(y_btm), sx_prw: ex_prw] #+1
        hri_hgt, hri_wdt = hist_reg_img.shape[:2]
        line_x = np.arange(0, ex_prw - sx_prw) #+1
        #*y_range = (0, int(rot_btm_kpt_y - rot_top_kpt_y))
        y_range = (0, int(y_btm - y_top))
        pixl_line_y = minmax_scale(pixs_agg, feature_range=y_range)
        poly_line_y = poly_func(x_range)
        poly_line_y = minmax_scale(poly_line_y, feature_range=y_range)
        ax_sp.clear()
        ax_sp.imshow(hist_reg_img, extent=[0, hri_wdt, 0, hri_hgt])
        ax_sp.plot(line_x, pixl_line_y, color='red', linewidth=2)  # spatial pixel line plot
        ax_sp.plot(line_x, poly_line_y, color='white', linewidth=2)  # polynomial approx. of line
        plt.axvline(x=x_mid-sx_prw, color='black', linestyle=':')  # vertical at global maxima
        plt.axvline(x=x_lft-sx_prw, color='black', linestyle='--')  # vertical at left local minima
        plt.axvline(x=x_rgt-sx_prw, color='black', linestyle='--')  # vertical at left local minima
        plt.pause(0.001)  # plt.show(block=False)n

    return np.mean(frm_roi_coord, axis=0).astype(np.int32), \
           frm_roi_coord.astype(np.int32)


def torso_poly_crop(frm_img, pillar_kpts, reg_margins, torso_adjust, frm_dim, fid, display=False):
    '''
    Used to crop torso region body zones with 4 pillar keypoints
    :param frm_img: resized frame image
    :param pillar_kpts: pair-of-pair keypoint coordinates
    :param reg_margins: pre-defined configuration for targeting body part within torso
    :param torso_adjust: pre-defined configuration for adjusting keypoints to proper position
    :param frm_dim: resized frame image dimension
    :param fid: frame ID number 0-15(aps) or 0-63(a3daps)
    :param display: whether or not to display intermediate images for debugging
    :return: roi bounding-polygon coordinates and it's center in the wrt the frame CS
    '''
    assert (len(pillar_kpts)==4), '{} == 4 ?'.format(len(pillar_kpts))  # pillar kpt pair-of-pairs
    frm_wdt, frm_hgt = frm_dim
    # reshape keypoints (4, 2) to (2, 2, 2). top: axis 0, index 0, bottom: axis 0, index 1
    pillar_kpts = pillar_kpts.reshape((2, 2, 2))
    # identify leftmost and rightmost keypoints in each pair
    min_x_idx_1, min_x_idx_2 = np.argmin(pillar_kpts[:, :, 0], axis=1)
    lft_kpt_1 = pillar_kpts[0, min_x_idx_1]            # top-lft kpt
    rgt_kpt_1 = pillar_kpts[0, (min_x_idx_1 + 1) % 2]  # top-rgt kpt
    lft_kpt_2 = pillar_kpts[1, min_x_idx_2]            # btm-lft kpt
    rgt_kpt_2 = pillar_kpts[1, (min_x_idx_2 + 1) % 2]  # btm-rgt kpt
    if display: kpts_cord_1 = np.int32([lft_kpt_1, rgt_kpt_1, rgt_kpt_2, lft_kpt_2])
    # adjust torso region
    sy_adj_f, ey_adj_f = torso_adjust[:2]
    torso_agg_wdt = (abs(lft_kpt_1[0]-rgt_kpt_1[0]) + abs(lft_kpt_2[0]-rgt_kpt_2[0])) / 2  # avg wdt
    torso_agg_hgt = (abs(lft_kpt_1[1]-lft_kpt_2[1]) + abs(rgt_kpt_1[1]-rgt_kpt_2[1])) / 2  # avg hgt
    lft_kpt_1[1] += torso_agg_hgt * sy_adj_f  # adjust top-lft kpt y position
    lft_kpt_2[1] += torso_agg_hgt * ey_adj_f  # adjust btm-lft kpt y position
    rgt_kpt_1[1] += torso_agg_hgt * sy_adj_f  # adjust top-rgt kpt y position
    rgt_kpt_2[1] += torso_agg_hgt * ey_adj_f  # adjust btm-rgt kpt y position
    if display: kpts_cord_2 = np.int32([lft_kpt_1, rgt_kpt_1, rgt_kpt_2, lft_kpt_2])
    # define 4 line functions for each side of polygon enclosing torso region
    # lines are defined by coefficients [m, c, d] in equation: y = m(x+d) + c
    lft_rgt_1_line = np.asarray([lft_kpt_1, rgt_kpt_1])  # top lft->rgt side
    lft_rgt_1_line_coef = line_from_two_points(lft_rgt_1_line)
    lft_rgt_2_line = np.asarray([lft_kpt_2, rgt_kpt_2])  # btm lft->rgt side
    lft_rgt_2_line_coef = line_from_two_points(lft_rgt_2_line)
    lft_1_2_line = np.asarray([lft_kpt_1, lft_kpt_2])  # lft top->btm side
    lft_1_2_line_coef = line_from_two_points(lft_1_2_line)
    rgt_1_2_line = np.asarray([rgt_kpt_1, rgt_kpt_2])  # rgt top->btm side
    rgt_1_2_line_coef = line_from_two_points(rgt_1_2_line)
    # target region of interest using reg_margins
    # execute x,y margin adjustments on the linear line equations
    sx_f, ex_f, sy_f, ey_f = reg_margins
    if sx_f is None:
        sx_f, ex_f = ex_f[fid]
    # vertical shift of top and bottom side lines
    lft_rgt_1_line_coef[1] += torso_agg_hgt * sy_f  # update 'c' to move lft_rgt_1_line up/down
    lft_rgt_2_line_coef[1] += torso_agg_hgt * ey_f  # update 'c' to move lft_rgt_2_line up/down
    # horizontal shift of left and right side lines
    lft_1_2_line_coef[2] += torso_agg_wdt * sx_f  # update 'd' to move lft_1_2_line left/right
    rgt_1_2_line_coef[2] += torso_agg_wdt * ex_f  # update 'd' to move rgt_1_2_line left/right
    # get any 2 points on each of the 4 (moved) lines
    lft_rgt_1_line_pts = two_points_from_line(lft_rgt_1_line_coef, lft_rgt_1_line[:, 0])
    lft_rgt_2_line_pts = two_points_from_line(lft_rgt_2_line_coef, lft_rgt_2_line[:, 0])
    lft_1_2_line_pts = two_points_from_line(lft_1_2_line_coef, lft_1_2_line[:, 0])
    rgt_1_2_line_pts = two_points_from_line(rgt_1_2_line_coef, rgt_1_2_line[:, 0])
    # compute intersections between lines to get 4 vertices of roi
    lft_kpt_1 = get_lines_intersect(lft_rgt_1_line_pts, lft_1_2_line_pts)  # top-lft kpt
    rgt_kpt_1 = get_lines_intersect(lft_rgt_1_line_pts, rgt_1_2_line_pts)  # top-rgt kpt
    lft_kpt_2 = get_lines_intersect(lft_rgt_2_line_pts, lft_1_2_line_pts)  # btm-lft kpt
    rgt_kpt_2 = get_lines_intersect(lft_rgt_2_line_pts, rgt_1_2_line_pts)  # btm-rgt kpt
    # organize keypoints coordinate. order: topLft->topRgt->btmRgt->btmLft
    frm_roi_coord = np.asarray([lft_kpt_1, rgt_kpt_1, rgt_kpt_2, lft_kpt_2])
    ##print ("frm_roi_coord:\n{}\n".format(np.around(frm_roi_coord, )))
    assert (np.all(frm_roi_coord>=0)), "frm_roi coordinates:\n{}".format(frm_roi_coord)
    assert (np.all(frm_roi_coord[:,0]<=frm_wdt)), "frm_roi x:{}".format(frm_roi_coord[:,0])
    assert (np.all(frm_roi_coord[:,1]<=frm_hgt)), "frm_roi y:{}".format(frm_roi_coord[:,1])

    # display images for debugging
    if display:
        # display frame image with keypoints pairs linked by a polygon
        frm_kpts_img = frm_img.copy()
        pts = kpts_cord_1.reshape((-1, 1, 2))  # need vertices coordinates in (rows, 1, 2) shape
        cv.polylines(frm_kpts_img, [pts], True, (0, 0, 0), 2)
        cv_display_image('image with keypoints', frm_kpts_img, 1)  # img with pillar kpts
        # display image with torso part bounding boxes (grown/shrunk limb)
        frm_torso_img = frm_img.copy()
        diag = np.int32(np.around(np.sqrt(frm_wdt**2 + frm_hgt**2), 1))  # frame diagonal
        pts = kpts_cord_2.reshape((-1, 1, 2))  # need vertices coordinates in (rows, 1, 2) shape
        cv.polylines(frm_torso_img, [pts], True, (0, 0, 0), 2)
        pts = frm_roi_coord.reshape((-1, 1, 2)).astype(np.int32)  # need vertices in (rows, 1, 2)
        cv.polylines(frm_torso_img, [pts], True, (200, 0, 0), 2)
        cv_display_image('image with part section', frm_torso_img, 1, diag)  # img with torso bbox

    return np.mean(frm_roi_coord, axis=0).astype(np.int32), \
           frm_roi_coord.astype(np.int32)

def line_from_two_points(point_pair):
    '''
    Computes coefficients of the line equation: y = mx + c
        given two points on the line
    :param point_pair: (x,y) coordinates of 2 points
    :return: [m, c, d=0] for equation:
    '''
    A = np.vstack([point_pair[:, 0], np.ones(2)]).T
    coeff = np.linalg.lstsq(A, point_pair[:, 1], rcond=None)[0]
    ##print ("line solution is y = {m:.1f}*x + {c:.1f}".format(m=coeff[0], c=coeff[1]))
    return np.append(coeff, 0)

def two_points_from_line(line_coeffs, pts_x):
    '''
    Computes y = m(x+d) + c
    :param line_coeffs: [m, c, d]
    :param pts_x: x1, x2
    :return: some two points on the line defined by line_coeffs
    '''
    m, c, d = line_coeffs
    x1, x2 = pts_x
    if x1==x2:
        ##print ("x1:{:.1f} == x2:{:.1f}".format(x1, x2))
        # must be a vertical line so cannot use line equation
        x1, x2 = pts_x + d  # move x by d (MUST BE +d)
        y1, y2 = 100, 200   # pick arbitrary y positions keeping x fixed
    else:
        y1, y2 = m*(pts_x - d) + c
    ##print ("line points: [({:.1f}, {:.1f}), ({:.1f}, {:.1f})]".format(x1, y1, x2, y2))
    return [[x1, y1], [x2, y2]]

def get_lines_intersect(line_1_pts, line_2_pts):
    '''
    Computes the intersection point between the lines derived from line points
    :param line_1_pts: 2 points on 1st line: [[x1, y1], [x2, y2]]
    :param line_2_pts: 2 points on 2nd line: [[x1, y1], [x2, y2]]
    :return: intersection point
    '''
    s = np.vstack([line_1_pts[0], line_1_pts[1], line_2_pts[0], line_2_pts[1]])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2) # point of intersection
    ##print ("intersection: ({:.1f}, {:.1f}, {:.1f})".format(x, y, z))
    assert (z!=0), "z=={} implies lines are parallel, but they should not".format(z)
    return [x/z, y/z]


def read_image(file, rgb=True, icef=0):
    '''
    Reads the image and may enhance the contrast or other transformations
    :param file: filename of image
    :param icef: image contrast enhancement factor
    :return: read and transformed image
    '''
    try:
        img = cv.imread(file)
        assert (np.max(img)>30>np.min(img)), 'min:{}, max:{}'.format(np.min(img), np.max(img))
        if rgb: img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = enhance_contrast(img, factor=icef, is_rgb=rgb)
    except: #IOError
        print('Image Read Error: filepath: {} may not exist'.format(file))
        sys.exit()
    return img


def bb_half_dims(crop_reg_wdt, crop_reg_hgt):
    x_l = crop_reg_wdt // 2
    x_r = x_l + (crop_reg_wdt % 2)
    y_t = crop_reg_hgt // 2
    y_b = y_t + (crop_reg_hgt % 2)
    return x_l, x_r, y_t, y_b

def bb_coords(x, y, x_l, x_r, y_t, y_b):
    # s_x = max(0, x - x_l)  # <0 will cause unintended behavior in numpy array slicing
    # e_x = s_x + x_l + x_r  # x + x_r
    # s_y = max(0, y - y_t)  # <0 will cause unintended behavior in numpy array slicing
    # e_y = s_y + y_t + y_b  # y + y_b
    s_x = x - x_l
    e_x = x + x_r
    s_y = y - y_t
    e_y = y + y_b
    return s_x, e_x, s_y, e_y

def pad_image_boundary(image, s_x, e_x, s_y, e_y, bg_pixels, img_dim):
    roi_s_x, roi_e_x, roi_s_y, roi_e_y = s_x, e_x, s_y, e_y
    img_wdt, img_hgt = img_dim  # or image.shape[:2]
    canvas_hgt, canvas_wdt = img_hgt, img_wdt
    img_s_y, img_e_y, img_s_x, img_e_x = 0, img_hgt, 0, img_wdt
    is_within_bounds = True

    # exclusive OR conditions (to the left of 'or')
    assert ((s_x<0)!=(e_x>img_wdt) or (s_x>=0 and e_x<=img_wdt)), '{} {} {}'.format(s_x, e_x, img_wdt)
    assert ((s_y<0)!=(e_y>img_hgt) or (s_y>=0 and e_y<=img_hgt)), '{} {} {}'.format(s_y, e_y, img_hgt)
    assert (image.shape[:2]==(img_dim[1], img_dim[0])), '{} vs. {}'.format(image.shape, img_dim)

    if s_x < 0:
        img_s_x = abs(s_x)
        img_e_x = img_wdt + img_s_x
        canvas_wdt = img_e_x  # canvas_wdt += img_s_x
        roi_s_x = 0
        roi_e_x = roi_e_x + img_s_x
        is_within_bounds = is_within_bounds and False
    elif e_x > img_wdt:
        canvas_wdt += e_x - img_wdt
        is_within_bounds = is_within_bounds and False

    if s_y < 0:
        img_s_y = abs(s_y)
        img_e_y = img_hgt + img_s_y
        canvas_hgt = img_e_y  # canvas_hgt += img_s_y
        roi_s_y = 0
        roi_e_y = roi_e_y + img_s_y
        is_within_bounds = is_within_bounds and False
    elif e_y > img_hgt:
        canvas_hgt += e_y - img_hgt
        is_within_bounds = is_within_bounds and False

    if is_within_bounds:
        return image, roi_s_x, roi_e_x, roi_s_y, roi_e_y

    assert (roi_s_x>=0 and roi_s_y>=0), "roi s_x:{}, s_y:{}".format(roi_s_x, roi_s_y)
    assert (img_s_x>=0 and img_s_x>=0), "img s_x:{}, s_y:{}".format(roi_s_x, img_s_y)
    padded_image = np.zeros((canvas_hgt, canvas_wdt, len(bg_pixels)), dtype=np.uint8)
    padded_image += np.uint8(bg_pixels)
    padded_image[img_s_y: img_e_y, img_s_x: img_e_x] = image
    return padded_image, roi_s_x, roi_e_x, roi_s_y, roi_e_y

def move_off_boundary(start_xy, end_xy, upper_bound, lower_bound=0):
    if start_xy < lower_bound:
        start_xy, end_xy = lower_bound, end_xy + abs(start_xy)
    elif end_xy > upper_bound:
        start_xy, end_xy = start_xy - (end_xy - upper_bound), upper_bound

    return start_xy, end_xy

def keypoints_meta(kpts_list, frm_kpts_meta):
    kpts_meta = list()
    if isinstance(kpts_list[0], list):
        for kpts_pair in kpts_list:
            for kpt in kpts_pair:
                kpts_meta.append(frm_kpts_meta[kpt])
    else:
        for kpt in kpts_list:
            kpts_meta.append(frm_kpts_meta[kpt])
    return np.asarray(kpts_meta)

def keypoints_meta_v2(x_kpts, y_kpts, frm_kpts_meta):
    x_meta = list()
    for kpt in x_kpts:
        x_meta.append(frm_kpts_meta[kpt])
    y_meta = list()
    for kpt in y_kpts:
        y_meta.append(frm_kpts_meta[kpt])
    return np.asarray(x_meta), np.asarray(y_meta)

def enhance_contrast(image, factor, is_rgb):
    '''
    Enhance image contrast to make objects in image more visible
    :param image: image with pixels ranging [0, 255]
    :param factor: image contrast enhancement factor
    :param is_rgb: whether image is in RGB color-space or BGR
    :return: contrast enhanced unit8 image
    '''
    if factor < 1: return image # drastic change when factor <= 0
    # enhance image contrast to make more visible
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=factor, tileGridSize=(5, 5))
    color_space_cvt = cv.COLOR_RGB2LAB if is_rgb else cv.COLOR_BGR2LAB
    lab = cv.cvtColor(image, color_space_cvt)  # convert to LAB color space
    l, a, b = cv.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv.merge((l2, a, b))  # merge channels
    color_space_cvt = cv.COLOR_LAB2RGB if is_rgb else cv.COLOR_LAB2BGR
    return cv.cvtColor(lab, color_space_cvt)  # convert back to RGB/BGR
