# -*- coding: utf-8 -*-
# @Time    : 2/20/2021 4:43 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : assign_anomaly.py
# @Software: src

'''
Assign manually annotated threat/anomaly objects to a body-zone and
generate a pixel-level mask of threat object within it's bounding-box.
'''

import os
import sys
import time
import pickle
import cv2 as cv
import numpy as np
import pandas as pd

from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from skimage.segmentation import slic, felzenszwalb, watershed, quickshift, mark_boundaries

sys.path.append('../')
from tf_neural_net.data_preprocess import read_image, keypoints_meta
from tf_neural_net.data_preprocess import limb_poly_crop, torso_poly_crop
from tf_neural_net.commons import Logger, cv_display_image
from tf_neural_net.commons import df_map_to_dict_map, frm_zone_kpts_map_type
from tf_neural_net.commons import ZONE_NAME_TO_ID, BODY_ZONES_GROUP
from tf_neural_net.commons import BDGRP_ORIENTED, BDGRP_ADJUSTMENT



def log_summary(zone_meta):
    msg = '\nThreat-to-Zone Assignment Summary'
    rows = ('', 'scans_with_threat', 'n_frame_occurrences',
            'n_clashes_per_zone', 'max_wdt', 'max_hgt', 'clash_rate')
    for row_idx, row_name in enumerate(rows):
        msg += '\n{:<20}'.format(row_name)
        for zone_name in ZONE_LIST:
            zone_idx = ZONE_TAG_TO_IDX[zone_name]
            if row_name=='': value = zone_name
            elif row_name=='clash_rate':
                value = round(zone_meta[zone_idx, 2]/zone_meta[zone_idx, 1], 3)
            else: value = zone_meta[zone_idx, row_idx-1]
            msg += ' {:<5} '.format(value)
    _logger.log_msg(msg), print(msg)


def superpixel_segmentation(image, n_segments=10, kernel=5):
    imgrgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = cv.bilateralFilter(image,9,75,75)
    # apply SLIC and extract (approximately) the supplied number of segments
    segments = slic(img.astype(np.float32), n_segments=n_segments, sigma=kernel, compactness=1)
    #segments = quickshift(img, ratio=0.5)
    #segments = felzenszwalb(img, scale=100)
    #segments = watershed(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    # show the output of SLIC
    fig = plt.figure("Superpixels -- {} segments".format(n_segments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(imgrgb, segments))
    plt.axis("off")
    plt.show()
    return image


def canny_edge_detection(image):
    img = cv.bilateralFilter(image,9,75,75)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img, 10, 50, L2gradient=True)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    return image


def segment_body_zones(scan_id, fid, poly_degree=6, ax_sp=None, display=False, db=50):
    # read frame image
    img_path = os.path.join(_data_root, scan_id, '{}.png'.format(fid))
    frm_img = read_image(img_path, rgb=False)
    zoi_poly_coords = np.zeros((N_ZONES, 4, 2), dtype=np.int32) # top-left --> bottom-left

    # Get keypoints' coordinates and confidence available for scan's frame
    column_name = 'Frame{}'.format(fid)
    cell = _kpts_df.loc[_kpts_df['scanID']==scan_id, column_name]
    frm_kpts_meta = eval(cell.values[0])

    for zone_name in ZONE_LIST:
        zone_frm_kpts = _zfk_map[zone_name][fid]
        if zone_frm_kpts is None: continue
        zone_idx = ZONE_TAG_TO_IDX[zone_name]
        grp_name = BODY_ZONES_GROUP[zone_name]
        bdpart_adjust = BDGRP_ADJUSTMENT[grp_name]
        region_meta = BDGRP_ORIENTED[grp_name]
        half_roi_wdt = region_meta[0]//2

        # get body zone metadata
        kpts_meta = keypoints_meta(zone_frm_kpts, frm_kpts_meta) # kpts_meta shape=(?, 3)
        kpts_coord = kpts_meta[:, :2]

        # derive region-of-interest (roi) bounding polygon coordinates and center
        if ZFK_MAP_TYPE[zone_name][fid]:  # pair pillar keypoints
            # oriented bounding-box crop using anchor keypoints pair
            assert (len(zone_frm_kpts) == 2), "{}: {} not (2,)".format(zone_name, zone_frm_kpts)
            (x, y), roi_coord_wrt_frm = \
                limb_poly_crop(frm_img, kpts_coord, bdpart_adjust, half_roi_wdt, poly_degree,
                               (FRM_WDT,FRM_HGT), FRM_CORNERS, BG_PIXELS, ax_sp, display, db=db)
        else:  # pair-of-pair anchor keypoints
            # corner keypoint region crop using 4 pillar keypoints
            assert (len(zone_frm_kpts[1]) == 2), "{}:{} not (2,2)".format(zone_name, zone_frm_kpts)
            (x, y), roi_coord_wrt_frm = \
                torso_poly_crop(frm_img, kpts_coord, region_meta[2:], bdpart_adjust,
                                (FRM_WDT,FRM_HGT), fid, display)
        # confirm x_coordinates of ALL roi vertices are not left-side-OOB or right-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,0]<0)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        assert (not np.all(roi_coord_wrt_frm[:,0]>FRM_WDT)), "roi_x:{}".format(roi_coord_wrt_frm[:,0])
        # confirm y_coordinates of ALL roi vertices are not top-side-OOB or bottom-side-OOB
        assert (not np.all(roi_coord_wrt_frm[:,1]<0)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        assert (not np.all(roi_coord_wrt_frm[:,1]>FRM_HGT)), "roi_y:{}".format(roi_coord_wrt_frm[:,1])
        zoi_poly_coords[zone_idx] = roi_coord_wrt_frm

    return zoi_poly_coords, frm_img


def extract_bbox_coordinates(bbox_coord):
    (x1, y1), (x2, y2) = bbox_coord[0], bbox_coord[1]
    assert(x1!=x2 and y1!=y2), '{}=={} or {}=={}'.format(x1, x2, y1, y2)
    s_x, e_x = min(x1, x2), max(x1, x2)
    s_y, e_y = min(y1, y2), max(y1, y2)
    return s_x, e_x, s_y, e_y

def bbox_to_polygon_coordinate(bbox_coord):
    s_x, e_x, s_y, e_y = extract_bbox_coordinates(bbox_coord)
    return np.asarray([[s_x, s_y],   # top-left
                       [e_x, s_y],   # top-right
                       [e_x, e_y],   # bottom-right
                       [s_x, e_y]])  # bottom-left


def compute_threat_zone_overlaps(threat_bb, zone_polygons, frm_img, display=False):
    if display: cv.rectangle(frm_img, threat_bb[0], threat_bb[1], color=(0,0,255))
    #print('threat_bb:{}'.format(threat_bb))
    anom_poly = bbox_to_polygon_coordinate(threat_bb)
    polygon1_shape = Polygon(anom_poly)
    zone_overlap = np.zeros(N_ZONES, dtype=np.float32)
    for idx, zone_poly in enumerate(zone_polygons):
        #print('{}. zone_poly:{}'.format(idx, zone_poly))
        if np.all(zone_poly==0): continue
        # Define each polygon
        polygon2_shape = Polygon(zone_poly)
        # Calculate intersection and union, and tne IOU
        polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
        zone_overlap[idx] = polygon_intersection
        if display:
            pts = zone_poly.reshape((-1, 1, 2))  # need vertices coordinates in (rows, 1, 2) shape
            reg_img = cv.polylines(frm_img.copy(), [pts], True, (0, 0, 0), 1)
            cv_display_image('threat vs. zone region', reg_img)
    return zone_overlap


def assign_threat_to_zone(anom_anot_csv):
    # (n_occurrences_per_zone, n_frames_occurrence, n_clashes_per_zone, max_wdt, max_hgt)
    zone_meta = np.zeros((N_ZONES, 5), dtype=np.int32)
    threat_bbox_dict = dict()
    for zone_name in ZONE_LIST:
        threat_bbox_dict[zone_name] = dict()

    threat_df = pd.read_csv(anom_anot_csv)
    n_rows = threat_df.shape[0]
    for df_idx, row in threat_df.iterrows():
        scan_id = row['ID']
        for fid in range(N_FRAMES):
            col_name = 'Frame{}'.format(fid)
            entry = row[col_name]
            if entry=='N/M': continue
            # the scan has a threat or more in this frame
            threat_bboxes = eval(entry)
            zone_polygons, frm_img = segment_body_zones(scan_id, fid)
            for bb_idx, threat_bb in enumerate(threat_bboxes):
                threat_id = '{:>4}-{}-{:<2}'.format(df_idx, bb_idx, fid)
                threat_assigned = False
                zone_overlaps = compute_threat_zone_overlaps(threat_bb, zone_polygons, frm_img)
                #assert#(np.max(zone_overlaps)>0), 'zone_overlaps:{}'.format(zone_overlaps)
                while not threat_assigned and np.max(zone_overlaps)>0:
                    assign_mode = -1
                    zone_idx = np.argmax(zone_overlaps)
                    zone_name = ZONE_IDX_TO_TAG[zone_idx]
                    if threat_bbox_dict[zone_name].get(scan_id, None) is None:
                        # no record of scan having any threat on the zone
                        threat_bbox_dict[zone_name][scan_id] = {fid:threat_bb}
                        assign_mode = 0
                    elif threat_bbox_dict[zone_name][scan_id].get(fid, None) is None:
                        # no record of scan having a threat on the zone in the frame
                        threat_bbox_dict[zone_name][scan_id][fid] = threat_bb
                        assign_mode = 1
                    else: # there's already a threat assigned to the scan-zone-frame
                        zone_meta[zone_idx, 2] += 1
                        msg = '{} multi-threat to zone clash for scanid:{}, zone:{:<4}, fid:{:<2}, ' \
                              'threat_bbox:{}'.format(threat_id, scan_id, zone_name, fid, threat_bb)
                        _logger.log_msg(msg), print(msg)
                        zone_overlaps[zone_idx] = 0  # deactivate zone

                    if assign_mode>=0:
                        # book-keeping
                        threat_assigned = True
                        if assign_mode==0: zone_meta[zone_idx, 0] += 1
                        zone_meta[zone_idx, 1] += 1
                        bb_wdt = abs(threat_bb[0][0] - threat_bb[1][0])
                        bb_hgt = abs(threat_bb[0][1] - threat_bb[1][1])
                        if bb_wdt>zone_meta[zone_idx][3]: zone_meta[zone_idx][3] = bb_wdt
                        if bb_hgt>zone_meta[zone_idx][4]: zone_meta[zone_idx][4] = bb_hgt

                if not threat_assigned:
                    msg = '{} could not assign threat for scanid:{}, zone:{:<4}, fid:{:<2}, ' \
                            'threat_bbox:{}'.format(threat_id, scan_id, zone_name, fid, threat_bb)
                    _logger.log_msg(msg), print(msg)

        if (df_idx+1)%100==0 or (df_idx+1)==n_rows:
            print('{:>4}/{} rows passed..'.format(df_idx+1, n_rows))

    log_summary(zone_meta)
    # save assignment and metadata
    threat_bbox_dict['zone_meta'] = zone_meta
    with open(ANOMALY_ZONE_ASSIGN_PATH, 'wb') as file_handle:
        pickle.dump(threat_bbox_dict, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    return threat_bbox_dict


def extract_threats(threat_bbox_dict):
    zone_meta = threat_bbox_dict['zone_meta']
    df_columns = ['scanID', 'count']
    for fid in range(N_FRAMES): df_columns.append('Frame{}'.format(fid))
    for zone_name, zone_threat_dict in threat_bbox_dict.items():
        zone_idx = ZONE_TAG_TO_IDX[zone_name]
        n_zone_occurrences, n_frames_occurrence, n_clashes, max_wdt, max_hgt = zone_meta[zone_idx]
        zone_threat_images = np.zeros((n_frames_occurrence, max_hgt, max_wdt, 3), dtype=np.uint8)
        zone_df = pd.DataFrame(columns=df_columns)
        array_idx = 0
        for scan_id, scan_threat_dict in zone_threat_dict.items():
            zone_df = zone_df.append({'scanID':scan_id}, ignore_index=True)
            count = 0
            for fid, threat_bb in scan_threat_dict.items():
                s_x, e_x, s_y, e_y = extract_bbox_coordinates(threat_bb)
                img_path = os.path.join(_data_root, scan_id, '{}.png'.format(fid))
                threat_img = read_image(img_path, rgb=False)[s_y:e_y, s_x:e_x]
                threat_img = superpixel_segmentation(threat_img)
                #threat_img = canny_edge_detection(threat_img)
                #threat_img = background_subtract(threat_img)
                wdt, hgt = e_x - s_x, e_y - s_y
                s_x, s_y = (max_wdt - wdt)//2, (max_hgt - hgt)//2
                zone_threat_images[array_idx, s_y:s_y+hgt, s_x:s_x+wdt] = threat_img
                zone_df.loc[zone_df['scanID']==scan_id, 'Frame{}'.format(fid)] = array_idx
                count += 1
                array_idx += 1
            zone_df.loc[zone_df['scanID']==scan_id, 'count'] = count

        np.save(ZONE_NP_PATH_TEMPLATE.format(zone_name), zone_threat_images)
        zone_df.to_csv(ZONE_DF_PATH_TEMPLATE.format(zone_name), encoding='utf-8', index=False)


def setup_and_run(time_str):
    global _kpts_df, _zfk_map, _data_root, _logger, N_FRAMES, N_ZONES, FRM_HGT, FRM_WDT, \
        FRM_CORNERS, ZONE_LIST, ZONE_TAG_TO_IDX, ZONE_IDX_TO_TAG, ZFK_MAP_TYPE, \
        ANOMALY_ZONE_ASSIGN_PATH, ZONE_DF_PATH_TEMPLATE, ZONE_NP_PATH_TEMPLATE, BG_PIXELS

    tsa_ext = 'aps'
    subset = 'train_set'
    dataset_root = '../../../datasets/tsa/{}_images/dataset'.format(tsa_ext)
    _data_root = os.path.join(dataset_root, subset)
    threat_dir = os.path.join(dataset_root, 'anom_anot')
    os.makedirs(threat_dir, exist_ok=True)
    ANOMALY_ZONE_ASSIGN_PATH = \
        os.path.join(threat_dir, 'anomaly-zone-assign_{}.pickle'.format(time_str))
    ZONE_NP_PATH_TEMPLATE = os.path.join(threat_dir, '{}_images.npy')
    ZONE_DF_PATH_TEMPLATE = os.path.join(threat_dir, '{}_dfmeta.csv')
    annotations = '../../Metadata/tsa_psc/stage1_labels_1_marked.csv'
    _logger = Logger(os.path.join(threat_dir, 'anomaly-zone-assign_{}.log'.format(time_str)))

    hpe_kpts_csv = '../../Metadata/hrnet_kpts/all_sets-w32_256x192-opt-ref-30-bav4.csv'
    _kpts_df = pd.read_csv(hpe_kpts_csv)
    zfk_map_csv = '../../Metadata/zfk_maps/fid_zones_kpts_map_v6.csv'
    map_df = pd.read_csv(zfk_map_csv)
    _zfk_map = df_map_to_dict_map(map_df, ZONE_NAME_TO_ID)
    ZFK_MAP_TYPE = frm_zone_kpts_map_type(_zfk_map)
    N_ZONES = 17
    N_FRAMES = 16
    ZONE_LIST = ZONE_NAME_TO_ID.keys()
    ZONE_IDX_TO_TAG = dict()
    ZONE_TAG_TO_IDX = dict()
    for zone_idx, zone_name in enumerate(ZONE_LIST):
        ZONE_IDX_TO_TAG[zone_idx] = zone_name
        ZONE_TAG_TO_IDX[zone_name] = zone_idx

    bgi = read_image('../../Metadata/tsa_psc/aps_bg.png', rgb=False, icef=0)
    BG_PIXELS = np.mean(bgi, axis=(0, 1)).astype(np.uint8, copy=False).tolist()
    FRM_HGT, FRM_WDT = 660, 512
    FRM_CORNERS = np.float32([[       0,       0, 1],     # top-left
                              [ FRM_WDT,       0, 1],     # top-right
                              [ FRM_WDT, FRM_HGT, 1],     # bottom-right
                              [       0, FRM_HGT, 1]]).T  # bottom-left


    #threat_bbox_dict = assign_threat_to_zone(annotations)

    pickled_dict = os.path.join(threat_dir, 'anomaly-zone-assign_2021-02-23-12-43.pickle')
    with open(pickled_dict, 'rb') as file_handle:
        threat_bbox_dict = pickle.load(file_handle)
    #log_summary(threat_bbox_dict['zone_meta'])

    extract_threats(threat_bbox_dict)

if __name__=='__main__':

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    setup_and_run(time_str)



