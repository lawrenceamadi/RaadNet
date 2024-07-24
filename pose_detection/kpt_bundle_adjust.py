
from __future__ import print_function

import os
import sys
import time
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import randint
from mpl_toolkits import mplot3d # <--- This is important for 3d plotting
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from sklearn.preprocessing import minmax_scale

sys.path.append('../')
from tf_neural_net.commons import move_figure
from tf_neural_net.data_preprocess import get_lines_intersect
from pose_detection.coco_kpts_zoning import valid_kypts_per_16frames, valid_kypts_per_64frames

np.set_printoptions(precision=4, suppress=False)



def print_ba_optimization_info(res, n_observations, n_cameras, n_points,
                               points_3d_prior, points_2d, camera_params, t_sec):
    if res is not None:
        error = np.sum(np.square(res.fun)) / 2
        residual_vec = np.int32(np.around(res.fun.reshape((n_observations, 2)), 0))
        print('\nresiduals at solution:\nx\n{}\ny\n{}\nshape: {}, square sum: {}'
              .format(residual_vec[:, 0], residual_vec[:, 1], residual_vec.shape, error))
        print('\nerror at solution: {}'.format(res.cost))
        print('\nalgorithm terminated because, {}'.format(ALGO_STATUS[res.status]))
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    n = np.size(camera_params) + np.size(points_3d_prior)
    m = np.size(points_2d)
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    print('3d points\n', np.int32(np.around(points_3d_prior, 0)))
    print("Bundle adjustment optimization took {0:.0f} seconds on avg.".format(t_sec))


def get_frame_config_constants(n_frames):
    if n_frames == 16:
        conf_beta = APS_CONF_BETA
        x_cardinal_fids = APS_X_CARDINAL_FIDS
        z_cardinal_fids = APS_Z_CARDINAL_FIDS
        valid_kpts_per_frm = valid_kypts_per_16frames()
    else:
        conf_beta = A3D_CONF_BETA
        x_cardinal_fids = A3D_X_CARDINAL_FIDS
        z_cardinal_fids = A3D_Z_CARDINAL_FIDS
        valid_kpts_per_frm = valid_kypts_per_64frames()
    return valid_kpts_per_frm, conf_beta, x_cardinal_fids, z_cardinal_fids


def flag_acceptable_kpt_observations(estimated_kpts_set, acceptable_type):
    # acceptable implies that the kpt in a frame is
    # both valid and considered confident or an inlier
    #assert(acceptable_type in ['confident_kpts','inlier_kpts','valid_only_kpts'])
    max_observations = 0
    # 3D points initial estimates should come from best a3daps
    for scan_estimated_kpts in estimated_kpts_set:
        # scan_estimated_kpts: shape: (14: kpts, 16/64: frames, 3: (x, y, conf))
        #assert(scan_estimated_kpts.shape == (N_KPTS,16,3) or (N_KPTS,64,3))
        #assert(scan_estimated_kpts.dtype == np.float32)
        #n_kpts, n_frames = scan_estimated_kpts.shape[:2]
        #max_observations += n_kpts * n_frames
        max_observations = N_KPTS*N_FRAMES
        valid_kpts_per_frm, __, __, __ = get_frame_config_constants(N_FRAMES)
        scan_estimated_kpts_conf = np.around(scan_estimated_kpts[:, :, 2], 4)
        if acceptable_type=='confident_kpts':
            # Note kpts with confidence score greater than a preset fixed confidence threshold
            is_confident_kpts = scan_estimated_kpts_conf > VALID_CONF_THRESH
            scan_observations = np.logical_and(is_confident_kpts, valid_kpts_per_frm)
        elif acceptable_type=='inlier_kpts':
            # Note kpts with confidence score that falls within the inlier confidence threshold
            q1, q3 = np.quantile(scan_estimated_kpts_conf, [0.25, 0.75], axis=1)  # compute Q1 & Q3
            iqr = q3 - q1  # compute inter-quartile-range
            low_inlier_thresh = q1 - 1.5*iqr
            expanded_low_inlier_thresh = \
                np.repeat(low_inlier_thresh, N_FRAMES).reshape((N_KPTS, N_FRAMES))
            is_inlier_kpts = scan_estimated_kpts_conf>=expanded_low_inlier_thresh
            scan_observations = np.logical_and(is_inlier_kpts, valid_kpts_per_frm)
        else: scan_observations = valid_kpts_per_frm
        n_kpts_observations = np.sum(np.int32(scan_observations), axis=-1)
        n_observations = np.sum(n_kpts_observations)
    #assert(0<=n_observations<=max_observations)
    return scan_observations, n_kpts_observations, n_observations


def error_per_keypoint(residuals, kpt_indices_list):
    error = np.square(residuals).reshape((-1, 2))
    pnt_wgt = np.zeros(len(kpt_indices_list))
    for i, kpt_indices in enumerate(kpt_indices_list):
        kpt_error = np.sum(error[kpt_indices, :]) / 2
        pnt_wgt[i] = kpt_error
    return pnt_wgt / (np.max(pnt_wgt) + 1e-7)


def plot_3d_pose(points3d_list, scan_id, error_list,
                 new_plot, fig_idx, wrt_path=None, display=False):
    '''Plot multiple 3d-pose stick figure. One per subplot'''
    plt.figure(POSE_3D_FIG[fig_idx])  # activate pose-3d figure

    # Plot new 3D-Pose
    if new_plot:
        sp_axs = SP_AXS[fig_idx]
        SP_FIG[fig_idx].suptitle('3D Poses of {}'.format(scan_id))
        avg_error = np.mean(error_list, axis=-1)
        color_err = np.asarray(error_list)/np.max(error_list)  # normalize across all poses

        for idx, points3d in enumerate(points3d_list):
            if fig_idx==0: sp_ax = sp_axs[idx]
            else:
                r_idx, c_idx = idx//3, idx%3
                sp_ax = sp_axs[r_idx, c_idx]
            # Clear subplot canvas
            sp_ax.clear()
            pose_label = '{} {:>5.1f}'.format(POSE3D_TYPES[idx], avg_error[idx])
            sp_ax.set(title=pose_label, xlim=(-HALF_WDT, HALF_WDT),
                      zlim=(FRM_HGT, 0), ylim=(0, FRM_WDT))
                      #xlabel='x', ylabel='z', zlabel='y')
            # Turn off tick labels
            sp_ax.set_yticklabels([]), sp_ax.set_xticklabels([]), sp_ax.set_zticklabels([])
            # Initialize pose variables
            line_cnt = 0
            points3d = np.int32(points3d)
            xKpts = points3d[:, 0]
            yKpts = points3d[:, 1]
            zKpts = points3d[:, 2]
            color = POSE3D_COLOR[idx]

            # Draw skeletal limbs (lines between keypoints)
            for limb, kptPair in LIMB_KPTS_PAIRS.items():
                kptA, kptB = kptPair
                kptAIdx, kptBIdx = KPTS_ID_INDEX[kptA], KPTS_ID_INDEX[kptB]
                if yKpts[kptAIdx]>=0 and yKpts[kptBIdx]>=0:
                    zline = [yKpts[kptAIdx], yKpts[kptBIdx]]
                    xline = [xKpts[kptAIdx], xKpts[kptBIdx]]
                    yline = [zKpts[kptAIdx], zKpts[kptBIdx]]
                    sp_ax.plot(xline, yline, zline, color, zdir='z')
                    line_cnt += 1

            # Data for three-dimensional keypoints
            validKptIndexes = np.argwhere(yKpts>=0).flatten()
            zData = yKpts[validKptIndexes]
            xData = xKpts[validKptIndexes]
            yData = zKpts[validKptIndexes]
            c_wgt = color_err[idx][validKptIndexes]
            sp_ax.scatter3D(xData, yData, zData, c=c_wgt, cmap='summer') #RdYlGn

        if wrt_path is not None:
            plt.savefig(wrt_path)
            plt.savefig(TEMP_FIG)
        if display: plt.pause(0.001)

    # Copy recently plotted 3D-Pose
    else: shutil.copyfile(TEMP_FIG, wrt_path)


def plot_2d_residuals(residuals, scan_id, fig_idx, title_tag, save=None,
                      valid_pts=None, confident_pts=None, selected_pts=None,
                      type='Projected', display=False):
    '''Plot euclidean distance error derived from 2d points' residuals'''
    pinpoint_kpt = selected_pts is not None
    n_points = len(residuals[0])
    plt.figure(RESIDUAL_FIG[fig_idx])  # activate residual figure
    plt.clf()
    plt.ylim(-10, 500)
    unit = 'mm' if UNIT_MM else 'px'
    plt.title('{} {} Keypoints Error'.format(title_tag, type))
    #plt.ylabel('Euclidean Dist. ({}) from Estimated 2D Keypoint'.format(unit))
    plt.ylabel('2D Keypoints Euclidean Dist. ({})'.format(unit))
    if pinpoint_kpt:
        plt.grid(which='both', axis='x', linestyle='-')
        plt.xticks(np.arange(0, n_points, 4))
        plt.xlabel('Keypoint per Frame\n{}'.format(scan_id))
    else: plt.xlabel('All 2D Keypoints\n{}'.format(scan_id))

    # plot each error line
    for idx, residual in enumerate(residuals):
        avg_error = np.mean(residuals[idx])
        line_label = '{} {:>5.1f}'.format(POSE3D_TYPES[idx], avg_error)
        plt.plot(residual, color=POSE3D_COLOR[idx], label=line_label)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if pinpoint_kpt:
        colors = list()
        for fid in range(len(valid_pts)):
            pt_color = 'black'
            if selected_pts[fid]: pt_color = 'green'
            if selected_pts[fid] and not confident_pts[fid]: pt_color = 'orange'
            if not selected_pts[fid] and confident_pts[fid]: pt_color = 'purple'
            if not valid_pts[fid]: pt_color = 'red'
            colors.append(pt_color)

        x_range = np.arange(n_points)
        for residual in residuals:
            plt.scatter(x_range, residual, s=10, c=colors, marker='o')

    if save is not None: plt.savefig(save)
    if display: plt.pause(0.001)


def aggregate_pnt_from_cardinal_frms(kid, cardinal_fids, pt_axis_per_frm,
                                     valid_kpts_per_frm, frm_wdt=None):
    '''Aggregate given 3d-point axis-coordinates from that of 2d-points in cardinal frames'''
    cnt = 0
    agg_pnt = 0
    for i, cardinal_fid in enumerate(cardinal_fids):
        if valid_kpts_per_frm[kid, cardinal_fid]:
            pnt = pt_axis_per_frm[cardinal_fid]
            if i==1 and frm_wdt is not None:
                pnt = frm_wdt - pnt # flipped frame
            agg_pnt += pnt
            cnt += 1
    return agg_pnt / cnt


def derive_3dkpt_prior_from_canrdinal_frms(kid, kpt_2d_per_frm, x_cardinal_fids,
                                           z_cardinal_fids, valid_kpts_per_frm):
    '''Get 3d-kpt position prior from 2d-kpts in cardinal frames
        - 3d-x from 2d-x of x_cardinal_fids,
        - 3d-y from 2d-y of x_cardinal_fids,
        - 3d-z from 2d-z of z_cardinal_fids
    '''
    w_x = aggregate_pnt_from_cardinal_frms(kid, x_cardinal_fids, kpt_2d_per_frm[:, 0],
                                           valid_kpts_per_frm, frm_wdt=FRM_WDT) - HALF_WDT
    w_z = aggregate_pnt_from_cardinal_frms(kid, z_cardinal_fids, kpt_2d_per_frm[:, 0],
                                           valid_kpts_per_frm, frm_wdt=FRM_WDT)
    w_y = aggregate_pnt_from_cardinal_frms(kid, x_cardinal_fids, kpt_2d_per_frm[:, 1],
                                           valid_kpts_per_frm, frm_wdt=None)
    return (w_x, w_y, w_z)


def circular_translate_vec(theta_deg):
    diameter = 2 * INNER_RADIUS
    #assert(0<=theta_deg<=360), 'theta_deg:{}'.format(theta_deg)
    alpha_deg = theta_deg if theta_deg<=180 else 360 - theta_deg
    #assert(0<=alpha_deg<=180), 'alpha_deg:{}'.format(alpha_deg)
    alpha_rad = np.deg2rad(alpha_deg)
    t_magn = np.sqrt((2*(INNER_RADIUS**2)) - (2*(INNER_RADIUS**2)) * np.cos(alpha_rad))
    #assert(0<=t_magn<=diameter), 't_magn:{}, diameter:{}'.format(t_magn, diameter)
    phi_magn = (180 - alpha_deg) / 2
    #assert(0<=phi_magn<=90), 'phi_magn:{}'.format(phi_magn)
    phi_sign = -1 if (180-theta_deg)<0 else 1
    #assert(phi_sign==1 or phi_sign==-1), 'phi_sign:{}'.format(phi_sign)
    phi_deg = phi_sign * phi_magn
    #assert(-90<=phi_deg<=90), 'phi_deg:{}'.format(phi_deg)
    phi_rad = np.deg2rad(phi_deg)
    cos_phi = np.cos(phi_rad)
    #assert(0<=cos_phi<=1), 'cos_phi:{}'.format(cos_phi)
    t_z = t_magn * cos_phi
    #assert(0<=t_z<=diameter), 't_z:{}, diameter:{}'.format(t_z, diameter)
    sin_phi = np.sin(phi_rad)
    #assert(-1<=sin_phi<=1), 'sin_phi:{}'.format(sin_phi)
    t_x = t_magn * sin_phi
    #assert(-diameter<=t_x<=diameter), 't_x:{}, diameter:{}'.format(t_x, diameter)
    #assert(-INNER_RADIUS<=t_x<=INNER_RADIUS), 't_x:{}, INNER_RADIUS:{}'.format(t_x, INNER_RADIUS)
    v_trans = np.array([t_x, 0, t_z])
    return v_trans


def rotate_about_yaxis_mtx(theta_deg):
    ###assert#(0<=theta_deg<=360), 'theta_deg:{}'.format(theta_deg)
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    rot_mtx = np.zeros((3, 3), dtype=np.float32)
    rot_mtx[0, :] = [cos_theta, 0, sin_theta]
    rot_mtx[1, 1] = 1
    rot_mtx[2, :] = [-sin_theta, 0, cos_theta]
    return rot_mtx


def wld2cam_transformation_matrix(angle_deg):
    #assert(0<=angle_deg<=360), 'angle_deg:{}'.format(angle_deg)
    translate_v = circular_translate_vec(angle_deg)
    rotate_mtx = rotate_about_yaxis_mtx(angle_deg)
    transform_mtx = np.zeros((3, 4), dtype=np.float32)
    transform_mtx[:3, :3] = rotate_mtx
    transform_mtx[:3, 3] = rotate_mtx.dot(-translate_v)
    return transform_mtx, -translate_v


def project_3d_to_2d(points_3d, n_kpts, n_cameras, n_observations):
    camera_indices = np.empty(n_observations, dtype=np.int32)
    point_indices = np.empty(n_observations, dtype=np.int32)
    points_2d = np.empty((n_observations, 2), dtype=np.float32)
    wld_kpt_3dh = np.ones((4, 1), dtype=np.float32)
    angle_step = 360 / n_cameras
    idx = 0
    for kid in range(n_kpts):
        point_index = kid
        wld_kpt_3d = points_3d[kid][:, np.newaxis]
        wld_kpt_3dh[:3, :] = wld_kpt_3d
        for fid in range(n_cameras):
            camera_index = fid
            camera_indices[idx] = camera_index
            point_indices[idx] = point_index
            angle_deg = fid * angle_step
            ext_tfm_mtx, __ = wld2cam_transformation_matrix(angle_deg)
            cam_kpt_3d = ext_tfm_mtx.dot(wld_kpt_3dh)
            c_x, c_y, c_z = cam_kpt_3d
            i_x = c_x + HALF_WDT
            i_y = c_y
            #assert(0<=i_x<=FRM_WDT), 'i_x:{}, FRM_WDT:{}'.format(i_x, FRM_WDT)
            #assert(0<=i_y<=FRM_HGT), 'i_y:{}, FRM_HGT:{}'.format(i_y, FRM_HGT)
            points_2d[idx] = [i_x, i_y]
            idx += 1
    return camera_indices, point_indices, points_2d


def define_camera_extrinsic_params(n_cameras):
    camera_params = np.empty((n_cameras, 3, 4), dtype=np.float32)
    angle_step = 360 / n_cameras
    for idx in range(n_cameras):
        angle_deg = idx * angle_step
        #assert(0<=angle_deg<=360), 'angle_deg:{}'.format(angle_deg)
        ext_tfm_mtx, __ = wld2cam_transformation_matrix(angle_deg)
        camera_params[idx] = ext_tfm_mtx
    return camera_params


def scale_points_confidence(points_conf, point_indices, n_points, sqrt=False):
    '''Scale confidence scores to (0,1] per keypoint then return square root'''
    kpt_indices_list = list()
    for kpt_idx in range(n_points):
        kpt_indices = np.argwhere(point_indices == kpt_idx)
        kpt_indices_list.append(kpt_indices)
        kpt_confs = points_conf[kpt_indices]
        kpt_confs = minmax_scale(kpt_confs, feature_range=(np.min(kpt_confs), 1), copy=False)
        points_conf[kpt_indices] = kpt_confs
    points_conf = np.repeat(points_conf, 2).reshape(-1, 2)
    if sqrt: return np.sqrt(points_conf)
    return points_conf, kpt_indices_list


def organize_all_scan_keypoints(estimated_kpts_set, n_observations,
                                acceptable_kpts_per_frm, pose3d_est_idx=0):
    # from a multiple set of estimated keypoints of a scan
    # Note. For orthographic projection, lens focal length is infinity
    # - Selects and uses valid and confident keypoints
    #assert(estimated_kpts_set[pose3d_est_idx].shape==(N_KPTS,64,3))
    points_3d_prior = np.empty((N_KPTS, 3), dtype=np.float32)
    camera_indices = np.empty(n_observations, dtype=np.int32)
    point_indices = np.empty(n_observations, dtype=np.int32)
    points_2d = np.empty((n_observations, 2), dtype=np.float32)
    points_confidence = np.empty(n_observations, dtype=np.float32)
    kpts_2d_metadata = dict() # dict of lists, for each kpt
    idx = 0
    for est_idx, scan_estimated_kpts in enumerate(estimated_kpts_set):
        n_kpts, n_frames = scan_estimated_kpts.shape[:2]
        valid_kpts_per_frm, conf_beta, x_cardinal_fids, z_cardinal_fids = \
            get_frame_config_constants(n_frames)
        kpts_meta_per_frm = np.zeros((n_kpts,n_frames,3), dtype=np.float32)  # (x,y,conf)

        for kid in range(n_kpts):
            kpt_idx_pool = dict()
            # set kid's camera_indices, point_indices, points_2d, points_confidence, and kpt_meta
            for fid in range(n_frames):
                x, y, conf = scan_estimated_kpts[kid][fid]
                kpts_meta_per_frm[kid, fid, :3] = [x, y, conf]
                conf = round(conf + conf_beta, 4)
                if acceptable_kpts_per_frm[kid][fid]:
                    if N_CAMERAS==16 or n_frames==64: camera_index = fid
                    else: camera_index = (fid*4)%64  # N_CAMERAS==64 and n_frames==16
                    camera_indices[idx] = camera_index
                    point_indices[idx] = kid
                    points_2d[idx] = [x, y]
                    points_confidence[idx] = conf # todo REMOVE***
                    # note kpt grp index for ransac
                    grp_idx = VPNT_GRP_INDEXES[fid]
                    if kpt_idx_pool.get(grp_idx) is None: kpt_idx_pool[grp_idx] = list()
                    kpt_idx_pool[grp_idx].append(idx)
                    idx += 1
            # set kid's 3d point prior
            if est_idx==pose3d_est_idx:
                points_3d_prior[kid] = \
                    derive_3dkpt_prior_from_canrdinal_frms(kid, kpts_meta_per_frm[kid,:,:2],
                                        x_cardinal_fids, z_cardinal_fids, valid_kpts_per_frm)

        kpts_2d_metadata[n_frames] = kpts_meta_per_frm

    #assert(idx==n_observations), 'idx:{}, n_observations:{}'.format(idx, n_observations)
    points_confidence = np.clip(points_confidence, 0, 1)

    #assert(np.all(0<=points_2d[:,0]) and np.all(points_2d[:,0]<=FRM_WDT)), '{}'.format(points_2d)
    #assert(np.all(0<=points_2d[:,1]) and np.all(points_2d[:,1]<=FRM_HGT)), '{}'.format(points_2d)
    #assert(np.all(-HALF_WDT<=points_3d_prior[:,0]) and np.all(points_3d_prior[:,0]<=HALF_WDT))
    #assert(np.all(0<=points_3d_prior[:,1]) and np.all(points_3d_prior[:,1]<=FRM_HGT))
    #assert(np.all(0<=points_confidence) and np.all(points_confidence<=1))
    return points_3d_prior, camera_indices, \
           point_indices, points_2d, points_confidence, kpts_2d_metadata


def organize_per_keypoints(estimated_kpts_set, n_observations,
                           acceptable_kpts_per_frm, pose3d_est_idx=0):
    # from a multiple set of estimated keypoints of a scan
    # Note. For orthographic projection, lens focal length is infinity
    #assert(estimated_kpts_set[pose3d_est_idx].shape==(N_KPTS,64,3))
    points_3d_prior = np.empty((N_KPTS, 3), dtype=np.float32)
    camera_indices = [None]*N_KPTS
    point_indices = [None]*N_KPTS
    points_2d = [None]*N_KPTS
    points_confidence = [None]*N_KPTS
    kpts_2d_metadata = dict() # dict of lists, for each kpt
    kpt_fid_2_index_map = dict()  # dict of dict, kpt->fid->x-input-index
    kpt_indexes_ransac_pool = dict() # dict of dict of lists, for each kpt and viewpoint group

    for est_idx, scan_estimated_kpts in enumerate(estimated_kpts_set):
        n_kpts, n_frames = scan_estimated_kpts.shape[:2]
        #assert(n_kpts==N_KPTS)
        valid_kpts_per_frm, conf_beta, x_cardinal_fids, z_cardinal_fids = \
            get_frame_config_constants(n_frames)
        kpts_meta_per_frm = np.zeros((n_kpts,n_frames,4), dtype=np.float32)  # (x,y,conf,0/1)

        for kid in range(n_kpts):
            kpt_observations = n_observations[kid]
            kpt_camera_indices = np.empty(kpt_observations, dtype=np.int32)
            kpt_point_indices = np.empty(kpt_observations, dtype=np.int32)
            kpt_points_2d = np.empty((kpt_observations, 2), dtype=np.float32)
            kpt_points_confidence = np.empty(kpt_observations, dtype=np.float32)
            kpt_idx_pool = dict()
            kpt_fid_idx_map = dict()
            idx = 0
            # set kid's camera_indices, point_indices, points_2d, points_confidence, and kpt_meta
            for fid in range(n_frames):
                x, y, conf = scan_estimated_kpts[kid][fid]
                kpts_meta_per_frm[kid, fid, :3] = [x, y, conf]
                conf = round(conf + conf_beta, 4)
                if acceptable_kpts_per_frm[kid][fid]:
                    if N_CAMERAS==16 or n_frames==64: camera_index = fid
                    else: camera_index = (fid*4)%64 # N_CAMERAS == 64 and n_frames == 16
                    kpt_camera_indices[idx] = camera_index
                    kpt_point_indices[idx] = kid #point_index
                    kpt_points_2d[idx] = [x, y]
                    kpt_points_confidence[idx] = conf  #todo ***REMOVE
                    #kpts_meta_per_frm[kid, fid, 3] = conf  # >0: implies kpt at frm is used for BA
                    # note kpt grp index for ransac
                    grp_idx = VPNT_GRP_INDEXES[fid]
                    if kpt_idx_pool.get(grp_idx) is None: kpt_idx_pool[grp_idx] = list()
                    kpt_idx_pool[grp_idx].append(idx)
                    kpt_fid_idx_map[fid] = idx
                    idx += 1
            # confirm assertions
            #assert(idx==kpt_observations), 'idx:{}, n_observations:{}'.format(idx, n_observations)
            #assert(np.all(0<=kpt_points_2d[:,0]) and np.all(kpt_points_2d[:,0]<FRM_WDT))
            #assert(np.all(0<=kpt_points_2d[:,1]) and np.all(kpt_points_2d[:,1]<FRM_HGT))
            #assert(np.all(0<=kpt_points_confidence) and np.all(kpt_points_confidence<=1))
            # set kid's 3d point prior
            if est_idx==pose3d_est_idx:
                points_3d_prior[kid] = \
                    derive_3dkpt_prior_from_canrdinal_frms(kid, kpts_meta_per_frm[kid,:,:2],
                                        x_cardinal_fids, z_cardinal_fids, valid_kpts_per_frm)
            # collect into parent data structure
            camera_indices[kid] = kpt_camera_indices
            point_indices[kid] = kpt_point_indices
            points_2d[kid] = kpt_points_2d
            points_confidence[kid] = np.clip(kpt_points_confidence, 0, 1)
            for grp_idx in range(N_VPNT_GRPS):
                kpt_idx_pool[grp_idx] = np.asarray(kpt_idx_pool[grp_idx])  # list-->ndarray
            kpt_indexes_ransac_pool[kid] = kpt_idx_pool
            kpt_fid_2_index_map[kid] = kpt_fid_idx_map

        kpts_2d_metadata[n_frames] = kpts_meta_per_frm

    #assert(np.all(-HALF_WDT<=points_3d_prior[:,0]) and np.all(points_3d_prior[:,0]<=HALF_WDT))
    #assert(np.all(0<=points_3d_prior[:,1]) and np.all(points_3d_prior[:,1]<=FRM_HGT))
    return points_3d_prior, camera_indices, point_indices, points_2d, \
           points_confidence, kpts_2d_metadata, kpt_indexes_ransac_pool, kpt_fid_2_index_map


def orthographic_project(points_3dh, camera_params, n_observations):
    """Convert 3-D points to 2-D by projecting onto images."""
    ext_tfm_matrices = camera_params #.reshape(-1, 3, 4)
    indexes = np.arange(n_observations)
    #assert(n_observations==ext_tfm_matrices.shape[0]), '{}'.format(ext_tfm_matrices.shape)
    cam_kpts_3d = np.dot(ext_tfm_matrices, points_3dh)
    cam_kpts_3d = cam_kpts_3d[indexes, :, indexes]
    points_proj = cam_kpts_3d[:, :2] + [HALF_WDT, 0]
    return points_proj


def get_line_world_pts_from_xpos(frm_xy_pos, fid, n_cameras):
    frm_x, frm_y = frm_xy_pos
    angle_step = 360 / n_cameras
    angle_deg = fid * angle_step
    tfm_mtx, trans_vec = wld2cam_transformation_matrix(360-angle_deg)
    cam_kpt_3dh = np.ones((4, 1), dtype=np.float32)
    cam_kpt_3dh[:3, :] = [[frm_x-HALF_WDT], [frm_y], [0]]  # [[255], [frm_y], [0]]
    wld_kpt_3d = tfm_mtx.dot(cam_kpt_3dh)
    w_x, w_y, w_z = wld_kpt_3d[:, 0]
    #assert(w_y==frm_y), 'w_y:{}, frm_y:{}'.format(w_y, frm_y)
    #assert(-OUTER_RADIUS<=w_x<=OUTER_RADIUS), 'w_x:{}, OUTER_RADIUS:{}'.format(w_x, OUTER_RADIUS)
    #assert((INNER_RADIUS-OUTER_RADIUS)<=w_z<=(INNER_RADIUS+OUTER_RADIUS)), 'w_z:{}'.format(w_z)
    line_pt1 = [w_x, w_z]
    frm_norm_vec = trans_vec + np.array([0, 0, INNER_RADIUS])  # vector normal to the frame
    frm_unit_vec = frm_norm_vec / INNER_RADIUS
    end_point = [w_x, 0, w_y] + (frm_unit_vec * (2*INNER_RADIUS))
    line_pt2 = [end_point[0], end_point[2]]
    return [line_pt1, line_pt2]


def aggregate_3dpose_from_2dposes(kpts_meta_per_frm, n_frames=64):
    """Compute 3dpose by mean aggregation of intersection
    of each keypoint ray projected from all 2d frames
    """
    agg_kpts_3d = np.zeros((N_KPTS, 3), dtype=np.int32)
    valid_kpts_per_frm, __, x_cardinal_fids, z_cardinal_fids = get_frame_config_constants(n_frames)

    for kid in range(N_KPTS):
        y_axis_coords = list()
        xz_intercepts = list()
        intercepts_conf = list()
        for fidA in range(n_frames):
            # investigate whether or not a good kpt (starting with fidA) pairing can be found
            if not valid_kpts_per_frm[kid, fidA]:
                continue  # skip if any keypoint in the pair is invalid
            next = 1
            fidB = (fidA+next)%n_frames
            while next < n_frames and \
                    abs(fidA-fidB)!=n_frames/2 and not valid_kpts_per_frm[kid, fidB]:
                # find next valid neighboring kpt that is not 180" apart
                fidB = (fidA+next)%n_frames
                next += 1
            if next==n_frames: continue  # a proper next kpt was not found

            # found good kpt pairing so project ray through both kpts and find intercept point
            lineA_conf = kpts_meta_per_frm[kid,fidA,2]
            lineB_conf = kpts_meta_per_frm[kid,fidB,2]
            inter_conf = (lineA_conf+lineB_conf) / 2
            lineA_pts = get_line_world_pts_from_xpos(kpts_meta_per_frm[kid,fidA,:2], fidA, n_frames)
            lineB_pts = get_line_world_pts_from_xpos(kpts_meta_per_frm[kid,fidB,:2], fidB, n_frames)
            intercept = get_lines_intersect(lineA_pts, lineB_pts)
            if -INNER_RADIUS<=intercept[0]<=INNER_RADIUS and 0<=intercept[1]<=2*INNER_RADIUS:
                # intersection point of both lines is within inner circle boundary
                # and is therefore an acceptable 3D keypoint location
                xz_intercepts.append(intercept)
                intercepts_conf.append(inter_conf)
                y_aggregate = (kpts_meta_per_frm[kid,fidA,1]+kpts_meta_per_frm[kid,fidA,1])/2
                y_axis_coords.append(y_aggregate)

        if len(xz_intercepts)>0:
            # transform values of intercepts_conf to sum to zero
            intercepts_conf = intercepts_conf / np.sum(intercepts_conf)
            #assert(np.around(np.sum(intercepts_conf), 0)==1), '{}'.format(np.sum(intercepts_conf))
            # Aggregate keypoint's 3D position as the weighted sum of all acceptable candidate
            # 3D keypoint (intersect). Weighted by the (sigmoid-like) confidence of the intercepts
            xz_intercepts = np.asarray(xz_intercepts)
            agg_kpts_3d[kid, 0] = np.sum(xz_intercepts[:, 0]*intercepts_conf)  # w_x
            agg_kpts_3d[kid, 2] = np.sum(xz_intercepts[:, 1]*intercepts_conf)  # w_z
            agg_kpts_3d[kid, 1] = np.sum(y_axis_coords*intercepts_conf)  # w_y

        else:
            # We expect at least one acceptable 3D point. However in the rare but likely case
            # that len(xz_intercepts)==0, we estimate w_x and w_z of kpt from cardinal frames
            agg_kpts_3d[kid] = \
                derive_3dkpt_prior_from_canrdinal_frms(kid, kpts_meta_per_frm[kid,:,:2],
                                    x_cardinal_fids, z_cardinal_fids, valid_kpts_per_frm)

    return agg_kpts_3d


def res_func(params, camera_params, n_points, n_observations,
             camera_indices, point_indices, points_2d, points_wgt):
    """
    Compute residuals.
    `params` contains 3-D coordinates only.
    """
    points_3d = params.reshape((n_points, 3))
    points_3dh = np.append(points_3d, np.ones((n_points, 1)), axis=1)
    # duplicates 3d points and camera parameters to match number of 2d point observations
    dup_points_3d = points_3dh[point_indices]
    dup_points_3d = np.swapaxes(dup_points_3d, 0, 1)
    dup_camera_params = camera_params[camera_indices]
    points_proj = orthographic_project(dup_points_3d, dup_camera_params, n_observations)
    if points_wgt is None: return (points_proj - points_2d).ravel()
    return ((points_proj - points_2d) * points_wgt).ravel()


def euc_func(residuals):
    # If UNIT_MM==True, euclidean distance error is computed in millimeters,
    # otherwise, euclidean distance error is computed in image pixels
    if UNIT_MM:
        residuals[:,0] = residuals[:,0] * (1./512)*1000  # x-Axis: 512 pixels --> 1 meter
        residuals[:,1] = residuals[:,1] * (2.0955/660)*1000  # y-Axis: 660 pixels --> 2.0955 meters
    euc_dist_error = np.sqrt(np.sum(np.square(residuals), axis=-1))
    return euc_dist_error  # shape=(N_KPTS, N_FRAMES)


def res_euc_func(flat_kpts3d, camera_params, n_points, n_observations,
                 camera_indices, point_indices, points_2d, points_wgt):
    '''Compute euclidean distance error from residuals'''
    residual = res_func(flat_kpts3d, camera_params, n_points, n_observations,
                  camera_indices, point_indices, points_2d, points_wgt)
    return euc_func(residual.reshape(-1, 2))


def points_only_bundle_adjustment_sparsity(n_points, point_indices):
    m = point_indices.size * 2
    n = n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(point_indices.size)
    for s in range(3):
        A[2 * i, point_indices * 3 + s] = 1
        A[2 * i + 1, point_indices * 3 + s] = 1
    return A


def get_estimated_keypoints(scanid, df, n_frames):
    kpts_meta = np.zeros((N_KPTS, n_frames, 3), np.float32)
    for fid in range(n_frames):
        column_name = 'Frame{}'.format(fid)
        cell = df.loc[df['scanID'] == scanid, column_name]
        frm_kpts_meta = eval(cell.values[0])  # eval() or ast.literal_eval()
        for kid in range(N_KPTS):
            kpt = KPTS_INDEX_ID[kid]
            kpts_meta[kid, fid, :] = frm_kpts_meta[kpt]
    return kpts_meta


def df_log_projected_keypoints(kpt_wdf, points_3d, kpts_confidence, kpts_error_per_frm,
                               all_kpts_avg_error, select_kpts_avg_error, n_frames=16):
    # log scan ID
    kpt_wdf.at[_df_idx, 'scanID'] = _scanid
    wld_kpt_3dh = np.ones((4, 1), dtype=np.float32)
    angle_step = 360 / n_frames

    # log 2d keypoints metadata in each frame
    for fid in range(n_frames):
        projected_kpts_meta = dict()
        angle_deg = fid * angle_step
        ext_tfm_mtx, __ = wld2cam_transformation_matrix(angle_deg)
        for kid in range(N_KPTS):
            kpt = KPTS_INDEX_ID[kid]
            wld_kpt_3d = points_3d[kid][:, np.newaxis]
            wld_kpt_3dh[:3, :] = wld_kpt_3d
            cam_kpt_3d = ext_tfm_mtx.dot(wld_kpt_3dh)
            c_x, c_y, c_z = np.around(cam_kpt_3d, 0)
            i_x = int(c_x + HALF_WDT)
            i_y = int(c_y)
            err = kpts_error_per_frm[kid][fid]
            conf = kpts_confidence[kid][fid]
            #assert(0<=i_x<=FRM_WDT), 'i_x:{}, FRM_WDT:{}'.format(i_x, FRM_WDT)
            #assert(0<=i_y<=FRM_HGT), 'i_y:{}, FRM_HGT:{}'.format(i_y, FRM_HGT)
            #assert(0<=conf<=1), 'conf:{}'.format(conf)
            projected_kpts_meta[kpt] = (i_x, i_y, conf, err)
        # log 2d keypoints metadata for frame
        column_name = 'Frame{}'.format(fid)
        kpt_wdf.at[_df_idx, column_name] = str(projected_kpts_meta)

    # log 3d keypoint metadata
    keypoints_3d = dict()
    points_3d = np.around(points_3d, 0).astype(np.int32)
    for kid in range(N_KPTS):
        kpt = KPTS_INDEX_ID[kid]
        w_x, w_y, w_z = points_3d[kid]
        a_er = all_kpts_avg_error[kid]
        s_er = select_kpts_avg_error[kid]
        keypoints_3d[kpt] = (w_x, w_y, w_z, a_er, s_er)
    kpt_wdf.at[_df_idx, '3dPose'] = str(keypoints_3d)


def get_2d_keypoints_from_3d_points(points_3d, n_kpts, n_frames=16, verify=True):
    #assert(n_kpts==points_3d.shape[0]), '{}'.format(points_3d.shape)
    wld_kpt_3dh = np.ones((4, 1), dtype=np.float32)
    angle_step = 360 / n_frames
    projected_2d_kpts = np.zeros((n_kpts, n_frames, 2), dtype=np.int32)

    for fid in range(n_frames):
        angle_deg = fid * angle_step
        ext_tfm_mtx, __ = wld2cam_transformation_matrix(angle_deg)
        for k_idx in range(n_kpts):
            wld_kpt_3d = points_3d[k_idx][:, np.newaxis]
            wld_kpt_3dh[:3, :] = wld_kpt_3d
            cam_kpt_3d = ext_tfm_mtx.dot(wld_kpt_3dh)
            c_x, c_y, c_z = np.around(cam_kpt_3d, 0)
            i_x = int(c_x + HALF_WDT)
            i_y = int(c_y)
            #assert(not verify or 0<=i_x<FRM_WDT), 'i_x:{}'.format(i_x)
            #assert(not verify or 0<=i_y<FRM_HGT), 'i_y:{}'.format(i_y)
            projected_2d_kpts[k_idx, fid] = [i_x, i_y]

    return projected_2d_kpts


def back_project_3d_points(points_3d, kpts_2d_metadata, n_frames=64):
    # derive 2d-kpts for each frame from the 3d-point
    projected_2d_kpts = \
        get_2d_keypoints_from_3d_points(points_3d, N_KPTS, n_frames=n_frames, verify=False)
    # Compute euclidean distance from projected kpts to frame 2d-kpts
    error_per_kpt_frm = euc_func(projected_2d_kpts - kpts_2d_metadata[n_frames][:,:,:2])
    all_kpts_avg_error = np.mean(error_per_kpt_frm, axis=-1)
    # shapes: (N_KPTS, N_FRAMES, 2), (N_KPTS, N_FRAMES), (N_KPTS,)
    return projected_2d_kpts, error_per_kpt_frm, all_kpts_avg_error


def toy_problem(verbose=2, n_frames=64, visualize_3dpose=True, show_info=True, show_plot=True):
    pose_3d = {'Nk' : [256, 200, 256],
               'RSh': [156, 200, 256], 'REb': [ 56, 150, 256], 'RWr': [156, 100, 256],
               'LSh': [356, 200, 256], 'LEb': [456, 150, 256], 'LWr': [356, 100, 256],
               'RHp': [200, 400, 256], 'RKe': [192, 500, 256], 'RAk': [182, 650, 256],
               'LHp': [312, 400, 256], 'LKe': [320, 500, 256], 'LAk': [330, 650, 256]}

    n_points = N_KPTS
    n_observations = n_frames * n_points
    points_3d_prior = np.empty((n_points, 3), dtype=np.float32)
    for kid in range(n_points):
        kpt = KPTS_INDEX_ID[kid]
        x, y, z = pose_3d[kpt]
        x = x - HALF_WDT
        points_3d_prior[kid] = [x, y, z]

    print('\n3d x points:\n{}'.format(np.int32(points_3d_prior[:, 0])))
    print('\n3d y points:\n{}'.format(np.int32(points_3d_prior[:, 1])))

    camera_indices, point_indices, points_2d = \
        project_3d_to_2d(points_3d_prior, n_points, N_CAMERAS, n_observations)
    print('\n2d x points:\n{}'.format(np.int32(points_2d[:, 0])))
    print('\n2d y points:\n{}'.format(np.int32(points_2d[:, 1])))

    x0 = points_3d_prior.ravel()
    # Bundle adjustment optimization
    A = points_only_bundle_adjustment_sparsity(n_points, point_indices)
    t0 = time.time()
    res = least_squares(res_func, x0, jac_sparsity=A, verbose=verbose,
                        x_scale='jac', ftol=1e-4, method='trf',
                        args=(CAMERA_PARAMS, n_points, n_observations,
                              camera_indices, point_indices, points_2d))
    t1 = time.time()
    params = res.x
    error = np.sum(np.square(res.fun))
    residual_vec = np.int32(np.around(res.fun.reshape((n_observations, 2)), 0))
    print('\nresiduals at solution:\nx\n{}\ny\n{}\nshape: {}, square sum: {}'
          .format(residual_vec[:, 0], residual_vec[:, 1], residual_vec.shape, error))
    print('\nerror at solution: {}'.format(res.cost))
    vba_points_3d = params.reshape((n_points, 3))
    print('\nalgorithm terminated because, {}'.format(ALGO_STATUS[res.status]))

    if show_info:
        print("n_cameras: {}".format(N_CAMERAS))
        print("n_points: {}".format(n_points))
        n = np.size(CAMERA_PARAMS) + np.size(points_3d_prior)
        m = np.size(points_2d)
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        print('3d points\n', np.int32(np.around(vba_points_3d, 0)))# + [HALF_WDT, 0, 0])
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

    if show_plot:
        # Before adjustment
        f0 = res_func(x0, CAMERA_PARAMS, n_points, n_observations,
                      camera_indices, point_indices, points_2d)
        plot_2d_residuals([f0, res.fun], 'toy-problem', 0, 'Residuals')

    if visualize_3dpose:
        plot_3d_pose([points_3d_prior, vba_points_3d], 'All', [0, 0], 'toy-problem', 0)


def get_plot_config(degree_of_correction, kpts_degree_of_correction, c_idx):
    global _most_correction, _least_correction, _worse_correction
    # Decide whether and how to log keypoint plots and residual/error plots
    log_plots_figures = [False]*(N_KPTS+1)
    log_plots_subdir, log_plots_kpt_tag = ['']*(N_KPTS+1), ['']*(N_KPTS+1)
    for kid in range(N_KPTS+1):
        kpt = KPTS_INDEX_ID.get(kid, '{}')
        if kid < N_KPTS:
            correction = kpts_degree_of_correction[kid]
        else: correction = degree_of_correction

        k_most_correct = _most_correction[c_idx, kid]
        k_least_correct = _least_correction[c_idx, kid]
        k_worse_correct = _worse_correction[c_idx, kid]

        if correction>=0 and np.min(k_most_correct)<correction:
            array_idx = np.argmin(k_most_correct)
            log_plots_figures[kid] = True
            log_plots_kpt_tag[kid] = '{}_{}.png'.format(kpt, array_idx)
            log_plots_subdir[kid] = 'most_correction'
            _most_correction[c_idx, kid, array_idx] = correction
        elif correction>=0 and np.max(k_least_correct)>correction:
            array_idx = np.argmax(k_least_correct)
            log_plots_figures[kid] = True
            log_plots_kpt_tag[kid] = '{}_{}.png'.format(kpt, array_idx)
            log_plots_subdir[kid] = 'least_correction'
            _least_correction[c_idx, kid, array_idx] = correction
        elif correction<0 and np.max(k_worse_correct)>correction:
            array_idx = np.argmax(k_worse_correct)
            log_plots_figures[kid] = True
            log_plots_kpt_tag[kid] = '{}_{}.png'.format(kpt, array_idx)
            log_plots_subdir[kid] = 'worse_correction'
            _worse_correction[c_idx, kid, array_idx] = correction
    return log_plots_figures, log_plots_kpt_tag, log_plots_subdir


def choose_random_observations(kpt_idx_pool, kpt_camera_indices, kpt_points_2d):
    # Choose a random subset of acceptable keypoints in frames according to the multi-viewpoint
    # group constraint (ensuring a min number of keypoints is selected from each viewpoint group)
    kpt_chosen_indexes = list()
    for grp_idx, grp_idx_array in kpt_idx_pool.items():
        n_grp_pnts = len(grp_idx_array)
        max_choice = min(n_grp_pnts, MAX_GRP_CHOICE)
        if max_choice<1: continue
        elif max_choice<MIN_GRP_CHOICE: n_choice = max_choice
        else: n_choice = randint(MIN_GRP_CHOICE, max_choice+1)
        # sample without duplicates
        indices_of_choices = random.sample(range(0, n_grp_pnts), n_choice)
        grp_choice_indexes = grp_idx_array[indices_of_choices]
        kpt_chosen_indexes.extend(grp_choice_indexes.tolist())

    return len(kpt_chosen_indexes), kpt_camera_indices[kpt_chosen_indexes], \
           kpt_points_2d[kpt_chosen_indexes]


def optimize_keypoint_bundle_adjust(x0, kpts_2d_coord, n_points,
                                    n_chosen_observations, chosen_camera_indices, chosen_points_2d,
                                    point_indices=None, points_weight=None, verbose=0):
    global _t_sec_sum, _n_iters
    # Bundle Adjustment (BA.) optimization
    if point_indices is None:
        # when only one (single) keypoint is being optimized
        #assert(n_points==1), 'n_points:{}'.format(n_points)
        point_indices =  np.zeros(n_chosen_observations, dtype=np.int32)
    A = points_only_bundle_adjustment_sparsity(n_points, point_indices)
    t0 = time.time()
    res = least_squares(res_func, x0, jac_sparsity=A, verbose=verbose,
                        x_scale='jac', ftol=1e-4, method='trf',
                        args=(CAMERA_PARAMS, n_points, n_chosen_observations, chosen_camera_indices,
                              point_indices, chosen_points_2d, points_weight))
    t_sec = time.time() - t0
    _t_sec_sum += t_sec
    _n_iters += 1

    # Compute euclidean dist. from projected kpts after BA., and degree of correction
    kpt_point_3d = res.x.reshape((n_points, 3))
    proj_2d_kpts = get_2d_keypoints_from_3d_points(kpt_point_3d, n_points, n_frames=64)
    kpts_err_per_frm = euc_func(proj_2d_kpts - kpts_2d_coord)
    kpts_avg_err = np.mean(kpts_err_per_frm, axis=-1)
    used_kpts_error = euc_func(res.fun.reshape(-1, 2))

    return kpt_point_3d, kpts_err_per_frm, kpts_avg_err, used_kpts_error


def all_kpts_vanilla_bundle_adjustment_3d_optimization(show_info=False):  # v2
    global _used_kpts_mean_error, _t_sec_sum, _n_iters
    # Vanilla Bundle Adjustment
    _t_sec_sum, _n_iters = 0, 0
    # initial estimated keypoints
    estimated_kpts_set = list()
    if ADJUST_A3D:
        a3d_est_kpts = get_estimated_keypoints(_scanid, a3d_df, n_frames=64)
        estimated_kpts_set.append(a3d_est_kpts)
    if ADJUST_APS:
        aps_est_kpts = get_estimated_keypoints(_scanid, aps_df, n_frames=16)
        estimated_kpts_set.append(aps_est_kpts)
    acceptable_kpts_per_frm, n_kpts_observations, n_observations = \
        flag_acceptable_kpt_observations(estimated_kpts_set, 'confident_kpts')
    points_3d_prior, camera_indices, point_indices, \
    points_2d, points_confidence, kpts_2d_metadata = \
        organize_all_scan_keypoints(estimated_kpts_set, n_observations, acceptable_kpts_per_frm)

    #assert(_valid_frame_observations>=n_observations), '{}'.format(_valid_frame_observations)
    n_points = N_KPTS

    # Back-project aggregated 3d-points to 2d-keypoints, and compute errors
    agg_points_3d = aggregate_3dpose_from_2dposes(kpts_2d_metadata[64][:,:,:3])
    p1_proj_2d_kpts, p1_err_per_kpt_frm, p1_all_kpts_avg_err = \
        back_project_3d_points(agg_points_3d, kpts_2d_metadata)
    p1_used_kpts_error = \
        res_euc_func(agg_points_3d.ravel(), CAMERA_PARAMS, n_points, n_observations,
                     camera_indices, point_indices, points_2d, None)
    _used_kpts_mean_error[0, _df_idx] = np.mean(p1_used_kpts_error)

    points_wgt, kpt_indices_list = \
        scale_points_confidence(points_confidence, point_indices, n_points)
    if not KPT_CONF_AS_BA_WGT: points_wgt = None

    # Back-project cardinal frame pose-prior 3d-points to 2d-keypoints, and compute errors
    x0 = points_3d_prior.ravel()
    p2_proj_2d_kpts, p2_err_per_kpt_frm, p2_all_kpts_avg_err = \
        back_project_3d_points(points_3d_prior, kpts_2d_metadata)
    p2_used_kpts_error = \
        res_euc_func(x0, CAMERA_PARAMS, n_points, n_observations,
                     camera_indices, point_indices, points_2d, points_wgt)
    _used_kpts_mean_error[1, _df_idx] = np.mean(p2_used_kpts_error)

    # Vanilla Bundle Adjustment (BA.) optimization
    vba_points_3d, p3_err_per_kpt_frm, p3_all_kpts_avg_err, p3_used_kpts_error = \
        optimize_keypoint_bundle_adjust(x0, kpts_2d_metadata[64][:,:,:2], n_points, n_observations,
                                        camera_indices, points_2d, point_indices, points_wgt)
    _used_kpts_mean_error[2, _df_idx] = np.mean(p3_used_kpts_error)

    if show_info:
        print_ba_optimization_info(None, n_observations, N_CAMERAS, n_points, points_3d_prior,
                                   points_2d, CAMERA_PARAMS, _t_sec_sum/_n_iters)

    return (agg_points_3d, p1_err_per_kpt_frm, p1_all_kpts_avg_err, p1_used_kpts_error), \
           (points_3d_prior, p2_err_per_kpt_frm, p2_all_kpts_avg_err, p2_used_kpts_error), \
           (vba_points_3d, p3_err_per_kpt_frm, p3_all_kpts_avg_err, p3_used_kpts_error), \
           n_observations, acceptable_kpts_per_frm, kpts_2d_metadata


def per_kpt_ransac_bundle_adjustment_3d_optimization(show_info=False):  # v3
    global _used_kpts_mean_error, _t_sec_sum, _n_iters
    # RANSAC Enabled Bundle Adjustment
    _t_sec_sum, _n_iters = 0, 0
    # initial estimated keypoints
    estimated_kpts_set = list()
    if ADJUST_A3D:
        a3d_est_kpts = get_estimated_keypoints(_scanid, a3d_df, n_frames=64)
        estimated_kpts_set.append(a3d_est_kpts)
    if ADJUST_APS:
        aps_est_kpts = get_estimated_keypoints(_scanid, aps_df, n_frames=16)
        estimated_kpts_set.append(aps_est_kpts)
    acceptable_kpts_per_frm, n_kpts_observations, n_observations = \
        flag_acceptable_kpt_observations(estimated_kpts_set, 'inlier_kpts')
    points_3d_prior, camera_indices, point_indices, points_2d, \
    points_confidence, kpts_2d_metadata, kpts_indexes_ransac_pool, kpts_fid_2_index_map = \
        organize_per_keypoints(estimated_kpts_set, n_kpts_observations, acceptable_kpts_per_frm)

    #assert(_valid_frame_observations>=n_observations), '{}'.format(_valid_frame_observations)
    n_points = N_KPTS #points_3d_prior.shape[0]
    ba_points_3d = np.empty((3, N_KPTS, 3), dtype=np.float32)
    p_err_per_kpt_frm = np.empty((3, N_KPTS, 64), dtype=np.float32)
    p_all_kpts_avg_err = np.empty((3, N_KPTS), dtype=np.float32)
    p_used_kpts_error = [list(), list(), list()]
    largest_inlier_kpts = np.zeros(N_FRAMES, dtype=np.bool)
    n_rsc1_observations, n_rsc2_observations = 0, 0

    # RANSAC Optimization per keypoint
    for kid in range(N_KPTS):
        x0 = points_3d_prior[kid].ravel()
        kpt_2d_coord = kpts_2d_metadata[64][kid,:,:2]
        kpt_camera_indices = camera_indices[kid]
        kpt_points_2d = points_2d[kid]
        kpt_observations = n_kpts_observations[kid]
        kpt_fid_idx_map = kpts_fid_2_index_map[kid]
        kpt_ba_x_indexes = kpts_indexes_ransac_pool[kid]

        # Vanilla Bundle Adjustment (BA.) optimization
        ba_points_3d[0, kid], p_err_per_kpt_frm[0, kid], \
        p_all_kpts_avg_err[0, kid], van2_used_kpts_error = \
            optimize_keypoint_bundle_adjust(x0, kpt_2d_coord, 1, kpt_observations,
                                            kpt_camera_indices, kpt_points_2d)
        p_used_kpts_error[0].extend(van2_used_kpts_error)

        rsc1_used_kpts_error = None
        max_inliers, least_error = 0, np.inf
        for iter_idx in range(MAX_RANSAC_ITERS):
            # randomly choose a subset of acceptable observations
            n_chosen_observations, chosen_camera_indices, chosen_points_2d = \
                choose_random_observations(kpt_ba_x_indexes, kpt_camera_indices, kpt_points_2d)
            #assert(MAX_GRP_CHOICE*N_VPNT_GRPS>=n_chosen_observations)

            kpt_point_3d, kpt_err_per_frm, kpts_avg_err, used_kpts_error = \
                optimize_keypoint_bundle_adjust(x0, kpt_2d_coord, 1, n_chosen_observations,
                                                chosen_camera_indices, chosen_points_2d)
            # Compute inlier vs. outlier kpts
            # outliers are defined as keypoints with: euc_error > Q3 + 1.5*IQR (upper-bound)
            q1, q3 = np.quantile(kpt_err_per_frm, [0.25, 0.75])  # compute Q1 & Q3
            iqr = q3 - q1  # compute inter-quartile-range
            outlier_error_thresh = q3 + 1.5*iqr
            is_inlier_kpt = kpt_err_per_frm[0] < outlier_error_thresh
            n_inliers = np.sum(is_inlier_kpt.astype(np.int32))

            # subset of acceptable keypoints that produce the most inliers with least error
            if n_inliers>max_inliers and kpts_avg_err<least_error:
                # RANSAC_v1: Best optimized 3d-pose with most inliers and
                # least avg. euclidean dist. error to all acceptable frame keypoints
                max_inliers, least_error = n_inliers, kpts_avg_err
                ba_points_3d[1, kid] = kpt_point_3d
                p_err_per_kpt_frm[1, kid] = kpt_err_per_frm
                p_all_kpts_avg_err[1, kid] = kpts_avg_err
                rsc1_used_kpts_error = used_kpts_error
                # RANSAC_v2: BA with inlier subset of acceptable keypoints discovered using RANSAC
                largest_inlier_kpts = is_inlier_kpt
                # Force stop RANSAC iteration when the all kpts are discovered to be inliers
                # if n_inliers==kpt_observations: break

        # RANSAC_v1
        n_rsc1_kpt_observations = len(rsc1_used_kpts_error)
        #assert(kpt_observations>=n_rsc1_kpt_observations), '{}'.format(n_rsc1_kpt_observations)
        n_rsc1_observations += n_rsc1_kpt_observations
        p_used_kpts_error[1].extend(rsc1_used_kpts_error)

        # RANSAC_v2: BA optimized 3d-pose with largest subset of
        # acceptable inlier keypoints discovered during RANSAC iterations
        is_inlier_and_acceptable = \
            np.logical_and(largest_inlier_kpts, acceptable_kpts_per_frm[kid])
        n_acceptable_inlier = np.sum(is_inlier_and_acceptable.astype(np.int32))
        inlier_kpt_fids = np.argwhere(is_inlier_and_acceptable).ravel()
        acceptable_inlier_x_indexes = list()
        for fid in inlier_kpt_fids:
            # if inlier fid is acceptable include it's corresponding BA x-input index
            acceptable_inlier_x_indexes.append(kpt_fid_idx_map[fid])
        chosen_points_2d = kpt_points_2d[acceptable_inlier_x_indexes]
        chosen_camera_indices = kpt_camera_indices[acceptable_inlier_x_indexes]
        ba_points_3d[2, kid], p_err_per_kpt_frm[2, kid], \
        p_all_kpts_avg_err[2, kid], rsc2_used_kpts_error = \
            optimize_keypoint_bundle_adjust(x0, kpt_2d_coord, 1, n_acceptable_inlier,
                                            chosen_camera_indices, chosen_points_2d)

        largest_inlier_kpts[:] = False  # reset all to false
        n_rsc2_kpt_observations = len(rsc2_used_kpts_error)
        #assert(kpt_observations>=n_rsc2_kpt_observations), '{}'.format(n_rsc2_kpt_observations)
        n_rsc2_observations += n_rsc2_kpt_observations
        p_used_kpts_error[2].extend(rsc2_used_kpts_error)

    #assert(np.all(n_observations>=[n_rsc1_observations, n_rsc2_observations]))
    for idx in range(3):
        p_used_kpts_error[idx] = np.asarray(p_used_kpts_error[idx])
        _used_kpts_mean_error[3+idx, _df_idx] = np.mean(p_used_kpts_error[idx])

    if show_info:
        print_ba_optimization_info(None, n_observations, N_CAMERAS, n_points, points_3d_prior,
                                   points_2d, CAMERA_PARAMS, _t_sec_sum/_n_iters)

    return (ba_points_3d[0], p_err_per_kpt_frm[0], p_all_kpts_avg_err[0], p_used_kpts_error[0]), \
           (ba_points_3d[1], p_err_per_kpt_frm[1], p_all_kpts_avg_err[1], p_used_kpts_error[1]), \
           (ba_points_3d[2], p_err_per_kpt_frm[2], p_all_kpts_avg_err[2], p_used_kpts_error[2]), \
           (n_observations, n_rsc1_observations, n_rsc2_observations), \
           acceptable_kpts_per_frm, kpts_2d_metadata


def evaluate_and_log_keypoints(subdir, fig_idx, kpt_wdf, p_points_3d, p_all_kpts_avg_errors,
                               p_all_kpts_error_per_frm, p_used_kpts_error_per_frm,
                               acceptable_kpts_per_frm, kpts_conf_per_frm,
                               plot_3dpose=True, plot_residual=True):

    # Compute degree of correction of all keypoints and each keypoint
    degree_of_correction = np.mean(p_used_kpts_error_per_frm[0]) - \
                           np.mean(p_used_kpts_error_per_frm[-1]) # todo: use p_all_kpts_error
    kpts_degree_of_correction = p_all_kpts_avg_errors[0] - p_all_kpts_avg_errors[-1]

    # Compute confidence of each kpt and log kpt meta in dataframe
    select_kpts_error = np.where(acceptable_kpts_per_frm, p_all_kpts_error_per_frm[-1], 0)
    n_select_frms_per_kpt = np.sum(acceptable_kpts_per_frm.astype(np.int32), axis=-1)
    select_kpts_avg_error = np.sum(select_kpts_error, axis=-1) / n_select_frms_per_kpt

    # Log Bundle-Adjusted 2d and 3d keypoints
    df_log_projected_keypoints(kpt_wdf, p_points_3d[-1],
                               np.around(kpts_conf_per_frm, 3),
                               np.around(p_all_kpts_error_per_frm[-1], 1),
                               np.around(p_all_kpts_avg_errors[-1], 2),
                               np.around(select_kpts_avg_error, 2))

    # Plot and save figures is certain criteria are met
    is_pose_plotted = False
    log_plots_figures, log_plots_kpt_tag, log_plots_subdir = \
        get_plot_config(degree_of_correction, kpts_degree_of_correction, fig_idx)

    for k_idx, log_plots in enumerate(log_plots_figures):
        plot_subdir = log_plots_subdir[k_idx]
        fig_filename = log_plots_kpt_tag[k_idx]
        title_tag = fig_filename[0: fig_filename.find('_')]

        if log_plots and plot_residual:
            if k_idx<N_KPTS: # plot errors per frm of a given kpt
                plt_path = os.path.join(plots_dir, subdir, 'kpt_error', plot_subdir, fig_filename)
                valid_pts = valid_kypts_per_64frames()[k_idx]
                confident_kpts = kpts_conf_per_frm[k_idx]>VALID_CONF_THRESH
                selected_fids = acceptable_kpts_per_frm[k_idx]
                title_tag += ' All Frame'
                residual_list = list()
                for pi_err_per_kpt_frm in p_all_kpts_error_per_frm:
                    residual_list.append(pi_err_per_kpt_frm[k_idx])
            else: # plot errors of all kpts in each frame
                valid_pts, confident_kpts, selected_fids = None, None, None
                # First plot error of only kpts used for BA optimization
                title = title_tag.format('Selected')
                fig_name = fig_filename.format('Used')
                plt_path = os.path.join(plots_dir, subdir, 'kpt_error', plot_subdir, fig_name)
                plot_2d_residuals(p_used_kpts_error_per_frm, _scanid, fig_idx, title, plt_path,
                            valid_pts, confident_kpts, selected_fids, 'Residual', display=True)
                # Then plot error of all kpts
                title_tag = title_tag.format('All')
                fig_filename = fig_filename.format('All')
                plt_path = os.path.join(plots_dir, subdir, 'kpt_error', plot_subdir, fig_filename)
                residual_list = list()
                for pi_err_per_kpt_frm in p_all_kpts_error_per_frm:
                    residual_list.append(pi_err_per_kpt_frm.ravel())

            plot_2d_residuals(residual_list, _scanid, fig_idx, title_tag, plt_path,
                              valid_pts, confident_kpts, selected_fids, display=True)

        if log_plots and plot_3dpose:
            plt_path = os.path.join(plots_dir, subdir, '3d_pose', plot_subdir, fig_filename)
            plot_3d_pose(p_points_3d, _scanid, p_all_kpts_avg_errors, wrt_path=plt_path,
                         fig_idx=fig_idx, display=True, new_plot=is_pose_plotted==False)
            is_pose_plotted = True


def iterate_over_scans():
    global _df_idx, _scanid, _valid_frame_observations, _used_kpts_mean_error
    n_scans = a3d_df.shape[0]
    _used_kpts_mean_error = np.zeros((N_POSE3D_TYPES, n_scans), dtype=np.float32)
    _valid_frame_observations = np.sum(np.int32(valid_kypts_per_64frames())) #todo: what if aps?
    n_van1_ba_observations, n_van2_ba_observations = 0, 0
    n_rsc1_ba_observations, n_rsc2_ba_observations = 0, 0
    random.seed(17*n_scans)
    np.random.seed(n_scans)

    for (_df_idx, row) in a3d_df.iterrows():
        #if _df_idx>1: break
        _scanid = row['scanID']
        print('{:>6}. {}'.format(_df_idx+1, _scanid))

        # run vanilla bundle adjustment for scan's keypoints
        (agg_points_3d, p1_err_per_kpt_frm, p1_all_kpts_avg_err, p1_used_kpts_error), \
        (points_3d_prior, p2_err_per_kpt_frm, p2_all_kpts_avg_err, p2_used_kpts_error), \
        (vba1_points_3d, p3_err_per_kpt_frm, p3_all_kpts_avg_err, p3_used_kpts_error), \
        n_vba1_observations, acceptable_kpts_per_frm, \
        kpts_2d_metadata = all_kpts_vanilla_bundle_adjustment_3d_optimization()
        n_van1_ba_observations += n_vba1_observations
        evaluate_and_log_keypoints( 'vanilla_ba', 0, vba_kpt_wdf,
                    [agg_points_3d, points_3d_prior, vba1_points_3d],
                    [p1_all_kpts_avg_err, p2_all_kpts_avg_err, p3_all_kpts_avg_err],
                    [p1_err_per_kpt_frm, p2_err_per_kpt_frm, p3_err_per_kpt_frm],
                    [p1_used_kpts_error, p2_used_kpts_error, p3_used_kpts_error],
                    acceptable_kpts_per_frm, kpts_2d_metadata[64][:,:,2])

        # run ransac enabled bundle adjustment for scan's keypoints
        (vba2_points_3d, p4_err_per_kpt_frm, p4_all_kpts_avg_err, p4_used_kpts_error), \
        (rba1_points_3d, p5_err_per_kpt_frm, p5_all_kpts_avg_err, p5_used_kpts_error), \
        (rba2_points_3d, p6_err_per_kpt_frm, p6_all_kpts_avg_err, p6_used_kpts_error), \
        (n_vba2_observations, n_rba1_observations, n_rba2_observations), acceptable_kpts_per_frm, \
        kpts_2d_metadata = per_kpt_ransac_bundle_adjustment_3d_optimization()
        n_van2_ba_observations += n_vba2_observations
        n_rsc1_ba_observations += n_rba1_observations
        n_rsc2_ba_observations += n_rba2_observations
        evaluate_and_log_keypoints( 'ransac_ba', 1, rba_kpt_wdf,
                    [agg_points_3d, vba1_points_3d, rba1_points_3d,
                     points_3d_prior, vba2_points_3d, rba2_points_3d],
                    [p1_all_kpts_avg_err, p3_all_kpts_avg_err, p5_all_kpts_avg_err,
                     p2_all_kpts_avg_err, p4_all_kpts_avg_err, p6_all_kpts_avg_err],
                    [p1_err_per_kpt_frm, p3_err_per_kpt_frm, p5_err_per_kpt_frm,
                     p2_err_per_kpt_frm, p4_err_per_kpt_frm, p6_err_per_kpt_frm],
                    [p1_used_kpts_error, p3_used_kpts_error, p5_used_kpts_error,
                     p2_used_kpts_error, p4_used_kpts_error, p6_used_kpts_error],
                    acceptable_kpts_per_frm, kpts_2d_metadata[64][:,:,2])

        if (_df_idx+1)%100==0 or (_df_idx+1)==n_scans:
            print('\n{:>4} SCANS PASSED..\n'.format(_df_idx+1))
            vba_kpt_wdf.to_csv(vba_wrt_kpt_csv, encoding='utf-8', index=False)
            rba_kpt_wdf.to_csv(rba_wrt_kpt_csv, encoding='utf-8', index=False)

    plt_path = os.path.join(plots_dir, 'scans_avg_residual.png')
    plot_2d_residuals(_used_kpts_mean_error, 'All-{}-scans'.format(n_scans),
                      0, 'Composite Scans', plt_path, display=True)
    print('\n  {} 2D keypoints used as observations for bundle adjustment'
          '\n\tOf {:,} total 2D keypoints:'
          '\n\tVanilla-1 BA: {:9,} valid & confident 2D keypoints used for BA'
          '\n\tVanilla-2 BA: {:9,} valid & inlier (by conf. score) 2D keypoints used for BA'
          '\n\t RANSAC-1 BA: {:9,} RANSAC chosen subset of valid & inlier (by conf. score) '
          'keypoints that generated largest inliers (by euc. dist.) and least euc. dist.error'
          '\n\t RANSAC-2 BA: {:9,} BA of largest inliers (by euc. dist.) discovered by RANSAC '
          'from the subset of valid & inlier (by conf. score) keypoints'.
          format(kpt_csv_prefix, n_scans*_valid_frame_observations, n_van1_ba_observations,
                 n_van2_ba_observations,  n_rsc1_ba_observations, n_rsc2_ba_observations))


if __name__ == "__main__":
    # Setup and configurations
    kpt_csv_prefix = 'all_sets-w32_256x192-rgb_wfp-opt-ref.x0110v3'
    ba_version = 'bav3'
    ADJUST_APS = False
    ADJUST_A3D = True
    KPT_CONF_AS_BA_WGT = False
    VALID_CONF_THRESH = 0.3
    N_KPTS = 15
    MAX_RANSAC_ITERS = 100

    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}.csv'
    if ADJUST_APS: aps_df = pd.read_csv(path_template.format('aps', kpt_csv_prefix))
    if ADJUST_A3D: a3d_df = pd.read_csv(path_template.format('a3daps', kpt_csv_prefix))
    columns = ['scanID']
    for fid in range(16): columns.append('Frame{}'.format(fid))
    vba_kpt_wdf = pd.DataFrame(columns=columns+['3dPose'])
    rba_kpt_wdf = pd.DataFrame(columns=columns+['3dPose'])
    path_template = '../../../datasets/tsa/{}_images/dataset/hrnet_kpts/{}-{}_{}.csv'
    vba_wrt_kpt_csv = path_template.format('aps', kpt_csv_prefix, 'van', ba_version) #***
    rba_wrt_kpt_csv = path_template.format('aps', kpt_csv_prefix, 'rsc', ba_version) #***
    plots_home_dir = '../../../datasets/tsa/ba_plots/'
    os.makedirs(plots_home_dir, exist_ok=True)
    plots_dir = os.path.join(plots_home_dir, '{}-{}'.format(kpt_csv_prefix, ba_version))

    img_dir_template = '../../../datasets/tsa/{}_images/dataset/{}'
    img_subset_dirs = {'aps':list(), 'a3daps':list()}
    for tsa_ext in img_subset_dirs.keys():
        for subset in ('train_set', 'test_set'):
            img_subset_dirs[tsa_ext].append(os.path.join(img_dir_template.format(tsa_ext, subset)))

    UNIT_MM = True  # True->millimeters, False->pixels
    N_FRAMES = 64 if ADJUST_A3D else 16
    FRM_HGT, FRM_WDT = 660, 512
    HALF_WDT = FRM_WDT / 2
    INNER_RADIUS = FRM_WDT / 2  # radius of inner circle (brown circle in sketch)
    OUTER_RADIUS = np.sqrt(2*(FRM_WDT**2)) / 2  # radius of outer (red) circle (in sketch)
    APS_X_CARDINAL_FIDS = (0, 8)
    APS_Z_CARDINAL_FIDS = (4, 12)
    A3D_X_CARDINAL_FIDS = (0, 32)
    A3D_Z_CARDINAL_FIDS = (16, 48)
    N_CAMERAS = N_FRAMES
    CAMERA_PARAMS =  define_camera_extrinsic_params(N_CAMERAS)
    APS_CONF_BETA = -0.3 # todo ***Remove
    A3D_CONF_BETA = 0
    SAVE_N_PLOTS = 10
    POSE3D_TYPES = ('3dPose-Agg', 'Simple-BA-1', 'Ransac-BA-1',
                    '3dPose-Prior', 'Simple-BA-2', 'Ransac-BA-2')
    POSE3D_COLOR = ('black', 'blueviolet', 'olive', 'brown', 'royalblue', 'gray', )
    N_POSE3D_TYPES = len(POSE3D_TYPES)
    KPTS_INDEX_ID = {0:'Nk',  1:'RSh', 2:'REb', 3:'RWr',  4:'LSh',  5:'LEb',  6:'LWr',
                     7:'RHp', 8:'RKe', 9:'RAk', 10:'LHp', 11:'LKe', 12:'LAk', 13:'MHp', 14:'Hd'}
    KPTS_ID_INDEX = {'Nk' :0,  'RSh':1, 'REb':2, 'RWr':3, 'RHp':7,  'RKe':8,  'RAk':9, 'Hd':14,
                     'MHp':13, 'LSh':4, 'LEb':5, 'LWr':6, 'LHp':10, 'LKe':11, 'LAk':12}
    LIMB_KPTS_PAIRS = {"RShoulder":('Nk','RSh'), "RBicep":('RSh','REb'), "RForearm":('REb','RWr'),
                       "LShoulder":('Nk','LSh'), "LBicep":('LSh','LEb'), "LForearm":('LEb','LWr'),
                       "RAbdomen":('RSh','RHp'), "RThigh":('RHp','RKe'), "RLeg":('RKe','RAk'),
                       "LAbdomen":('LSh','LHp'), "LThigh":('LHp','LKe'), "LLeg":('LKe','LAk'),
                       "RWaist":('MHp','RHp'), "LWaist":('MHp','LHp'),
                       "Spine":('MHp','Nk'), "Skull":('Nk','Hd')}

    POSE_FIG_HGT = 4#.8 # default hgt is 4.8"
    POSE_FIG_WDT = 4#.8 # default wdt is 6.4"
    ALGO_STATUS = {-1: 'improper input parameters status returned from MINPACK.',
                   0 : 'the maximum number of function evaluations is exceeded.',
                   1 : 'gtol termination condition is satisfied.',
                   2 : 'ftol termination condition is satisfied.',
                   3 : 'xtol termination condition is satisfied.',
                   4 : 'Both ftol and xtol termination conditions are satisfied.'}

    # RANSAC variables
    N_VPNT_GRPS = 4  # number of viewpoint(frame) groups
    VPNT_GRP_SIZE = N_FRAMES//N_VPNT_GRPS
    MIN_GRP_CHOICE = VPNT_GRP_SIZE//4
    MAX_GRP_CHOICE = VPNT_GRP_SIZE//2  # note available valid choice may be less
    VPNT_GRP_SFIDS = np.arange(N_FRAMES//8, N_FRAMES, VPNT_GRP_SIZE)  # start-fids
    VPNT_GRP_EFIDS = (VPNT_GRP_SFIDS + VPNT_GRP_SIZE) % N_FRAMES  # end-fids
    VPNT_GRP_INDEXES = np.zeros(N_FRAMES, dtype=np.int32)
    for idx in range(N_FRAMES):
        fid = (VPNT_GRP_SFIDS[0]+idx)%N_FRAMES
        grp_idx = idx//VPNT_GRP_SIZE
        VPNT_GRP_INDEXES[fid] = grp_idx

    # figure log tracker and directories
    _most_correction = np.zeros((2, N_KPTS+1, SAVE_N_PLOTS), dtype=np.float32)
    _worse_correction = np.zeros((2, N_KPTS+1, SAVE_N_PLOTS), dtype=np.float32)
    _least_correction = np.full((2, N_KPTS+1, SAVE_N_PLOTS), fill_value=np.inf, dtype=np.float32)
    for subdir_1 in ('vanilla_ba', 'ransac_ba'):
        directory = os.path.join(plots_dir, subdir_1)
        os.makedirs(directory, exist_ok=True)
        for subdir_2 in ('3d_pose', 'kpt_error'):
            directory = os.path.join(plots_dir, subdir_1, subdir_2)
            os.makedirs(directory, exist_ok=True)
            for subdir_3 in ('most_correction', 'least_correction', 'worse_correction'):
                directory = os.path.join(plots_dir, subdir_1, subdir_2, subdir_3)
                os.makedirs(directory, exist_ok=True)

    # 3D-Pose & Residual plot setup
    TEMP_FIG = os.path.join(plots_dir, 'temp_fig.png')
    POSE_3D_FIG, RESIDUAL_FIG = ['']*2, ['']*2
    SP_FIG, SP_AXS = [None]*2, [None]*2
    for i, (rows, cols) in enumerate([(1, 3), (2, 3)]):
        POSE_3D_FIG[i] = '3d-Pose-{}'.format(i+1)
        SP_FIG[i], SP_AXS[i] = \
            plt.subplots(nrows=rows, ncols=cols, num=POSE_3D_FIG[i],
                         figsize=(cols*POSE_FIG_WDT, rows*POSE_FIG_HGT),
                         subplot_kw={'projection':'3d'})
        SP_FIG[i].subplots_adjust(left=0.0, right=1.0, wspace=-0.05)
        move_figure(SP_FIG[i], 700+(i*1250), (i+1)*25)
        RESIDUAL_FIG[i] = 'Residual-{}'.format(i+1)
        fig = plt.figure(RESIDUAL_FIG[i])
        move_figure(fig, 600+(i*650), 500)

    #toy_problem()
    iterate_over_scans()
    plt.close()
    if os.path.exists(TEMP_FIG): os.remove(TEMP_FIG)


"""
Notes
 1. all_sets-w32_256x192-rgb_nfp-opt-ref.30110v2-bav2.csv
      - Vanilla BA. with initial pose as cardinal pose-prior
      - kpt_csv_prefix:'all_sets-w32_256x192-rgb_nfp-opt-ref.30110v2'
      - ba_version:'bav2', ADJUST_APS:False, ADJUST_A3D:True
      - KPT_CONF_AS_BA_WGT:False, VALID_CONF_THRESH:0.3, N_KPTS:15
      - roughly 2.16M (valid & confident) of 2,197,590 (valid) 2D keypoints were used for vanilla BA
2. all_sets-w32_256x192-rgb_wfp-opt-ref.30110v2-bav2.csv
      - Vanilla BA. with initial pose as cardinal pose-prior 
      - kpt_csv_prefix:'all_sets-w32_256x192-rgb_wfp-opt-ref.30110v2'
      - ba_version:'bav2', ADJUST_APS:False, ADJUST_A3D:True
      - KPT_CONF_AS_BA_WGT:False, VALID_CONF_THRESH:0.3, N_KPTS:15
      - 2,083,567 (valid & confident) of 2,197,590 (valid) 2D keypoints were used for vanilla BA
3. all_sets-w32_256x192-rgb_wfp-opt-ref.x0110v3-bav2.csv
      - Vanilla BA. with initial pose as cardinal pose-prior
      - kpt_csv_prefix:'all_sets-w32_256x192-rgb_wfp-opt-ref.x0110v3'
      - ba_version:'bav2', ADJUST_APS:False, ADJUST_A3D:True
      - KPT_CONF_AS_BA_WGT:False, VALID_CONF_THRESH:0.3, N_KPTS:15
      - 2,083,567 (valid & confident) of 2,197,590 (valid) 2D keypoints were used for vanilla BA
4. all_sets-w32_256x192-rgb_wfp-opt-ref.x0110v3-(rsc/van)_bav3.csv
      - Vanilla-V1 BA. & RANSAC-V2 BA with initial pose as cardinal pose-prior 
      - kpt_csv_prefix:'all_sets-w32_256x192-rgb_wfp-opt-ref.x0110v3'
      - ba_version:'bav3', ADJUST_APS:False, ADJUST_A3D:True
      - KPT_CONF_AS_BA_WGT:False, VALID_CONF_THRESH:0.3, N_KPTS:15
      - Vanilla-1 BA: 2,083,567 valid & confident 2D keypoints used for BA
      - Vanilla-2 BA: 2,147,870 valid & inlier (by conf. score) 2D keypoints used for BA
      - RANSAC-1 BA :   923,330 RANSAC chosen subset of valid & inlier (by conf. score) keypoints
      - RANSAC-2 BA : 2,074,468 BA of largest inliers (by euc. dist.) discovered by RANSAC
"""