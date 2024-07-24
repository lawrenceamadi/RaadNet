'''
    Script implements common, custom stand-alone functions often used in other scripts
'''
##print('\nCommons Script Called\n')
import os
import sys
import h5py
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')
from pose_detection import coco_kpts_zoning as znkp

# OLD CODE
# =======================================================================================
def get_overlap(caxis1, caxis2, taxis1, taxis2):
    # measures the magnitude of overlap on an axis
    if taxis1 <= caxis1 and caxis1 <= taxis2 or taxis1 <= caxis2 and caxis2 <= taxis2:
        # there is partial overlap: box intersect threat region
        if taxis1 <= caxis1 and caxis1 <= taxis2:
            return min(taxis2, caxis2) - caxis1 # must always be positive
        elif taxis1 <= caxis2 and caxis2 <= taxis2:
            return caxis2 - max(taxis1, caxis1)
    elif caxis1 <= taxis1 and caxis2 >= taxis2:
        # there is complete overlap: box covers entire threat region
        return taxis2 - taxis1
    else:
        # there is no overlap: box and threat region do not intersect
        return 0

def get_label(y1, y2, x1, x2, refPtList, areaThresh=0.5):
    # given the threat regions in an image function decides whether
    # a particular region is part of the threat regions
    if refPtList == []:
        return 0

    for region in refPtList:
        xr1 = min(region[0][0], region[1][0])
        xr2 = max(region[0][0], region[1][0])
        yr1 = min(region[0][1], region[1][1])
        yr2 = max(region[0][1], region[1][1])

        threat_area = (xr2 - xr1) * (yr2 - yr1)
        x_overlap = get_overlap(x1, x2, xr1, xr2)
        y_overlap = get_overlap(y1, y2, yr1, yr2)
        overlap_area = x_overlap * y_overlap
        overlap_fraction = overlap_area / threat_area

        if overlap_fraction > areaThresh:
            return 1
    return 0

def model_name(rootdirForModels, modelNamePrefix, startSurfix=0, new=True):
    i = startSurfix
    filename = os.path.join(rootdirForModels, modelNamePrefix + str(i))
    if new:
        while os.path.exists(filename):
            i += 1
            filename = os.path.join(rootdirForModels, modelNamePrefix + str(i))
    return filename

def create_dir(path):
    # create directory if it does not exist
    if os.path.exists(path) == False:
        os.mkdir(path)

def to_unit_vector(vector):
    return vector / np.linalg.norm(vector)

def norm_to_min_1(vector):
    return vector / np.min(vector)

def bin_class_weight(binClassDist, factor=1, decimalPlace=2):
    '''
    Converts binary class distribution to weights such that the class with most counts has a weight of 1
        and the class with the lesser count has weight greater than 1. To balance the distribution
        for example index 0 : clean class count, index 1: threat class count
    :param binClassDist:    array/list containing the count for two classes
    :param decimalPlace:    decimal place to round computation to
    :return:                array/list of the same size as input with class corresponding weights
    '''
    minIndex, maxIndex = (1, 0) if binClassDist[1] <= binClassDist[0] else (0, 1)
    weight = np.ones(shape=(2), dtype=np.float32)
    weight[minIndex] = factor * (binClassDist[maxIndex] / binClassDist[minIndex])
    return np.around(weight, decimalPlace)

def class_distribution(labelVector):
    '''
    Returns count of each unique class in vector
        count at index:i represents count for class:i
    :param labelVector: vector containing class of samples
    :return:            array of length=# of unique classes
    '''
    return np.bincount(labelVector)

def multi_output_weight(labelMatrix, outputNames, defaultHits=1, factor=1, normalize=None):
    '''
    Computes and returns weights to balance imbalance class distribution in dataset
    :param labelMatrix: 2D array of labels for zones, shape: (sample, numOfZones)
    :param outputNames:
    :param defaultHits: value (#) to assign if class has no instance in dataset
    :param normalize:   what type of normalization to apply to computed weight
    :return:
    '''
    perOutputWeight = {} #np.ones(shape=(labelMatrix.shape[1]), dtype=np.float32)
    assert(labelMatrix.shape[1] == len(outputNames))

    for i in range(labelMatrix.shape[1]):
        classDist = class_distribution(labelMatrix[:, i])

        if classDist.shape[0] == 2:
            classWght = bin_class_weight(classDist, factor=factor)
        else:
            classWght = [defaultHits, defaultHits]

        if normalize == 'UNIT':
            classWght = to_unit_vector(classWght)
        elif normalize == 'MIN1':
            classWght = norm_to_min_1(classWght)

        perClassWeight = {0: classWght[0], 1: classWght[1]}
        perOutputWeight[outputNames[i]] = perClassWeight

    return perOutputWeight


# NEW CODE
# =======================================================================================

class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file

    def log_msg(self, msg,):
        with open(self.log_file, 'a+') as log:
            print(msg, file=log)
            log.close()

def order_by_proximity(integer_list, n_frames=16):
    def find_adj_grp(start_num, end_num):
        for i in range(len(group_list)):
            if grp_not_chosen[i]:
                new_grp = group_list[i]
                s_num, e_num = new_grp[0], new_grp[-1]
                if ((start_num - e_num) % n_frames) == 1:
                    grp_not_chosen[i] = False
                    return True, new_grp, True
                elif ((s_num - end_num) % n_frames) == 1:
                    grp_not_chosen[i] = False
                    return True, new_grp, False
        return False, None, None

    def merge_adjacent_groups():
        ordered_list = list()
        for i in range(len(group_list)):
            if grp_not_chosen[i]:
                grp = group_list[i]
                grp_not_chosen[i] = False
                ret = True
                while ret:
                    ret, adj_grp, left = find_adj_grp(grp[0], grp[-1])
                    if ret:
                        if left:
                            adj_grp.extend(grp)
                            grp = adj_grp
                        else: grp.extend(adj_grp)
                ordered_list.extend(grp)
        return ordered_list

    def find_next_num(num):
        left = (num - 1) % n_frames
        right = (num + 1) % n_frames
        for i in range(len(integer_list)):
            if num_not_chosen[i]:
                number = integer_list[i]
                if number == left or number == right:
                    num_not_chosen[i] = False
                    return True, number
        return False, None

    group_list = list()
    num_not_chosen = np.full((len(integer_list)), fill_value=True, dtype=np.bool)
    for i in range(len(integer_list)):
        if num_not_chosen[i]:
            num = integer_list[i]
            num_not_chosen[i] = False
            group = [num]
            ret = True
            while ret:
                ret, next_num = find_next_num(num)
                if ret:
                    group.append(next_num)
                    num = next_num
            group_list.append(group)

    grp_not_chosen = np.full((len(group_list)), fill_value=True, dtype=np.bool)
    return merge_adjacent_groups()

def get_lr_schedule(lr_schedule, lr_epochs):
    assert (len(lr_schedule) == len(lr_epochs))
    scheduling = list() # list of tuples: [(epoch to start, learning rate), ..]
    for i in range(len(lr_schedule)):
        scheduling.append((lr_epochs[i], lr_schedule[i]))
    return scheduling

def scheduler(schedule_key, schedule_value):
    assert (len(schedule_value) == len(schedule_key))
    schedule = dict()
    for i in range(len(schedule_value)):
        schedule[schedule_key[i]] = schedule_value[i]
    return schedule

def df_map_to_dict_map(map_df, zone_name_to_id):
    '''
    organize cell contents of map_df in a dict-of-dict structure zone->fid->kpts
    :param map_df: dataframe of keypoints used to segment/locate body part
    :param zone_name_to_id: dict of zone_name->zone_id
    :return: dict-of-dict
    '''
    zfk_map = dict()
    zones = list(zone_name_to_id.keys())
    for index, row in map_df.iterrows():
        fid_y = row['Fid']  # y or fid
        if index > 0: fid_y = int(fid_y)  # fid
        for zone_name in zones:
            zone_map = zfk_map.get(zone_name, None)
            # create dict for zone if not already present
            if zone_map is None:
                zone_map = dict()
                zfk_map[zone_name] = zone_map
            zone_map[fid_y] = eval(row[zone_name])
    return zfk_map

def frm_zone_kpts_map_type(zfk_map):
    '''
    creates a dict-of-dict indicating whether the zone->frm->kpts map per zone per frame is:
        1. True: (keypoint pair) anchor keypoints used for limb segmentation
        2. False: (pair of keypoint pairs) pillar keypoints used for torso part segmentation
        3. None: (invisible body part in frame) no keypoints
    :param fzk_map: dict-of-dict of frame-zone-keypoints mapping
    :return: dict-of-dict
    '''
    map_type = dict()
    for zone_name, zone_map in zfk_map.items():
        zone_map_type = dict()
        for fid_y, kpts_list in zone_map.items():
            if kpts_list is None:
                zone_map_type[fid_y] = None
                continue

            assert (len(kpts_list) == 2), "pair/pair-of-pair kpts. Not {}".format(len(kpts_list))
            elem_1, elem_2 = kpts_list
            if isinstance(elem_1, list):
                assert (isinstance(elem_2, list)), "second element must be a list too"
                zone_map_type[fid_y] = False  # false: implies pillar keypoints pair-of-pair
            else:
                assert (isinstance(elem_1, str) and isinstance(elem_2, str)), "both must be strings"
                zone_map_type[fid_y] = True  # true: implies anchor keypoint pairs
        map_type[zone_name] = zone_map_type

    return map_type

def df_map_to_valid_zones(map_df, zone_name, n_frames):
    '''
    Collects lists (bool & fids) of valid zones.
        assumes frame id appear in increasing order in dataframe
    :param map_df: dataframe of map
    :param zone_name: name of zone also matching column name in dataframe
    :return: 1. list of boolean, 2. list of valid frames for zone
    '''
    zone_bool_list = list()
    zone_fids_list = list()
    for index, row in map_df.iterrows():
        if index > 0: # skip first index because it is row for 'y' kpts
            assert (0 <= index < n_frames+1) # +1 because first row is for 'y' not fid
            is_valid = False if eval(row[zone_name]) is None else True
            zone_bool_list.append(is_valid)
            if is_valid:
                fid = int(row['Fid'])
                assert (fid == (index-1))
                zone_fids_list.append(fid)

    assert (len(zone_bool_list) == n_frames)
    return zone_bool_list, zone_fids_list

def df_map_to_valid_kpts(map_df, kpt_dict_list):
    # assumes frame id appear in increasing order in dataframe
    # kpt_dict_list: key: kpt_name, value: empty list
    kpts = list(kpt_dict_list.keys())
    columns = list(map_df) # create a list of dataframe columns

    for index, row in map_df.iterrows():
        # search entire row for kpt
        for kpt_name in kpts:
            is_valid = False
            for cid in range(1, len(columns)):
                zone_name = columns[cid]
                kpt_list = eval(row[zone_name])
                if kpt_name in kpt_list:
                    is_valid = True
                    break
            kpt_dict_list[kpt_name] = kpt_dict_list[kpt_name].append(is_valid)
    return kpt_dict_list

def get_grp_ordered_zones(present_zone_names, map_df, ext_format, logger=None):
    grp_zone_frames = dict()
    msg = ''
    for zone_name in present_zone_names:
        n_frames = DATA_NUM_FRAMES[ext_format]
        zone_bool, zone_frames = df_map_to_valid_zones(map_df, zone_name, n_frames)
        zone_frames = order_by_proximity(zone_frames, n_frames=n_frames)
        grp_zone_frames[zone_name] = zone_frames
        msg += '\n{:<5} {:>2} frames: {}'.format(zone_name, len(zone_frames), zone_frames)
    if logger is not None: logger.log_msg(msg)
    print(msg)
    return grp_zone_frames

def aps_to_a3daps_fid(aps_fid):
    '''
    Converts aps-fid to corresponding a3daps-fid
    :param aps_fid: aps frame id
    :return: corresponding a3daps frame id
    '''
    return (aps_fid * 4) % 64 # favored to (aps_fid * 4 - 1) % 64

def replace_entries_with_a3daps_fids(map_df):
    '''
    Replace aps-fids entry in Fid column with corresponding a3daps-fids
    :param map_df: dataframe of keypoints used to segment/locate body part
    :return: map_df with a3daps frame IDs
    '''
    for index, row in map_df.iterrows():
        cell_entry = row['Fid']  # y or fid
        if index > 0:
            aps_fid = int(cell_entry)  # fid
            a3daps_fid = aps_to_a3daps_fid(aps_fid)
            map_df.at[index, 'Fid'] = a3daps_fid
    return map_df

def rename_columns_with_a3daps_fids(kpts_df, col_prefix='Frame'):
    '''
    Renames some columns in dataframe by replacing aps-fid with a3daps-fid
    :param kpts_df: dataframe of keypoints' locations
    :param col_prefix: prefix of columns in dataframe to rename
    :return: dataframe with columns renamed
    '''
    rename_cols = dict()
    s_idx = len(col_prefix)
    for col_name in kpts_df.columns:
        if col_name.find(col_prefix) >= 0:
            aps_fid = int(col_name[s_idx:])
            a3daps_fid = aps_to_a3daps_fid(aps_fid)
            rename_cols[col_name] = '{}{}'.format(col_prefix, a3daps_fid)
    return kpts_df.rename(columns=rename_cols)

def replace_keys_with_a3daps_fids(torso_config):
    for bd_grp, tuple_value in torso_config.items():
        list_value = list(tuple_value)
        for idx, entry in enumerate(tuple_value):
            if isinstance(entry, dict):
                new_dict = dict()
                for aps_fid, value in entry.items():
                    a3daps_fid = aps_to_a3daps_fid(aps_fid)
                    new_dict[a3daps_fid] = value
                list_value[idx] = new_dict
        torso_config[bd_grp] = tuple(list_value)
    return torso_config

def map_aps_frames_2_a3daps(zone_ordered_fids, logger=None):
    msg = '\nMapping aps frame ids to a3daps frame ids'
    for zone_name, aps_frame_ids in zone_ordered_fids.items():
        a3daps_frame_ids = list()
        for aps_fid in aps_frame_ids:
            a3daps_fid = aps_to_a3daps_fid(aps_fid)
            a3daps_frame_ids.append(a3daps_fid)
        zone_ordered_fids[zone_name] = a3daps_frame_ids
        msg += '\n\t{:<4}{:>50}\t-->{:>50}'.\
                format(zone_name, str(aps_frame_ids), str(a3daps_frame_ids))
    if logger is not None: logger.log_msg(msg)
    print(msg)
    return zone_ordered_fids

def duplicate_frames_v1(zone_ordered_frames, ips, logger=None):
    # All zones must have the same number of images per sample
    # Hence, duplicate frames where necessary until equal
    msg = '\nDuplicating zone frames:'
    ordered_zone_frameids = dict()
    for zone_name, unique_zone_fids in zone_ordered_frames.items():
        ###print(zone_name, unique_zone_fids)
        # number of frames must be a multiple of the sequence group size
        assert (len(unique_zone_fids) % 3 == 0) # 3: cfg.MODEL.IMAGES_PER_SEQ_GRP
        assert (len(unique_zone_fids) <= ips) # IMGS_PER_SAMPLE
        # Important to copy list so that changes does not affect original unique_zone_ids list
        zone_frames = unique_zone_fids.copy()
        for i in range(ips - len(zone_frames)):
            zone_frames.append(zone_frames[i])
        ordered_zone_frameids[zone_name] = zone_frames
        msg += '\n\t{:<4}{:>50}\t-->{:>50}'.format(zone_name,
                                                   str(unique_zone_fids), str(zone_frames))
    if logger is not None: logger.log_msg(msg)
    print(msg)
    return ordered_zone_frameids

def duplicate_frames_v2(zone_ordered_frames, ipsmp, ipset, mingap, fid_upbound, logger=None):
    # All zones must have the same number of images per sample
    # Hence, duplicate frames where necessary until equal
    msg = '\nDuplicating zone frames:'
    ordered_zone_frameids = dict()
    for zone_name, unique_zone_fids in zone_ordered_frames.items():
        duplicate_fids = unique_zone_fids
        assert(len(unique_zone_fids) <= ipsmp) # IMGS_PER_SAMPLE
        remainder = ipsmp - len(unique_zone_fids)
        if remainder > 0:
            # separate into groups of neighboring fids
            groups = list()
            neighbors = [unique_zone_fids[0]]
            for i in range(1, len(unique_zone_fids)):
                fid = unique_zone_fids[i]
                lft_fid = (fid - mingap) % fid_upbound
                rgt_fid = (fid + mingap) % fid_upbound
                prev_fid = unique_zone_fids[i - 1]
                #if abs(unique_zone_fids[i - 1] - fid) == mingap:
                if prev_fid == lft_fid or prev_fid == rgt_fid:
                    neighbors.append(fid)
                else:
                    assert (len(neighbors) >= ipset)
                    groups.append(neighbors)
                    neighbors = [fid]
            groups.append(neighbors)

            # todo: sort groups in decreasing order of size.
            #  so that the set with most variability is duplicated most often
            duplicate_fids = list()
            grp_meta = [0] * len(groups)
            remainder = ipsmp
            sid = 0
            while remainder > 0:
                set = groups[sid]
                idx = grp_meta[sid]
                set_size = len(set)
                step = 1 if set_size < (ipset * 2) else ipset
                if idx + ipset <= set_size:
                    duplicate_fids.extend(set[idx: idx + ipset])
                    grp_meta[sid] = idx + step
                    remainder -= ipset
                else:
                    reorder_set = list()
                    for i in range(set_size):
                        fid = set[(i + 1) % set_size]
                        reorder_set.append(fid)
                    groups[sid] = reorder_set
                    grp_meta[sid] = 0
                sid = (sid + 1) % len(groups)

            # verify all unique frames are accounted for
            for fid in unique_zone_fids: assert (fid in duplicate_fids)
            assert (len(duplicate_fids) == ipsmp)

        ordered_zone_frameids[zone_name] = duplicate_fids
        msg += '\n\t{:<4}{:>50}\t-->{:>50}'.format(zone_name,
                                                   str(unique_zone_fids), str(duplicate_fids))
    if logger is not None: logger.log_msg(msg)
    print(msg)
    return ordered_zone_frameids

def map_zones_frameids_to_indexes(zone_unique_ordered_frames):
    # returns dict of dicts. Must be unique frame ids passed zone_ordered_frames
    unique_fid_to_idx = dict()
    for zone_name, unique_zone_fids in zone_unique_ordered_frames.items():
        # number of frames must be a multiple of the sequence group size
        assert(len(unique_zone_fids) <= 12)  # IMGS_PER_SAMPLE
        fid_to_idx = dict()
        for idx, fid in enumerate(unique_zone_fids):
            assert(fid_to_idx.get(fid, -1) < 0) # fid must be unique in list
            fid_to_idx[fid] = idx
        unique_fid_to_idx[zone_name] = fid_to_idx
    return unique_fid_to_idx

def remove_files(directory, substrings):
    dir_contents = os.listdir(directory)
    for content_name in dir_contents:
        for word in substrings:
            if content_name.find(word) >= 0:
                os.remove(os.path.join(directory, content_name))
                break

def monitor_batch_norm(model, tag):
    print('\n{}'.format(tag))
    for layer in model.layers:
        l_name = layer.name

        if l_name.find('fe_time_dist') >= 0:
            print('\n{}'.format(l_name))
            for fe_layer in layer.layer.layers:
                fl_name = fe_layer.name
                if fl_name.find('bn') >= 0 or fl_name.find('BN') > 0:
                    p_wgt = fe_layer.get_weights()
                    print('\t{:<30}\tgamma:{:>23}\trun-mean:{:>23}\tbeta:{:>23}\trun-var:{:>23}'.
                          format(fl_name, p_wgt[0][0], p_wgt[2][0], p_wgt[1][0], p_wgt[3][0]))

        elif l_name.find('grp_seq_time_dist') >= 0:
            print('\n{}'.format(l_name))
            for td_layer in layer.layer.layers:
                tl_name = td_layer.name
                if tl_name.find('batch') >= 0:
                    p_wgt = td_layer.get_weights()
                    print('\t{:<30}\tgamma:{:>23}\trun-mean:{:>23}\tbeta:{:>23}\trun-var:{:>23}'.
                          format(tl_name, p_wgt[0][0], p_wgt[2][0], p_wgt[1][0], p_wgt[3][0]))

def parse_to_hdf5(dict_of_array, file_name):
    hdf5_store = h5py.File(file_name, "a")
    for key, array in dict_of_array.items():
        hdf5_store.create_dataset(key, data=array, compression="gzip")
    return hdf5_store

def zone_index_metadata(zone_names, zone_frames):
    zone_index_meta = dict()
    s_idx = 0
    for i, zone_name in enumerate(zone_names):
        n_zone_imgs = len(zone_frames[zone_name])
        e_idx = s_idx + n_zone_imgs
        zone_index_meta[zone_name] = (i, s_idx, e_idx)
        s_idx = e_idx
    return zone_index_meta

def map_zone_to_group(group_to_zone_dict):
    # given a dict of key: body group, value: tuple of associated body zones
    # produce a dictionary of key: body zone, value: associated body group
    zone_to_group = dict()
    for key, value in group_to_zone_dict.items():
        for zone_name in value: zone_to_group[zone_name] = key
    return zone_to_group

def get_subnet_output_names(subnet_tags, output_tag):
    subnet_out_names = list()
    for tag in subnet_tags:
        output_name = '{}_{}'.format(tag, output_tag)
        subnet_out_names.append(output_name)
    return subnet_out_names

def truncate_strings(string_list, len=3):
    trunc_string_list = list()
    for string in string_list:
        trunc_string_list.append(string[:len])
    return trunc_string_list

def list_from_dict_values(var_dict):
    return list(set(var_dict.values()))

def zone_index_id_map(body_grp_list):
    # derive zones from present groups
    if len(body_grp_list) < 10:
        body_zones = []
        for body_grp in body_grp_list:
            body_zones.extend(BODY_GROUP_ZONES[body_grp])
    else:
        body_zones = list(ZONE_NAME_TO_ID.keys())
        assert (len(body_grp_list) == 10)
        assert (len(body_zones) == 17)
    # map zone ids to indexes
    zone_indx_to_id = dict()
    zone_id_to_indx = dict()
    for idx, zone_name in enumerate(body_zones):
        zone_indx_to_id[idx] = zone_name
        zone_id_to_indx[zone_name] = idx
        #print('{:<4} <--> {:>2}'.format(zone_name, idx))
    # map body zone_idx to body group_idx
    zone_to_grp_indx = dict()
    for z_idx, zone_name in enumerate(body_zones):
        g_idx = body_grp_list.index(BODY_ZONES_GROUP[zone_name])
        assert (g_idx>=0), "{} not in {}".format(BODY_ZONES_GROUP[zone_name], body_grp_list)
        zone_to_grp_indx[z_idx] = g_idx

    return zone_indx_to_id, zone_id_to_indx, zone_to_grp_indx, body_zones

def anomaly_objects_location(scanid, threat_obj_df, n_frames):
    idx = threat_obj_df.index[threat_obj_df['ID'] == scanid].tolist()[0]
    marked = threat_obj_df.at[idx, 'Marked']
    #print(idx, threat_obj_df.at[idx[0], 'Frame1'])
    #print(threat_obj_df.loc[threat_obj_df['ID'] == scanid, 'Frame1'])
    if marked == 2: return None

    scan_anom_obj_loc = list()
    for fid in range(n_frames):
        threat_objs_coords = []
        frame = 'Frame{}'.format(fid)
        #cell = threat_obj_df.loc[threat_obj_df['ID'] == scanid, frame].values[0]
        cell = threat_obj_df.at[idx, frame]
        if cell != "N/M":
            threat_objs_coords = eval(cell)

        # create numpy array to hold threat object bounding boxes
        n_threat_bbs = len(threat_objs_coords)
        if n_threat_bbs > 0:
            threat_bbs = np.empty((n_threat_bbs, 4))
            # parse threat bounding-boxes one by one
            for i, bb_coord in enumerate(threat_objs_coords):
                bb_s_x = min(bb_coord[0][0], bb_coord[1][0])
                bb_e_x = max(bb_coord[0][0], bb_coord[1][0])
                bb_s_y = min(bb_coord[0][1], bb_coord[1][1])
                bb_e_y = max(bb_coord[0][1], bb_coord[1][1])
                threat_bbs[i] = [bb_s_x, bb_e_x, bb_s_y, bb_e_y]
            # add to parent list
            scan_anom_obj_loc.append(threat_bbs)
        else: scan_anom_obj_loc.append(None)
    return scan_anom_obj_loc

# def get_crop_window_dims(present_zones, zone_id_to_indx, bdgrp_crop_config):
#     # Set up tf.data input pipeline
#     zone_crop_dims = np.zeros((len(present_zones), 2), dtype=np.int16)
#     for zone_name in present_zones:
#         grp_name = BODY_ZONES_GROUP[zone_name]
#         zone_type_indx = zone_id_to_indx[zone_name]
#         zone_crop_dims[zone_type_indx] = bdgrp_crop_config[grp_name][:2]
#     return zone_crop_dims

def get_roi_dims(present_zones, zone_name_to_indx, bdgrp_crop_config, scale_dim_f):
    zone_roi_dims = np.zeros((len(present_zones), 2), dtype=np.int32)
    for zone_name in present_zones:
        idx = zone_name_to_indx[zone_name]
        grp_name = BODY_ZONES_GROUP[zone_name]
        roi_dim = bdgrp_crop_config[grp_name][:2] * scale_dim_f
        #roi_dim = np.ceil(BDGRP_CROP_CONFIG[grp_name][:2] * scale_dim_f)
        zone_roi_dims[idx] = roi_dim

    return zone_roi_dims

def smart_crop_window(roi_wdim, nci_wdim,
                      max_xy_shift, rot_angle, zoom_ftr, scale_dim_f, force_sqr):
    # roi_wdim: region of interest, eg. 128x96
    # nci_wdim: network-cropped-image dimension (fixed) for all images passed to model, eg. 160x160
    # return prw_wdim: preloaded region window dimension
    nci_wdt, nci_hgt = nci_wdim
    roi_wdt, roi_hgt = roi_wdim
    assert (nci_hgt>=roi_hgt and nci_wdt>=roi_wdt), '{} vs. {}'.format(nci_wdim, roi_wdim)

    # scale dimensions
    wdt_sf, hgt_sf = scale_dim_f
    nci_hgt = int(nci_hgt * hgt_sf)
    nci_wdt = int(nci_wdt * wdt_sf)
    roi_wdt = int(roi_wdt * wdt_sf)
    roi_hgt = int(roi_hgt * hgt_sf)

    max_x_shift, max_y_shift = max_xy_shift
    angle = np.radians(rot_angle) # convert degree angle to radians
    assert (0<=rot_angle<=45), 'rot_angle:{}'.format(rot_angle)
    assert (zoom_ftr<1), 'zoom_ftr:{}'.format(zoom_ftr)

    if force_sqr and roi_wdt != roi_hgt:
        # resulting win_wdt & win_hgt may not be equal due to difference of x & y shift
        # however, a centered region of (nci_wdt, nci_hgt) will be cropped during image gen.
        nci_wdt = max(nci_wdt, nci_hgt)
        nci_hgt = nci_wdt

    # # 1. compute padded width and length, accommodating shift augmentation
    # pad_hgt = nci_hgt + 2*max_y_shift  # shift by factor of roi_hgt
    # pad_wdt = nci_wdt + 2*max_x_shift  # shift by factor of roi_wdt
    #
    # # 2. adjust crop window dimension, accommodating rotation augmentation
    # win_wdt = pad_wdt*np.cos(angle) + pad_hgt*np.sin(angle)
    # win_hgt = pad_hgt*np.cos(angle) + pad_wdt*np.sin(angle)
    #
    # # 3. adjust crop window dimension, accommodating zoom-in augmentation
    # zoom_in_ftr = 1 - zoom_ftr  # zoom_in_ftr <1 implies scale-down image
    # win_wdt /= zoom_in_ftr
    # win_hgt /= zoom_in_ftr

    # 2. adjust crop window dimension, accommodating rotation augmentation
    win_wdt = nci_wdt*np.cos(angle) + nci_hgt*np.sin(angle)
    win_hgt = nci_hgt*np.cos(angle) + nci_wdt*np.sin(angle)

    # 3. adjust crop window dimension, accommodating zoom-in augmentation
    zoom_in_ftr = 1 - zoom_ftr  # zoom_in_ftr <1 implies scale-down image
    win_wdt /= zoom_in_ftr
    win_hgt /= zoom_in_ftr

    # 1. compute padded width and length, accommodating shift augmentation
    win_wdt += 2*max_x_shift  # shift by factor of roi_wdt
    win_hgt += 2*max_y_shift  # shift by factor of roi_hgt

    return [int(round(win_wdt, 0)), int(round(win_hgt, 0))]

def define_region_windows(present_zones, zone_name_to_indx,
                          bdgrp_crop_config, nci_region_dim, channels,
                          max_xy_shifts, rot_angle, zoom_factor, scale_dim_factor, force_square):
    # define pre-loaded region crop windows
    prw_shapes = dict()  # key: zone_name, value: (H, W, C), eg. (164, 164, 3)
    prw_dims = np.zeros((len(present_zones), 2), dtype=np.int32)
    for i, zone_name in enumerate(present_zones):
        zone_idx = zone_name_to_indx[zone_name]
        grp_name = BODY_ZONES_GROUP[zone_name]
        roi_wdim = bdgrp_crop_config[grp_name][:2]
        z_xy_sft = [max_xy_shifts[0][i], max_xy_shifts[1][i]]
        crop_dim = smart_crop_window(roi_wdim, nci_region_dim, z_xy_sft,
                                     rot_angle, zoom_factor, scale_dim_factor, force_square)
        prw_shapes[zone_name] = (crop_dim[1], crop_dim[0], channels)
        prw_dims[zone_idx] = crop_dim
    return prw_shapes, prw_dims

def adjust_xy_shift(nci_dim, shift_aug_ftr, present_zones, zone_tag_to_idx, zone_roi_dims):
    nci_hgt, nci_wdt = nci_dim
    x_sht_f, y_sht_f = shift_aug_ftr
    assert (0 <= x_sht_f <= 1 and 0 <= y_sht_f <= 1)
    n_zones = len(present_zones)
    max_aug_x_shift = np.zeros((n_zones), dtype=np.int32)
    max_aug_y_shift = np.zeros((n_zones), dtype=np.int32)

    for i, zone_name in enumerate(present_zones):
        # assert that x-y-shift augmentation stays within bounds of net-cropped-image
        idx = zone_tag_to_idx[zone_name]
        roi_wdt, roi_hgt = zone_roi_dims[idx]
        max_x_shift = int(roi_wdt * x_sht_f)
        max_y_shift = int(roi_hgt * y_sht_f)
        room_for_x_shift = (nci_wdt - roi_wdt) // 2
        room_for_y_shift = (nci_hgt - roi_hgt) // 2

        # adjust x-y-shift augmentation to keep RoI within NcI
        # This property is desirable to ensure that RoI does not fall Out-of-Bounds of the
        # Network-Crop-Image during x/y shift augmentation, especially when NO wanderingRoI
        # (ie, RtC & NcI are wandering while RoI stays fixed at location)
        adj_max_x_sft = min(room_for_x_shift, max_x_shift)
        adj_max_y_sft = min(room_for_y_shift, max_y_shift)
        max_aug_x_shift[i] = adj_max_x_sft
        max_aug_y_shift[i] = adj_max_y_sft

    return [max_aug_x_shift, max_aug_y_shift]

# def map_roi_coordinates(present_zones, img_shape, nci_dim, bdgrp_crop_config):
#     img_hgt, img_wdt = img_shape[:2]
#     nci_wdt, nci_hgt = nci_dim
#     wdt_ftr = nci_wdt / img_wdt
#     hgt_ftr = nci_hgt / img_hgt
#     zone_roi_coord = dict()
#     for zone_name in present_zones:
#         grp_name = BODY_ZONES_GROUP[zone_name]
#         grp_wdt, grp_hgt = bdgrp_crop_config[grp_name][:2]
#         wdt_mgn = (nci_wdt - grp_wdt) / (2 * wdt_ftr)
#         x2 = wdt_mgn + (grp_wdt / wdt_ftr)
#         hgt_mgn = (nci_hgt - grp_hgt) / (2 * hgt_ftr)
#         y2 = hgt_mgn + (grp_hgt / hgt_ftr)
#         zone_roi_coord[zone_name] = np.int32([wdt_mgn, x2, hgt_mgn, y2]) # [x1, x2, y1, y2]
#     return zone_roi_coord

def get_clip_values(n_decimals):
    min_clip = 10**-n_decimals
    max_clip = 1 - min_clip
    return (min_clip, max_clip)

def log_bdgrp_seg_config(present_groups, bdgrp_crop_config, logger):
    # log body group crop region configuration
    msg = '\nBDGRP_CROP_CONFIG:'
    for grp_name, window_meta in bdgrp_crop_config.items():
        if grp_name in present_groups:
            msg += '\n\t{:<10}{}'.format(grp_name, window_meta)
    # log body group crop region adjustment
    msg += '\n\nBDGRP_ADJUSTMENT:'
    for grp_name, adjust_meta in BDGRP_ADJUSTMENT.items():
        if grp_name in present_groups:
            msg += '\n\t{:<10}{}'.format(grp_name, adjust_meta)
    logger.log_msg(msg)

def get_pseudo_data_naming_config(cfg, n_unique_ids, d_set, subset_tags=None):
    # data config filename pattern: set-n_scans-image_size-region_dim
    tag0 = 'a3daps' if cfg.DATASET.ROOT.find('a3daps')>=0 else 'aps'
    if d_set=='train': tag1_0 = 'trn'
    elif d_set=='valid': tag1_0 = 'val'
    else: tag1_0 = 'tst'
    tag1_1 = ''
    if subset_tags is not None:
        subset_tags.sort()  # ascending order
        for tag_num in subset_tags: tag1_1 += str(tag_num)
    tag1 = '-{}.{}.{}'.format(tag1_0, tag1_1, n_unique_ids)
    tag2 = '-sclpix' if cfg.DATASET.SCALE_PIXELS else '-orgpix'
    if eval('cfg.{}.AUGMENTATION'.format(d_set.upper())):
        tag3 = '-aug{}{}.{}{}'.format(("%.2f" % cfg.AUGMENT.X_SHIFT).replace('0.','.'),
                ("%.2f" % cfg.AUGMENT.Y_SHIFT).replace('0.','.'), cfg.AUGMENT.ROTATE,
                ("%.2f" % cfg.AUGMENT.S_ZOOM).replace('0.','.'))
    else: tag3 = '-no_aug'
    tag4 = '-tbox' if cfg.LABELS.USE_THREAT_BBOX_ANOTS else ''
    if cfg.MODEL.EXTRA.ROI_TYPE=='oriented':
        tag5 = '-orient{}'.format(cfg.MODEL.EXTRA.POLYNOMIAL_DEGREE)
    else: tag5 = '-{}'.format(cfg.MODEL.EXTRA.ROI_TYPE)
    data_cfg_name = '{}{}{}-n{}-r{}{}{}{}'.format(tag0, tag1, tag2,
                    cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.REGION_DIM[0], tag3, tag4, tag5)
    return data_cfg_name

def cv_display_image(win_name, image, code=0, min_dim=150):
    hgt, wdt = image.shape[:2]
    c = image.shape[2] if len(image.shape)==3 else 1
    if (wdt < min_dim or hgt < min_dim):
        img = np.zeros((min_dim, min_dim, c), dtype=np.uint8)
        s_x, s_y = (min_dim - wdt)//2, (min_dim - hgt)//2
        img[s_y: s_y+hgt, s_x: s_x+wdt] = image
    else: img = image
    cv.imshow(win_name, img) # place new image in window
    key = cv.waitKey(code) # display window and pause by code
    if key == ord('q'): sys.exit(0)

def cv_close_windows(win_name=None):
    if win_name is not None:
        cv.destroyWindow(win_name)
    else: cv.destroyAllWindows()

def plot_image_pixel_histogram(image, ax_sp, title):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax_sp.clear()
    ax_sp.set_title(title)
    ax_sp.plot(cdf_normalized, color='b')
    ax_sp.hist(image.flatten(),256,[0,256], color='r')
    plt.xlim([0,256])
    ax_sp.legend(('cdf','histogram'), loc='upper left')
    #plt.show()
    plt.pause(0.001)

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


# CONSTANTS
BODY_GROUP_ZONES = {'Bicep'  : ['RBp', 'LBp'],
                    'Forearm': ['RFm', 'LFm'],
                    'Chest'  : ['UCh'],
                    'Abs'    : ['RAb', 'LAb'],
                    'UThigh' : ['URTh', 'ULTh'],
                    'Groin'  : ['Gr'],
                    'LThigh' : ['LRTh', 'LLTh'],
                    'Calf'   : ['RCf', 'LCf'],
                    'Ankle'  : ['RAk', 'LAk'],
                    'Back'   : ['UBk']}

BODY_ZONES_GROUP = map_zone_to_group(BODY_GROUP_ZONES)
BODY_GROUPS_LIST = list_from_dict_values(BODY_ZONES_GROUP)

# multiples of 16              wdt,  hgt, x_s_f, y_s_f,
BDGRP_ALIGNED = {'Bicep'  : (  112,   96,     0,   0.1), # t3: (112,112,0,0)
                 'Forearm': (  112,  112,     0,  -0.1), # t3: (112,112,0,0)
                 'Chest'  : (  128,   96,     0,  0.15),
                 'Abs'    : (  112,  112,     0,     0), # t4: (128,112,0,0)        *****
                 'UThigh' : (  112,  112,     0,   0.2), # t2: (128,128,0,0)
                 # Groin is longer than UThigh to include more of inner thigh
                 'Groin'  : (   96,  128,     0,  0.25), # t4: (80,128,0,0.25)      *****
                 'LThigh' : (   96,   96,     0, -0.15), # t1: (112,96,0,0)         "-0.2
                 'Calf'   : (   96,   80,     0,  -0.2), # t2: (96,96,0,-0.1)
                 'Ankle'  : (   96,   80,     0,  -0.2), #                          "-0.3
                 'Back'   : (  128,   96,     0,  0.15)} # t4: (128,96,0,0.1/0.05)  *****

# anchor keypoints per body-zone for cropping oriented (pre-loaded window) bounding-box
# 2 anchor keypoints: pair (common in limbs), use histogram to determine width of bbox
# 4 pillar keypoints: box region with keypoints and crop a section of the boxed region
# +: implies move down/right, -: implies move up/left
X_GR = {15:(  1/4,    0), 0:(  1/8, -1/8), 1:(    0, -1/4),
         7:(  1/4,    0), 8:(  1/8, -1/8), 9:(    0, -1/4)}
#                               wdt,  hgt,  s_x,  e_x,  s_y,  e_y,
BDGRP_ORIENTED = {'Bicep'  : (   80,  112, None, None, None, None),
                  'Forearm': (   80,  112, None, None, None, None),
                  'Chest'  : (  160,   96,    0,    0,    0, -3/5), # top:2/5th of torso
                  'Abs'    : (  160,  112, -1/5,  1/5,  2/5,    0), # lower:3/5th rgt/lft of torso
                  'UThigh' : (  128,  112, None, None, None, None), # top:3/5th of thigh
                  'Groin'  : (  112,  112, None, X_GR,    0, -2/5), # mid:3/4 top:2/5th of thigh
                  'LThigh' : (   96,   96, None, None, None, None), # lower:2/5th of thigh
                  'Calf'   : (   96,   80, None, None, None, None), # top:3/5th of lower leg
                  'Ankle'  : (   96,   80, None, None, None, None), # lower:2/5th of lower leg
                  'Back'   : (  160,   96,    0,    0,    0, -3/5)} # top:2/5th of torso

# vertical adjustment of body part
# +: implies move down, -: implies move up
# keypoint position               top,  bot,  s_y,  e_y,
BDGRP_ADJUSTMENT = {'Bicep'  : (    0,    0,    0,    0),
                    'Forearm': (    0,    0,    0,    0),
                    'Chest'  : ( -1/8, -1/8, None, None), # shoulder-hip torso
                    'Abs'    : ( -1/8, -1/8,  1/8,    0), # shoulder-hip torso & neck-to-midHip limb
                    'UThigh' : ( -1/6,    0,    0, -2/5), # hip-to-knee limb
                    'Groin'  : ( -1/6,    0, None, None), # hip-to-knee limb
                    'LThigh' : ( -1/6,    0,  3/5,    0), # hip-to-knee limb
                    'Calf'   : (    0,    0,    0, -3/7), # knee-to-ankle limb -- was:(0,0,0,-2/5)
                    'Ankle'  : (    0,  1/8,  1/2,    0), # knee-to-ankle limb -- was:(0,0,3/5,0)
                    'Back'   : ( -1/8, -1/8, None, None)} # shoulder-hip torso

# Image augmentation defaults   x_sht  y_sht  rotat  szoom, hflip  p_bgt  n_con
GROUP_AUG_CONFIG = {'Bicep'  : [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'Forearm': [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'Chest'  : [True,  True,  True,  True,  True,  True,  True ], # F:   ,rot
                    'Abs'    : [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'UThigh' : [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'Groin'  : [True,  True,  True,  True,  True,  True,  True ], # F:   ,rot
                    'LThigh' : [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'Calf'   : [True,  True,  True,  True,  True,  True,  True ], # F:hfl
                    'Ankle'  : [True,  True,  True,  True,  True,  True,  True ], # F:hfl,rot,y_s
                    'Back'   : [True,  True,  True,  True,  True,  True,  True ]} # F:rot

ZONE_NAME_TO_ID = {'RBp' : 'Zone1',  # 1
                   'RFm' : 'Zone2',  # 2
                   'LBp' : 'Zone3',  # 3
                   'LFm' : 'Zone4',  # 4
                   'UCh' : 'Zone5',  # 5
                   'RAb' : 'Zone6',  # 6
                   'LAb' : 'Zone7',  # 7
                   'URTh': 'Zone8',  # 8
                   'Gr'  : 'Zone9',  # 9
                   'ULTh': 'Zone10', # 10
                   'LRTh': 'Zone11', # 11
                   'LLTh': 'Zone12', # 12
                   'RCf' : 'Zone13', # 13
                   'LCf' : 'Zone14', # 14
                   'RAk' : 'Zone15', # 15
                   'LAk' : 'Zone16', # 16
                   'UBk' : 'Zone17'} # 17

NO_AUG_DEFAULTS = {'X_SHIFT': 0, 'Y_SHIFT': 0, 'ROTATE': 0,
                   'N_CONTRAST': 0, 'BRIGHTNESS': 0, 'P_CONTRAST': 0}
NETWORK_PREFIX = {'combined':'Comb{}', 'body_zones':'BdZone', 'body_groups':'BdGrps'}
SUBNET_CONFIGS = ['body_zones', 'body_groups']
BG_BORDER_PIXEL = {'aps_BGR': [87, 4, 68], 'a3daps': []}
DATA_NUM_FRAMES = {'aps': 16, 'a3daps': 64}
TSA_PIXELS_RANGE = {'aps':   {'red':(0, 255), 'green':(0, 255), 'blue':(0, 165)},
                    'a3daps':{'red':(0, 255), 'green':(0, 255), 'blue':(0, 165)}}
