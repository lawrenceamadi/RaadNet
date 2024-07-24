import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from tf_neural_net.commons import df_map_to_dict_map, ZONE_NAME_TO_ID


def zone_confidence(kpt_tuple, frm_kpts_meta):
    if kpt_tuple is None: return 0, None
    kpts_conf = list()
    for kpt in kpt_tuple:
        conf = frm_kpts_meta[kpt][2]
        assert (0 <= conf <= 1)
        kpts_conf.append(conf)
    return np.mean(np.asarray(kpts_conf)), kpts_conf


def compile_zone_confidence():
    global conf_dfs
    used_kpts_conf = list()
    row_per_zone = dict()
    per_zone_tally = dict()
    n_scans = kpts_df.shape[0]
    for i, row in kpts_df.iterrows():
        scan_id = row['scanID']
        for zone_name in zone_name_list:
            row_per_zone[zone_name] = {'scanID': scan_id}
            per_zone_tally[zone_name] = list()

        for fid in range(n_frames):
            col_name = 'Frame{}'.format(fid)
            frm_znkpt_map = fzk_map[fid] # dict
            frm_kpts_meta = eval(row[col_name])
            for zone_name in zone_name_list:
                zone_kpts = frm_znkpt_map[zone_name]
                zone_conf, kpts_conf = zone_confidence(zone_kpts, frm_kpts_meta)
                row_per_zone[zone_name][col_name] = zone_conf
                if kpts_conf is not None:
                    per_zone_tally[zone_name].append(zone_conf)
                    used_kpts_conf.extend(kpts_conf)

        for zone_name in zone_name_list:
            zone_avg = np.around(np.mean(np.asarray(per_zone_tally[zone_name])), 6)
            row_per_zone[zone_name]['Avg'] = zone_avg
            #print(row_per_zone[zone_name])
            zone_df = conf_dfs[zone_name]
            conf_dfs[zone_name] = zone_df.append(row_per_zone[zone_name], ignore_index=True)

        if (i +  1) % 100 == 0: print('{:>4}/{} scans processed..'.format(i + 1, n_scans))

    # save dataframes
    for zone_name in zone_name_list:
        file_path = os.path.join(csv_dir, '{}.csv'.format(zone_name))
        conf_dfs[zone_name].to_csv(file_path, encoding='utf-8', index=False, sep=',')

    np.save(os.path.join(csv_dir, 'kpts_confidence.npy'), np.asarray(used_kpts_conf))


if __name__ == '__main__':
    global conf_dfs
    n_frames = 16
    # '../../../datasets/tsa/aps_images/dataset/hrnet_kpts/all_sets-w32_...'
    fr_kpt_csv = sys.argv[1]
    kpts_df = pd.read_csv(fr_kpt_csv)
    csv_dir = fr_kpt_csv[0: fr_kpt_csv.find('.csv')]
    os.makedirs(csv_dir, exist_ok=True)
    fzkmap_csv = '../../../datasets/tsa/aps_images/dataset/fid_zones_kpts_map_v1.csv'
    map_df = pd.read_csv(fzkmap_csv)
    fzk_map = df_map_to_dict_map(map_df, ZONE_NAME_TO_ID)
    zone_name_list = list(ZONE_NAME_TO_ID.keys())
    columns = ['scanID']
    for fid in range(n_frames): columns.append('Frame{}'.format(fid))
    columns.append('Avg')
    conf_dfs = dict()
    for zone_name in zone_name_list:
        conf_dfs[zone_name] = pd.DataFrame(columns=columns)

    compile_zone_confidence()