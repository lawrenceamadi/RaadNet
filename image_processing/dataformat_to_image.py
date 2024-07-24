'''
    Convert TSA dataset format (.aps, .a3daps, .a3d, .ahi)
    to images
'''

import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import cv2 as cv
import pandas as pd

sys.path.append('../')
from neural_net import commons as com
from image_processing import transformations as imgtf



# display image animation
def plot_image(path):
    data = read_data(path)
    #matplotlib.image.imsave('name.png', data)
    fig = matplotlib.pyplot.figure(figsize = (13,13))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=200, blit=True)


# save .a3d to video
def save_to_vtk(data, filepath):
    from pyevtk.hl import gridToVTK
    """
    save the 3d data to a .vtk file.
    Parameters
    ------------
    data : 3d np.array 3d matrix that we want to visualize
    filepath : str where to save the vtk model, do not include vtk extension, it does automatically
    """
    x = np.arange(data.shape[0] + 1)
    y = np.arange(data.shape[1] + 1)
    z = np.arange(data.shape[2] + 1)
    gridToVTK(filepath, x, y, z, cellData={'data': data.copy()})


# convert image animation to video: NOT TESTED YET!!!
def dump_to_video(dbpath, video_path):
    data = read_data(dbpath)
    w, h, n = data.shape
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter(video_path, fourcc, 2.0, (w, h))

    for i in range(n):
        img = np.flipud(data[:,:,i].transpose())
        norm = plt.Normalize()
        img = norm(img)
        img = plt.cm.viridis(img)
        img = (255.0 * img).astype(np.uint8)
        out.write(img)

    out.release()


def volumetric_image():
    # animation to preview images
    matplotlib.rc('animation', html='html5')
    readDir = '../../../Passenger-Screening-Challenge/Data/dataformat_samples/'
    scanfile = os.path.join(readDir, '0043db5e8c819bffc15261b1f1ac5e42.a3d')
    # convert animation to video
    dumpfile = os.path.join(readDir, 'visualize', '0043db5e8c819bffc15261b1f1ac5e42')
    imgdata = read_data(scanfile)
    imgdata /= np.max(imgdata)
    save_to_vtk(imgdata, dumpfile)
    #print(imgdata)
    for y in range(imgdata.shape[-1]):
        cv.imshow('cross-section', imgdata[:, :, y])
        if cv.waitKey(10) == ord('q'): sys.exit()
    print('done')


def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h

def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    #print(nx, ny, nt, h['data_scale_factor'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()

    if extension != '.ahi':
        return data
    else:
        return real, imag


def save_images(readScanfile, wrtScanDir):
    # saving image as png file
    com.create_dir(wrtScanDir)
    data = read_data(readScanfile)
    #print(data.shape)
    # read all 16 frames and save each frame as image file
    for i in range(data.shape[-1]):
        img = np.flipud(data[:,:,i].transpose())
        #cv.imshow('img', img)
        #if cv.waitKey(0) == ord('q'): sys.exit()
        matplotlib.pyplot.imsave(os.path.join(wrtScanDir, str(i)+'.png'),img)


def generate_scans(readDir, wrtDir, ext='.aps'):
    listoffiles = os.listdir(readDir)

    # efficiency update 02/15/2018, prevents processing of duplicate files
    currentFile = "id"

    for index, file in enumerate(listoffiles):
        scanid = file[0:file.find(ext)]
        if currentFile != file:
            currentFile = file
            rfile = os.path.join(readDir, file)
            wpath = os.path.join(wrtDir, scanid)
            save_images(rfile, wpath)

        if (index + 1) % 100 == 0:
            print('\t{:>4}/{} processed..'.format(index + 1, len(listoffiles)))

    print ("Done!")

def generate_scans_with_csv(readcsv, readDir, wrtDir, ext='.aps'):
    # compiles and saves scans with gt labelling
    df = pd.read_csv(readcsv)

    # efficiency update 02/15/2018, prevents processing of duplicate files
    currentFile = "id"

    for index, row in df.iterrows():
        scanid = row["scanID"]
        file = scanid + ext  # extracts filename without zone
        if currentFile != file:
            currentFile = file
            rfile = os.path.join(readDir, file)
            wpath = os.path.join(wrtDir, scanid)
            save_images(rfile, wpath)

        if (index + 1) % 100 == 0:
            print('\t{:>4}/1147 processed..'.format(index + 1))

    print ("Done!")

def show_sidebyside(readcsv, apsDir, a3dapsDir):
    # display images from both data formats side by side
    df = pd.read_csv(readcsv)
    imgtf.create_display_window('a3daps', 100, 10, x_size=512, y_size=660)
    imgtf.create_display_window('aps', 100, 530, x_size=512, y_size=660)

    for index, row in df.iterrows():
        scanid = row["scanID"]
        mode = 10

        for i in range(64):
            a3dapsFile = '{}.png'.format(i)
            apsFile = '{}.png'.format(int(i / 4))
            a3dapsPath = os.path.join(a3dapsDir, scanid, a3dapsFile)
            apsPath = os.path.join(apsDir, scanid, apsFile)
            cv.imshow('a3daps', cv.imread(a3dapsPath))
            cv.imshow('aps', cv.imread(apsPath))
            key = cv.waitKey(mode)
            if key == ord('q'): sys.exit()
            if key == ord('p'): mode = 0
            if key == ord('c'): mode = 10



if __name__ == '__main__':
    #readcsv = '../../Data/tsa_psc/dataSetDistribution.csv'
    ds_ext = 'a3daps_images' #***
    readDir = '../../../datasets/tsa/{}/dataset/stage2/'.format(ds_ext)
    #dir1 = '../../../Passenger-Screening-Challenge/Data/aps_images/dataset/set_classified_1147/'
    dir2 = '../../../datasets/tsa/{}/dataset/test_set'.format(ds_ext)
    generate_scans(readDir, dir2, ext='.a3daps')
    #show_sidebyside(readcsv, dir1, dir2)
