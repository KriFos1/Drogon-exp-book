import sys
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as pat
#import matplotlib.collections as mcoll
#from matplotlib.colors import ListedColormap
import pickle
from scipy.interpolate import interp1d
from scipy.io import loadmat
#import datetime as dt
#import cv2
import os
from misc import grdecl
import matplotlib.dates as mdates


# Set paths and find results
path_to_files = '.'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)
seis_data = ['sim2seis', 'bulkimp']
non_scalar = seis_data + ['rft']
date_format = mdates.DateFormatter('%b, %Y')


def plot_prod():
    """
    Plot all production data
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    data_dates = np.genfromtxt('reportdates.csv', delimiter=',', dtype='datetime64[s]')
    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)

    pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
    pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']
    ref_data = []
    if os.path.exists(str(path_to_files) + '/ref_data.p'):
        with open(str(path_to_files) + '/ref_data.p', 'rb') as f:
            ref_data = pickle.load(f)

    # Time_step
    tot_key = [el for el in obs[0].keys() if el not in non_scalar]
    x_days = [data_dates[i] for i in assim_index]
    ne = pred1[0][list(pred1[0].keys())[0]].shape[1]  # get the ensemble size from here

    for k in tot_key:

        # Find a well number
        n = tot_key.index(k)
        my_data = tot_key[n]
        print(my_data)
        t1, t2 = my_data.split(':')

        data_obs = []
        data1 = []
        data2 = []
        ref = []
        for ind, i in enumerate(assim_index):
            data_obs.append(obs[i][my_data])
            data1.append(pred1[i][my_data])
            data2.append(pred2[i][my_data])
            if ref_data:
                if my_data in ref_data[ind].keys():
                    ref.append(ref_data[ind][my_data])
                else:
                    ref.append(None)

        n_d_obs = np.empty(0)
        x_d = np.empty(0)
        n_d1 = np.empty((ne, 0))
        x_d1 = np.empty(0)
        n_d2 = np.empty((ne, 0))
        x_d2 = np.empty(0)
        n_d_ref = np.empty(0)
        x_d_ref = np.empty(0)
        for ind, i in enumerate(assim_index):
            if data_obs[ind] is not None:
                n_d_obs = np.append(n_d_obs, data_obs[ind])
 #               x_d = np.append(x_d, x_days[ind])
            if ref_data and ref[ind] is not None:
                n_d_ref = np.append(n_d_ref, ref[ind])
 #               x_d_ref = np.append(x_d_ref, x_days[ind])
            if data_obs[ind] is not None:
                n_d1 = np.append(n_d1, data1[ind].transpose(), axis=1)
 #               x_d1 = np.append(x_d1, x_days[ind])
            if data_obs[ind] is not None:
                n_d2 = np.append(n_d2, data2[ind].transpose(), axis=1)
 #               x_d2 = np.append(x_d2, x_days[ind])

        f = plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(x_days, np.percentile(n_d1, 90, axis=0), 'k')
        plt.plot(x_days, np.percentile(n_d1, 100, axis=0), ':k')
        plt.plot(x_days, np.percentile(n_d1, 10, axis=0), 'k')
        plt.plot(x_days, np.percentile(n_d1, 0, axis=0), ':k')
        p1 = plt.plot(x_days, n_d_obs, '.r')
        p2 = None
        if ref_data:
            p2 = plt.plot(x_d_ref, n_d_ref, 'g')
        ax1.fill_between(x_days, np.percentile(n_d1, 100, axis=0), np.percentile(n_d1, 0, axis=0), facecolor='lightgrey')
        ax1.fill_between(x_days, np.percentile(n_d1, 90, axis=0), np.percentile(n_d1, 10, axis=0), facecolor='grey')
        p3 = ax1.fill(np.nan, np.nan, 'lightgrey')
        p4 = ax1.fill(np.nan, np.nan, 'grey')
        p5 = plt.plot(x_days, np.mean(n_d1, axis=0), 'orange')
        if p2:
            ax1.legend([(p1[0],), (p2[0],), (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'ref', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 8}, bbox_to_anchor=(1, -0.5), ncol=2)
        else:
            ax1.legend([(p1[0],), (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 8}, bbox_to_anchor=(1, -0.5), ncol=2)
        plt.title(str(t1) + ' initial forcast, at Well: ' + str(t2))
        if t1 == 'WWPR' and t2 == 'A1':
            ylim = [0, 4500]
        elif t1 == 'WWPR' and t2 == 'A2':
            ylim = [0, 2000]
        elif t1 == 'WWPR' and t2 == 'A3':
            ylim = [0, 3000]
        elif t1 == 'WWPR' and t2 == 'A4':
            ylim = [0, 3500]
        elif t1 == 'WOPR' and t2 == 'A1':
            ylim = [0, 5000]
        elif t1 == 'WOPR' and t2 == 'A2':
            ylim = [0, 4500]
        elif t1 == 'WOPR' and t2 == 'A3':
            ylim = [0, 6000]
        elif t1 == 'WOPR' and t2 == 'A4':
            ylim = [0, 4500]
        else:
            ylim = plt.gca().get_ylim()
        ax1.set_ylim(ylim)
        ax1.xaxis.set_major_formatter(date_format)
         # Rotate labels for better fit
        f.autofmt_xdate()
        #plt.xlabel('Days')
        if "WBHP" in my_data:
            plt.ylabel('Bar')
        else:
            plt.ylabel('Sm3/Day')

        ax2 = plt.subplot(2, 1, 2)
        plt.plot(x_days, np.percentile(n_d2, 90, axis=0), 'k')
        plt.plot(x_days, np.percentile(n_d2, 100, axis=0), ':k')
        plt.plot(x_days, np.percentile(n_d2, 10, axis=0), 'k')
        plt.plot(x_days, np.percentile(n_d2, 0, axis=0), ':k')
        plt.plot(x_days, n_d_obs, '.r')
        if ref_data:
            plt.plot(x_d_ref, n_d_ref, 'g')
        ax2.fill_between(x_days, np.percentile(n_d2, 100, axis=0), np.percentile(n_d2, 0, axis=0), facecolor='lightgrey')
        ax2.fill_between(x_days, np.percentile(n_d2, 90, axis=0), np.percentile(n_d2, 10, axis=0), facecolor='grey')
        plt.plot(x_days, np.mean(n_d2, axis=0), 'orange')
        plt.title(str(t1) + ' final forcast, at Well: ' + str(t2))
        f.tight_layout(pad=0.5)
        ax2.set_ylim(ylim)
        ax2.xaxis.set_major_formatter(date_format)
         # Rotate labels for better fit                                                                                                                                                
        f.autofmt_xdate()
        #plt.xlabel('Days')
        if "WBHP" in my_data:
            plt.ylabel('Bar')
        else:
            plt.ylabel('Sm3/Day')
        if save_figure is True:
            plt.savefig(str(path_to_figures) + '/' + str(t2) + '_' + str(t1) + '.png', format='png')

    plt.show()

def plot_sim2seis(f_dim,num_iter=1,layer=1):
    """
    Plot all seismic data
    """
    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    data_dates = np.genfromtxt('reportdates.csv', delimiter=',', dtype='datetime64[s]')
    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)

    pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
    pred2 = np.load(str(path_to_files) + f'/posterior_forecast.npz', allow_pickle=True)['pred_data']

    tot_key = [el for el in obs[0].keys() if el in seis_data]
    x_days = [data_dates[i] for i in assim_index]
    vintage = [(x_days[c],c) for c in [numb for numb,el in enumerate(obs) if el[tot_key[0]] is not None]]

    dim = np.prod(f_dim)

    trans_dict = {}

    def _lookup(kw):
        return trans_dict[kw] if kw in trans_dict else kw

    # Write a quantity to the grid as a grdecl file
    def _write_to_file(name, props): 
        with open(path_to_figures + '/' + name + '.grdecl', 'wb') as fileobj:
            grdecl._write_kw(fileobj, name, props, _lookup, dim)

    
    for date, elem in vintage:
        # observed data
        plt.figure()
        plt.title('Observed seismic data at time: ' + str(date) + ', layer: ' + str(layer))
        plt.imshow(obs[elem][tot_key[0]].reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/observed_seis_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('observed_seis_' + str(date),obs[elem][tot_key[0]])

        #mean
        plt.figure()
        plt.title('Mean prior seismic data at time: ' + str(date) + ', layer: ' + str(layer))
        plt.imshow(pred1[elem][tot_key[0]].mean(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/prior_seis_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('prior_seis_mean' + str(date),pred1[elem][tot_key[0]].mean(axis=1))

        # mean residual
        plt.figure()
        plt.title('Mean prior seismic residual data at time: ' + str(date) + ', layer: ' + str(layer))
        plt.imshow((pred1[elem][tot_key[0]] - obs[elem][tot_key[0]].reshape(-1,1)).mean(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/prior_seis_residual_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('residual_seis_prior' + str(date),(pred1[elem][tot_key[0]] - obs[elem][tot_key[0]].reshape(-1,1)).mean(axis=1))

        #standard deviation
        plt.figure()
        plt.title('Standard deviation of prior seismic data at time: ' + str(date) + ', layer: ' + str(layer))
        plt.imshow(pred1[elem][tot_key[0]].std(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/prior_seis_std_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('prior_seis_std' + str(date),pred1[elem][tot_key[0]].std(axis=1))

        #mean post
        plt.figure()
        plt.title('Mean posterior seismic data at time: ' + str(date) + ', layer: ' + str(layer)) 
        plt.imshow(pred2[elem][tot_key[0]].mean(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/posterior_seis_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('post_seis_mean' + str(date),pred2[elem][tot_key[0]].mean(axis=1))

        # mean residual post
        plt.figure()
        plt.title('Mean posterior seismic residual data at time: ' + str(date) + ', layer: ' + str(layer)) 
        plt.imshow((pred2[elem][tot_key[0]] - obs[elem][tot_key[0]].reshape(-1,1)).mean(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/posterior_seis_residual_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('post_seis_residual' + str(date),(pred2[elem][tot_key[0]] - obs[elem][tot_key[0]].reshape(-1,1)).mean(axis=1))

        #standard deviation post
        plt.figure()
        plt.title('Standard deviation of posterior seismic data at time: ' + str(date) +
            ', layer: ' + str(layer)) 
        plt.imshow(pred2[elem][tot_key[0]].std(axis=1).reshape(f_dim)[layer-1,:,:], cmap='seismic')
        plt.colorbar()
        plt.savefig(str(path_to_figures) + '/posterior_seis_std_' + str(date) + '_layer' + str(layer) + '.png', format='png')
        plt.close()

        _write_to_file('post_seis_std' + str(date),pred2[elem][tot_key[0]].std(axis=1))

    #plt.close('all')
    #


#plot_sim2seis([31,73,46],4,27)
plot_prod()
