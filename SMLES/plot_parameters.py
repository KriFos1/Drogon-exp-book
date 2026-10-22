import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
from scipy.stats import norm
from misc import ecl, grdecl
from scipy.io import loadmat


# Set paths and find results
path_to_files = '.'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)


def plot_layer(field, f_dim, iter=1, layer_no=1):
    """
    Plot parameters in given layer

    Input:
        - field   : string specifying the property
        - f_dim   : dimension of the property (2d or 3d)
        - iter    : plot results at this iteration
        - layer_no: plot for this layer

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    if os.path.exists(str(path_to_files) + '/actnum.npz'):
        actnum = np.load(str(path_to_files) + '/actnum.npz')['actnum']
    else: 
        actnum = np.ones(np.prod(f_dim), dtype=bool)

    # Load debug steps
    field_post = np.zeros(f_dim)
    field_post[:] = np.nan
    field_post_std = np.zeros(f_dim)
    field_post_std[:] = np.nan
    post = np.concatenate([el[field] for el in  np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)['state']],axis=1)
    if 'perm' in field:
        np.clip(post, -5, 8.5, out=post)
        post = np.exp(post)
    elif 'poro' in field:
        np.clip(post, 0.01, 1, out=post)
    field_post[actnum.reshape(f_dim)] = post.mean(1)
    field_post_layer = field_post[layer_no - 1, :, :]
    field_post_std[actnum.reshape(f_dim)] = post.std(axis=1, ddof=1)
    field_post_std_layer = field_post_std[layer_no - 1, :, :]
    max_post_std = np.nanmax(field_post_std_layer)
    min_post_std = np.nanmin(field_post_std_layer)

    # Load Prior field
    prior = np.load(str(path_to_files) + '/prior.npz')[field]
    field_prior = np.zeros(f_dim)
    field_prior_std = np.zeros(f_dim)
    field_prior[:] = np.nan
    field_prior_std[:] = np.nan
    if 'perm' in field:
        np.clip(prior, -5, 8.5, out=prior)
        prior = np.exp(prior)
    elif 'poro' in field:
        np.clip(prior, 0.01, 1, out=prior)
    field_prior[actnum.reshape(f_dim)] = prior.mean(1)
    field_prior_layer = field_prior[layer_no - 1, :, :]
    field_prior_std[actnum.reshape(f_dim)] = prior.std(axis=1, ddof=1)
    field_prior_std_layer = field_prior_std[layer_no - 1, :, :]
    max_prior_std = np.nanmax(field_prior_std_layer)
    min_prior_std = np.nanmin(field_prior_std_layer)

    # Plotting
    if os.path.exists('utm_res.mat'):
        sx = loadmat('utm_res.mat')['sx_res']
        sy = loadmat('utm_res.mat')['sy_res']
    else:
        sy = np.linspace(0, f_dim[1], num=f_dim[1]+1)
        sx = np.linspace(0, f_dim[2], num=f_dim[2]+1)

    # Load wells if present
    wells = None
    if os.path.exists('wells.npz'):
        wells = np.load('wells.npz')['wells']

    plt.figure()
    plt.pcolormesh(sx, sy, field_prior_layer, cmap='jet', shading='flat')
    plt.colorbar()
    if 'perm' in field:
        plt.clim(vmin=0, vmax=5000)
    elif 'poro' in field:
        plt.clim(vmin=0, vmax=.7)
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Prior, ' + field
    filename = str(path_to_figures) + '/' + field + '_prior'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, field_post_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if 'perm' in field:
        plt.clim(vmin=0, vmax=5000)
    elif 'poro' in field:
        plt.clim(vmin=0, vmax=.7)
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior, ' + field
    filename = str(path_to_figures) + '/' + field + '_post'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    field_diff = field_post_layer - field_prior_layer
    plt.pcolormesh(sx, sy, field_diff, cmap='jet', shading='auto')
    plt.colorbar()
    if 'perm' in field:
        plt.clim(vmin=0, vmax=5000)
    elif 'poro' in field:
        plt.clim(vmin=0, vmax=.7)
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior - Prior, ' + field
    filename = str(path_to_figures) + '/' + field + '_diff'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    # std
    np.array([np.minimum(min_prior_std, min_post_std), np.maximum(max_prior_std, max_post_std)])
    plt.figure()
    plt.pcolormesh(sx, sy, field_prior_std_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if 'perm' in field:
        plt.clim(vmin=0, vmax=1000)
    elif 'poro' in field:
        plt.clim(vmin=0.0, vmax=0.12)
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Prior std ' + field
    filename = str(path_to_figures) + '/' + field + '_std_prior'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, field_post_std_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if 'perm' in field:
        plt.clim(vmin=0, vmax=1000)
    elif 'poro' in field:
        plt.clim(vmin=0.0, vmax=0.12)
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior std ' + field
    filename = str(path_to_figures) + '/' + field + '_std_post'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.show()


def plot_avg_field(field, ecl_file, iter=1, trunc=None, max_depth=100):
    """
    Plot averaged parameters

    Input:
        - field    : string specifying the property
        - ecl_file : path to an eclipse init file
        - iter     : plot results at this iteration
        - truc     : when plotting differences, only show values larger than trunc
        - max_depth: compute the average over this depth

    % Copyright (c) 2023 NORCE, All Rights Reserved.

    """

    if trunc is None:
        trunc = {}
    ecl_init = ecl.EclipseInit(ecl_file)
    dz = ecl_init.cell_data('DZ')
    porv = ecl_init.cell_data('PORV')
    f_dim = [ecl_init.nk, ecl_init.nj, ecl_init.ni]

    if os.path.exists(str(path_to_files) + '/actnum.npz'):
        actnum = np.load(str(path_to_files) + '/actnum.npz')['actnum']
    else:
        actnum = np.ones(np.prod(f_dim), dtype=bool)

    # Load Prior field
    if 'swat' in field or 'pres' in field:
        type = field[0:4]
        base_ind = int(field[5])
        monitor_ind = int(field[7])
        data = np.load('dynamic_data_0.npz', allow_pickle=True)
        act = data['act']
        field_prior = data[type][monitor_ind] - data[type][base_ind]
        for member in range(len(act)):
            field_prior[member, ~act[member, :]] = np.nan
        field_prior = np.nanmean(field_prior, axis=0)
    else:
        prior = np.load(str(path_to_files) + '/prior.npz')[field]
        field_prior = np.zeros(f_dim)
        field_prior[:] = np.nan
        if 'perm' in field:
            prior = np.exp(prior)
        elif 'multz' in field:
            prior = norm.cdf(prior) * 2
        field_prior[actnum.reshape(f_dim)] = prior.mean(1)

    # Load debug steps
    if 'swat' in field or 'pres' in field:
        if os.path.exists('dynamic_data.npz'):
            type = field[0:4]
            base_ind = int(field[5])
            monitor_ind = int(field[7])
            data = np.load('dynamic_data.npz', allow_pickle=True)
            act = data['act']
            field_post = data[type][monitor_ind] - data[type][base_ind]
            for member in range(len(act)):
                field_post[member, ~act[member, :]] = np.nan
            field_post = np.nanmean(field_post, axis=0)
        else:
            field_post = np.zeros(field_prior.shape)
    else:
        field_post = np.zeros(f_dim)
        field_post[:] = np.nan
        post = np.concatenate([el[field] for el in  np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)['state']],axis=1)
        if 'perm' in field:
            post = np.exp(post)
        elif 'multz' in field:
            post = norm.cdf(post) * 2
        field_post[actnum.reshape(f_dim)] = post.mean(1)

    # loop over all columns (dz is in shape nz,ny,nx)
    field_post_avg = np.zeros(f_dim[1:])
    field_post_avg[:] = np.nan
    field_prior_avg = np.zeros(f_dim[1:])
    field_prior_avg[:] = np.nan
    for i in range(ecl_init.ni):
        for j in range(ecl_init.nj):
            depth_column = dz[:, j, i].data
            porv_column = porv[:, j, i].data
            depth = np.cumsum(depth_column.data)
            depth_index = np.asarray(depth <= max_depth).nonzero()[0][-1]
            porv_total = np.cumsum(porv_column.data)[depth_index]
            vec_prior = np.multiply(field_prior[0:depth_index, j, i], porv_column[0:depth_index])
            vec_post = np.multiply(field_post[0:depth_index, j, i], porv_column[0:depth_index])
            if depth[depth_index] > 0:
                field_post_avg[j, i] = np.nansum(vec_post) / porv_total
                field_prior_avg[j, i] = np.nansum(vec_prior) / porv_total

    max_post = np.nanmax(field_post_avg)
    min_post = np.nanmin(field_post_avg)
    max_prior = np.nanmax(field_prior_avg)
    min_prior = np.nanmin(field_prior_avg)

    # Plotting
    if os.path.exists('utm_res.mat'):
        sx = loadmat('utm_res.mat')['sx_res']
        sy = loadmat('utm_res.mat')['sy_res']
    else:
        sx = np.linspace(0, f_dim[1], num=f_dim[1])
        sy = np.linspace(0, f_dim[2], num=f_dim[2])
    cl = np.array([np.minimum(min_prior, min_post), np.maximum(max_prior, max_post)])
    prefix = ''
    if 'swat' in field or 'pres' in field:
        prefix = '$\Delta$'

    # Load wells if present
    wells = None
    if os.path.exists('wells.npz'):
        wells = np.load('wells.npz')['wells']

    plt.figure()
    field_prior_avg = field_prior_avg
    im = plt.pcolormesh(sx, sy, field_prior_avg, cmap='jet', shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Prior ' + prefix + field)
    if save_figure is True:
        filename = str(path_to_figures) + '/' + field + '_prior_avg'
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    field_post_avg = field_post_avg
    im = plt.pcolormesh(sx, sy, field_post_avg, cmap='jet', shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Posterior ' + prefix + field)
    if save_figure is True:
        filename = str(path_to_figures) + '/' + field + '_post_avg'
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    data_diff = field_post_avg - field_prior_avg
    if field in trunc.keys():
        data_diff[np.abs(data_diff) < trunc[field]] = np.nan
    im = plt.pcolormesh(sx, sy, data_diff, cmap='seismic', shading='auto')
    cl_value = np.nanmax(np.abs(data_diff))
    cl_diff = np.array([-cl_value, cl_value])
    im.set_clim(cl_diff)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    if field in trunc.keys():
        plt.title('Posterior - Prior, ' + prefix + field + ' (trunc ' + str(trunc[field]) + ')')
    else:
        plt.title('Posterior - Prior, ' + prefix + field)
    if save_figure is True:
        filename = str(path_to_figures) + '/' + field + '_diff_avg'
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.show()


def plot_scalar(param, iter=None):
    """
    Plot scalar parameters

    Input:
        - param   : string spesifying the property
        - iter    : plot results at this iteration

    % Copyright (c) 2023 NORCE, All Rights Reserved.

    """


    # Load debug steps
    post = np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)['state'][()][param]
    post = post.flatten()
    post_mean = np.mean(post)

    # Load Prior field
    prior = np.load(str(path_to_files) + '/prior.npz')[param]
    prior = prior.flatten()
    prior_mean = np.mean(prior)

    # Plotting
    plt.figure()
    plt.hist(prior, 10, density=True, facecolor='b', alpha=0.3, label='Prior')
    plt.hist(post, 10, density=True, facecolor='g', alpha=0.3, label='Posterior')
    ylim = plt.gca().get_ylim()
    plt.plot(prior_mean * np.ones(2), np.array(ylim), 'b')
    plt.plot(post_mean * np.ones(2), np.array(ylim), 'g')
    plt.legend()
    plt.title('Distribution for ' + param)
    if save_figure is True:
        plt.savefig(str(path_to_figures) + '/' + param)

    plt.show()


def export_to_grid(propname):
    """
    Export a property to .grdecl file (for inspection in e.g., ResInsight)

    Input:
        - propname: string spesifying property name

    % Copyright (c) 2023 NORCE, All Rights Reserved.

    """

    # Load posterior property
    with open('posterior_state_estimate.p','rb') as f:
        post = pickle.load(f)[-1][propname]
        
    #np.concatenate([el[field] for el in  np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)['state']],axis=1)
    if 'perm' in propname:
        post = np.exp(post)

    # Load prior property
    prior = np.load(str(path_to_files) + '/prior_4933.npz')[propname]
    if 'perm' in propname:
        prior = np.exp(prior)

    # Active gridcells
    if os.path.exists(str(path_to_files) + '/actnum.npz'):
        actnum = np.load(str(path_to_files) + '/actnum.npz')['actnum']
    else:
        actnum = np.ones(prior.shape[0], dtype=bool)

    # Make the mean property on full grid
    field_post = np.zeros(actnum.shape)
    field_prior = np.zeros(actnum.shape)
    field_post[actnum] = post.mean(1)
    field_prior[actnum] = prior.mean(1)
    dim = len(actnum)

    trans_dict = {}

    def _lookup(kw):
        return trans_dict[kw] if kw in trans_dict else kw

    # Write a quantity to the grid as a grdecl file
    with open(path_to_files + '/prior_mean_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'prior_mean_'+propname, field_prior, _lookup, dim)
    with open(path_to_files + '/posterior_mean_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'posterior_mean_'+propname, field_post, _lookup, dim)

    # Make the std property on full grid                                                                                                                                               
    field_post = np.zeros(actnum.shape)
    field_prior = np.zeros(actnum.shape)
    field_post[actnum] = post.std(1)
    field_prior[actnum] = prior.std(1)
    dim = len(actnum)

    # Write a quantity to the grid as a grdecl file                                                                                                                                     
    with open(path_to_files + '/prior_std_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'prior_std_'+propname, field_prior, _lookup, dim)
    with open(path_to_files + '/posterior_std_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'posterior_std_'+propname, field_post, _lookup, dim)

export_to_grid('permx')
export_to_grid('poro')

#plot_layer('permx', [31,73,46],4,1)
#plot_layer('poro', [31,73,46],4,1)
#plot_layer('satnum', [31,73,46],4,1)
