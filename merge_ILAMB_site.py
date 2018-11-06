from ILAMB.Variable import Variable
from ILAMB.Confrontation import Confrontation
from ILAMB.ModelResult import ModelResult
import ILAMB.Post as post
import ILAMB.ilamblib as il
from taylorDiagram import plot_Taylor_graph
import numpy as np
import waipy
import matplotlib.pyplot as plt
from PyEMD import EEMD
from hht import hht
from hht import plot_imfs
from hht import plot_frequency
import os
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import os
from netCDF4 import Dataset
from scipy import stats, linalg
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import proj3d
import pandas as pd
import seaborn as sns; sns.set()

col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']

def change_x_tick(obs, tt, site_id, ax0):
    tmask = np.where(obs.mask == False)[0]
    if tmask.size > 0:
        tmin, tmax = tmask[[0, -1]]
    else:
        tmin = 0;
        tmax = obs.time.size - 1
    t = tt[tmin:(tmax + 1)]
    ind = np.where(t % 365 < 30.)[0]
    ticks= t[ind] - (t[ind] % 365)
    nums = int(len(ticks)/6)
    if nums>=1:
        ticks = ticks[::nums]
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)

def Plot_TimeSeries(obs,mod,site_id):
    print('Process on TimeSeries ' + 'No.' + str(site_id) + '!')
    # print(obs.lat[site_id], obs.lon[site_id])
    x = np.ma.masked_invalid(obs.data[:, site_id])
    y = np.ma.masked_invalid(mod.data[:, site_id])
    t = mod.time
    xx = x.compressed()
    yy = y[~x.mask]
    mods = []
    mods.append(yy)
    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('Time series')
    ax0 = fig0.add_subplot(2,1,1)
    fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    ax0.plot(t[~x.mask], xx, 'k')
    ax0.plot(t[~x.mask], yy, '-', label="Model ")
    ax0.set_xlabel('Time')
    ax0.set_ylabel(obs.unit)
    change_x_tick(x, t, site_id, ax0)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0


def Plot_TimeSeries_multiple_mod(obs, mmod, site_id):
    print('Process on TimeSeries ' + 'No.' + str(site_id) + '!')
    # print(obs.lat[site_id], obs.lon[site_id])
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time
    xx = x.compressed()
    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('Time series')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax0.plot(t[~x.mask], xx, 'k-', label='Observed')
    ax0.set_xlabel('Time')
    ax0.set_ylabel(obs.unit)
    change_x_tick(x, t, site_id, ax0)

    mods = []
    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        yy = y[~x.mask]
        mods.append(yy)
        ax0.plot(t[~x.mask], yy, '-', label="Model " + str(i + 1), color=col[i])

    fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0


def Plot_PDF_CDF(obs,mod,site_id):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    t = mod.time
    x = np.ma.masked_invalid(obs.data[:, site_id])
    y = np.ma.masked_invalid(mod.data[:, site_id])

    # xx = x.compressed()
    # yy = y[~x.mask]

    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('PDF and CDF')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    h_obs_sorted = np.ma.sort(y).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), '-', label="Model ")
    ax1.plot(h_obs_sorted, p1_data, '-', label="Model ")

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0



def Plot_PDF_CDF_multiple_mod(obs,mmod,site_id):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time

    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('PDF and CDF')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        h_obs_sorted = np.ma.sort(y).compressed()
        p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
        p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
        x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
        ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[i])
        ax1.plot(h_obs_sorted, p1_data, label="Model "+str(i+1), color=col[i])

    ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0

def Plot_PDF_CDF_multiple_mod_seasonal(obs,mmod,site_id):
    print('Process on PDF&CDF ' + 'No.' + str(site_id) + '!')
    x = np.ma.masked_invalid(obs.data[:, site_id])
    t = obs.time

    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('PDF and CDF')
    ax0 = fig0.add_subplot(2, 1, 1)
    ax1 = fig0.add_subplot(2, 1, 2)

    h_obs_sorted = np.ma.sort(x).compressed()
    p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
    p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
    ax0.plot(x_h, p_h / float(sum(p_h)), 'k-', label='Observed')
    ax1.plot(h_obs_sorted, p1_data, 'k-', label='Observed')

    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        h_obs_sorted = np.ma.sort(y).compressed()
        p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
        p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)))  # bin it into n = N/10 bins
        x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
        ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[i])
        ax1.plot(h_obs_sorted, p1_data, label="Model "+str(i+1), color=col[i])

    ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])

    return fig0

def Plot_Wavelet(obs, obst, mod, modt, site_id, unit):
    print('Process on Wavelet ' + 'No.' + str(site_id) + '!')
    data = np.ma.masked_invalid(obs.data[:, site_id])
    # y = np.ma.masked_invalid(mod.data[:, site_id])
    time_data = obst[~data.mask]
    # print(data.compressed().shape, time_data.shape)
    fig3 = plt.figure(figsize=(6, 6))
    result = waipy.cwt(data.compressed(), 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Obs')
    ax1,ax2,ax3,ax5 = waipy.wavelet_plot('Obs', time_data, data.compressed(), 0.03125, result, fig3, unit=unit)
    change_x_tick(data, modt, site_id, ax2)
    return fig3

def Plot_Wavelet_multiple_mod(obs, obst, site_id, unit, model_name='Obs'):
    print('Process on Wavelet ' + 'No.' + str(site_id) + '!')
    data = np.ma.masked_invalid(obs.data[:, site_id])
    time_data = obst[~data.mask]
    fig3 = plt.figure(figsize=(10, 6))
    result = waipy.cwt(data.compressed(), 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name= model_name)
    ax1, ax2, ax3, ax5 = waipy.wavelet_plot(model_name, time_data, data.compressed(), 0.03125, result, fig3, unit=unit)
    change_x_tick(data, obst, site_id, ax2)
    return fig3


def Plot_IMF(obs, obst, mod, modt, site_id, unit):
    print('Process on IMF ' + 'No.' + str(site_id) + '!')
    data = np.ma.masked_invalid(obs.data[:, site_id])
    time = obst[~data.mask]
    y = np.ma.masked_invalid(mod.data[:, site_id])
    time_y = modt[~y.mask]
    d_mod = []
    d_mod.append(y)
    fig1 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig2 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig3 = plt.figure(figsize=(5, 5))
    fig4 = plt.figure(figsize=(5, 5))

    fig1.subplots_adjust(wspace=0.5, hspace=0.3)
    fig2.subplots_adjust(wspace=0.5, hspace=0.3)

    eemd = EEMD(trials=5)
    imfs = eemd.eemd(data.compressed())
    fig3, freq = hht(data.compressed(), imfs, time, 1, fig3)
    if len(imfs) >= 1:
        fig1 = plot_imfs(data.compressed(), imfs, time_samples=time, fig=fig1, no=1, m=len(d_mod))
        fig2 = plot_frequency(data.compressed(), freq.T, time_samples=time, fig=fig2, no=1, m=len(d_mod))
    imfs2 = eemd.eemd(y.compressed())
    fig4, freq2 = hht(y.compressed(), imfs2, time_y, 1, fig4)
    if len(imfs) >= 1:
        fig1 = plot_imfs(y.compressed(), imfs2, time_samples=time_y, fig=fig1, no=2, m=len(d_mod))
        fig2 = plot_frequency(y.compressed(), freq2.T, time_samples=time_y, fig=fig2, no=2, m=len(d_mod))
    return fig1, fig2, fig3, fig4


def Plot_IMF_multiple_mod(obs, ot, mmod, mt, site_id):
    print('Process on Decomposer_IMF_' + str(site_id) + '!')
    data0 = np.ma.masked_invalid(obs.data[:, site_id])
    data0 = np.ma.masked_where(data0==0.00, data0)

    # time = ot[~data.mask]
    d_mod = []
    for i, mod in enumerate(mmod):
        y = np.ma.masked_invalid(mod.data[:, site_id])
        time_y = mt[~y.mask]
        d_mod.append(y)

    fig0 = plt.figure(figsize=(8, 4))
    fig3 = plt.figure(figsize=(5, 5))
    data = np.ma.masked_invalid(obs.data[:, site_id])
    data = np.ma.masked_where(data==0.00, data)

    time = ot[~data.mask]
    data = data.compressed()
    eemd = EEMD(trials=5)
    print(len(data), len(data0))
    if len(data) > 0:
        imfs = eemd.eemd(data)
        # print('obs',imfs.shape)
        if len(imfs) >= 1:
            ax0 = fig0.add_subplot(1, 2, 1)
            ax0.plot(time, (imfs[len(imfs) - 1]), 'k-', label='Observed')

            # d_d_obs = np.asarray([str(1850 + int(x) / 365) + (
            #     '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x
            #                       in
            #                       time])
            d_d_obs = np.asarray([str(1850 + 1 + int(x) / 365) for x
                                  in
                                  time])
            ax0.xaxis.set_ticks(
                [time[0], time[2 * len(time) / 5],
                 time[4 * len(time) / 5]])
            ax0.set_xticklabels(
                [d_d_obs[0], d_d_obs[2 * len(d_d_obs) / 5],
                 d_d_obs[4 * len(d_d_obs) / 5]])

        ## hht spectrum
        if len(imfs) >= 1:
            fig3, freq = hht(data, imfs, time, 1, fig3, inityear=1850)


    fig1 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig2 = plt.figure(figsize=(4 * (len(d_mod) + 1), 8))
    fig1.subplots_adjust(wspace=0.5, hspace=0.3)
    fig2.subplots_adjust(wspace=0.5, hspace=0.3)

    if len(data) > 0:
        if len(imfs) >= 1:
            fig1 = plot_imfs(data, imfs, time_samples=time, fig=fig1, no=1, m=len(d_mod),inityear=1850)
            fig2 = plot_frequency(data, freq.T, time_samples=time, fig=fig2, no=1, m=len(d_mod),inityear=1850)

        models1 = []
        datamask = []
        data1 = imfs[len(imfs) - 1]
        for m in range(len(d_mod)):
            ## hht spectrum
            eemd = EEMD(trials=5)
            fig4 = plt.figure(figsize=(5, 5))
            data2 = d_mod[m][~data0.mask]
            imfs = eemd.eemd(data2.compressed())
            # print('mod'+str(m), imfs.shape)
            if len(imfs) >= 1:
                fig4, freq = hht(data2.compressed(), imfs, time[~data2.mask], 1, fig4,inityear=1850)

            if len(imfs) >= 1:
                fig1 = plot_imfs(data2.compressed(), imfs, time_samples=time[~data2.mask], fig=fig1, no=m + 2,
                                 m=len(d_mod), inityear=1850)
                fig2 = plot_frequency(data2.compressed(), freq.T, time_samples=time[~data2.mask], fig=fig2, no=m + 2,
                                      m=len(d_mod),inityear=1850)
                ax0.plot(time[~data2.mask], (imfs[len(imfs) - 1]), '-', label='Model' + str(m + 1), c=col[m])
                models1.append(imfs[len(imfs) - 1])
                datamask.append(data2)

        ax0.set_xlabel('Time')
        # ax0.set_ylabel('(' + obs.unit + ')')
        ax0.yaxis.tick_right()
        ax0.yaxis.set_label_position("right")
        ax0.legend(bbox_to_anchor=(-0.05, 1), shadow=False, fontsize='medium')
        plot_Taylor_graph(data1, models1, fig0, 122, datamask=datamask)
    else:
        print("'Data's length is too short !")


    fig0.subplots_adjust(left=0.1, hspace=0.25, wspace=0.55)

    return fig0, fig1, fig2, fig3


def CycleReshape(var, cycle_length=None):  # cycle_length [days]
    if cycle_length == None:
        cycle_length = 1.
    dt = ((var.time_bnds[:, 1] - var.time_bnds[:, 0]).mean())
    spd = int(round(cycle_length / dt))
    if spd <= 0:
        print('The cycle number is smaller than the given data cycle')
    elif spd == 1:
        begin = 0
    else:
        begin = np.argmin(var.time[:(spd - 1)] % spd) # begin starts from 8?
    end = begin + int(var.time[begin:].size / float(spd)) * spd
    shp = (-1, spd) + var.data.shape[1:]  # reshape (-1, spd, site)
    cycle = var.data[begin:end, :].reshape(shp)  # reshape(time, cycle, site)
    # what is the meaning of tbnd? var.time_bnds[begin:end, :].shape  (time, 2)
    tbnd = var.time_bnds[begin:end, :].reshape((-1, spd, 2)) # % cycle_length
    tbnd = tbnd[:, 0, :]
    # tbnd[-1, 1] = 1.
    t = tbnd.mean(axis=1)
    return cycle, t, tbnd  # bounds on time

        # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
def Plot_TimeSeries_cycle(obs, mod, site_id, cycle_length):

    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
    print('Process on Cycle Means ' + 'No.' + str(site_id) + '!')
    x = odata[:, :, site_id]
    y = mdata[:, :, site_id]

    fig0 = plt.figure(figsize=(8, 8))
    plt.suptitle('Cycles means')
    xx = x[:,0]
    yy = y[:,0]

    mask1 = xx.mask | yy.mask
    xx = np.ma.masked_where(mask1, xx)
    yy = np.ma.masked_where(mask1, yy)
    xx = xx[~xx.mask]
    yy = yy[~yy.mask]
    t = ot[~xx.mask]

    # print(xx.shape, yy.shape, t.shape)
    mods = []
    mods.append(yy)
    ax0 = fig0.add_subplot(211)
    ax0.plot(t, xx, 'k')
    ax0.plot(t, yy)
    ax0.set_xlabel('Time')
    ax0.set_ylabel(obs.unit + '(Cycle ' + str(0) + ')')
    # change_x_tick(xx, t, site_id, ax0)
    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    # print(x.shape)
    # for i in range(len(x[:, 0])):
    #     xx = x[i, :].compressed()
    #     yy = y[i, :][~x[i, :].mask]
    #     mods = []
    #     mods.append(yy)
    #     ax0 = fig0.add_subplot(len(x[:, 0]), 2, i*2+1)
    #     num = len(x[:, 0])*100 +2*10+i*2+2
    #     print(num)
    #     fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, num, bbox_to_anchor=(1, 0.45), datamask=None)
    #     ax0.plot(t[~x[i, :].mask], xx, 'k')
    #     ax0.plot(t[~x[i, :].mask], yy)
    #     ax0.set_xlabel('Time')
    #     ax0.set_ylabel(obs.unit+'(Cycle '+str(i)+')')
    #     change_x_tick(mod, site_id, ax0)
    return fig0


def Plot_TimeSeries_cycle_multiple_mod(obs, mmod, site_id, cycle_length):

    print('Process on Cycle Means ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = np.ma.masked_invalid(odata[:, :, site_id])
    t = ot
    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('Cycles means')
    xx = np.mean(x, axis=1)
    ax0 = fig0.add_subplot(211)
    ax0.plot(t, xx, 'k', label="Obs")
    # change_x_tick(xx, t, site_id, ax0)
    mods = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        yy = np.mean(y, axis=1)
        ax0.plot(t, yy, label=" Model "+str(i+1), color=col[i])
        mods.append(yy[~xx.mask])
    ax0.set_xlabel('Time')
    ax0.set_ylabel(obs.unit + '(Cycle ' + str(0) + ')')
    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    fig0, samples0 = plot_Taylor_graph(xx[~xx.mask], mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    # fix x-ticks
    nums = int(len(t)/5)
    ticks = t[::nums] - (t[::nums] % 365)
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0


def Plot_TimeSeries_cycle_multiple_mod_season(obs, mmod, site_id, cycle_length):

    print('Process on Cycle Means ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = np.ma.masked_invalid(odata[:, :, site_id])
    t = ot
    fig0 = plt.figure(figsize=(6, 5))
    plt.suptitle('Cycles means')
    xx = np.mean(x, axis=1)
    ax0 = fig0.add_subplot(111)
    ax0.plot(t, xx, 'k', label="Obs")
    # change_x_tick(xx, t, site_id, ax0)
    mods = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        yy = np.mean(y, axis=1)
        ax0.plot(t, yy, label=" Model "+str(i+1), color=col[i])
        mods.append(yy[~xx.mask])
    ax0.set_xlabel('Time')
    ax0.set_ylabel(obs.unit + '(Cycle ' + str(0) + ')')
    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    # fig0, samples0 = plot_Taylor_graph(xx[~xx.mask], mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    ax0.legend(bbox_to_anchor=(1.3, 0), shadow=False)
    # fix x-ticks
    nums = int(len(t)/5)
    ticks = t[::nums] - (t[::nums] % 365)
    ticklabels = (ticks / 365. + 1850.).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    return fig0


def Plot_TimeSeries_cycle_reshape(obs, mod, site_id, cycle_length):

    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
    print('Process on Cycle Means Reshape ' + 'No.' + str(site_id) + '!')
    t = ot
    x = odata[:, :, site_id]
    y = mdata[:, :, site_id]
    mask1 = x.mask | y.mask
    x = np.ma.masked_where(mask1, x)
    y = np.ma.masked_where(mask1, y)

    xx = x.mean(axis=0)
    yy = y.mean(axis=0)
    print(xx.shape,yy.shape)
    mods = []
    mods.append(yy)
    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('Cycles means')
    ax0 = fig0.add_subplot(2, 1, 1)
    fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)
    ax0.plot(range(len(x.mean(axis=0))), x.mean(axis=0), 'k')
    ax0.fill_between(range(len(x.mean(axis=0))), x.mean(axis=0) - x.std(axis=0),
                     x.mean(axis=0) + x.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     facecolor='gray',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)
    ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0))
    ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                     y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     facecolor='blue',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)
    # ax0.set_xlabel(obs.unit+'(Cycle)')
    ax0.set_ylabel('Mean')
    return fig0



def Plot_TimeSeries_cycle_reshape_multiple_mod(obs, mmod, site_id, cycle_length,xname="Hours of a day"):

    print('Process on Cycle Means Reshape ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = odata[:, :, site_id]
    t = ot

    fig0 = plt.figure(figsize=(6, 9))
    plt.suptitle('Cycles means')
    ax0 = fig0.add_subplot(2, 1, 1)

    # x = np.ma.masked_where(mask1, x)
    ax0.plot(range(len(x.mean(axis=0))), x.mean(axis=0), 'k', label='Obs')
    ax0.fill_between(range(len(x.mean(axis=0))), x.mean(axis=0) - x.std(axis=0),
                     x.mean(axis=0) + x.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     facecolor='gray',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    xx = x.mean(axis=0)
    mods = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        # mask1 = x.mask | y.mask
        # y = np.ma.masked_where(mask1, y)
        yy = y.mean(axis=0)

        mods.append(yy)
        ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod '+str(i+1))
        ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                         y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                         facecolor='blue',
                         linewidth=0.5, linestyle='dashdot', antialiased=True)
    ax0.set_xlabel(xname)
    ax0.set_ylabel('Mean')

    nums = int(len(x.mean(axis=0))/6)
    ticks = np.arange(len(x.mean(axis=0)))
    if nums>=1:
        ticks = ticks[::nums]
    ticklabels = ticks.astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)

    return fig0


def Plot_TimeSeries_cycle_reshape_multiple_mod_season(obs, mmod, site_id, cycle_length,xname="Hours of a day"):

    print('Process on Cycle Means Reshape ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=cycle_length)
    x = odata[:, :, site_id]
    t = ot

    fig0 = plt.figure(figsize=(6, 6))
    plt.suptitle('Cycles means')
    ax0 = fig0.add_subplot(1, 1, 1)

    # x = np.ma.masked_where(mask1, x)
    ax0.plot(range(len(x.mean(axis=0))), x.mean(axis=0), 'k', label='Obs')
    ax0.fill_between(range(len(x.mean(axis=0))), x.mean(axis=0) - x.std(axis=0),
                     x.mean(axis=0) + x.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                     facecolor='gray',
                     linewidth=0.5, linestyle='dashdot', antialiased=True)

    xx = x.mean(axis=0)
    mods = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=cycle_length)
        y = mdata[:, :, site_id]
        # mask1 = x.mask | y.mask
        # y = np.ma.masked_where(mask1, y)
        yy = y.mean(axis=0)

        mods.append(yy)
        ax0.plot(range(len(y.mean(axis=0))), y.mean(axis=0), label='Mod '+str(i+1))
        ax0.fill_between(range(len(y.mean(axis=0))), y.mean(axis=0) - y.std(axis=0),
                         y.mean(axis=0) + y.std(axis=0), alpha=0.2, edgecolor='#1B2ACC',
                         facecolor='blue',
                         linewidth=0.5, linestyle='dashdot', antialiased=True)
    ax0.set_xlabel(xname)
    ax0.set_ylabel('Mean')

    nums = int(len(x.mean(axis=0))/6)
    ticks = np.arange(len(x.mean(axis=0)))
    if nums>=1:
        ticks = ticks[::nums]
    ticklabels = ticks.astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)

    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
    # fig0, samples0 = plot_Taylor_graph(xx, mods, fig0, 212, bbox_to_anchor=(1, 0.45), datamask=None)

    return fig0


def GetSeasonMask(obs, mod, site_id, season_kind):
    print('This function is used to mask seasons!')
    # The input data is annual data
    odata, ot, otb = CycleReshape(obs, cycle_length=365.)
    mdata, mt, mtb = CycleReshape(mod, cycle_length=365.)
    print('Process on mask Seasonal ' + 'No.' + str(site_id) + '!')
    x = odata[:, :, site_id]
    y = mdata[:, :, site_id]
    begin = 0
    end = 0 + int(x[0, :].size / float(4)) * 4
    mask = np.ones_like(x)
    if season_kind == 1:
        mask[:,begin:end/4]=0
        xx = np.ma.masked_where(mask, x)
        yy = np.ma.masked_where(mask, x)
    elif season_kind == 2:
        mask[:,end/4:2*end/4]=0
        xx = np.ma.masked_where(mask, x)
        yy = np.ma.masked_where(mask, x)
    elif season_kind == 3:
        mask[:,2*end/4:3*end/4]=0
        xx = np.ma.masked_where(mask, x)
        yy = np.ma.masked_where(mask, x)
    else:
        mask[:, 3 * end / 4:end] = 0
        xx = np.ma.masked_where(mask, x)
        yy = np.ma.masked_where(mask, x)
    return xx.reshape(1,-1), yy.reshape(1,-1)


def GetSeasonMask_multiplemod(obs, mmod, site_id, season_kind):
    print('This function is used to mask seasons!')
    # The input data is annual data
    print('Process on mask Seasonal ' + 'No.' + str(site_id) + '!')
    odata, ot, otb = CycleReshape(obs, cycle_length=365.)

    x = odata[:, :, site_id]
    begin = 0
    end = 0 + int(x[0, :].size / float(4)) * 4
    mask = np.ones_like(x)
    if season_kind == 1:
        mask[:,begin:end/4] = 0
        xx = np.ma.masked_where(mask, x)
    elif season_kind == 2:
        mask[:,end/4:2*end/4]=0
        xx = np.ma.masked_where(mask, x)
    elif season_kind == 3:
        mask[:,2*end/4:3*end/4]=0
        xx = np.ma.masked_where(mask, x)
    else:
        mask[:, 3 * end / 4:end] = 0
        xx = np.ma.masked_where(mask, x)
    obs2 = Variable(name=obs.name, unit=obs.unit, time=obs.time, data=(xx.reshape(1,-1).T))
    mmod2 = []
    for i, mod in enumerate(mmod):
        mdata, mt, mtb = CycleReshape(mod, cycle_length=365.)
        y = mdata[:, :, site_id]
        mask = np.ones_like(x)
        if season_kind == 1:
            mask[:,begin:end/4]=0
            yy = np.ma.masked_where(mask, y)
        elif season_kind == 2:
            mask[:,end/4:2*end/4]=0
            yy = np.ma.masked_where(mask, y)
        elif season_kind == 3:
            mask[:,2*end/4:3*end/4]=0
            yy = np.ma.masked_where(mask, y)
        else:
            mask[:, 3 * end / 4:end] = 0
            yy = np.ma.masked_where(mask, y)
        # newmod.append(yy.reshape(1,-1))
        mmod2.append(Variable(name=mod.name, unit=mod.unit, time=mod.time, data=(yy.reshape(1, -1).T)))
    # print("obs_data", (obs2.data[:,0]).shape)
    # print("obs_time1", (obs.data[:,0]).shape)
    return obs2, mmod2


def Plot_response2(obs1, mod1, obs2, mod2, site_id):
    print('Process on Response ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])

    y1 = np.ma.masked_invalid(mod1.data[:, site_id])
    y2 = np.ma.masked_invalid(mod2.data[:, site_id])

    xx1 = x1.compressed()
    xx2 = x2.compressed()

    yy1 = y1[~x1.mask]
    yy2 = y2[~x2.mask]

    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('Response 2 variables')
    ax0 = fig0.add_subplot(1, 1, 1)
    ax0.plot(x1, x2, 'k.')
    ax0.plot(y1, y2,'.')
    ax0.set_xlabel(obs1.name+'\n' +obs1.unit)
    ax0.set_ylabel(obs2.name+'\n' +obs2.unit)
    plt.suptitle(
        'Site:' + str(site_id) + ':  ' + obs1.name + ' vs ' + obs2.name + 'Response',
        fontsize=24)
    return fig0


def Plot_response4(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, site_id):
    print('Process on Response4 ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x1 = np.ma.masked_where(x1 == 0, x1)
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])
    x2 = np.ma.masked_where(x2 == 0, x2)
    x3 = np.ma.masked_invalid(obs3.data[:, site_id])
    x3 = np.ma.masked_where(x3 == 0, x3)
    x4 = np.ma.masked_invalid(obs4.data[:, site_id])
    x4 = np.ma.masked_where(x4 == 0, x4)

    y1 = np.ma.masked_invalid(mod1.data[:, site_id])
    y1 = np.ma.masked_where(y1 == 0, y1)
    y2 = np.ma.masked_invalid(mod2.data[:, site_id])
    y2 = np.ma.masked_where(y2 == 0, y2)
    y3 = np.ma.masked_invalid(mod3.data[:, site_id])
    y3 = np.ma.masked_where(y3 == 0, y3)
    y4 = np.ma.masked_invalid(mod4.data[:, site_id])
    y4 = np.ma.masked_where(y4 == 0, y4)

    mask1 = x1.mask | x2.mask | x3.mask | x4.mask
    mask2 = y1.mask | y2.mask | y3.mask | y4.mask

    fig0 = plt.figure(figsize=(8, 5))
    plt.suptitle('Response 4 variables')
    ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
    cax = ax0.scatter(x1[~mask1],x2[~mask1],x3[~mask1], c=x4[~mask1], cmap=plt.hot())

    ax0.set_title( obs4.name+'\n' + obs4.unit + '')
    ax0.set_xlabel(obs1.name+'\n' + obs1.unit + '', linespacing=3.2)
    ax0.set_ylabel(obs2.name+'\n' + obs2.unit + '', linespacing=3.2)
    ax0.set_zlabel(obs3.name+'\n' + obs3.unit + '', linespacing=3.2)
    z = x4[~mask1]
    axes = [ax0]
    cbar = fig0.colorbar(cax, ticks=[min(z), (max(z) + min(z)) / 2, max(z)], orientation='vertical',
                         label=obs4.name, ax=axes, shrink=1)
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    plt.suptitle(
        'Site:' + str(site_id) + ':  ' + obs1.name + ' (' + obs2.name + ', ' + obs3.name + ', ' + obs4.name + ') Response',
        fontsize=24)
    ax0.xaxis.labelpad = 5
    ax0.yaxis.labelpad = 5
    ax0.zaxis.labelpad = 5

    return fig0

def correlation_matrix(datas):
    frame = pd.DataFrame(datas)
    A = frame.corr(method='pearson', min_periods=1)
    corr = np.ma.corrcoef(A)
    mask = np.zeros_like(A)
    mask[np.triu_indices_from(mask)] = True

    return corr, mask


def partial_corr(C):
#     """
    #     Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    #     for the remaining variables in C.
    #     Parameters
    #     ----------
    #     C : array-like, shape (n, p)
    #         Array with the different variables. Each column of C is taken as a variable
    #     Returns
    #     -------
    #     P : array-like, shape (p, p)
    #         P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
    #         for the remaining variables in C.

#     """
    # """
    # Partial Correlation in Python (clone of Matlab's partialcorr)
    # This uses the linear regression approach to compute the partial
    # correlation (might be slow for a huge number of variables).The code is adopted from
    # https://gist.github.com/fabianp/9396204419c7b638d38f
    # Date: Nov 2014
    # Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    # Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    # """

    C = np.column_stack([C, np.ones(C.shape[0])])

    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr[0:C.shape[1] - 1, 0:C.shape[1] - 1]


def plot_4_variable_corr(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, site_id):
    print('Process on Corr4 ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x1 = np.ma.masked_where(x1 == 0, x1)
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])
    x2 = np.ma.masked_where(x2 == 0, x2)
    x3 = np.ma.masked_invalid(obs3.data[:, site_id])
    x3 = np.ma.masked_where(x3 == 0, x3)
    x4 = np.ma.masked_invalid(obs4.data[:, site_id])
    x4 = np.ma.masked_where(x4 == 0, x4)

    y1 = np.ma.masked_invalid(mod1.data[:, site_id])
    y1 = np.ma.masked_where(y1 == 0, y1)
    y2 = np.ma.masked_invalid(mod2.data[:, site_id])
    y2 = np.ma.masked_where(y2 == 0, y2)
    y3 = np.ma.masked_invalid(mod3.data[:, site_id])
    y3 = np.ma.masked_where(y3 == 0, y3)
    y4 = np.ma.masked_invalid(mod4.data[:, site_id])
    y4 = np.ma.masked_where(y4 == 0, y4)

    mask1 = x1.mask | x2.mask | x3.mask | x4.mask
    mask2 = y1.mask | y2.mask | y3.mask | y4.mask

    v0 = np.asarray([x1[~mask1],x2[~mask1],x3[~mask1],x4[~mask1]]).T


    # print('v0:',v0)

    corr0 = partial_corr(v0)
    # print('corr0:',corr0)
    fig0 = plt.figure(figsize=(8, 5))
    plt.suptitle('Correlations 4 variables')
    ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
    ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Obs', marker='^')

    return fig0

def plot_4_variable_corr_multiple(obs1, mod1_1, obs2, mod2_1, obs3, mod3_1, obs4, mod4_1, site_id):
    print('Process on Corr4 ' + 'No.' + str(site_id) + '!')
    x1 = np.ma.masked_invalid(obs1.data[:, site_id])
    x1 = np.ma.masked_where(x1 == 0, x1)
    x2 = np.ma.masked_invalid(obs2.data[:, site_id])
    x2 = np.ma.masked_where(x2 == 0, x2)
    x3 = np.ma.masked_invalid(obs3.data[:, site_id])
    x3 = np.ma.masked_where(x3 == 0, x3)
    x4 = np.ma.masked_invalid(obs4.data[:, site_id])
    x4 = np.ma.masked_where(x4 == 0, x4)

    mask1 = x1.mask | x2.mask | x3.mask | x4.mask
    v0 = np.asarray([x1[~mask1],x2[~mask1],x3[~mask1],x4[~mask1]]).T
    corr0 = partial_corr(v0)
    # print('corr0:',corr0)
    fig0 = plt.figure(figsize=(5, 5))
    plt.suptitle('Correlations 4 variables')
    ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
    ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Obs', marker='^')

    for i, mod1 in enumerate(mod1_1):
        y1 = np.ma.masked_invalid(mod1_1[i].data[:, site_id])
        y1 = np.ma.masked_where(y1 == 0, y1)
        y2 = np.ma.masked_invalid(mod2_1[i].data[:, site_id])
        y2 = np.ma.masked_where(y2 == 0, y2)
        y3 = np.ma.masked_invalid(mod3_1[i].data[:, site_id])
        y3 = np.ma.masked_where(y3 == 0, y3)
        y4 = np.ma.masked_invalid(mod4_1[i].data[:, site_id])
        y4 = np.ma.masked_where(y4 == 0, y4)
        mask2 = y1.mask | y2.mask | y3.mask | y4.mask
        v0 = np.asarray([y1[~mask2], y2[~mask2], y3[~mask2], y4[~mask2]]).T
        corr0 = partial_corr(v0)
        ax0.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Mod'+str(i), marker='^')

    ax0.dist = 12
    ax0.tick_params(labelsize=5)
    ax0.legend(bbox_to_anchor=(1.03, 1), loc=2, fontsize=8)
    ax0.set_xlabel(obs1.name + '\n' + obs1.unit + '')
    ax0.set_ylabel(obs2.name + '\n' + obs2.unit + '')
    ax0.set_zlabel(obs3.name + '\n' + obs3.unit + '')
    plt.suptitle(
        'Site:' + str(site_id) + ':  ' + obs1.name + ' (' + obs2.name + ', ' + obs3.name + ', ' + obs4.name + ') Partial correlation',
        fontsize=24)
    ax0.xaxis.labelpad = 5
    ax0.yaxis.labelpad = 5
    ax0.zaxis.labelpad = 5
    return fig0

output_path = '/Users/lli51/Downloads/ILAMB_sample/output/'

# obs1 = Variable(filename='/Users/lli51/Downloads/alldata/obs_FSH_model_ilamb.nc4', variable_name='FSH')
# mod1 = Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSH_model_ilamb.nc4', variable_name='FSH')
#
# obs2=Variable(filename='/Users/lli51/Downloads/alldata/obs_NEE_model_ilamb.nc4', variable_name='NEE')
# mod2=Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_NEE_model_ilamb.nc4', variable_name='NEE')
#
# obs3=Variable(filename='/Users/lli51/Downloads/alldata/obs_ER_model_ilamb.nc4', variable_name='ER')
# mod3=Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_ER_model_ilamb.nc4', variable_name='ER')
#
# obs4=Variable(filename='/Users/lli51/Downloads/alldata/obs_FSDS_model_ilamb.nc4', variable_name='FSDS')
# mod4=Variable(filename='/Users/lli51/Downloads/alldata/171206_ELMv0_CN_FSDS_model_ilamb.nc4', variable_name='FSDS')


obs1 = Variable(filename='/Users/lli51/Downloads/allfiles/obs_GPP_model_ilamb.nc4', variable_name='GPP')
mod1 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_GPP_model_ilamb.nc4', variable_name='GPP')

mod11 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CNP_GPP_model_ilamb.nc4', variable_name='GPP')
mod12 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CN_GPP_model_ilamb.nc4', variable_name='GPP')
mod13 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_GPP_model_ilamb.nc4', variable_name='GPP')
mod14 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_GPP_model_ilamb.nc4', variable_name='GPP')
# mod11 = obs1
# mod12 = obs1
# mod13 = obs1
# mod14 = obs1

mod1_1 = [mod11,mod12,mod13,mod14]

obs2=Variable(filename='/Users/lli51/Downloads/allfiles/obs_NEE_model_ilamb.nc4', variable_name='NEE')
mod2=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_NEE_model_ilamb.nc4', variable_name='NEE')

mod21=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CNP_NEE_model_ilamb.nc4', variable_name='NEE')
mod22=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CN_NEE_model_ilamb.nc4', variable_name='NEE')
mod23=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_NEE_model_ilamb.nc4', variable_name='NEE')
mod24=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_NEE_model_ilamb.nc4', variable_name='NEE')

mod2_1 = [mod21,mod22,mod23,mod24]

obs3=Variable(filename='/Users/lli51/Downloads/allfiles/obs_ER_model_ilamb.nc4', variable_name='ER')
mod3=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_ER_model_ilamb.nc4', variable_name='ER')

mod31=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CNP_ER_model_ilamb.nc4', variable_name='ER')
mod32=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CN_ER_model_ilamb.nc4', variable_name='ER')
mod33=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_ER_model_ilamb.nc4', variable_name='ER')
mod34=Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_ER_model_ilamb.nc4', variable_name='ER')

mod3_1 = [mod31,mod32,mod33,mod34]

obs4 = Variable(filename='/Users/lli51/Downloads/allfiles/obs_FSH_model_ilamb.nc4', variable_name='FSH')
mod4 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_FSH_model_ilamb.nc4', variable_name='FSH')

mod41 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CNP_FSH_model_ilamb.nc4', variable_name='FSH')
mod42 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv1_CN_FSH_model_ilamb.nc4', variable_name='FSH')
mod43 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_FSH_model_ilamb.nc4', variable_name='FSH')
mod44 = Variable(filename='/Users/lli51/Downloads/allfiles/171206_ELMv0_CN_FSH_model_ilamb.nc4', variable_name='FSH')

mod4_1 = [mod41,mod42,mod43,mod44]

lats = obs1.lat
lons = obs1.lon

for siteid in range(len(lats)):
    region = "lat" + str(lats[siteid]) + "lon" + str(lons[siteid])

    # # Time series output
    # # fig0 = Plot_TimeSeries(obs1, mod1, siteid)
    # fig0 = Plot_TimeSeries_multiple_mod(obs1, mod1_1, siteid)
    # fig0.savefig(os.path.join(output_path, "%s_timeseries_hourly.png" % (region)),bbox_inches='tight')

    # # fig7 = Plot_TimeSeries_cycle(obs1, mod1, siteid, 90.)
    # fig71 = Plot_TimeSeries_cycle_multiple_mod(obs1, mod1_1, siteid, 1.)
    # fig71.savefig(os.path.join(output_path, "%s_timeseries_daily.png" % (region)),bbox_inches='tight')
    #
    # # fig7 = Plot_TimeSeries_cycle(obs1, mod1, siteid, 90.)
    # fig72 = Plot_TimeSeries_cycle_multiple_mod(obs1, mod1_1, siteid, 30.)
    # fig72.savefig(os.path.join(output_path, "%s_timeseries_monthly.png" % (region)),bbox_inches='tight')
    #
    # # fig7 = Plot_TimeSeries_cycle(obs1, mod1, siteid, 90.)
    # fig73 = Plot_TimeSeries_cycle_multiple_mod(obs1, mod1_1, siteid, 365.)
    # fig73.savefig(os.path.join(output_path, "%s_timeseries_yearly.png" % (region)),bbox_inches='tight')
    #
    #
    # fig74 = Plot_TimeSeries_cycle_multiple_mod(obs1, mod1_1, siteid, 90.)
    # fig74.savefig(os.path.join(output_path, "%s_timeseries_seasonly.png" % (region)),bbox_inches='tight')

    # # Cycle means output
    # # fig8 = Plot_TimeSeries_cycle_reshape(obs1, mod1, siteid, 90.)
    # fig8 = Plot_TimeSeries_cycle_reshape_multiple_mod(obs1, mod1_1, siteid, 1.)
    # fig8.savefig(os.path.join(output_path, "%s_timeseries_hourofday.png" % (region)),bbox_inches='tight')
    #
    # odata, ot, otb = CycleReshape(obs1, cycle_length=1)
    # obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
    # mod1_2 = []
    # for i, mod in enumerate(mod1_1):
    #     mdata, mt, mtb = CycleReshape(mod, cycle_length=1.)
    #     mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
    # fig81 = Plot_TimeSeries_cycle_reshape_multiple_mod(obs1_2, mod1_2, siteid, 365., xname="Days of a year")
    # fig81.savefig(os.path.join(output_path, "%s_timeseries_dayofyear.png" % (region)),bbox_inches='tight')
    #
    # odata, ot, otb = CycleReshape(obs1, cycle_length=30.)
    # obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
    # mod1_2 = []
    # for i, mod in enumerate(mod1_1):
    #     mdata, mt, mtb = CycleReshape(mod, cycle_length=30.)
    #     mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
    # fig82 = Plot_TimeSeries_cycle_reshape_multiple_mod(obs1_2, mod1_2, siteid, 365., xname="Months of a year")
    # fig82.savefig(os.path.join(output_path, "%s_timeseries_monthofyear.png" % (region)),bbox_inches='tight')
    #
    # odata, ot, otb = CycleReshape(obs1, cycle_length=90.)
    # obs1_2 = Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(odata, axis=1))
    # mod1_2 = []
    # for i, mod in enumerate(mod1_1):
    #     mdata, mt, mtb = CycleReshape(mod, cycle_length=90.)
    #     mod1_2.append(Variable(name=obs1.name,unit=obs1.unit,time=ot,data=np.mean(mdata, axis=1)))
    # fig83 = Plot_TimeSeries_cycle_reshape_multiple_mod(obs1_2, mod1_2, siteid, 365., xname="Seasons of a year")
    # fig83.savefig(os.path.join(output_path, "%s_timeseries_seasonofyear.png" % (region)),bbox_inches='tight')

    # # PDF and CDF output
    # # fig1 = Plot_PDF_CDF(obs1, mod1, siteid)
    # fig1 = Plot_PDF_CDF_multiple_mod(obs1, mod1_1, siteid)
    # fig1.savefig(os.path.join(output_path, "%s_PDFCDF.png" % (region)),bbox_inches='tight')


    # ## Wavelet output
    # odata, ot, otb = CycleReshape(obs1, cycle_length=30.) # when it goes to 30 days with hourly data, there are totally 720 samples [292, 720, 34]
    # # # fig2 = Plot_Wavelet(odata[:,siteid,:],ot, 0, 0, siteid, obs1.unit)
    # fig2 = Plot_Wavelet_multiple_mod(np.mean(odata, axis=1), ot, siteid, obs1.unit, model_name='Obs')
    # fig2.savefig(os.path.join(output_path, "%s_wavelet.png" % (region)), bbox_inches='tight')
    #
    # for i, mod in enumerate(mod1_1):
    #     mdata, mt, mtb = CycleReshape(mod, cycle_length=30.)
    #     fig2 = Plot_Wavelet_multiple_mod(np.mean(mdata, axis=1), mt, siteid, obs1.unit, model_name='Mod'+str(i))
    #     fig2.savefig(os.path.join(output_path, ("%s_wavelet_Mod"+str(i) + ".png") % (region)), bbox_inches='tight')


    # odata, ot, otb = CycleReshape(obs1, cycle_length=30.) # when it goes to 30 days with hourly data, there are totally 720 samples [292, 720, 34]
    # mods = []
    # for i, mod in enumerate(mod1_1):
    #     mdata, mt, mtb = CycleReshape(mod, cycle_length=30.)
    #     mods.append(np.mean(mdata, axis=1))
    # # fig3,fig4,fig5,fig6 = Plot_IMF(odata[:,siteid,:],ot, mdata[:,siteid,:], mt, siteid, obs1.unit)
    # fig3,fig4,fig5,fig6 = Plot_IMF_multiple_mod(np.mean(odata, axis=1), ot, mods, mt, siteid)
    #
    # fig3.savefig(os.path.join(output_path, "%s_IMF1.png" % (region)),bbox_inches='tight')
    # fig4.savefig(os.path.join(output_path, "%s_IMF2.png" % (region)),bbox_inches='tight')
    # fig5.savefig(os.path.join(output_path, "%s_IMF3.png" % (region)),bbox_inches='tight')
    # fig6.savefig(os.path.join(output_path, "%s_IMF4.png" % (region)),bbox_inches='tight')

    # fig9 = Plot_response2(obs1, mod1, obs2, mod2, siteid)
    # fig9.savefig(os.path.join(output_path, "%s_response.png" % (region)),bbox_inches='tight')

    # fig10 = Plot_response4(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, siteid)
    # fig10.savefig(os.path.join(output_path, "%s_response4.png" % (region)),bbox_inches='tight')

    # # fig11 = plot_4_variable_corr(obs1, mod1, obs2, mod2, obs3, mod3, obs4, mod4, siteid)
    # fig11 = plot_4_variable_corr_multiple(obs1, mod1_1, obs2, mod2_1, obs3, mod3_1, obs4, mod4_1, siteid)
    # fig11.savefig(os.path.join(output_path, "%s_corr4.png" % (region)),bbox_inches='tight')
    #
    # GetSeasonMask(data_o, data_m, 0, 1)
    # obs2_s1, mmod2_s1 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 1)
    # fig12 = Plot_TimeSeries_multiple_mod(obs2_s1, mmod2_s1, 0)
    # fig12 = Plot_TimeSeries_cycle_multiple_mod_season(obs2_s1, mmod2_s1, 0, 365.) # only one output is given
    # fig12.savefig(os.path.join(output_path, "%s_timeseries_s1.png" % (region)), bbox_inches='tight')
    # fig81 = Plot_TimeSeries_cycle_reshape_multiple_mod_season(obs2_s1, mmod2_s1, 0, 1.)
    # fig81.savefig(os.path.join(output_path, "%s_timeseries_hourofday_s1.png" % (region)),bbox_inches='tight')

    # obs2_s2, mmod2_s2 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 2)
    # fig12 = Plot_TimeSeries_multiple_mod(obs2_s1, mmod2_s1, 0)
    # fig12 = Plot_TimeSeries_cycle_multiple_mod_season(obs2_s2, mmod2_s2, 0, 365.) # only one output is given
    # fig12.savefig(os.path.join(output_path, "%s_timeseries_s2.png" % (region)), bbox_inches='tight')
    # fig83 = Plot_TimeSeries_cycle_reshape_multiple_mod_season(obs2_s2, mmod2_s2, 0, 1.)
    # fig83.savefig(os.path.join(output_path, "%s_timeseries_hourofday_s2.png" % (region)),bbox_inches='tight')

    # obs2_s3, mmod2_s3 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 3)
    # fig12 = Plot_TimeSeries_multiple_mod(obs2_s1, mmod2_s1, 0)
    # fig12 = Plot_TimeSeries_cycle_multiple_mod_season(obs2_s3, mmod2_s3, 0, 365.) # only one output is given
    # fig12.savefig(os.path.join(output_path, "%s_timeseries_s3.png" % (region)), bbox_inches='tight')
    # fig83 = Plot_TimeSeries_cycle_reshape_multiple_mod_season(obs2_s3, mmod2_s3, 0, 1.)
    # fig83.savefig(os.path.join(output_path, "%s_timeseries_hourofday_s3.png" % (region)),bbox_inches='tight')

    # obs2_s4, mmod2_s4 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 4)
    # fig12 = Plot_TimeSeries_multiple_mod(obs2_s1, mmod2_s1, 0)
    # fig12 = Plot_TimeSeries_cycle_multiple_mod_season(obs2_s4, mmod2_s4, 0, 365.) # only one output is given
    # fig12.savefig(os.path.join(output_path, "%s_timeseries_s4.png" % (region)), bbox_inches='tight')
    # fig84 = Plot_TimeSeries_cycle_reshape_multiple_mod_season(obs2_s4, mmod2_s4, 0, 1.)
    # fig84.savefig(os.path.join(output_path, "%s_timeseries_hourofday_s4.png" % (region)),bbox_inches='tight')




    obs2_1, mmod2_1 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 1)
    obs2_2, mmod2_2 = GetSeasonMask_multiplemod(obs2, mod2_1, siteid, 1)
    obs2_3, mmod2_3 = GetSeasonMask_multiplemod(obs3, mod3_1, siteid, 1)
    obs2_4, mmod2_4 = GetSeasonMask_multiplemod(obs4, mod4_1, siteid, 1)
    # fig10_s = Plot_response4(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], obs2_3, mmod2_3[0], obs2_4, mmod2_4[0], 0)
    # fig10_s.savefig(os.path.join(output_path, "%s_response4_s1.png" % (region)),bbox_inches='tight')

    fig9_s = Plot_response2(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], 0)
    fig9_s.savefig(os.path.join(output_path, "%s_response_s1.png" % (region)),bbox_inches='tight')


    obs2_1, mmod2_1 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 2)
    obs2_2, mmod2_2 = GetSeasonMask_multiplemod(obs2, mod2_1, siteid, 2)
    obs2_3, mmod2_3 = GetSeasonMask_multiplemod(obs3, mod3_1, siteid, 2)
    obs2_4, mmod2_4 = GetSeasonMask_multiplemod(obs4, mod4_1, siteid, 2)
    # fig10_s = Plot_response4(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], obs2_3, mmod2_3[0], obs2_4, mmod2_4[0], 0)
    # fig10_s.savefig(os.path.join(output_path, "%s_response4_s2.png" % (region)),bbox_inches='tight')

    fig9_s = Plot_response2(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], 0)
    fig9_s.savefig(os.path.join(output_path, "%s_response_s2.png" % (region)),bbox_inches='tight')


    obs2_1, mmod2_1 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 3)
    obs2_2, mmod2_2 = GetSeasonMask_multiplemod(obs2, mod2_1, siteid, 3)
    obs2_3, mmod2_3 = GetSeasonMask_multiplemod(obs3, mod3_1, siteid, 3)
    obs2_4, mmod2_4 = GetSeasonMask_multiplemod(obs4, mod4_1, siteid, 3)
    # fig10_s = Plot_response4(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], obs2_3, mmod2_3[0], obs2_4, mmod2_4[0], 0)
    # fig10_s.savefig(os.path.join(output_path, "%s_response4_s3.png" % (region)),bbox_inches='tight')

    fig9_s = Plot_response2(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], 0)
    fig9_s.savefig(os.path.join(output_path, "%s_response_s3.png" % (region)),bbox_inches='tight')


    obs2_1, mmod2_1 = GetSeasonMask_multiplemod(obs1, mod1_1, siteid, 4)
    obs2_2, mmod2_2 = GetSeasonMask_multiplemod(obs2, mod2_1, siteid, 4)
    obs2_3, mmod2_3 = GetSeasonMask_multiplemod(obs3, mod3_1, siteid, 4)
    obs2_4, mmod2_4 = GetSeasonMask_multiplemod(obs4, mod4_1, siteid, 4)
    # fig10_s = Plot_response4(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], obs2_3, mmod2_3[0], obs2_4, mmod2_4[0], 0)
    # fig10_s.savefig(os.path.join(output_path, "%s_response4_s4.png" % (region)),bbox_inches='tight')

    fig9_s = Plot_response2(obs2_1, mmod2_1[0], obs2_2, mmod2_2[0], 0)
    fig9_s.savefig(os.path.join(output_path, "%s_response_s4.png" % (region)),bbox_inches='tight')





