from score_post import time_basic_score3
import numpy as np
import matplotlib.pyplot as plt
from score_post import time_basic_score5
from taylorDiagram import plot_Taylor_graph_day_cycle
from taylorDiagram import plot_Taylor_graph_three_cycle

fontsize = 13
plt.rcParams.update({'font.size': 12})
lengendfontsize = 12
col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']

def max_none(a, b):
    if a is None:
        a = float('-inf')
    if b is None:
        b = float('-inf')
    return max(a, b)

def min_none(a, b):
    if a is None:
        a = float('inf')
    if b is None:
        b = float('inf')
    return min(a, b)

def plot_day_cycle_categories(fig0, obs, mod, j, rect0, rect1, rect2, rect3, rect4, rect, ref_times):
    # organize the data for taylor gram and plot
    [s_obs, h_obs, d_obs, m_obs, y_obs, s_t_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs] = obs
    [s_mod, h_mod, d_mod, m_mod, y_mod, s_t_mod, h_t_mod, d_t_mod, m_t_mod, y_t_mod] = mod
    data0 = s_obs[j, :][~s_obs[j, :].mask]
    data1 = h_obs[j, :][~s_obs[j, :].mask]
    data2 = d_obs[j, :][~s_obs[j, :].mask]
    data3 = m_obs[j, :][~s_obs[j, :].mask]
    data4 = y_obs[j, :][~s_obs[j, :].mask]


    s_t_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs = s_t_obs[~s_obs[j, :].mask], h_t_obs[~s_obs[j, :].mask], d_t_obs[~s_obs[j, :].mask], m_t_obs[~s_obs[j, :].mask], y_t_obs[~s_obs[j, :].mask]
    models1, models2, models3, models4, models5 = [], [], [], [], []
    h1, h2, h3, h4, h0 = None, None, None, None, None
    h1s, h2s, h3s, h4s, h0s = None, None, None, None, None

    if len(data1) > 0 and len(data2) > 0 and len(data3) > 0 and len(data4) > 0 and len(data0) > 0:
        h1, h2, h3, h4, h0 = max_none(np.ma.max(data1), h1), max_none(np.ma.max(data2), h2), max_none(np.ma.max(data3), h3), max_none(np.ma.max(data4), h4), max_none(np.ma.max(data0), h0)
        h1s, h2s, h3s, h4s, h0s = min_none(np.ma.min(data1), h1s), min_none(np.ma.min(data2), h2s), min_none(np.ma.min(data3), h3s), min_none(np.ma.min(data4), h4s), min_none(np.ma.min(data0), h0s)


    for i in range(len(d_mod)):
        models1.append(h_mod[i][j, :][~s_obs[j, :].mask])
        models2.append(d_mod[i][j, :][~s_obs[j, :].mask])
        models3.append(m_mod[i][j, :][~s_obs[j, :].mask])
        models4.append(y_mod[i][j, :][~s_obs[j, :].mask])
        models5.append(s_mod[i][j, :][~s_obs[j, :].mask])
        if len(data1) > 0 and len(data2) > 0 and len(data3) > 0 and len(data4) > 0 and len(data0) > 0:
            h1, h2, h3, h4, h0 = max_none(np.ma.max(h_mod[i][j, :][~s_obs[j, :].mask]), h1), max_none(np.ma.max(d_mod[i][j, :][~s_obs[j, :].mask]), h2), max_none(np.ma.max(m_mod[i][j, :][~s_obs[j, :].mask]), h3), max_none(np.ma.max(y_mod[i][j, :][~s_obs[j, :].mask]),
                                                                                                         h4), max_none(np.ma.max(s_mod[i][j, :][~s_obs[j, :].mask]), h0)
            h1s, h2s, h3s, h4s, h0s = min_none(np.ma.min(h_mod[i][j, :][~s_obs[j, :].mask]), h1s), min_none(np.ma.min(d_mod[i][j, :][~s_obs[j, :].mask]), h2s), min_none(np.ma.min(m_mod[i][j, :][~s_obs[j, :].mask]), h3s), min_none(np.ma.min(y_mod[i][j, :][~s_obs[j, :].mask]),
                                                                                                     h4s), min_none(np.ma.min(s_mod[i][j, :][~s_obs[j, :].mask]), h0s)


    fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_day_cycle(data1, data2, data3, data4, data0, models1, models2, models3, models4, models5, fig0, rect=rect, ref_times=ref_times, bbox_to_anchor=(1.00, 0.33))

    ax0 = fig0.add_subplot(rect1)
    ax1 = fig0.add_subplot(rect2)
    ax2 = fig0.add_subplot(rect3)
    ax3 = fig0.add_subplot(rect4)
    ax4 = fig0.add_subplot(rect0)

    # print('data size', len(h_t_obs), len(d_t_obs), len(m_t_obs), len(y_t_obs), len(s_t_obs))
    # print('data size', len(data0), len(data1), len(data2), len(data3), len(data4))

    ax0.plot(h_t_obs, data1, 'k-', label='Observed')
    ax1.plot(d_t_obs, data2, 'k-', label='Observed')
    ax2.plot(m_t_obs, data3, 'k-', label='Observed')
    ax3.plot(y_t_obs, data4, 'k-', label='Observed')
    ax4.plot(s_t_obs, data0, 'k-', label='Observed')

    for i in range(len(h_mod)):
        ax0.plot(h_t_obs, models1[i], '-', label="Model " + str(i + 1), color=col[i])
        ax1.plot(d_t_obs, models2[i], '-', label="Model " + str(i + 1), color=col[i])
        ax2.plot(m_t_obs, models3[i], '-', label="Model " + str(i + 1), color=col[i])
        ax3.plot(y_t_obs, models4[i], '-', label="Model " + str(i + 1), color=col[i])
        ax4.plot(s_t_obs, models5[i], '-', label="Model " + str(i + 1), color=col[i])

    if len(data1) > 0 and len(data2) > 0 and len(data3) > 0 and len(data4) > 0 and len(data0) > 0:
        ax0.set_ylim(h1s-0.5*abs(h1s),  h1+0.5*abs(h1))
        ax1.set_ylim(h2s-0.5*abs(h2s),  h2+0.5*abs(h2))
        ax2.set_ylim(h3s-0.5*abs(h3s),  h3+0.5*abs(h3))
        ax3.set_ylim(h4s-0.5*abs(h4s),  h4+0.5*abs(h4))
        ax4.set_ylim(h0s-0.5*abs(h0s),  h0+0.5*abs(h0))


    ax0.set_yticklabels([])
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig0, ax0, ax1, ax2, ax3, ax4, [samples1, samples2, samples3, samples4, samples5]

def plot_four_cycle_categories(fig0, obs, mod, j, rect1, rect2, rect3, rect, ref_times):
    # organize the data for taylor gram and plot
    [h_obs, d_obs, m_obs, h_t_obs, d_t_obs, m_t_obs] = obs
    [h_mod, d_mod, m_mod, h_t_mod, d_t_mod, m_t_mod] = mod
    data1 = h_obs[j, :][~h_obs[j, :].mask]
    data2 = d_obs[j, :][~d_obs[j, :].mask]
    data3 = m_obs[j, :][~m_obs[j, :].mask]
    h_t_obs, d_t_obs, m_t_obs = h_t_obs[~h_obs[j, :].mask], d_t_obs[~d_obs[j, :].mask], m_t_obs[~m_obs[j, :].mask]
    models1, models2, models3 = [], [], []
    for i in range(len(d_mod)):
        models1.append(h_mod[i][j, :][~h_obs[j, :].mask])
        models2.append(d_mod[i][j, :][~d_obs[j, :].mask])
        models3.append(m_mod[i][j, :][~m_obs[j, :].mask])
        # if len(h_mod[i][j, :][~h_obs[j, :].mask]) > 0:
        #     h_m = max(np.max(h_mod[i][j, :][~h_obs[j, :].mask]), h_m)
        #     d_m = max(np.max(d_mod[i][j, :][~d_obs[j, :].mask]), d_m)
        #     m_m = max(np.max(m_mod[i][j, :][~m_obs[j, :].mask]), m_m)
        #     h_m_s = min(np.min(h_mod[i][j, :][~h_obs[j, :].mask]), h_m_s)
        #     d_m_s = min(np.min(d_mod[i][j, :][~d_obs[j, :].mask]), d_m_s)
        #     m_m_s = min(np.min(m_mod[i][j, :][~m_obs[j, :].mask]), m_m_s)

    fig0, samples1, samples2, samples3 = plot_Taylor_graph_three_cycle(data1, data2, data3, models1, models2, models3, fig0, rect=rect, ref_times=ref_times, bbox_to_anchor=(1.15, 0.35))

    ax0 = fig0.add_subplot(rect1)
    ax1 = fig0.add_subplot(rect2)
    ax2 = fig0.add_subplot(rect3)

    ax0.plot(h_t_obs, data1, 'k-', label='Observed')
    ax1.plot(d_t_obs, data2, 'k-', label='Observed')
    ax2.plot(m_t_obs, data3, 'k-', label='Observed')
    # ax3.plot(y_t_obs, data4, 'k-', label='Observed')
    for i in range(len(h_mod)):
        ax0.plot(h_t_obs, models1[i], '-', label="Model " + str(i + 1), color=col[i])
        ax1.plot(d_t_obs, models2[i], '-', label= "Model " + str(i + 1), color=col[i])
        ax2.plot(m_t_obs, models3[i], '-', label= "Model " + str(i + 1), color=col[i])
        # ax3.plot(y_t_obs, models4[i], '-', label= "Model " + str(i + 1))
    # fig0.legend(line,labels, loc='upper left')
    return fig0, ax0, ax1, ax2, [samples1, samples2, samples3]

class cycle_post(object):

    def __init__(self, variable, site_name, filedir, h_unit_obs, d_unit_obs, m_unit_obs, y_unit_obs):
        self.variable = variable
        self.sitename = site_name
        self.filedir = filedir
        self.h_unit_obs, self.d_unit_obs, self.m_unit_obs, self.y_unit_obs = h_unit_obs, d_unit_obs,m_unit_obs, y_unit_obs

    def plot_three_cycle(self, hour_np, day_np, month_np, season_np, mhour_np, mday_np, mmonth_np,
                                     mseason_np):

        day_mean_np, day_error_np = day_np.mean(axis=1), day_np.std(axis=1)
        month_mean_np, month_error_np = month_np.mean(axis=1), month_np.std(axis=1)
        season_mean_np, season_error_np = season_np.mean(axis=2).T, season_np.std(axis=2).T

        mday_mean_np, mmonth_mean_np, mseason_mean_np = [], [], []
        mday_err_np, mmonth_err_np, mseason_err_np = [], [], []

        m_xasix2, m_xasix3, m_xasix4 = [],[],[]
        day_time = np.arange(1, 366)
        month_time = np.arange(1, 13)#np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov', 'Dec'])
        season_time = np.arange(1, 5)#np.array(['DJF', 'MMA', 'JJA', 'SON'])

        for m in range(len(mhour_np)):
            mday_mean_np.append(mday_np[m].mean(axis=1))
            mmonth_mean_np.append(mmonth_np[m].mean(axis=1))
            mseason_mean_np.append(mseason_np[m].mean(axis=2).T)
            mday_err_np.append(mday_np[m].std(axis=1))
            mmonth_err_np.append(mmonth_np[m].std(axis=1))
            mseason_err_np.append(mseason_np[m].std(axis=2).T)
            m_xasix2.append(day_time)
            m_xasix3.append(month_time)
            m_xasix4.append(season_time)

            """ plot the mean and deviation timeseries  """
        obs = [day_mean_np, month_mean_np, season_mean_np, day_time, month_time, season_time]
        mod = [mday_mean_np, mmonth_mean_np, mseason_mean_np, m_xasix2, m_xasix3, m_xasix4]
        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            fig1 = plt.figure(figsize=(6, 10))
            print('Process on four_cycles_' + site + '_No.' + str(j) + '!')
            ''' Observations data need to use masked '''
            fig1, ax1, ax2, ax3, samples = plot_four_cycle_categories(fig1, obs, mod, j, 611, 612, 613, 212, 10)

            model_score = time_basic_score3(samples)
            scores.append(model_score)


            ax1.fill_between(day_time, day_mean_np[j, :] - day_error_np[j, :],
                                day_mean_np[j, :] + day_error_np[j, :], alpha=0.2, edgecolor='#1B2ACC',
                                facecolor='gray',
                                linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax1.set_title('Daily mean and standard deviation')
            ax2.fill_between(month_time, month_mean_np[j, :] - month_error_np[j, :],
                                month_mean_np[j, :] + month_error_np[j, :], alpha=0.2, edgecolor='#1B2ACC',
                                facecolor='gray',
                                linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax2.set_title('Monthly mean and standard deviation')
            ax3.fill_between(season_time, season_mean_np[j, :] - season_error_np[j, :],
                                season_mean_np[j, :] + season_error_np[j, :], alpha=0.2, edgecolor='#1B2ACC',
                                facecolor='gray',
                                linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax3.set_title('Yearly mean and standard deviation')
            # for m in range(len(mhour_np)):
            #     ax1.fill_between(day_time, mday_mean_np[m][j, :] - mday_err_np[m][j, :],
            #                      mday_mean_np[m][j, :] + mday_err_np[m][j, :], alpha=0.2, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     # ax1.set_title('Daily mean and standard deviation')
            #     ax2.fill_between(month_time, mmonth_mean_np[m][j, :] - mmonth_err_np[m][j, :],
            #                      mmonth_mean_np[m][j, :] + mmonth_err_np[m][j, :], alpha=0.2, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     # ax2.set_title('Monthly mean and standard deviation')
            #     ax3.fill_between(season_time, mseason_mean_np[m][j, :] - mseason_err_np[m][j, :],
            #                      mseason_mean_np[m][j, :] + mseason_err_np[m][j, :], alpha=0.2, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            # fontsize = 26
            plt.suptitle('Hourly, daily and monthly cycles')
            ax1.set_xlabel('Day of a year', fontsize=fontsize)
            ax1.set_ylabel(self.variable + '\n' + self.d_unit_obs+'', fontsize=fontsize)
            ax2.set_xlabel('Month of a year', fontsize=fontsize)
            ax2.set_ylabel(self.variable + '\n' + self.m_unit_obs+'', fontsize=fontsize)
            ax3.set_xlabel('Season of a year', fontsize=fontsize)
            ax3.set_ylabel(self.variable + '\n' + self.m_unit_obs+'', fontsize=fontsize)
            ax1.set_xlim([1, 365])
            ax2.set_xlim([1, 12])
            ax3.set_xlim([1, 4])
            ax1.xaxis.set_ticks(range(1, 366, 30))
            ax1.set_xticklabels([str(x) + '' for x in range(1, 366, 30)])
            ax2.xaxis.set_ticks(range(1, 13))
            ax2.set_xticklabels([str(x) + '' for x in range(1, 13)])
            ax3.xaxis.set_ticks(range(1, 5))
            ax3.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax1.legend(bbox_to_anchor=(1.05, 0.4), loc=2, borderaxespad=0., fontsize=lengendfontsize)

            # ax1.legend(loc='upper right', shadow=False, fontsize='medium')
            # ax2.legend(loc='upper right', shadow=False, fontsize='medium')
            # ax3.legend(loc='upper right', shadow=False, fontsize='medium')
            plt.tight_layout(rect=[0, 0.01, 1, 0.97])
            fig1.savefig(self.filedir  + self.variable + '/' + site + '_time_series_' + self.variable + '.png', bbox_inches='tight')
            plt.close('all')
        scores = np.asarray(scores)
        return scores

    def plot_days_cycle(self, hour_np_s1, hour_np_s2, hour_np_s3, hour_np_s4, mhour_np_s1, mhour_np_s2, mhour_np_s3, mhour_np_s4, hour_np, mhour_np):


        hour_mean_np, hour_error_np = hour_np.mean(axis=1), hour_np.std(axis=1)
        mhour_mean_np, mhour_err_np = [], []

        hour_mean_np_s1, hour_error_np_s1 = hour_np_s1.mean(axis=0), hour_np_s1.std(axis=0)
        hour_mean_np_s2, hour_error_np_s2 = hour_np_s2.mean(axis=0), hour_np_s2.std(axis=0)
        hour_mean_np_s3, hour_error_np_s3 = hour_np_s3.mean(axis=0), hour_np_s3.std(axis=0)
        hour_mean_np_s4, hour_error_np_s4 = hour_np_s4.mean(axis=0), hour_np_s4.std(axis=0)
        mhour_mean_np_s1, mhour_mean_np_s2, mhour_mean_np_s3, mhour_mean_np_s4 = [], [], [], []
        mhour_err_np_s1, mhour_err_np_s2, mhour_err_np_s3, mhour_err_np_s4 = [],[],[],[]

        Time_scale = np.arange(1, 25)
        m_xasix = []
        for m in range(len(mhour_np_s1)):
            mhour_mean_np_s1.append(mhour_np_s1[m].mean(axis=0))
            mhour_mean_np_s2.append(mhour_np_s2[m].mean(axis=0))
            mhour_mean_np_s3.append(mhour_np_s3[m].mean(axis=0))
            mhour_mean_np_s4.append(mhour_np_s4[m].mean(axis=0))
            mhour_mean_np.append(mhour_np[m].mean(axis=1))
            mhour_err_np_s1.append(mhour_np_s1[m].std(axis=0))
            mhour_err_np_s2.append(mhour_np_s2[m].std(axis=0))
            mhour_err_np_s3.append(mhour_np_s3[m].std(axis=0))
            mhour_err_np_s4.append(mhour_np_s4[m].std(axis=0))
            mhour_err_np.append(mhour_np[m].std(axis=1))
            m_xasix.append(Time_scale)

        obs = [hour_mean_np, hour_mean_np_s1, hour_mean_np_s2, hour_mean_np_s3, hour_mean_np_s4, Time_scale, Time_scale, Time_scale, Time_scale, Time_scale]
        mod = [mhour_mean_np, mhour_mean_np_s1, mhour_mean_np_s2, mhour_mean_np_s3, mhour_mean_np_s4, m_xasix, m_xasix, m_xasix, m_xasix,
               m_xasix]
        scores = []
        for j, site in enumerate(self.sitename):

            if self.sitename.mask[j]:
                continue
            fig0 = plt.figure(figsize=(7, 9))
            print('Process on day_cycle_' + site + '_No.' + str(j) + '!')
            ''' Observations data need to use masked '''
            fig0.subplots_adjust(wspace=0.03, hspace=0.1)
            fig0, ax0, ax1, ax2, ax3, ax4, samples = plot_day_cycle_categories(fig0, obs, mod, j, 811, 812, 813, 814, 815, 313, 20)

            model_score = time_basic_score5(samples)
            scores.append(model_score)

            ax0.fill_between(Time_scale[~hour_mean_np[j, :].mask], hour_mean_np_s1[j, :][~hour_mean_np[j, :].mask] - hour_error_np_s1[j, :][~hour_mean_np[j, :].mask],
                                   hour_mean_np_s1[j, :][~hour_mean_np[j, :].mask] + hour_error_np_s1[j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
                                   facecolor='gray',
                                   linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax0.set_title('DJF')
            ax1.fill_between(Time_scale[~hour_mean_np[j, :].mask], hour_mean_np_s2[j, :][~hour_mean_np[j, :].mask] - hour_error_np_s2[j, :][~hour_mean_np[j, :].mask],
                                   hour_mean_np_s2[j, :][~hour_mean_np[j, :].mask] + hour_error_np_s2[j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
                                   facecolor='gray',
                                   linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax1.set_title('MAM')
            ax2.fill_between(Time_scale[~hour_mean_np[j, :].mask], hour_mean_np_s3[j, :][~hour_mean_np[j, :].mask] - hour_error_np_s3[j, :][~hour_mean_np[j, :].mask],
                                   hour_mean_np_s3[j, :][~hour_mean_np[j, :].mask] + hour_error_np_s3[j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
                                   facecolor='gray',
                                   linewidth=0.5, linestyle='dashdot', antialiased=True)
            # ax2.set_title('JJA')
            ax3.fill_between(Time_scale[~hour_mean_np[j, :].mask], hour_mean_np_s4[j, :][~hour_mean_np[j, :].mask] - hour_error_np_s4[j, :][~hour_mean_np[j, :].mask],
                                   hour_mean_np_s4[j, :][~hour_mean_np[j, :].mask] + hour_error_np_s4[j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
                                   facecolor='gray',
                                   linewidth=0.5, linestyle='dashdot', antialiased=True)

            ax4.fill_between(Time_scale[~hour_mean_np[j, :].mask], hour_mean_np[j, :][~hour_mean_np[j, :].mask] - hour_error_np[j, :][~hour_mean_np[j, :].mask],
                             hour_mean_np[j, :][~hour_mean_np[j, :].mask] + hour_error_np[j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
                             facecolor='gray',
                             linewidth=0.5, linestyle='dashdot', antialiased=True)


            # for m in range(len(mhour_mean_np_s1)):
            #     ax0.fill_between(Time_scale[~hour_mean_np[j, :].mask], mhour_mean_np_s1[m][j, :][~hour_mean_np[j, :].mask] - mhour_err_np_s1[m][j, :][~hour_mean_np[j, :].mask],
            #                      mhour_mean_np_s1[m][j, :][~hour_mean_np[j, :].mask] + mhour_err_np_s1[m][j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     # ax0.set_title('DJF')
            #     ax1.fill_between(Time_scale[~hour_mean_np[j, :].mask], mhour_mean_np_s2[m][j, :][~hour_mean_np[j, :].mask] - mhour_err_np_s2[m][j, :][~hour_mean_np[j, :].mask],
            #                      mhour_mean_np_s2[m][j, :][~hour_mean_np[j, :].mask] + mhour_err_np_s2[m][j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     # ax1.set_title('MAM')
            #     ax2.fill_between(Time_scale[~hour_mean_np[j, :].mask], mhour_mean_np_s3[m][j, :][~hour_mean_np[j, :].mask] - mhour_err_np_s3[m][j, :][~hour_mean_np[j, :].mask],
            #                      mhour_mean_np_s3[m][j, :][~hour_mean_np[j, :].mask] + mhour_err_np_s3[m][j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     # ax2.set_title('JJA')
            #     ax3.fill_between(Time_scale[~hour_mean_np[j, :].mask], mhour_mean_np_s4[m][j, :][~hour_mean_np[j, :].mask] - mhour_err_np_s4[m][j, :][~hour_mean_np[j, :].mask],
            #                      mhour_mean_np_s4[m][j, :][~hour_mean_np[j, :].mask] + mhour_err_np_s4[m][j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
            #                      linewidth=0.5, linestyle='dashdot', antialiased=True)
            #     ax4.fill_between(Time_scale[~hour_mean_np[j, :].mask], mhour_mean_np[m][j, :][~hour_mean_np[j, :].mask] - mhour_err_np[m][j, :][~hour_mean_np[j, :].mask],
            #                  mhour_mean_np[m][j, :][~hour_mean_np[j, :].mask]+ mhour_err_np[m][j, :][~hour_mean_np[j, :].mask], alpha=0.1, edgecolor='#1B2ACC',
            #                  linewidth=0.5, linestyle='dashdot', antialiased=True)

            ax4.set_title('Diurnal cycle')
            ax0.set_ylabel('DJF', fontsize=fontsize)
            ax0.yaxis.set_label_position("right")
            ax1.set_ylabel('MAM', fontsize=fontsize)
            ax1.yaxis.set_label_position("right")
            ax2.set_ylabel('JJA', fontsize=fontsize)
            ax2.yaxis.set_label_position("right")
            ax3.set_ylabel('SON', fontsize=fontsize)
            ax3.yaxis.set_label_position("right")
            ax4.set_ylabel('Diurnal', fontsize=fontsize)
            ax4.yaxis.set_label_position("right")
            ax0.grid(False)
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax4.grid(False)

            fig0.text(0.04, 0.7, self.variable + '(' + self.h_unit_obs + ')', va='center', rotation='vertical')
            ax0.set_xlim([1, 24])
            ax1.set_xlim([1, 24])
            ax2.set_xlim([1, 24])
            ax3.set_xlim([1, 24])
            ax4.set_xlim([1, 24])

            ax0.set_xticklabels([])
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            # ax3.set_xticklabels([])
            ax4.set_xticklabels([])

            ax0.legend(bbox_to_anchor=(1.3, 0.7), borderaxespad=0., fontsize=lengendfontsize)

            ax3.xaxis.set_ticks(range(1, 25))
            ax3.set_xticklabels([str(x) + '' for x in range(1, 25)])
            # fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
            fig0.savefig(self.filedir + self.variable + '/' + site + '_' + 'day_' + self.variable + '.png', bbox_inches='tight')
            plt.close('all')
        scores = np.asarray(scores)
        return scores




def cycle_analysis(variable_name, h_unit_obs, d_unit_obs,m_unit_obs, y_unit_obs, h_site_name_obs, filedir, o_h_s1, o_h_s2, o_h_s3, o_h_s4, m_h_s1, m_h_s2, m_h_s3, m_h_s4, o_hour_data, o_daily_data, o_monthly_data, o_seasonly_data, m_hour_data, m_daily_data, m_monthly_data, m_seasonly_data, hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod):
    # plot all cycle graph, day, season, four
    f1 = cycle_post(variable_name, h_site_name_obs, filedir, h_unit_obs, d_unit_obs, m_unit_obs, y_unit_obs)
    scores_day_cycle = f1.plot_days_cycle(o_h_s1, o_h_s2, o_h_s3, o_h_s4, m_h_s1, m_h_s2, m_h_s3, m_h_s4, o_hour_data, m_hour_data )
    scores_three_cycle = f1.plot_three_cycle(o_hour_data, o_daily_data, o_monthly_data, o_seasonly_data, m_hour_data, m_daily_data, m_monthly_data, m_seasonly_data)
    return scores_day_cycle, scores_three_cycle
