import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from score_post import time_basic_score3
from taylorDiagram import plot_Taylor_graph_time_basic
from taylorDiagram import plot_Taylor_graph_season_cycle
from score_post import time_basic_score5


start_year = 1991
end_year = 2014

fontsize = 10
plt.rcParams.update({'font.size': 10})
lengendfontsize = 10
col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']

test_variables = ['GPP', 'NEE', 'ET', 'EFLX_LH_TOT', 'ER','ET']

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

def day_seasonly_process(hour_data):
    # return shape (season, site, year)
    # data = data.reshape(len(data),len(data[0])/12, 4, 3)
    hour_data_s1, hour_data_s2, hour_data_s3, hour_data_s4 = [], [], [], []
    for y in range(len(hour_data[0]) / 365):
        for d in range(0, 365):
            if d <= 58:
                hour_data_s1.append(hour_data[0:len(hour_data), y * 365 + d])
            elif d <= 151:
                hour_data_s2.append(hour_data[0:len(hour_data), y * 365 + d])
            elif d <= 242:
                hour_data_s3.append(hour_data[0:len(hour_data), y * 365 + d])
            elif d <= 334:
                hour_data_s4.append(hour_data[0:len(hour_data), y * 365 + d])
            else:
                hour_data_s1.append(hour_data[0:len(hour_data), y * 365 + d])

    hour_data_s1 = np.asarray(hour_data_s1)
    hour_data_s2 = np.asarray(hour_data_s2)
    hour_data_s3 = np.asarray(hour_data_s3)
    hour_data_s4 = np.asarray(hour_data_s4)

    hour_data_s1 = np.ma.masked_invalid(hour_data_s1)
    hour_data_s2 = np.ma.masked_invalid(hour_data_s2)
    hour_data_s3 = np.ma.masked_invalid(hour_data_s3)
    hour_data_s4 = np.ma.masked_invalid(hour_data_s4)

    hour_data_s1 = np.ma.fix_invalid(hour_data_s1)
    hour_data_s2 = np.ma.fix_invalid(hour_data_s2)
    hour_data_s3 = np.ma.fix_invalid(hour_data_s3)
    hour_data_s4 = np.ma.fix_invalid(hour_data_s4)

    hour_data_s1 = np.ma.masked_where(hour_data_s1 > 9.96921e+12, hour_data_s1)
    hour_data_s2 = np.ma.masked_where(hour_data_s2 > 9.96921e+12, hour_data_s2)
    hour_data_s3 = np.ma.masked_where(hour_data_s3 > 9.96921e+12, hour_data_s3)
    hour_data_s4 = np.ma.masked_where(hour_data_s4 > 9.96921e+12, hour_data_s4)

    return hour_data_s1,hour_data_s2,hour_data_s3,hour_data_s4

def day_models_seasonly_process(m):
    hour_data_s1, hour_data_s2, hour_data_s3, hour_data_s4 = [],[],[],[]
    for i in range(len(m)):
        s1,s2,s3,s4 = day_seasonly_process(m[i])
        hour_data_s1.append(s1)
        hour_data_s2.append(s2)
        hour_data_s3.append(s3)
        hour_data_s4.append(s4)
    return hour_data_s1, hour_data_s2, hour_data_s3, hour_data_s4



def plot_time_basics_categories(fig0, obs, mod, j, rect1, rect2, rect3, rect, ref_times):
    # organize the data for taylor gram and plot
    [h_obs, d_obs, m_obs, y_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs] = obs
    [h_mod, d_mod, m_mod, y_mod, h_t_mod, d_t_mod, m_t_mod, y_t_mod] = mod

    data1 = h_obs[j, :][~h_obs[j, :].mask]
    data2 = d_obs[j, :][~d_obs[j, :].mask]
    data3 = m_obs[j, :][~m_obs[j, :].mask]

    models1, models2, models3 = [], [], []
    h_m, d_m, m_m, h_m_s, d_m_s, m_m_s = None, None, None, None, None, None
    for i in range(len(d_mod)):
        models1.append(h_mod[i][j, :][~h_obs[j, :].mask])
        models2.append(d_mod[i][j, :][~d_obs[j, :].mask])
        models3.append(m_mod[i][j, :][~m_obs[j, :].mask])

    fig0, samples1, samples2, samples3 = plot_Taylor_graph_time_basic(data1, data2, data3, models1, models2, models3, fig0, rect=rect, ref_times=ref_times, bbox_to_anchor=(0.9, 0.45))


    ax0 = fig0.add_subplot(rect1)
    ax1 = fig0.add_subplot(rect2)
    ax2 = fig0.add_subplot(rect3)


    if len(data1) > 0:
        cm = plt.cm.get_cmap('RdYlBu')
        h_y = (max(np.max(data1), h_m) * 1.1 * np.ones(len(h_obs[j, :])))
        ax0.scatter(h_t_obs, h_y, c=h_obs[j, :].mask, marker='s', cmap=cm, s=1)
        d_y = (max(np.max(data2), d_m) * 1.1 * np.ones(len(d_obs[j, :])))
        ax1.scatter(d_t_obs, d_y, c=d_obs[j, :].mask, marker='s', cmap=cm, s=1)
        m_y = (max(np.max(data3), m_m) * 1.1 * np.ones(len(m_obs[j, :])))
        ax2.scatter(m_t_obs, m_y, c=m_obs[j, :].mask, marker='s', cmap=cm, s=1)
        ax0.set_ylim(min_none(np.min(data1), h_m_s), max_none(np.max(data1), h_m) * 1.15)
        ax1.set_ylim(min_none(np.min(data2), d_m_s), max_none(np.max(data2), d_m) * 1.15)
        ax2.set_ylim(min_none(np.min(data3), m_m_s), max_none(np.max(data3), m_m) * 1.15)

    else:
        # cm = plt.cm.get_cmap('RdYlBu')
        h_y = (1 * np.ones(len(h_t_obs)))
        ax0.scatter(h_t_obs, h_y, c=h_obs[j, :].mask, marker='s', cmap='Blues', s=1)
        d_y = (1* np.ones(len(d_t_obs)))
        ax1.scatter(d_t_obs, d_y, c=d_obs[j, :].mask, marker='s', cmap='Blues', s=1)
        m_y = (1 * np.ones(len(m_t_obs)))
        ax2.scatter(m_t_obs, m_y, c=m_obs[j, :].mask, marker='s', cmap='Blues', s=1)


    h_t_obs, d_t_obs, m_t_obs = h_t_obs[~h_obs[j, :].mask], d_t_obs[~d_obs[j, :].mask], m_t_obs[~m_obs[j, :].mask]

    ax0.plot(h_t_obs, data1, 'k-', label='Observed')
    ax1.plot(d_t_obs, data2, 'k-', label='Observed')
    ax2.plot(m_t_obs, data3, 'k-', label='Observed')

    for i in range(len(h_mod)):
        ax0.plot(h_t_obs, models1[i], '-', label= "Model " + str(i + 1), color=col[i])
        ax1.plot(d_t_obs, models2[i], '-', label= "Model " + str(i + 1), color=col[i])
        ax2.plot(m_t_obs, models3[i], '-', label= "Model " + str(i + 1), color=col[i])

    return fig0, ax0, ax1, ax2, [samples1, samples2, samples3]



def plot_season_cycle_categories(fig0, obs, mod, j, rect0, rect1, rect2, rect3, rect4, rect, ref_times):

    # organize the data for taylor gram and plot
    [s_obs, h_obs, d_obs, m_obs, y_obs, s_t_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs] = obs
    [s_mod, h_mod, d_mod, m_mod, y_mod, s_t_mod, h_t_mod, d_t_mod, m_t_mod, y_t_mod] = mod

    data1 = h_obs[j, :][~s_obs[j, :].mask]
    data2 = d_obs[j, :][~s_obs[j, :].mask]
    data3 = m_obs[j, :][~s_obs[j, :].mask]
    data4 = y_obs[j, :][~s_obs[j, :].mask]
    data0 = s_obs[j, :][~s_obs[j, :].mask]

    s_t_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs = s_t_obs[~s_obs[j, :].mask], h_t_obs[~s_obs[j, :].mask], d_t_obs[
        ~s_obs[j, :].mask], m_t_obs[~s_obs[j, :].mask], y_t_obs[~s_obs[j, :].mask]


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

    fig0, samples1, samples2, samples3, samples4, samples5 = plot_Taylor_graph_season_cycle(data1, data2, data3, data4,
                                                                                         data0, models1, models2,
                                                                                         models3, models4, models5,
                                                                                         fig0, rect=rect,
                                                                                         ref_times=ref_times,
                                                                                         bbox_to_anchor=(1.01, 0.33))
    ax0 = fig0.add_subplot(rect1)
    ax1 = fig0.add_subplot(rect2)
    ax2 = fig0.add_subplot(rect3)
    ax3 = fig0.add_subplot(rect4)
    ax4 = fig0.add_subplot(rect0)


    ax0.plot(h_t_obs, data1, 'k-', label='Observed')
    ax1.plot(d_t_obs, data2, 'k-', label='Observed')
    ax2.plot(m_t_obs, data3, 'k-', label='Observed')
    ax3.plot(y_t_obs, data4, 'k-', label='Observed')
    ax4.plot(s_t_obs, data0, 'k-', label='Observed')


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

    for i in range(len(h_mod)):
        ax0.plot(h_t_obs, models1[i], '-', label="Model " + str(i + 1), color=col[i])
        ax1.plot(d_t_obs, models2[i], '-', label="Model " + str(i + 1), color=col[i])
        ax2.plot(m_t_obs, models3[i], '-', label="Model " + str(i + 1), color=col[i])
        ax3.plot(y_t_obs, models4[i], '-', label="Model " + str(i + 1), color=col[i])
        ax4.plot(s_t_obs, models5[i], '-', label="Model " + str(i + 1), color=col[i])
    # print(m_t_obs)
    m_d_obs = np.asarray([str(start_year + int(x) / 365) for x in m_t_obs])#
    # print(m_d_obs)
    # hello
    if len(data1) > 0 and len(data2) > 0 and len(data3) > 0 and len(data4) > 0 and len(data0) > 0:
        ax3.xaxis.set_ticks(
            [m_t_obs[0], m_t_obs[len(m_t_obs) / 5], m_t_obs[2 * len(m_t_obs) / 5], m_t_obs[3 * len(m_t_obs) / 5],
             m_t_obs[4 * len(m_t_obs) / 5]])

        ax3.set_xticklabels(
            [m_d_obs[0], m_d_obs[len(m_d_obs) / 5], m_d_obs[2 * len(m_d_obs) / 5], m_d_obs[3 * len(m_d_obs) / 5],
             m_d_obs[4 * len(m_d_obs) / 5]])

    return fig0, ax0, ax1, ax2, ax3, ax4, [samples1, samples2, samples3, samples4, samples5]




class basic_post(object):

    def __init__(self, variable, site_name, filedir, h_unit_obs, d_unit_obs, m_unit_obs, y_unit_obs):
        self.variable = variable
        self.sitename = site_name
        self.filedir = filedir
        self.h_unit_obs, self.d_unit_obs, self.m_unit_obs, self.y_unit_obs = h_unit_obs, d_unit_obs, m_unit_obs, y_unit_obs


    def plot_time_series(self, hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod, score=True):
        [h_obs, h_t_obs, _] = hour_obs
        [h_mod, h_t_mod, _] = hour_mod
        [m_obs, m_t_obs, _] = month_obs
        [m_mod, m_t_mod, _] = month_mod
        [d_obs, d_t_obs, _] = day_obs
        [d_mod, d_t_mod, _] = day_mod
        [y_obs, y_t_obs, _] = year_obs
        [y_mod, y_t_mod, _] = year_mod

        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on time_basic_' + site + '_No.' + str(j) + '!')
            obs = [h_obs, d_obs, m_obs, y_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs]
            mod = [h_mod, d_mod, m_mod, y_mod, h_t_mod, d_t_mod, m_t_mod, y_t_mod]

            if score:
                fig0 = plt.figure(figsize=(8, 11))

                fig0, ax0, ax1, ax2, samples = plot_time_basics_categories(fig0, obs, mod, j, 611, 612, 613, 212, 10)

                model_score = time_basic_score3(samples)
                scores.append(model_score)
                plt.suptitle('Time series')
                ax0.set_xlabel('Hourly', fontsize=fontsize)
                ax0.set_ylabel(self.variable + '\n' + self.h_unit_obs + '', fontsize=fontsize)
                ax1.set_xlabel('Daily', fontsize=fontsize)
                ax1.set_ylabel(self.variable + '\n' + self.d_unit_obs + '', fontsize=fontsize)
                ax2.set_xlabel('Monthly', fontsize=fontsize)
                ax2.set_ylabel(self.variable + '\n' + self.m_unit_obs + '', fontsize=fontsize)

                ax0.grid(False)
                ax1.grid(False)
                ax2.grid(False)

                m_d_obs = np.asarray(
                    [str(start_year + x / 12) + ('0' + str(x % 12 + 1) if x % 12 < 9 else str(x % 12 + 1)) for x in
                     np.arange(0, 12 * (end_year - start_year + 1))])
                d_d_obs = np.asarray([str(start_year + x / 365) + (
                '0' + str(x % 365 / 31 + 1) if x % 365 / 31 < 9 else str(x % 365 / 31 + 1)) for x in
                                      np.arange(0, (365 * (end_year - start_year + 1)))])
                h_d_obs = np.asarray([str(start_year + (x / 24) / 365) + (
                '0' + str((x / 24) % 365 / 31 + 1) if (x / 24) % 365 / 31 < 9 else str((x / 24) % 365 / 31 + 1)) for x
                                      in np.arange(0, (365 * (end_year - start_year + 1) * 24))])

                ax0.xaxis.set_ticks([h_t_obs[0], h_t_obs[len(h_t_obs) / 5], h_t_obs[2 * len(h_t_obs) / 5],
                                     h_t_obs[3 * len(h_t_obs) / 5], h_t_obs[4 * len(h_t_obs) / 5]])
                ax1.xaxis.set_ticks([d_t_obs[0], d_t_obs[len(d_t_obs) / 5], d_t_obs[2 * len(d_t_obs) / 5],
                                     d_t_obs[3 * len(d_t_obs) / 5], d_t_obs[4 * len(d_t_obs) / 5]])
                ax2.xaxis.set_ticks([m_t_obs[0], m_t_obs[len(m_t_obs) / 5], m_t_obs[2 * len(m_t_obs) / 5],
                                     m_t_obs[3 * len(m_t_obs) / 5], m_t_obs[4 * len(m_t_obs) / 5]])

                ax0.set_xticklabels([h_d_obs[0], h_d_obs[len(h_d_obs) / 5], h_d_obs[2 * len(h_d_obs) / 5],
                                     h_d_obs[3 * len(h_d_obs) / 5], h_d_obs[4 * len(h_d_obs) / 5]])
                ax1.set_xticklabels([d_d_obs[0], d_d_obs[len(d_d_obs) / 5], d_d_obs[2 * len(d_d_obs) / 5],
                                     d_d_obs[3 * len(d_d_obs) / 5], d_d_obs[4 * len(d_d_obs) / 5]])
                ax2.set_xticklabels([m_d_obs[0], m_d_obs[len(m_d_obs) / 5], m_d_obs[2 * len(m_d_obs) / 5],
                                     m_d_obs[3 * len(m_d_obs) / 5], m_d_obs[4 * len(m_d_obs) / 5]])

                ax0.legend(bbox_to_anchor=(1.20, -0.5), shadow=False, fontsize=lengendfontsize)
                if len(self.variable) < 12:
                    fig0.tight_layout(rect=[0, 0.01, 1, 0.97])
                else:
                    fig0.subplots_adjust(wspace=0, hspace=1.0)

                fig0.savefig(self.filedir + self.variable + '/' + site + '_' + 'time_basic' +'_' + self.variable + '.png', bbox_inches='tight')
                plt.close('all')

        scores = np.asarray(scores)
        return scores

    def plot_season_cycle(self, o_seasonly_data, m_seasonly_data, year_obs, year_mod, month_obs, score=True):
        [y_obs, y_t_obs, y_unit_obs] = year_obs
        [y_mod, y_t_mod, y_unit_mod] = year_mod
        [m_obs, m_t_obs, m_unit_obs] = month_obs
        y_fit = []
        if self.variable in test_variables:
            for m in range(len(y_mod)):
                y_fit.append(y_mod[m]/12.0)
            y_obs = y_obs/12.0
            y_mod = y_fit

        mhour_mean_np_s1, mhour_mean_np_s2, mhour_mean_np_s3, mhour_mean_np_s4 = [], [], [], []
        m_xasix = []
        time = m_t_obs.reshape(len(m_t_obs) / 12, 12)

        for m in range(len(m_seasonly_data)):
            mhour_mean_np_s1.append(m_seasonly_data[m][0, :, :])
            mhour_mean_np_s2.append(m_seasonly_data[m][1, :, :])
            mhour_mean_np_s3.append(m_seasonly_data[m][2, :, :])
            mhour_mean_np_s4.append(m_seasonly_data[m][3, :, :])
            m_xasix.append(time[:, 0])

        obs = [y_obs, o_seasonly_data[0, :, :], o_seasonly_data[1, :, :], o_seasonly_data[2, :, :],
               o_seasonly_data[3, :, :], y_t_obs, time[:, 0], time[:, 0],
               time[:, 0], time[:, 0]]

        mod = [y_mod, mhour_mean_np_s1, mhour_mean_np_s2, mhour_mean_np_s3, mhour_mean_np_s4, y_t_mod, m_xasix, m_xasix,
               m_xasix,
               m_xasix]

        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on season_cycle_' + site + '_No.' + str(j) + '!')
            fig5 = plt.figure(figsize=(6, 10))
            fig5.subplots_adjust(wspace=0.03, hspace=0.1)
            fig5, ax0, ax1, ax2, ax3, ax4, samples = plot_season_cycle_categories(fig5, obs, mod, j, 811, 812, 813, 814,
                                                                                  815, 313, 3)

            model_score = time_basic_score5(samples)
            scores.append(model_score)
            # left, width = .25, .6
            # bottom, height = .25, .5
            # right = left + width
            # top = bottom + height

            ax0.set_ylabel('DJF', fontsize=fontsize)
            ax0.yaxis.set_label_position("right")
            ax1.set_ylabel('MAM', fontsize=fontsize)
            ax1.yaxis.set_label_position("right")
            ax2.set_ylabel('JJA', fontsize=fontsize)
            ax2.yaxis.set_label_position("right")
            ax3.set_ylabel('SON', fontsize=fontsize)
            ax3.yaxis.set_label_position("right")
            ax4.set_ylabel('Annual', fontsize=fontsize)
            ax4.yaxis.set_label_position("right")

            ax0.grid(False)
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax4.grid(False)

            fig5.text(0.04, 0.7, self.variable + '(' + self.m_unit_obs + ')', va='center', rotation='vertical')

            ax0.set_xticklabels([])
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            # ax3.set_xticklabels([])
            ax4.set_xticklabels([])
            [s_obs, h_obs, d_obs, m_obs, y_obs, s_t_obs, h_t_obs, d_t_obs, m_t_obs, y_t_obs] = obs

            data1 = h_obs[j, :][~s_obs[j, :].mask]
            data2 = d_obs[j, :][~s_obs[j, :].mask]
            data3 = m_obs[j, :][~s_obs[j, :].mask]
            data4 = y_obs[j, :][~s_obs[j, :].mask]
            data0 = s_obs[j, :][~s_obs[j, :].mask]

            if len(data1) > 0 and len(data2) > 0 and len(data3) > 0 and len(data4) > 0 and len(data0) > 0:
                if site == 'AT-Neu':
                    ax0.legend(bbox_to_anchor=(1.3, 0.7), borderaxespad=0., fontsize=lengendfontsize)
                else:
                    ax0.legend(bbox_to_anchor=(1.1, 0.7), borderaxespad=0., fontsize=lengendfontsize)
            # if len(self.variable) < 12:
            #     fig5.tight_layout(rect=[0, 0.01, 1, 0.95])
            # else:
            #
            # plt.tight_layout(rect=[0, 0.01, 1, 0.98])
            ax4.set_title('Annual and seasonal time series')

            fig5.savefig(self.filedir + self.variable + '/' + site + '_season_' + self.variable + '.png',
                     bbox_inches='tight')
            plt.close('all')
        scores = np.asarray(scores)
        return scores

    def plot_cdf_pdf(self, hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod, score=True):

        [h_obs, h_t_obs, _] = hour_obs
        [h_mod, h_t_mod, _] = hour_mod
        [m_obs, m_t_obs, _] = month_obs
        [m_mod, m_t_mod, _] = month_mod
        [d_obs, d_t_obs, _] = day_obs
        [d_mod, d_t_mod, _] = day_mod
        [y_obs, y_t_obs, _] = year_obs
        [y_mod, y_t_mod, _] = year_mod
        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on CDF_' + site + '_No.' + str(j) + '!')
            h_obs_sorted = np.ma.sort(h_obs[j, :]).compressed()
            d_obs_sorted = np.ma.sort(d_obs[j, :]).compressed()
            m_obs_sorted = np.ma.sort(m_obs[j, :]).compressed()
            y_obs_sorted = np.ma.sort(y_obs[j, :]).compressed()
            # print(h_obs[j,:].shape)
            # print(h_obs_sorted)
            p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
            p2_data = 1. * np.arange(len(d_obs_sorted)) / (len(d_obs_sorted) - 1)
            p3_data = 1. * np.arange(len(m_obs_sorted)) / (len(m_obs_sorted) - 1)
            p4_data = 1. * np.arange(len(y_obs_sorted)) / (len(y_obs_sorted) - 1)
            fig1 = plt.figure(figsize=(6, 9))
            ax4 = fig1.add_subplot(4, 1, 1)
            ax5 = fig1.add_subplot(4, 1, 2)
            ax6 = fig1.add_subplot(4, 1, 3)
            ax7 = fig1.add_subplot(4, 1, 4)
            fig2 = plt.figure(figsize=(6, 9))
            ax0 = fig2.add_subplot(4, 1, 1)
            ax1 = fig2.add_subplot(4, 1, 2)
            ax2 = fig2.add_subplot(4, 1, 3)
            ax3 = fig2.add_subplot(4, 1, 4)


            ax4.plot(h_obs_sorted, p1_data, 'k-', label='Observed')
            ax5.plot(d_obs_sorted, p2_data, 'k-', label='Observed')
            ax6.plot(m_obs_sorted, p3_data, 'k-', label='Observed')
            ax7.plot(y_obs_sorted, p4_data, 'k-', label='Observed')

            if np.int(len(h_obs_sorted)/2160) > 0:
                p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)/2160))# bin it into n = N/10 bins
                x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
                p_d, x_d = np.histogram(d_obs_sorted, bins=np.int(len(d_obs_sorted)/90))  # bin it into n = N/10 bins
                x_d = x_d[:-1] + (x_d[1] - x_d[0]) / 2  # convert bin edges to centers
                p_m, x_m = np.histogram(m_obs_sorted, bins=np.int(len(m_obs_sorted)/1))  # bin it into n = N/1 bins
                x_m = x_m[:-1] + (x_m[1] - x_m[0]) / 2  # convert bin edges to centers
                p_y, x_y = np.histogram(y_obs_sorted, bins=np.int(len(y_obs_sorted)/1))  # bin it into n = N/1 bins
                x_y = x_y[:-1] + (x_y[1] - x_y[0]) / 2  # convert bin edges to centers

                ax0.plot(x_h, p_h/float(sum(p_h)), 'k-', label='Observed')
                ax1.plot(x_d, p_d/float(sum(p_d)), 'k-', label='Observed')
                ax2.plot(x_m, p_m/float(sum(p_m)), 'k-', label='Observed')
                ax3.plot(x_y, p_y/float(sum(p_y)), 'k-', label='Observed')

            model_score = []
            import scipy
            for i in range(len(d_mod)):
                ax4.plot(np.ma.sort((h_mod[i][j, :][~h_obs[j, :].mask])), p1_data, label="Model "+str(i+1), color=col[i])
                ax5.plot(np.ma.sort((d_mod[i][j, :][~d_obs[j, :].mask])), p2_data, label="Model "+str(i+1), color=col[i])
                ax6.plot(np.ma.sort((m_mod[i][j, :][~m_obs[j, :].mask])), p3_data, label="Model "+str(i+1), color=col[i])
                ax7.plot(np.ma.sort((y_mod[i][j, :][~y_obs[j, :].mask])), p4_data, label="Model "+str(i+1), color=col[i])
                if np.int(len(h_obs_sorted) / 2160) > 0:
                    # print(np.ma.sort((h_mod[i][j, :][~h_obs[j, :].mask])))
                    # print(len(h_obs_sorted) / 2160)
                    p_h, x_h = np.histogram(np.ma.sort((h_mod[i][j, :][~h_obs[j, :].mask])).compressed(), bins=len(h_obs_sorted) / 2160)  # bin it into n = N/10 bins
                    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
                    p_d, x_d = np.histogram(np.ma.sort((d_mod[i][j, :][~d_obs[j, :].mask])).compressed(), bins=len(d_obs_sorted) / 90)  # bin it into n = N/10 bins
                    x_d = x_d[:-1] + (x_d[1] - x_d[0]) / 2  # convert bin edges to centers.compressed()
                    p_m, x_m = np.histogram(np.ma.sort((m_mod[i][j, :][~m_obs[j, :].mask])).compressed(), bins=len(m_obs_sorted) / 3)  # bin it into n = N/10 bins
                    x_m = x_m[:-1] + (x_m[1] - x_m[0]) / 2  # convert bin edges to centers
                    p_y, x_y = np.histogram(np.ma.sort((y_mod[i][j, :][~y_obs[j, :].mask])).compressed(), bins=len(y_obs_sorted) / 1)  # bin it into n = N/10 bins
                    x_y = x_y[:-1] + (x_y[1] - x_y[0]) / 2  # convert bin edges to centers
                    ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[i])
                    ax1.plot(x_d, p_d / float(sum(p_d)), label="Model "+str(i+1), color=col[i])
                    ax2.plot(x_m, p_m / float(sum(p_m)), label="Model "+str(i+1), color=col[i])
                    ax3.plot(x_y, p_y / float(sum(p_y)), label="Model "+str(i+1), color=col[i])

                # k1, b1 = scipy.stats.ks_2samp(h_obs[j, :].compressed(), h_mod[i][j, :][~h_obs[j, :].mask])
                # k2, b2 = scipy.stats.ks_2samp(d_obs[j, :].compressed(), d_mod[i][j, :][~d_obs[j, :].mask])
                # k3, b3 = scipy.stats.ks_2samp(m_obs[j, :].compressed(), m_mod[i][j, :][~m_obs[j, :].mask])
                # k4, b4 = scipy.stats.ks_2samp(y_obs[j, :].compressed(), y_mod[i][j, :][~y_obs[j, :].mask])
                # model_score.append(1-min(b1,b2,b3,b4)/max(b1,b2,b3,b4))
                model_score =[]
            scores.append(model_score)
            fontsize = 12

            plt.suptitle('PDF and CDF')
            ax4.set_ylabel('CDF (Hourly)',fontsize=fontsize)
            ax4.set_xlabel(self.variable + '( ' + self.h_unit_obs + ' )', fontsize=fontsize)
            ax5.set_ylabel('CDF (Daily)',fontsize=fontsize)
            ax5.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax6.set_ylabel('CDF (Monthly)',fontsize=fontsize)
            ax6.set_xlabel(self.variable + '( ' + self.m_unit_obs + ' )', fontsize=fontsize)
            ax7.set_ylabel('CDF (Annually)',fontsize=fontsize)
            ax7.set_xlabel(self.variable + '( ' + self.y_unit_obs + ' )', fontsize=fontsize)
            ax4.grid(False)
            ax5.grid(False)
            ax6.grid(False)
            ax7.grid(False)


            ax4.legend(bbox_to_anchor=(1.23,-0.5), shadow=False, fontsize=lengendfontsize)
            fig1.tight_layout(rect=[0, 0.01, 1, 0.97])

            fig1.savefig(
                self.filedir + self.variable + '/' + site + '_' + 'cdf' + '_' + self.variable + '.png',
                bbox_inches='tight')

            ax0.set_ylabel('PDF (Hourly)',fontsize=fontsize)
            ax0.set_xlabel(self.variable + '( ' + self.h_unit_obs + ' )', fontsize=fontsize)
            ax1.set_ylabel('PDF (Daily)',fontsize=fontsize)
            ax1.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax2.set_ylabel('PDF (Monthly)',fontsize=fontsize)
            ax2.set_xlabel(self.variable + '( ' + self.m_unit_obs + ' )', fontsize=fontsize)
            ax3.set_ylabel('PDF (Annually)',fontsize=fontsize)
            ax3.set_xlabel(self.variable + '( ' + self.y_unit_obs + ' )', fontsize=fontsize)

            ax0.grid(False)
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)

            ax0.legend(bbox_to_anchor=(1.23,-0.5), shadow=False, fontsize=lengendfontsize)

            fig2.tight_layout(rect=[0, 0.01, 1, 0.95])

            fig2.savefig(
                self.filedir  + self.variable + '/' + site + '_' + 'pdf' + '_' + self.variable + '.png',
                bbox_inches='tight')
            plt.close('all')
        scores = np.asarray(scores)
        return scores

    def plot_season_cdf_pdf(self, day_obs, day_mod, score=True):
        [d_obs, d_t_obs, _] = day_obs
        [d_mod, d_t_mod, _] = day_mod
        h_obs, d_obs, m_obs, y_obs = day_seasonly_process(d_obs)
        model1,model2,model3,model4  = day_models_seasonly_process(d_mod)

        # print(season_data.shape)
        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on season_CDF_' + site + '_No.' + str(j) + '!')
            h_obs_sorted = np.ma.sort(h_obs[:, j]).compressed()
            d_obs_sorted = np.ma.sort(d_obs[:, j]).compressed()
            m_obs_sorted = np.ma.sort(m_obs[:, j]).compressed()
            y_obs_sorted = np.ma.sort(y_obs[:, j]).compressed()
            p1_data = 1. * np.arange(len(h_obs_sorted)) / (len(h_obs_sorted) - 1)
            p2_data = 1. * np.arange(len(d_obs_sorted)) / (len(d_obs_sorted) - 1)
            p3_data = 1. * np.arange(len(m_obs_sorted)) / (len(m_obs_sorted) - 1)
            p4_data = 1. * np.arange(len(y_obs_sorted)) / (len(y_obs_sorted) - 1)
            fig1 = plt.figure(figsize=(6, 9))
            ax4 = fig1.add_subplot(4, 1, 1)
            ax5 = fig1.add_subplot(4, 1, 2)
            ax6 = fig1.add_subplot(4, 1, 3)
            ax7 = fig1.add_subplot(4, 1, 4)
            fig2 = plt.figure(figsize=(6, 9))
            ax0 = fig2.add_subplot(4, 1, 1)
            ax1 = fig2.add_subplot(4, 1, 2)
            ax2 = fig2.add_subplot(4, 1, 3)
            ax3 = fig2.add_subplot(4, 1, 4)


            ax4.plot(h_obs_sorted, p1_data, 'k-', label='Observed')
            ax5.plot(d_obs_sorted, p2_data, 'k-', label='Observed')
            ax6.plot(m_obs_sorted, p3_data, 'k-', label='Observed')
            ax7.plot(y_obs_sorted, p4_data, 'k-', label='Observed')

            if np.int(len(h_obs_sorted)/20) > 0:
                p_h, x_h = np.histogram(h_obs_sorted, bins=np.int(len(h_obs_sorted)/20))# bin it into n = N/10 bins
                x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
                p_d, x_d = np.histogram(d_obs_sorted, bins=np.int(len(d_obs_sorted)/20))  # bin it into n = N/10 bins
                x_d = x_d[:-1] + (x_d[1] - x_d[0]) / 2  # convert bin edges to centers
                p_m, x_m = np.histogram(m_obs_sorted, bins=np.int(len(m_obs_sorted)/20))  # bin it into n = N/1 bins
                x_m = x_m[:-1] + (x_m[1] - x_m[0]) / 2  # convert bin edges to centers
                p_y, x_y = np.histogram(y_obs_sorted, bins=np.int(len(y_obs_sorted)/20))  # bin it into n = N/1 bins
                x_y = x_y[:-1] + (x_y[1] - x_y[0]) / 2  # convert bin edges to centers

                ax0.plot(x_h, p_h/float(sum(p_h)), 'k-', label='Observed')
                ax1.plot(x_d, p_d/float(sum(p_d)), 'k-', label='Observed')
                ax2.plot(x_m, p_m/float(sum(p_m)), 'k-', label='Observed')
                ax3.plot(x_y, p_y/float(sum(p_y)), 'k-', label='Observed')

            model_score = []
            import scipy
            for i in range(len(d_mod)):
                ax4.plot(np.ma.sort((model1[i][:, j][~h_obs[:, j].mask])), p1_data, label="Model "+str(i+1), color=col[i])
                ax5.plot(np.ma.sort((model2[i][:, j][~d_obs[:, j].mask])), p2_data, label="Model "+str(i+1), color=col[i])
                ax6.plot(np.ma.sort((model3[i][:, j][~m_obs[:, j].mask])), p3_data, label="Model "+str(i+1), color=col[i])
                ax7.plot(np.ma.sort((model4[i][:, j][~y_obs[:, j].mask])), p4_data, label="Model "+str(i+1), color=col[i])
                if np.int(len(h_obs_sorted) / 20) > 0:
                    # print(np.ma.sort((h_mod[i][j, :][~h_obs[j, :].mask])))
                    # print(len(h_obs_sorted) / 2160)
                    p_h, x_h = np.histogram(np.ma.sort((model1[i][:, j][~h_obs[:, j].mask])).compressed(), bins=len(h_obs_sorted) / 20)  # bin it into n = N/10 bins
                    x_h = x_h[:-1] + (x_h[1] - x_h[0]) / 2  # convert bin edges to centers
                    p_d, x_d = np.histogram(np.ma.sort((model2[i][:, j][~d_obs[:, j].mask])).compressed(), bins=len(d_obs_sorted) / 20)  # bin it into n = N/10 bins
                    x_d = x_d[:-1] + (x_d[1] - x_d[0]) / 2  # convert bin edges to centers.compressed()
                    p_m, x_m = np.histogram(np.ma.sort((model3[i][:, j][~m_obs[:, j].mask])).compressed(), bins=len(m_obs_sorted) / 20)  # bin it into n = N/10 bins
                    x_m = x_m[:-1] + (x_m[1] - x_m[0]) / 2  # convert bin edges to centers
                    p_y, x_y = np.histogram(np.ma.sort((model4[i][:, j][~y_obs[:, j].mask])).compressed(), bins=len(y_obs_sorted) / 20)  # bin it into n = N/10 bins
                    x_y = x_y[:-1] + (x_y[1] - x_y[0]) / 2  # convert bin edges to centers
                    ax0.plot(x_h, p_h / float(sum(p_h)), label="Model "+str(i+1), color=col[i])
                    ax1.plot(x_d, p_d / float(sum(p_d)), label="Model "+str(i+1), color=col[i])
                    ax2.plot(x_m, p_m / float(sum(p_m)), label="Model "+str(i+1), color=col[i])
                    ax3.plot(x_y, p_y / float(sum(p_y)), label="Model "+str(i+1), color=col[i])

                # k1, b1 = scipy.stats.ks_2samp(h_obs[j, :].compressed(), h_mod[i][j, :][~h_obs[j, :].mask])
                # k2, b2 = scipy.stats.ks_2samp(d_obs[j, :].compressed(), d_mod[i][j, :][~d_obs[j, :].mask])
                # k3, b3 = scipy.stats.ks_2samp(m_obs[j, :].compressed(), m_mod[i][j, :][~m_obs[j, :].mask])
                # k4, b4 = scipy.stats.ks_2samp(y_obs[j, :].compressed(), y_mod[i][j, :][~y_obs[j, :].mask])
                # model_score.append(1-min(b1,b2,b3,b4)/max(b1,b2,b3,b4))
                model_score =[]
            scores.append(model_score)
            fontsize = 12

            plt.suptitle('Seasonal PDF and CDF')
            ax4.set_ylabel('CDF (DJF)',fontsize=fontsize)
            ax4.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax5.set_ylabel('CDF (MAM)',fontsize=fontsize)
            ax5.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax6.set_ylabel('CDF (JJA)',fontsize=fontsize)
            ax6.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax7.set_ylabel('CDF (SOP)',fontsize=fontsize)
            ax7.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax4.grid(False)
            ax5.grid(False)
            ax6.grid(False)
            ax7.grid(False)


            ax4.legend(bbox_to_anchor=(1.23,-0.5), shadow=False, fontsize=lengendfontsize)
            fig1.tight_layout(rect=[0, 0.01, 1, 0.97])

            fig1.savefig(
                self.filedir + self.variable + '/' + site + '_' + 'season_cdf' + '_' + self.variable + '.png',
                bbox_inches='tight')

            ax0.set_ylabel('PDF (DJF)',fontsize=fontsize)
            ax0.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax1.set_ylabel('PDF (MAM)',fontsize=fontsize)
            ax1.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax2.set_ylabel('PDF (JJA)',fontsize=fontsize)
            ax2.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)
            ax3.set_ylabel('PDF (SOP)',fontsize=fontsize)
            ax3.set_xlabel(self.variable + '( ' + self.d_unit_obs + ' )', fontsize=fontsize)

            ax0.grid(False)
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)

            ax0.legend(bbox_to_anchor=(1.23, -0.5), shadow=False, fontsize=lengendfontsize)

            fig2.tight_layout(rect=[0, 0.01, 1, 0.95])

            fig2.savefig(
                self.filedir  + self.variable + '/' + site + '_' + 'season_pdf' + '_' + self.variable + '.png',
                bbox_inches='tight')
            plt.close('all')
        scores = np.asarray(scores)
        return scores



def time_analysis(variable_name, h_unit_obs, d_unit_obs,m_unit_obs, y_unit_obs, h_site_name_obs, filedir, hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod, o_seasonly_data, m_seasonly_data):
    f1 = basic_post(variable_name, h_site_name_obs, filedir, h_unit_obs, d_unit_obs, m_unit_obs, y_unit_obs)
    # scores_time_series = f1.plot_time_series(hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod, score=True)
    # scores_season_cycle = f1.plot_season_cycle(o_seasonly_data, m_seasonly_data, year_obs, year_mod, month_obs, score=True)
    # scores_pdf_cdf = f1.plot_cdf_pdf(hour_obs, hour_mod, day_obs, day_mod, month_obs, month_mod, year_obs, year_mod, score=True)
    season_score_pdf_cdf = f1.plot_season_cdf_pdf(day_obs, day_mod, score=True)
    return season_score_pdf_cdf, season_score_pdf_cdf, season_score_pdf_cdf