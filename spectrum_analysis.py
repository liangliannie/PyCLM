from taylorDiagram import plot_Taylor_graph
import matplotlib.pyplot as plt
from PyEMD import EEMD
from hht import hht
from hht import plot_imfs
from hht import plot_frequency
import waipy
from scipy.stats import norm
import numpy as np
# from matplotlib.ticker import FuncFormatter

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fY' % (x/365.0)

plt.rcParams.update({'font.size': 10})
fontsize = 13
lengendfontsize = 12
col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm']
start_year = 1991
end_year = 2014


# from matplotlib import colors as mcolors
# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
#                 for name, color in colors.items())
# col = [name for hsv, name in by_hsv][]
# def plot_imfs(signal, imfs, time_samples = None, fig=None):
#     ''' Author jaidevd https://github.com/jaidevd/pyhht/blob/dev/pyhht/visualization.py '''
#     '''Original function from pyhht, but without plt.show()'''
#     n_imfs = imfs.shape[0]
#     # print(np.abs(imfs[:-1, :]))
#     # axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
#     # Plot original signal
#     ax = plt.subplot(n_imfs + 1, 1, 1)
#     ax.plot(time_samples, signal)
#     ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
#     ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
#                    labelbottom=False)
#     ax.grid(False)
#     ax.set_ylabel('Signal')
#     ax.set_title('Empirical Mode Decomposition')
#
#     # Plot the IMFs
#     for i in range(n_imfs - 1):
#         # print(i + 2)
#         ax = plt.subplot(n_imfs + 1, 1, i + 2)
#         ax.plot(time_samples, imfs[i, :])
#         # ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
#         ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
#                        labelbottom=False)
#         ax.grid(False)
#         ax.set_ylabel('imf' + str(i + 1), fontsize=fontsize)
#
#     # Plot the residue
#     ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
#     ax.plot(time_samples, imfs[-1, :], 'r')
#     ax.axis('tight')
#     ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
#                    labelbottom=False)
#     ax.grid(False)
#     ax.set_ylabel('res.', fontsize=fontsize)
#     return ax

class spectrum_post(object):

    def __init__(self, filedir, h_site_name_obs, day_obs, day_mod, variable_name):
        [d_obs, d_t_obs, d_unit_obs] = day_obs
        [d_mod, d_t_mod, d_unit_mod] = day_mod
        self.d_obs = d_obs
        self.d_mod = d_mod
        self.d_t_obs = d_t_obs
        self.d_unit_obs = d_unit_obs
        self.sitename = h_site_name_obs
        self.variable = variable_name
        self.filedir = filedir


    def plot_decomposer_imf(self):

        d_obs = self.d_obs
        d_mod = self.d_mod
        d_t_obs = self.d_t_obs
        scores = []

        for j, site in enumerate(self.sitename):

            if self.sitename.mask[j]:
                continue
            print('Process on Decomposer_IMF_' + site + '_No.' + str(j) + '!')
            fig0 = plt.figure(figsize=(8, 4))
            fig3 = plt.figure(figsize=(5, 5))
            data = d_obs[j, :].compressed()
            time = d_t_obs[~d_obs[j, :].mask]
            eemd = EEMD(trials=5)


            if len(data) > 0:
                imfs = eemd.eemd(data)
                # print('obs',imfs.shape)
                if len(imfs) >= 1:
                    ax0 = fig0.add_subplot(1, 2, 1)
                    ax0.plot(time, (imfs[len(imfs) - 1]), 'k-', label='Observed')

                    d_d_obs = np.asarray([str(start_year + int(x) / 365) + (
                        '0' + str(int(x) % 365 / 31 + 1) if int(x) % 365 / 31 < 9 else str(int(x) % 365 / 31 + 1)) for x
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
                    fig3, freq = hht(data, imfs, time, 1, fig3)



            fig3.savefig(self.filedir + self.variable + '/' + site + '_hht_IMF_observed' + self.variable + '.png', bbox_inches='tight')

            fig1 = plt.figure(figsize=(4 * (len(d_mod)+1), 8))
            fig2 = plt.figure(figsize=(4 * (len(d_mod)+1), 8))
            fig1.subplots_adjust(wspace=0.5, hspace=0.3)
            fig2.subplots_adjust(wspace=0.5, hspace=0.3)

            if len(data) > 0:
                if len(imfs) >= 1:
                    fig1 = plot_imfs(data, imfs, time_samples=time, fig=fig1, no=1, m=len(d_mod))
                    fig2 = plot_frequency(data, freq.T, time_samples=time, fig=fig2, no=1, m=len(d_mod))

                models1 = []
                datamask = []
                data1 = imfs[len(imfs) - 1]
                for m in range(len(d_mod)):
                    ## hht spectrum
                    eemd = EEMD(trials=5)
                    fig3 = plt.figure(figsize=(5, 5))
                    data2 = d_mod[m][j, :][~d_obs[j, :].mask]
                    imfs = eemd.eemd(data2.compressed())
                    # print('mod'+str(m), imfs.shape)
                    if len(imfs) >= 1:
                        fig3, freq = hht(data2.compressed(), imfs, time[~data2.mask], 1, fig3)
                    fig3.savefig(self.filedir + self.variable + '/' + site + '_hht_IMF_model' + str(m + 1) + self.variable + '.png', bbox_inches='tight')
                    if len(imfs) >= 1:
                        fig1 = plot_imfs(data2.compressed(), imfs, time_samples=time[~data2.mask], fig=fig1, no=m+2, m=len(d_mod))
                        fig2 = plot_frequency(data2.compressed(), freq.T, time_samples=time[~data2.mask], fig=fig2, no=m+2, m=len(d_mod))
                        ax0.plot(time[~data2.mask], (imfs[len(imfs) - 1]), '-', label='Model' + str(m+1), c=col[m])
                        models1.append(imfs[len(imfs) - 1])
                        datamask.append(data2)

                ax0.set_xlabel('Time', fontsize=fontsize)
                ax0.set_ylabel('' + self.variable + '(' + self.d_unit_obs + ')', fontsize=fontsize)
                ax0.yaxis.tick_right()
                ax0.yaxis.set_label_position("right")
                ax0.legend(bbox_to_anchor=(-0.05, 1), shadow=False, fontsize='medium')
                plot_Taylor_graph(data1, models1, fig0, 122, datamask=datamask)
            else:
                print("'Data's length is too short !")

            fig1.savefig(self.filedir + self.variable + '/' + site + '_Decompose_IMF_' + self.variable + '.png', bbox_inches='tight')
            fig2.savefig(self.filedir + self.variable + '/' + site + '_deviation_IMF_' + self.variable + '.png', bbox_inches='tight')

            fig0.subplots_adjust(left=0.1, hspace=0.25, wspace=0.55)
            fig0.savefig(self.filedir  + self.variable + '/' + site + '_' + 'IMF' + '_' + self.variable + '.png', bbox_inches='tight')

            plt.close('all')

        return scores

    def plot_wavelet(self):
        d_obs = self.d_obs
        d_mod = self.d_mod
        d_t_obs = self.d_t_obs

        """ plot data wavelet """
        scores = []
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on Wavelet_' + site + '_No.' + str(j) + '!')
            data = d_obs[j, :].compressed()
            fig3 = plt.figure(figsize=(8, 8))
            if len(data) > 0:
                time_data = d_t_obs[~d_obs[j, :].mask]
                # time_data = d_t_obs
                result = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Obs')
                waipy.wavelet_plot('Obs', time_data, data, 0.03125, result, fig3, unit=self.d_unit_obs)
                # plt.tight_layout()
            for m in range(len(d_mod)):
                fig4 = plt.figure(figsize=(8, 8))
                data2 = d_mod[m][j, :][~d_obs[j, :].mask]
                data = data2.compressed() - d_obs[j, :].compressed()[~data2.mask]
                if len(data) > 0:
                    result = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Obs - Mod' + str(m+1))
                    waipy.wavelet_plot('Obs - Mod' + str(m+1), time_data[~data2.mask], data, 0.03125, result, fig4, unit=self.d_unit_obs, m=m)
                    # plt.tight_layout()
                fig4.savefig(self.filedir + self.variable + '/' + site + 'model' + str(m) + '_wavelet_' + self.variable + '.png', bbox_inches='tight')
            fig3.savefig(self.filedir + self.variable + '/' + site + '_Wavelet_' + self.variable + '.png',
                         bbox_inches='tight')
            plt.close('all')
        return scores


    def plot_spectrum(self):
        import waipy, math
        import numpy as np
        import matplotlib.pyplot as plt
        d_obs = self.d_obs
        d_mod = self.d_mod
        d_t_obs = self.d_t_obs
        scores = []
        """ Plot global wavelet spectrum """
        # col = ['palevioletred', 'm', 'plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen',
        #        'darkgreen', 'olivedrab', 'gold', 'tan', 'red']
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on Spectrum_' + site + '_No.' + str(j) + '!')
            data = d_obs[j, :].compressed()
            if len(data) > 0:
                result = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Data')
                loc_o, scale_o = norm.fit(result['global_ws'])
                fig4 = plt.figure(figsize=(4, 4))
                ax4 = fig4.add_subplot(1, 1, 1)
                # f1, sxx1 = waipy.fft(data)
                # ax.plot(np.log2(1 / f1 * result['dt']), sxx1, 'red', label='Fourier spectrum')
                # plt.suptitle(self.variable + ' ( ' + self.d_unit_obs + ' )', fontsize=8)
                ax4.plot(np.log2(result['period']), result['global_ws'], 'k-', label='Wavelet spectrum')
                ax4.plot(np.log2(result['period']), result['global_signif'], 'r--', label='95% confidence spectrum')
            model_score = []

            for m in range(len(d_mod)):
                data2 = d_mod[m][j, :][~d_obs[j, :].mask]
                data = data2.compressed()
                # data = d_mod[m][j, :][~d_obs[j, :].mask]
                if len(data) > 0:
                    result_temp = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Data')
                    loc_m, scale_m = norm.fit(result_temp['global_ws'])
                    model_score.append(abs(loc_m-loc_o))
                    ax4.plot(np.log2(result_temp['period']), result_temp['global_ws'], label='Model' + str(m), c=col[m])
                else:
                    model_score.append(0.5)
            model_score = [i/max(model_score) for i in model_score]
            scores.append(model_score)

            ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=lengendfontsize)
            # ax4.set_ylim(0, 1.25 * np.max(result['global_ws']))
            ax4.set_ylabel('Power', fontsize=fontsize)
            ax4.set_title('Global Wavelet Spectrum', fontsize=fontsize)
            y_min = int(min(np.log2(result['period'][0]), np.log2(result_temp['period'][0])))
            y_max = int(max(np.log2(result['period'][-1]) + 1, np.log2(result_temp['period'][-1]) + 1))
            yt = range(y_min, y_max, 3)  # create the vector of period
            Yticks = [float(math.pow(2, p)) for p in yt]  # make 2^periods
            ax4.set_xticks(yt)
            ax4.set_xticklabels(Yticks)
            ax4.set_xlim(xmin=(np.log2(np.min(result['period']))), xmax=(np.log2(np.max(result['period']))))
            plt.tight_layout()
            fig4.savefig(self.filedir  + self.variable + '/' + site + '_spectrum_' + self.variable + '.png', bbox_inches='tight')
            plt.close('all')
        return scores


    def plot_taylor_gram(self):
        d_obs = self.d_obs
        d_mod = self.d_mod
        d_t_obs = self.d_t_obs
        scores = []
        """  Taylor diagram """
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on Taylor_' + site + '_No.' + str(j) + '!')
            data1 = d_obs[j, :].compressed()
            models1 = []
            fig7 = plt.figure(figsize=(8, 8))
            for m in range(len(d_mod)):
                models1.append(d_mod[m][j, :][~d_obs[j, :].mask])

            # print(data1.shape, models1[0].shape)
            plot_Taylor_graph(data1, models1, fig7, 111)
            fig7.savefig(self.filedir + self.variable + '/' +site + '_taylor_' + self.variable + '.png', bbox_inches='tight')
            plt.close('all')
        return scores

    def plot_spectrum_score(self):
        import waipy, math
        import numpy as np
        import matplotlib.pyplot as plt
        d_obs = self.d_obs
        d_mod = self.d_mod
        d_t_obs = self.d_t_obs
        scores = []
        """ Plot global wavelet spectrum """
        for j, site in enumerate(self.sitename):
            if self.sitename.mask[j]:
                continue
            print('Process on Spectrum_' + site + '_No.' + str(j) + '!')
            data = d_obs[j, :].compressed()
            result = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Data')
            loc_o, scale_o = norm.fit(result['global_ws'])
            model_score = []
            for m in range(len(d_mod)):
                data = d_mod[m][j, :][~d_obs[j, :].mask]
                result_temp = waipy.cwt(data, 1, 1, 0.125, 2, 4 / 0.125, 0.72, 6, mother='Morlet', name='Data')
                loc_m, scale_m = norm.fit(result_temp['global_ws'])
                model_score.append(abs(loc_m - loc_o))
            model_score = model_score / max(model_score)
            scores.append(model_score)
        return scores

def spectrum_analysis(filedir,h_site_name_obs, day_obs, day_mod, variable_name):
    # plot frequency frequency
    f2 = spectrum_post(filedir, h_site_name_obs, day_obs, day_mod, variable_name)
    scores_decomposeimf = f2.plot_decomposer_imf()
    score_wavelet = f2.plot_wavelet()
    score_spectrum = f2.plot_spectrum()
    return scores_decomposeimf, score_wavelet, score_spectrum