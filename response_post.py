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
from variable_post import seasonly_process
from variable_post import models_seasonly_process
from scipy import signal
col = ['plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen', 'darkgreen',
       'olivedrab', 'gold', 'tan', 'red', 'palevioletred', 'm', 'plum']

from math import isnan
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnchoredText

plt.rcParams.update({'font.size': 12})
lengendfontsize = 8
responselist = ['SNOW', 'WIND', 'Humidity', 'Pressure', 'RAIN', 'FSDS', 'ET', 'TBOT']
Root_Dir2 = '/Users/lli51/Downloads/'


def correlation_matrix(datas):
    frame = pd.DataFrame(datas)
    A = frame.corr(method='pearson', min_periods=1)
    corr = np.ma.corrcoef(A)
    mask = np.zeros_like(A)
    mask[np.triu_indices_from(mask)] = True

    return corr, mask



def visualize3DData (X, fig, ax):
    """Visualize data in 3d plot with popover next to mouse position.

    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
    Returns:
        None
    """

    def distance(point, event):
        """Return distance between mouse position and given data point

        Args:
            point (np.array): np.array of shape (3,), with x,y,z in data coords
            event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
        Returns:
            distance (np.float64): distance (in screen coords) between mouse pos and data point
        """
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


    def calcClosestDatapoint(X, event):
        """"Calculate which data point is closest to the mouse position.

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            event (MouseEvent) - mouse event (containing mouse position)
        Returns:
            smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
        """
        distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)


    def annotatePlot(X, index):
        """Create popover label in 3d chart

        Args:
            X (np.array) - array of points, of shape (numPoints, 3)
            index (int) - index (into points array X) of item which should be printed
        Returns:
            None
        """
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate( "({},{},{})".format(X[index, 0], X[index, 1], X[index, 2]) % index,
            xy = (x2, y2), xytext = (-0.2, -0.1), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.1', fc = 'yellow', alpha = 0.5))#,
            # arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        fig.canvas.draw()


    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(X, event)
        annotatePlot (X, closestIndex)

    # ax.scatter(X[:,0], X[:,1], X[:,2], depthshade = False, picker = True)
    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion

    return fig, ax

def annotion(x, y, z, fig4, ax4, type=False):
    if type:
        fig4, ax4 = visualize3DData(np.asarray([[x, y, z]]), fig4, ax4)
    else:
        label3 = '(%1.2f,\n %1.2f,\n %1.2f)' % (x, y, z)
        ax4.text(x, y, z, label3, fontsize=4.5, rotation=90, alpha=0.7)
    return fig4, ax4

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

def read_file(Root_Dir, name):
    return Dataset(Root_Dir + name)

def data_extract(o, variable):
    ''' Read a certain variable for obs return np.masked.array for only data,
     time is not masked '''
    data = o.variables[variable][:]
    data = np.ma.masked_invalid(data)
    data = np.ma.masked_where(data >= 1.0e+12, data)
    data = np.ma.masked_array(data, data.mask, fill_value=np.inf)
    data = np.ma.fix_invalid(data)
    unit = o.variables[variable].units
    time = o.variables['time'][:]

    return data, time, unit

def models_data_extract(o, m, variable):
    ''' Read multiple models for a certain variable, return list of models data,
     model data is masked based on the observed data '''
    data1, time1, unit1 = [], [], []
    # for i in range(len(m)):
    #     if variable in responselist:
    #         data = o
    #         time, unit = [], []
    #     else:
    #         data, time, unit = data_extract(m[i], variable)
    #         np.ma.masked_where(np.ma.getmask(o), data)
    #
    #     data1.append(data)
    #     time1.append(time)
    #     unit1.append(unit)
    for i in range(len(m)):
        if variable in responselist:
            data = o
            time, unit = [], []
        else:
            data, time, unit = data_extract(m[i], variable)
            if len(data[0]) != len(o[0]):
                data = data[:, :-1]
            np.ma.masked_where(np.ma.getmask(o), data)
        data1.append(data)
        time1.append(time)
        unit1.append(unit)

    return data1, time1, unit1


def hour_seasonly_process(data):
    # return shape (season, site, year)
    hour_data = data.reshape(len(data), len(data[0])/24, 24)
    hour_data_s1, hour_data_s2, hour_data_s3, hour_data_s4 = [], [], [], []
    for y in range(len(data[0])/24/365):
        for d in range(0, 365):
            if d <= 58:
                hour_data_s1.append(hour_data[0:len(data), y*365+d, 0:24])
            elif d <= 151:
                hour_data_s2.append(hour_data[0:len(data), y*365+d, 0:24])
            elif d <= 242:
                hour_data_s3.append(hour_data[0:len(data), y*365+d, 0:24])
            elif d <= 334:
                hour_data_s4.append(hour_data[0:len(data), y*365+d, 0:24])
            else:
                hour_data_s1.append(hour_data[0:len(data), y*365+d, 0:24])

    season_data = [hour_data_s1, hour_data_s2, hour_data_s3, hour_data_s4]
    season_data = np.asarray(season_data)
    season_data = np.ma.masked_invalid(season_data)
    season_data = np.ma.fix_invalid(season_data)
    season_data = np.ma.masked_where(season_data >= 1.0e+12, season_data)

    return season_data


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


def month_seasonly_process(data):
    # return shape (season, site, year)
    # data = data.reshape(len(data),len(data[0])/12, 4, 3)
    m, n = len(data), len(data[0])/12
    # print(data.shape, m, n)
    data = data.reshape(m, n, 12)
    # data = np.ma.masked_array(data, data.mask, fill_value=np.Inf)
    # season_data = np.ma.zeros(m,n, 4)
    season1 = (data[0:m, 0:n, 0]+data[0:m, 0:n, 10]+data[0:m, 0:n, 11])/3.0
    season2 = (data[0:m, 0:n, 1]+data[0:m, 0:n, 2]+data[0:m, 0:n, 3])/3.0
    season3 = (data[0:m, 0:n, 4]+data[0:m, 0:n, 5]+data[0:m, 0:n, 6])/3.0
    season4 = (data[0:m, 0:n, 7]+data[0:m, 0:n, 8]+data[0:m, 0:n, 9])/3.0
    season_data = [season1, season2, season3, season4]
    season_data = np.asarray(season_data)
    season_data = np.ma.masked_invalid(season_data)
    season_data = np.ma.fix_invalid(season_data)
    season_data = np.ma.masked_where(season_data >= 1.0e+12, season_data)

    return season_data

def month_models_seasonly_process(m):
    month_data = []
    for i in range(len(m)):
        month_data.append(month_seasonly_process(m[i]))
    return month_data


def hour_models_seasonly_process(m):
    month_data = []
    for i in range(len(m)):
        month_data.append(hour_seasonly_process(m[i]))
    return month_data

def binPlot(X, Y, label, ax=None, numBins=8, xmin=None, xmax=None, c=None):

    '''  Adopted from  http://peterthomasweir.blogspot.com/2012/10/plot-binned-mean-and-mean-plusminus-std.html '''
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    bins = np.linspace(xmin, xmax, numBins + 1)
    xx = np.array([np.mean((bins[binInd], bins[binInd + 1])) for binInd in range(numBins)])
    yy = np.array([np.mean(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    yystd = 0.5*np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd + 1])]) for binInd in range(numBins)])
    if label == 'Observed':
        ax.plot(xx, yy, 'k-', label=label)
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color='k')
    else:
        ax.plot(xx, yy, '-', label=label, c=c)
        ax.errorbar(xx, yy, yerr=yystd, fmt='o', elinewidth=2, capthick=1, capsize=4, color=c)


    # patchHandle.set_facecolor([.8, .8, .8])
    # patchHandle.set_edgecolor('none')
    return xx, yy, yystd

def plot(variable1, variable2, ax0, ax1, label=None,c=None):
    x, y = variable1, variable2
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y)
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    # #############################################################################
    # Plot result
    segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(x))]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(0.5 * np.ones(len(x)))

    if label == 'Observed':
        ax0.plot(x, y, 'k.', markersize=12, label=label)
        # ax8[number].plot(x, y_, 'g.-', markersize=12)
        ax0.plot(x, lr.predict(x[:, np.newaxis]), 'k-', label='Linear' + label)
    else:
        ax0.plot(x, y, '.', markersize=12, label=label, c=c)
        # ax8[number].plot(x, y_, 'g.-', markersize=12)
        ax0.plot(x, lr.predict(x[:, np.newaxis]), '-', label='Linear' + label, c=c)
    # plt.gca().add_collection(lc)
    binPlot(x, y, label, ax1, 15, c=c)


# site_name = ['AU-Tum', 'AT-Neu']
def plot_all_response(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list1, variable_list2):
    directory = filedir + 'response' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    scores = []
    for i in range(len(variable_list1)):
        variable1, variable2 = variable_list1[i], variable_list2[i]
        if variable1 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,
                                                                                       'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable1)
            h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)
        else:
            h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable1)
            h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)

        if variable2 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,
                                                                                       'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o, variable2)
            h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)
        else:
            h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o, variable2)
            h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)


        h_mod1.append(h_obs1), d_mod1.append(d_obs1), m_mod1.append(m_obs1), y_mod1.append(y_obs1)
        h_mod2.append(h_obs2), d_mod2.append(d_obs2), m_mod2.append(m_obs2), y_mod2.append(y_obs2)

        for j, site in enumerate(h_site_name_obs1):
            if h_site_name_obs1.mask[j]:
                continue
            print('Process on response_' + site + '_No.' + str(j) + '!')
            fig8 = plt.figure(figsize=(8, 13))
            fig9 = plt.figure(figsize=(8, 13))
            ax80 = fig8.add_subplot(411)
            ax90 = fig9.add_subplot(411)
            ax81 = fig8.add_subplot(412)
            ax91 = fig9.add_subplot(412)
            ax82 = fig8.add_subplot(413)
            ax92 = fig9.add_subplot(413)
            ax83 = fig8.add_subplot(414)
            ax93 = fig9.add_subplot(414)
            for m in range(len(h_mod1)):
                # print(m)
                h1, h2, d1, d2 = h_mod1[m][j, :], h_mod2[m][j, :], d_mod1[m][j, :], d_mod2[m][j, :]
                m1, m2, y1, y2 = m_mod1[m][j, :], m_mod2[m][j, :], y_mod1[m][j, :], y_mod2[m][j, :]
                # print(h1.mask, h2.mask)
                mask1 = h1.mask | h2.mask
                mask2 = d1.mask | d2.mask
                mask3 = m1.mask | m2.mask
                mask4 = y1.mask | y2.mask
                h1 = np.ma.masked_where(mask1, h1)
                h2 = np.ma.masked_where(mask1, h2)
                d1 = np.ma.masked_where(mask2, d1)
                d2 = np.ma.masked_where(mask2, d2)
                m1 = np.ma.masked_where(mask3, m1)
                m2 = np.ma.masked_where(mask3, m2)
                y1 = np.ma.masked_where(mask4, y1)
                y2 = np.ma.masked_where(mask4, y2)
                # print(h1.mask, h2.mask)
                # print(h1.compressed().shape, h2.compressed().shape)
                if m == len(h_mod1)-1:
                    if len(h1.compressed()) != 0:
                        plot(h1.compressed(), h2.compressed(), ax80, ax90, label= 'Observed')
                        plot(d1.compressed(), d2.compressed(), ax81, ax91, label= 'Observed')
                        plot(m1.compressed(), m2.compressed(), ax82, ax92, label= 'Observed')
                        plot(y1.compressed(), y2.compressed(), ax83, ax93, label= 'Observed')
                else:
                    if len(h1.compressed()) != 0:
                        plot(h1.compressed(), h2.compressed(), ax80, ax90, label= "Model "+str(m+1), c=col[m])
                        plot(d1.compressed(), d2.compressed(), ax81, ax91, label= "Model "+str(m+1), c=col[m])
                        plot(m1.compressed(), m2.compressed(), ax82, ax92, label= "Model "+str(m+1), c=col[m])
                        plot(y1.compressed(), y2.compressed(), ax83, ax93, label= "Model "+str(m+1), c=col[m])
            # plt.suptitle(variable1 + 'vs' + variable2, fontsize=8)
            ax80.set_xlabel(variable1+' \n' + h_unit_obs1+' ')
            ax81.set_xlabel(variable1+' \n' + d_unit_obs1+' ')
            ax82.set_xlabel(variable1+' \n' + m_unit_obs1+' ')
            ax83.set_xlabel(variable1+' \n' + y_unit_obs1+' ')
            ax80.set_ylabel(variable2+'(Hourly) \n' + h_unit_obs2+' ')
            ax81.set_ylabel(variable2+'(Daily) \n' + d_unit_obs2+' ')
            ax82.set_ylabel(variable2+'(Monthly) \n' + m_unit_obs2+' ')
            ax83.set_ylabel(variable2+'(Yearly) \n' + y_unit_obs2+' ')

            ax90.set_xlabel(variable1+' \n' + h_unit_obs1+' ')
            ax91.set_xlabel(variable1+' \n' + d_unit_obs1+' ')
            ax92.set_xlabel(variable1+' \n' + m_unit_obs1+' ')
            ax93.set_xlabel(variable1+' \n' + y_unit_obs1+' ')
            ax90.set_ylabel(variable2+'(Hourly) \n' + h_unit_obs2+' ')
            ax91.set_ylabel(variable2+'(Daily) \n' + d_unit_obs2+' ')
            ax92.set_ylabel(variable2+'(Monthly) \n' + m_unit_obs2+' ')
            ax93.set_ylabel(variable2+'(Yearly) \n' + y_unit_obs2+' ')
            ax80.legend(bbox_to_anchor=(1.3, -0.5), shadow=False, fontsize='medium')
            ax90.legend(bbox_to_anchor=(1.25,-0.5), shadow=False, fontsize='medium')
            fig8.suptitle(site+ ':  '+variable1+' vs ' + variable2 + ' Response')
            fig9.suptitle(site+ ':  '+variable1+' vs ' + variable2 + ' Response bin')
            fig8.tight_layout(rect=[0, 0.01, 1, 0.95])
            fig9.tight_layout(rect=[0, 0.01, 1, 0.95])

            fig8.savefig(filedir + 'response' + '/' + site +'_' + variable1 + '_vs_' + variable2 + '_Response' + '.png', bbox_inches='tight')
            fig9.savefig(filedir + 'response' + '/' + site +'_' + variable1 + '_vs_' + variable2 + '_Response_Bin' + '.png', bbox_inches='tight')
            # print(filedir + 'response' + '/' + site +'_' + variable1 + '_vs_' + variable2 + '_Response' + '.png')
            plt.close('all')
    return scores

def plot_4_variable_response2(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):
    plt.rcParams.update({'font.size': 5})
    directory = filedir + 'response_3d' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    scores = []

    for i in range(len(variable_list)):
        variable1, variable2, variable3, variable4 = variable_list[i]
        if variable1 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,  'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o1, variable1)
            h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o1, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o1, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o1, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)
        else:
            h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable1)
            h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)

        if variable2 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o1, variable2)
            h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o1, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o1, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o1, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)
        else:
            h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o, variable2)
            h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)

        if variable3 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o1, variable3)
            h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o1, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o1, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o1, variable3)
            y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)
        else:
            h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o, variable3)
            h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o, variable3)
            y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)

        if variable4 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o1, variable4)
            h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o1, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o1, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o1, variable4)
            y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)
        else:
            h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o, variable4)
            h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o, variable4)
            y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)

        h_mod1.insert(0, h_obs1), d_mod1.insert(0, d_obs1), m_mod1.insert(0, m_obs1), y_mod1.insert(0, y_obs1)
        h_mod2.insert(0, h_obs2), d_mod2.insert(0, d_obs2), m_mod2.insert(0, m_obs2), y_mod2.insert(0, y_obs2)
        h_mod3.insert(0, h_obs3), d_mod3.insert(0, d_obs3), m_mod3.insert(0, m_obs3), y_mod3.insert(0, y_obs3)
        h_mod4.insert(0, h_obs4), d_mod4.insert(0, d_obs4), m_mod4.insert(0, m_obs4), y_mod4.insert(0, y_obs4)
        for j, site in enumerate(h_site_name_obs1):

            if h_site_name_obs1.mask[j]:
                continue

            print('Process on response_3d_' + site + '_No.' + str(j) + '!')
            fig8 = plt.figure(figsize=(16, 6*len(h_mod1)))

            for m in range(len(h_mod1)):
                if m == 0:
                    anchored_text = " Observed"
                else:
                    anchored_text = " Model" + str(m)

                ax80 = fig8.add_subplot(len(h_mod1)+1, 4, m*4 + 1, projection='3d')
                ax81 = fig8.add_subplot(len(h_mod1)+1, 4, m*4 + 2, projection='3d')
                ax82 = fig8.add_subplot(len(h_mod1)+1, 4, m*4 + 3, projection='3d')
                ax83 = fig8.add_subplot(len(h_mod1)+1, 4, m*4 + 4, projection='3d')

                # print(m)
                h1, h2, d1, d2 = h_mod1[m][j, :], h_mod2[m][j, :], d_mod1[m][j, :], d_mod2[m][j, :]
                m1, m2, y1, y2 = m_mod1[m][j, :], m_mod2[m][j, :], y_mod1[m][j, :], y_mod2[m][j, :]
                h3, h4, d3, d4 = h_mod3[m][j, :], h_mod4[m][j, :], d_mod3[m][j, :], d_mod4[m][j, :]
                m3, m4, y3, y4 = m_mod3[m][j, :], m_mod4[m][j, :], y_mod3[m][j, :], y_mod4[m][j, :]

                # print(h1.mask, h2.mask)
                mask1 = h1.mask | h2.mask | h3.mask | h4.mask
                mask2 = d1.mask | d2.mask | d3.mask | d4.mask
                mask3 = m1.mask | m2.mask | m3.mask | m4.mask
                mask4 = y1.mask | y2.mask | y3.mask | y4.mask

                h1 = np.ma.masked_where(mask1, h1)
                h2 = np.ma.masked_where(mask1, h2)
                h3 = np.ma.masked_where(mask1, h3)
                h4 = np.ma.masked_where(mask1, h4)

                d1 = np.ma.masked_where(mask2, d1)
                d2 = np.ma.masked_where(mask2, d2)
                d3 = np.ma.masked_where(mask2, d3)
                d4 = np.ma.masked_where(mask2, d4)

                m1 = np.ma.masked_where(mask3, m1)
                m2 = np.ma.masked_where(mask3, m2)
                m3 = np.ma.masked_where(mask3, m3)
                m4 = np.ma.masked_where(mask3, m4)

                y1 = np.ma.masked_where(mask4, y1)
                y2 = np.ma.masked_where(mask4, y2)
                y3 = np.ma.masked_where(mask4, y3)
                y4 = np.ma.masked_where(mask4, y4)
                #
                if len(h1.compressed()) != 0:

                    cax = ax80.scatter(h1.compressed(), h2.compressed(), h3.compressed(), c=h4.compressed(), cmap=plt.hot())
                    ax80.set_title('Hourly- ' + variable4 + '\n' + h_unit_obs4 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax80.set_xlabel(variable1 + '\n' + h_unit_obs1 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_ylabel(variable2 + '\n' + h_unit_obs2 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_zlabel(variable3 + '\n' + h_unit_obs3 + '', fontsize=lengendfontsize, linespacing=3.2)

                    ax81.scatter(d1.compressed(), d2.compressed(), d3.compressed(), c=d4.compressed(), cmap=plt.hot())
                    ax81.set_title('Daily- ' + variable4 + '\n' + d_unit_obs4 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax81.set_xlabel(variable1 + '\n' + d_unit_obs1 + '', fontsize=lengendfontsize)
                    ax81.set_ylabel(variable2 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax81.set_zlabel(variable3 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax82.scatter(m1.compressed(), m2.compressed(), m3.compressed(), c=m4.compressed(), cmap=plt.hot())
                    ax82.set_title('Monthly- ' + variable4 + '\n' + m_unit_obs4 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax82.set_xlabel(variable1 + '\n' + m_unit_obs1 + '', fontsize=lengendfontsize)
                    ax82.set_ylabel(variable2 + '\n' + m_unit_obs2 + '', fontsize=lengendfontsize)
                    ax82.set_zlabel(variable3 + '\n' + m_unit_obs3 + '', fontsize=lengendfontsize)
                    ax83.scatter(y1.compressed(), y2.compressed(), y3.compressed(), c=y4.compressed(), cmap=plt.hot())
                    ax83.set_title('Yearly- ' + variable4 + '\n' + y_unit_obs4 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax83.set_xlabel(variable1 + '\n' + y_unit_obs1 + '', fontsize=lengendfontsize)
                    ax83.set_ylabel(variable2 + '\n' + y_unit_obs2 + '', fontsize=lengendfontsize)
                    ax83.set_zlabel(variable3 + '\n' + y_unit_obs3 + '', fontsize=lengendfontsize)
                    ax80.tick_params(labelsize=5)
                    ax81.tick_params(labelsize=5)
                    ax82.tick_params(labelsize=5)
                    ax83.tick_params(labelsize=5)

                    ax80.dist = 13
                    ax81.dist = 13
                    ax82.dist = 13
                    ax83.dist = 13

                    z = h4.compressed()
                    axes = [ax80, ax81, ax82, ax83]
                    cbar = fig8.colorbar(cax, ticks=[min(z), (max(z) + min(z)) / 2, max(z)], orientation='vertical',
                                         label=variable4, ax=axes, shrink=0.75)
                    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
            plt.suptitle(site + ':  ' + variable1 +' vs '+ variable2 +' vs '+ variable3 +' vs '+ variable4 + ' Response', fontsize=24)
            fig8.savefig(filedir + 'response_3d' + '/' + site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'Observed' + '.png', bbox_inches='tight')
            plt.close('all')
    return scores


def plot_4_variable_seasonal(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):
    plt.rcParams.update({'font.size': 5})
    directory = filedir + 'response_3d' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    scores = []

    for i in range(len(variable_list)):
        variable1, variable2, variable3, variable4 = variable_list[i]
        if variable1 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,  'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o1, variable1)
            # h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o1, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o1, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            # y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o1, variable1)
            # y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)
        else:
            # h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable1)
            # h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            # y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            # y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)

        if variable2 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o1, variable2)
            # h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o1, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o1, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            # y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o1, variable2)
            # y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)
        else:
            # h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o, variable2)
            # h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            # y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            # y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)

        if variable3 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o1, variable3)
            # h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o1, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o1, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            # y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o1, variable3)
            # y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)
        else:
            # h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o, variable3)
            # h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            # y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o, variable3)
            # y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)

        if variable4 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o1, variable4)
            # h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o1, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o1, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            # y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o1, variable4)
            # y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)
        else:
            # h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o, variable4)
            # h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            # y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o, variable4)
            # y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)

        o_season1 = month_seasonly_process(m_obs1)
        o_season2 = month_seasonly_process(m_obs2)
        o_season3 = month_seasonly_process(m_obs3)
        o_season4 = month_seasonly_process(m_obs4)

        m_season1 = month_models_seasonly_process(m_mod1)
        m_season2 = month_models_seasonly_process(m_mod2)
        m_season3 = month_models_seasonly_process(m_mod3)
        m_season4 = month_models_seasonly_process(m_mod4)

        m_season1.insert(0, o_season1)
        m_season2.insert(0, o_season2)
        m_season3.insert(0, o_season3)
        m_season4.insert(0, o_season4)

        for j, site in enumerate(h_site_name_obs1):

            if h_site_name_obs1.mask[j]:
                continue

            print('Process on response_3d_' + site + '_No.' + str(j) + '!')
            fig8 = plt.figure(figsize=(16, 6*len(m_season1)))

            for m in range(len(m_season1)):
                if m == 0:
                    anchored_text = " Observed"
                else:
                    anchored_text = " Model" + str(m)

                ax80 = fig8.add_subplot(len(m_season1)+1, 4, m*4 + 1, projection='3d')
                ax81 = fig8.add_subplot(len(m_season1)+1, 4, m*4 + 2, projection='3d')
                ax82 = fig8.add_subplot(len(m_season1)+1, 4, m*4 + 3, projection='3d')
                ax83 = fig8.add_subplot(len(m_season1)+1, 4, m*4 + 4, projection='3d')

                # print(m)
                h1, h2, d1, d2 = m_season1[m][0, j, :], m_season2[m][0, j, :], m_season3[m][0, j, :], m_season4[m][0, j, :]
                m1, m2, y1, y2 = m_season1[m][1, j, :], m_season2[m][1, j, :], m_season3[m][1, j, :], m_season4[m][1, j, :]
                h3, h4, d3, d4 = m_season1[m][2, j, :], m_season2[m][2, j, :], m_season3[m][2, j, :], m_season4[m][2, j, :]
                m3, m4, y3, y4 = m_season1[m][3, j, :], m_season2[m][3, j, :], m_season3[m][3, j, :], m_season4[m][3, j, :]


                # print(h1.mask, h2.mask)
                mask1 = h1.mask | h2.mask | h3.mask | h4.mask
                mask2 = d1.mask | d2.mask | d3.mask | d4.mask
                mask3 = m1.mask | m2.mask | m3.mask | m4.mask
                mask4 = y1.mask | y2.mask | y3.mask | y4.mask

                h1 = np.ma.masked_where(mask1, h1)
                h2 = np.ma.masked_where(mask1, h2)
                h3 = np.ma.masked_where(mask1, h3)
                h4 = np.ma.masked_where(mask1, h4)

                d1 = np.ma.masked_where(mask2, d1)
                d2 = np.ma.masked_where(mask2, d2)
                d3 = np.ma.masked_where(mask2, d3)
                d4 = np.ma.masked_where(mask2, d4)

                m1 = np.ma.masked_where(mask3, m1)
                m2 = np.ma.masked_where(mask3, m2)
                m3 = np.ma.masked_where(mask3, m3)
                m4 = np.ma.masked_where(mask3, m4)

                y1 = np.ma.masked_where(mask4, y1)
                y2 = np.ma.masked_where(mask4, y2)
                y3 = np.ma.masked_where(mask4, y3)
                y4 = np.ma.masked_where(mask4, y4)
                #

                if len(h1.compressed()) != 0:

                    cax = ax80.scatter(h1.compressed(), h2.compressed(), h3.compressed(), c=h4.compressed(), cmap=plt.hot())
                    ax80.set_title('DJF - ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax80.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize, linespacing=3.2)

                    ax81.scatter(d1.compressed(), d2.compressed(), d3.compressed(), c=d4.compressed(), cmap=plt.hot())
                    ax81.set_title('MAM- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax81.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax81.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax81.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax82.scatter(m1.compressed(), m2.compressed(), m3.compressed(), c=m4.compressed(), cmap=plt.hot())
                    ax82.set_title('JJA- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax82.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax82.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax82.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax83.scatter(y1.compressed(), y2.compressed(), y3.compressed(), c=y4.compressed(), cmap=plt.hot())
                    ax83.set_title('SON- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax83.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax83.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax83.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax80.tick_params(labelsize=5)
                    ax81.tick_params(labelsize=5)
                    ax82.tick_params(labelsize=5)
                    ax83.tick_params(labelsize=5)

                    ax80.dist = 13
                    ax81.dist = 13
                    ax82.dist = 13
                    ax83.dist = 13

                    z = h4.compressed()
                    axes = [ax80, ax81, ax82, ax83]
                    cbar = fig8.colorbar(cax, ticks=[min(z), (max(z) + min(z)) / 2, max(z)], orientation='vertical',
                                         label=variable4, ax=axes, shrink=0.75)
                    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
            plt.suptitle(site + ':  ' + variable1 +' vs '+ variable2 +' vs '+ variable3 +' vs '+ variable4 + ' Response', fontsize=24)
            fig8.savefig(filedir + 'response_3d' + '/' + site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'seasonal' + '.png', bbox_inches='tight')
            plt.close('all')
    return scores


def plot_4_variable_day_seasonal(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):
    plt.rcParams.update({'font.size': 5})
    directory = filedir + 'response_3d' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    scores = []

    for i in range(len(variable_list)):
        variable1, variable2, variable3, variable4 = variable_list[i]
        if variable1 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,  'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o1, variable1)
            # h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o1, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            # m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o1, variable1)
            # m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            # y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o1, variable1)
            # y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)
        else:
            # h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable1)
            # h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable1)
            d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable1)
            d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable1)
            # m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            # m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            # y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            # y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)

        if variable2 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o1, variable2)
            # h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o1, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            # m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o1, variable2)
            # m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            # y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o1, variable2)
            # y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)
        else:
            # h_obs2, h_t_obs2, h_unit_obs2 = data_extract(h_o, variable2)
            # h_mod2, h_t_mod2, h_unit_mod2 = models_data_extract(h_obs2, h_models, variable2)
            d_obs2, d_t_obs2, d_unit_obs2 = data_extract(d_o, variable2)
            d_mod2, d_t_mod2, d_unit_mod2 = models_data_extract(d_obs2, d_models, variable2)
            # m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            # m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            # y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            # y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)

        if variable3 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o1, variable3)
            # h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o1, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            # m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o1, variable3)
            # m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            # # y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o1, variable3)
            # y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)
        else:
            # h_obs3, h_t_obs3, h_unit_obs3 = data_extract(h_o, variable3)
            # h_mod3, h_t_mod3, h_unit_mod3 = models_data_extract(h_obs3, h_models, variable3)
            d_obs3, d_t_obs3, d_unit_obs3 = data_extract(d_o, variable3)
            d_mod3, d_t_mod3, d_unit_mod3 = models_data_extract(d_obs3, d_models, variable3)
            # m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o, variable3)
            # m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            # y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o, variable3)
            # y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)

        if variable4 in responselist:
            h_o1, d_o1, m_o1, y_o1 = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')
            # h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o1, variable4)
            # h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o1, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            # m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o1, variable4)
            # m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            # y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o1, variable4)
            # y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)
        else:
            # h_obs4, h_t_obs4, h_unit_obs4 = data_extract(h_o, variable4)
            # h_mod4, h_t_mod4, h_unit_mod4 = models_data_extract(h_obs4, h_models, variable4)
            d_obs4, d_t_obs4, d_unit_obs4 = data_extract(d_o, variable4)
            d_mod4, d_t_mod4, d_unit_mod4 = models_data_extract(d_obs4, d_models, variable4)
            # m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o, variable4)
            # m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            # y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o, variable4)
            # y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)

        o_season11, o_season12, o_season13, o_season14 = day_seasonly_process(d_obs1)
        o_season21, o_season22, o_season23, o_season24 = day_seasonly_process(d_obs2)
        o_season31, o_season32, o_season33, o_season34 = day_seasonly_process(d_obs3)
        o_season41, o_season42, o_season43, o_season44 = day_seasonly_process(d_obs4)

        m_season11, m_season12, m_season13, m_season14 = day_models_seasonly_process(d_mod1)
        m_season21, m_season22, m_season23, m_season24 = day_models_seasonly_process(d_mod2)
        m_season31, m_season32, m_season33, m_season34 = day_models_seasonly_process(d_mod3)
        m_season41, m_season42, m_season43, m_season44 = day_models_seasonly_process(d_mod4)

        m_season11.insert(0, o_season11)
        m_season21.insert(0, o_season21)
        m_season31.insert(0, o_season31)
        m_season41.insert(0, o_season41)

        m_season12.insert(0, o_season12)
        m_season22.insert(0, o_season22)
        m_season32.insert(0, o_season32)
        m_season42.insert(0, o_season42)

        m_season13.insert(0, o_season13)
        m_season23.insert(0, o_season23)
        m_season33.insert(0, o_season33)
        m_season43.insert(0, o_season43)

        m_season14.insert(0, o_season14)
        m_season24.insert(0, o_season24)
        m_season34.insert(0, o_season34)
        m_season44.insert(0, o_season44)

        for j, site in enumerate(h_site_name_obs1):

            if h_site_name_obs1.mask[j]:
                continue

            print('Process on response_3d_' + site + '_No.' + str(j) + '!')
            fig8 = plt.figure(figsize=(16, 6*len(m_season11)))

            for m in range(len(m_season11)):
                if m == 0:
                    anchored_text = " Observed"
                else:
                    anchored_text = " Model" + str(m)

                ax80 = fig8.add_subplot(len(m_season11)+1, 4, m*4 + 1, projection='3d')
                ax81 = fig8.add_subplot(len(m_season11)+1, 4, m*4 + 2, projection='3d')
                ax82 = fig8.add_subplot(len(m_season11)+1, 4, m*4 + 3, projection='3d')
                ax83 = fig8.add_subplot(len(m_season11)+1, 4, m*4 + 4, projection='3d')

                # print(m)
                h1, h2, d1, d2 = m_season11[m][:, j], m_season21[m][:, j], m_season12[m][:, j], m_season22[m][:, j]
                m1, m2, y1, y2 = m_season13[m][:, j], m_season23[m][:, j], m_season14[m][:, j], m_season24[m][:, j]
                h3, h4, d3, d4 = m_season31[m][:, j], m_season41[m][:, j], m_season32[m][:, j], m_season42[m][:, j]
                m3, m4, y3, y4 = m_season33[m][:, j], m_season43[m][:, j], m_season34[m][:, j], m_season44[m][:, j]


                # print(h1.mask, h2.mask)
                mask1 = h1.mask | h2.mask | h3.mask | h4.mask
                mask2 = d1.mask | d2.mask | d3.mask | d4.mask
                mask3 = m1.mask | m2.mask | m3.mask | m4.mask
                mask4 = y1.mask | y2.mask | y3.mask | y4.mask

                h1 = np.ma.masked_where(mask1, h1)
                h2 = np.ma.masked_where(mask1, h2)
                h3 = np.ma.masked_where(mask1, h3)
                h4 = np.ma.masked_where(mask1, h4)

                d1 = np.ma.masked_where(mask2, d1)
                d2 = np.ma.masked_where(mask2, d2)
                d3 = np.ma.masked_where(mask2, d3)
                d4 = np.ma.masked_where(mask2, d4)

                m1 = np.ma.masked_where(mask3, m1)
                m2 = np.ma.masked_where(mask3, m2)
                m3 = np.ma.masked_where(mask3, m3)
                m4 = np.ma.masked_where(mask3, m4)

                y1 = np.ma.masked_where(mask4, y1)
                y2 = np.ma.masked_where(mask4, y2)
                y3 = np.ma.masked_where(mask4, y3)
                y4 = np.ma.masked_where(mask4, y4)
                #

                if len(h1.compressed()) != 0:

                    cax = ax80.scatter(h1.compressed(), h2.compressed(), h3.compressed(), c=h4.compressed(), cmap=plt.hot())
                    ax80.set_title('DJF - ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax80.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize, linespacing=3.2)
                    ax80.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize, linespacing=3.2)

                    ax81.scatter(d1.compressed(), d2.compressed(), d3.compressed(), c=d4.compressed(), cmap=plt.hot())
                    ax81.set_title('MAM- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax81.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax81.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax81.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax82.scatter(m1.compressed(), m2.compressed(), m3.compressed(), c=m4.compressed(), cmap=plt.hot())
                    ax82.set_title('JJA- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax82.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax82.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax82.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax83.scatter(y1.compressed(), y2.compressed(), y3.compressed(), c=y4.compressed(), cmap=plt.hot())
                    ax83.set_title('SON- ' + variable4 + '\n' + d_unit_obs1 + '' + anchored_text
                                   , fontsize=lengendfontsize)
                    ax83.set_xlabel(variable1 + '\n' + d_unit_obs2 + '', fontsize=lengendfontsize)
                    ax83.set_ylabel(variable2 + '\n' + d_unit_obs3 + '', fontsize=lengendfontsize)
                    ax83.set_zlabel(variable3 + '\n' + d_unit_obs4 + '', fontsize=lengendfontsize)
                    ax80.tick_params(labelsize=5)
                    ax81.tick_params(labelsize=5)
                    ax82.tick_params(labelsize=5)
                    ax83.tick_params(labelsize=5)

                    ax80.dist = 13
                    ax81.dist = 13
                    ax82.dist = 13
                    ax83.dist = 13

                    z = h4.compressed()
                    axes = [ax80, ax81, ax82, ax83]
                    cbar = fig8.colorbar(cax, ticks=[min(z), (max(z) + min(z)) / 2, max(z)], orientation='vertical',
                                         label=variable4, ax=axes, shrink=0.75)
                    cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
            plt.suptitle(site + ':  ' + variable1 +' vs '+ variable2 +' vs '+ variable3 +' vs '+ variable4 + ' Response', fontsize=24)
            fig8.savefig(filedir + 'response_3d' + '/' + site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'day_seasonal' + '.png', bbox_inches='tight')
            plt.close('all')
    return scores

def plot_4_variable_corr(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):
    col = ['k','palevioletred', 'm', 'plum', 'darkorchid', 'blue', 'navy', 'deepskyblue', 'darkcyan', 'seagreen',
           'darkgreen', 'olivedrab', 'gold', 'tan', 'red', 'orange', 'yellow']
    directory = filedir + 'response_3d' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    scores = []
    for i in range(len(variable_list)):
        variable1, variable2, variable3, variable4 = variable_list[i]

        if variable1 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2,
                                                                                   'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)
        else:

            m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable1)
            m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable1)
            y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable1)
            y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable1)

        if variable2 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)
        else:

            m_obs2, m_t_obs2, m_unit_obs2 = data_extract(m_o, variable2)
            m_mod2, m_t_mod2, m_unit_mod2 = models_data_extract(m_obs2, m_models, variable2)
            y_obs2, y_t_obs2, y_unit_obs2 = data_extract(y_o, variable2)
            y_mod2, y_t_mod2, y_unit_mod2 = models_data_extract(y_obs2, y_models, variable2)

        if variable3 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o, variable3)
            y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)
        else:
            m_obs3, m_t_obs3, m_unit_obs3 = data_extract(m_o, variable3)
            m_mod3, m_t_mod3, m_unit_mod3 = models_data_extract(m_obs3, m_models, variable3)
            y_obs3, y_t_obs3, y_unit_obs3 = data_extract(y_o, variable3)
            y_mod3, y_t_mod3, y_unit_mod3 = models_data_extract(y_obs3, y_models, variable3)

        if variable4 in responselist:
            h_o, d_o, m_o, y_o = read_file(Root_Dir2, 'test_hourly.nc'), read_file(Root_Dir2, 'test_daily.nc'), read_file(
                Root_Dir2, 'test_monthly.nc'), read_file(Root_Dir2, 'test_annual.nc')

            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o, variable4)
            y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)
        else:

            m_obs4, m_t_obs4, m_unit_obs4 = data_extract(m_o, variable4)
            m_mod4, m_t_mod4, m_unit_mod4 = models_data_extract(m_obs4, m_models, variable4)
            y_obs4, y_t_obs4, y_unit_obs4 = data_extract(y_o, variable4)
            y_mod4, y_t_mod4, y_unit_mod4 = models_data_extract(y_obs4, y_models, variable4)

        o_season_data1 = month_seasonly_process(m_obs1)
        m_season_data1 = month_models_seasonly_process(m_mod1)
        o_season_data2 = month_seasonly_process(m_obs2)
        m_season_data2 = month_models_seasonly_process(m_mod2)
        o_season_data3 = month_seasonly_process(m_obs3)
        m_season_data3 = month_models_seasonly_process(m_mod3)
        o_season_data4 = month_seasonly_process(m_obs4)
        m_season_data4 = month_models_seasonly_process(m_mod4)

        markersize = 45
        from scipy.stats.stats import pearsonr
        for j, site in enumerate(h_site_name_obs1):
            if h_site_name_obs1.mask[j]:
                continue
            # if j==1:
            #     break
            print('Process on response_rr_' + site + '_No.' + str(j) + '!')
            fig4 = plt.figure(figsize=(6, 6))
            ax4 = fig4.add_subplot(111, projection='3d')

            o_season1 = np.ma.masked_invalid(o_season_data1[0, j, :])
            o_season2 = np.ma.masked_invalid(o_season_data2[0, j, :])
            o_season3 = np.ma.masked_invalid(o_season_data3[0, j, :])
            o_season4 = np.ma.masked_invalid(o_season_data4[0, j, :])
            o_season1 = np.ma.masked_where(o_season1 >= 1.0e+18, o_season1)
            o_season2 = np.ma.masked_where(o_season2 >= 1.0e+18, o_season2)
            o_season3 = np.ma.masked_where(o_season3 >= 1.0e+18, o_season3)
            o_season4 = np.ma.masked_where(o_season4 >= 1.0e+18, o_season4)


            mask1 = y_obs1[j, :].mask | y_obs2[j, :].mask | y_obs3[j, :].mask | y_obs4[j, :].mask
            mask2 = o_season1.mask | o_season2.mask | o_season3.mask | o_season4.mask #| y_obs4[j, :].mask

            # print(y_obs1[j, ~mask1], y_obs2[j, ~mask1], y_obs3[j, ~mask1], y_obs4[j, ~mask1])

            if np.sum(mask1) >=1:
                v0 = np.asarray([y_obs1[j, ~mask1], y_obs2[j, ~mask1], y_obs3[j, ~mask1], y_obs4[j, ~mask1]]).T
                if len(v0) >=1:
                    if np.isnan(v0).any():
                        print('NaN exists in correlation')
                    else:
                        corr0 = partial_corr(v0)
                        if not np.isnan([corr0[0, 1], corr0[0, 2], corr0[0, 3]]).any():
                            ax4.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Obs Annal', marker='^', c=col[0],
                                        s=markersize)
                            # fig4, ax4 = annotion(corr0[0, 1], corr0[0, 2], corr0[0, 3], fig4, ax4)

            if np.sum(mask2) >= 1:

                v1 = np.asarray([o_season_data1[0, j, ~mask2], o_season_data2[0, j, ~mask2], o_season_data3[0, j, ~mask2],
                                 o_season_data4[0, j, ~mask2]]).T
                v2 = np.asarray([o_season_data1[1, j, ~mask2], o_season_data2[1, j, ~mask2], o_season_data3[1, j, ~mask2],
                                 o_season_data4[1, j, ~mask2]]).T
                v3 = np.asarray([o_season_data1[2, j, ~mask2], o_season_data2[2, j, ~mask2], o_season_data3[2, j, ~mask2],
                                 o_season_data4[2, j, ~mask2]]).T
                v4 = np.asarray([o_season_data1[3, j, ~mask2], o_season_data2[3, j, ~mask2], o_season_data3[3, j, ~mask2],
                                 o_season_data4[3, j, ~mask2]]).T

                if len(v1) >= 1:
                    if np.isnan(v1).any():
                        print('NaN exists in correlation')
                    else:
                        corr1 = partial_corr(v1)
                        if not np.isnan([corr1[0, 1], corr1[0, 2], corr1[0, 3]]).any():
                            ax4.scatter(corr1[0, 1], corr1[0, 2], corr1[0, 3], label='Obs DJF', marker='<', c=col[0],
                                        s=markersize)
                            # fig4, ax4 = annotion(corr1[0, 1], corr1[0, 2], corr1[0, 3], fig4, ax4)

                    if np.isnan(v2).any():
                        print('NaN exists in correlation')
                    else:
                        corr2 = partial_corr(v2)
                        if not np.isnan([corr2[0, 1], corr2[0, 2], corr2[0, 3]]).any():

                            ax4.scatter(corr2[0, 1], corr2[0, 2], corr2[0, 3], label='Obs MAM', marker='>', c=col[0],
                                        s=markersize)
                            # fig4, ax4 = annotion(corr2[0, 1], corr2[0, 2], corr2[0, 3], fig4, ax4)

                    if np.isnan(v3).any():
                        print('NaN exists in correlation')
                    else:
                        corr3 = partial_corr(v3)
                        if not np.isnan([corr3[0, 1], corr3[0, 2], corr3[0, 3]]).any():
                            ax4.scatter(corr3[0, 1], corr3[0, 2], corr3[0, 3], label='Obs JJA', marker='*', c=col[0],
                                        s=markersize)

                            # fig4, ax4 = annotion(corr3[0, 1], corr3[0, 2], corr3[0, 3], fig4, ax4)

                    if np.isnan(v4).any():
                        print('NaN exists in correlation')
                    else:
                        corr4 = partial_corr(v4)
                        if not np.isnan([corr4[0, 1], corr4[0, 2], corr4[0, 3]]).any():
                            ax4.scatter(corr4[0, 1], corr4[0, 2], corr4[0, 3], label='Obs SON', marker='o', c=col[0],
                                        s=markersize)
                            # fig4, ax4 = annotion(corr4[0, 1], corr4[0, 2], corr4[0, 3], fig4, ax4)

            for m in range(len(m_mod1)):
                if np.sum(mask1) >= 1:
                    v0 = np.asarray([y_mod1[m][j, ~mask1], y_mod2[m][j, ~mask1], y_mod3[m][j, ~mask1], y_mod4[m][j, ~mask1]]).T
                    if len(v0) >= 1:
                        if np.isnan(v0).any():
                            print('NaN exists in correlation')
                        else:
                            corr0 = partial_corr(v0)
                            if not np.isnan([corr0[0, 1], corr0[0, 2], corr0[0, 3]]).any():
                                ax4.scatter(corr0[0, 1], corr0[0, 2], corr0[0, 3], label='Mod ' + str(m + 1) + ' Annal',
                                            marker='^',
                                            c=col[m + 1], s=markersize)
                                # fig4, ax4 = annotion(corr0[0, 1], corr0[0, 2], corr0[0, 3], fig4, ax4)

                if np.sum(mask2) >= 1:
                    v1 = np.asarray(
                        [m_season_data1[m][0, j, ~mask2], m_season_data2[m][0, j, ~mask2], m_season_data3[m][0, j, ~mask2],
                         m_season_data4[m][0, j, ~mask2]]).T
                    v2 = np.asarray(
                        [m_season_data1[m][1, j, ~mask2], m_season_data2[m][1, j, ~mask2], m_season_data3[m][1, j, ~mask2],
                         m_season_data4[m][1, j, ~mask2]]).T
                    v3 = np.asarray(
                        [m_season_data1[m][2, j, ~mask2], m_season_data2[m][2, j, ~mask2], m_season_data3[m][2, j, ~mask2],
                         m_season_data4[m][2, j, ~mask2]]).T
                    v4 = np.asarray(
                        [m_season_data1[m][3, j, ~mask2], m_season_data2[m][3, j, ~mask2], m_season_data3[m][3, j, ~mask2],
                         m_season_data4[m][3, j, ~mask2]]).T
                    if len(v1) >= 1:
                        if np.isnan(v1).any():
                            print('NaN exists in correlation')
                        else:
                            corr1 = partial_corr(v1)
                            if not np.isnan([corr1[0, 1], corr1[0, 2], corr1[0, 3]]).any():
                                ax4.scatter(corr1[0, 1], corr1[0, 2], corr1[0, 3], label='Mod ' + str(m + 1) + ' DJF',
                                            marker='<', c=col[m + 1], s=markersize)
                                # fig4, ax4 = annotion(corr1[0, 1], corr1[0, 2], corr1[0, 3], fig4, ax4)

                        if np.isnan(v2).any():
                            print('NaN exists in correlation')
                        else:
                            corr2 = partial_corr(v2)
                            if not np.isnan([corr2[0, 1], corr2[0, 2], corr2[0, 3]]).any():
                                ax4.scatter(corr2[0, 1], corr2[0, 2], corr2[0, 3], label='Mod ' + str(m + 1) + ' MAM',
                                            marker='>', c=col[m + 1], s=markersize)
                                # fig4, ax4 = annotion(corr2[0, 1], corr2[0, 2], corr2[0, 3], fig4, ax4)

                        if np.isnan(v3).any():
                            print('NaN exists in correlation')
                        else:
                            corr3 = partial_corr(v3)
                            if not np.isnan([corr3[0, 1], corr3[0, 2], corr3[0, 3]]).any():
                                ax4.scatter(corr3[0, 1], corr3[0, 2], corr3[0, 3], label='Mod ' + str(m + 1) + ' JJA',
                                            marker='*', c=col[m + 1], s=markersize)
                                # fig4, ax4 = annotion(corr3[0, 1], corr3[0, 2], corr3[0, 3], fig4, ax4)

                        if np.isnan(v4).any():
                            print('NaN exists in correlation')
                        else:
                            corr4 = partial_corr(v4)
                            if not np.isnan([corr4[0, 1], corr4[0, 2], corr4[0, 3]]).any():
                                ax4.scatter(corr4[0, 1], corr4[0, 2], corr4[0, 3], label='Mod ' + str(m + 1) + ' SON',
                                            marker='o', c=col[m + 1], s=markersize)
                                # fig4, ax4 = annotion(corr4[0, 1], corr4[0, 2], corr4[0, 3], fig4, ax4)

            ax4.dist = 12
            ax4.tick_params(labelsize=5)
            ax4.legend(bbox_to_anchor=(1.03, 1), loc=2, fontsize=lengendfontsize)
            ax4.set_xlabel(variable2 + '\n' + m_unit_obs2 + '')
            ax4.set_ylabel(variable3 + '\n' + m_unit_obs3 + '')
            ax4.set_zlabel(variable4 + '\n' + m_unit_obs4 + '')
            ax4.xaxis.labelpad = 10
            ax4.yaxis.labelpad = 10
            ax4.zaxis.labelpad = 10
            # ax80.set_xlim(-1,1)
            # ax80.set_ylim(-1,1)
            # ax81.set_xlim(-1,1)
            # ax81.set_ylim(-1,1)
            # ax82.set_xlim(-1,1)
            # ax82.set_ylim(-1,1)
            # ax83.set_xlim(-1,1)
            # ax83.set_ylim(-1,1)

            plt.suptitle(site + ':  ' + variable1 +' ('+ variable2 +', '+ variable3 +', '+ variable4 + ') Partial correlation', fontsize=24)
            fig4.savefig(filedir + 'response_3d' + '/' + site + variable1 +variable2 + variable3 + variable4 +'3d_Response_corr' + 'Observed' + '.png', bbox_inches='tight')

            plt.close('all')


def plot_variable_matrix(data, variable_list, figname):
    fig, axes = plt.subplots(len(variable_list), len(variable_list), sharex=True, sharey=True,
                             figsize=(6, 6))
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    ax_cb = fig.add_axes([.91, .3, .03, .4])

    for j in range(len(variable_list)):  # models
        for k in range(j, len(variable_list)):
            array = data[:, k, j]  ## Put n sites in one picture
            nam = ['Obs('+str(array[0])[:4]+')']
            nam.extend(['M'+str(i)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            df_cm = pd.DataFrame(array)
            annot = [nam[i] for i in range(len(array))]
            # ax.pie(array, autopct=lambda(p): '{v:d}'.format(p * sum(list(array)) / 100), startangle=90,colors=my_cmap(my_norm(color_vals)))
            sns.heatmap(df_cm, annot=np.array([annot]).T, cmap='PuBuGn', cbar=k == 0, ax=axes[k][j],
                       vmin=-1, vmax=1, fmt = '',
                       cbar_ax=None if k else ax_cb)
            # print(i, j)
            if j == 0:
                axes[k][j].set_ylabel((variable_list[k]))
            if k == len(variable_list) - 1:
                axes[k][j].set_xlabel(variable_list[j])
            # axes[i][j].axis('off')
            axes[k][j].set_yticklabels([])
            axes[k][j].set_xticklabels([])
    plt.suptitle('Variable correlation:' + ' vs '.join(variable_list))
    fig.savefig(figname, bbox_inches='tight')
    plt.close('all')


def plot_variable_matrix_trend_and_detrend(data, dtrend_data, variable_list, figname, categories=None):
    fig, axes = plt.subplots(len(variable_list), len(variable_list), sharex=True, sharey=True,
                             figsize=(6, 6))
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    ax_cb = fig.add_axes([.91, .3, .03, .4])

    for j in range(len(variable_list)):  # models
        for k in range(j, len(variable_list)):
            array = data[:, k, j]  ## Put n sites in one picture
            nam = ['Obs('+str(array[0])[:4]+')']
            nam.extend(['M'+str(i)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            df_cm = pd.DataFrame(array)
            annot = [nam[i] for i in range(len(array))]
            # ax.pie(array, autopct=lambda(p): '{v:d}'.format(p * sum(list(array)) / 100), startangle=90,colors=my_cmap(my_norm(color_vals)))
            sns.heatmap(df_cm, annot=np.array([annot]).T, cmap='Spectral', cbar=k == 0, ax=axes[k][j],
                       vmin=-1, vmax=1, fmt = '',
                       cbar_ax=None if k else ax_cb)
            # print(i, j)
            if j == 0:
                axes[k][j].set_ylabel((variable_list[k]))
            if k == len(variable_list) - 1:
                axes[k][j].set_xlabel(variable_list[j])
            # axes[i][j].axis('off')
            axes[k][j].set_yticklabels([])
            axes[k][j].set_xticklabels([])

    for j in range(len(variable_list)):  # models
        for k in range(0, j):
            array = dtrend_data[:, k, j]  ## Put n sites in one picture
            nam = ['DObs('+str(array[0])[:4]+')']
            nam.extend(['DM'+str(i)+'('+str(array[i+1])[:4]+')' for i in range(len(array)-1)])
            df_cm = pd.DataFrame(array)
            annot = [nam[i] for i in range(len(array))]
            # ax.pie(array, autopct=lambda(p): '{v:d}'.format(p * sum(list(array)) / 100), startangle=90,colors=my_cmap(my_norm(color_vals)))
            sns.heatmap(df_cm, annot=np.array([annot]).T, cmap='Spectral', ax=axes[k][j],cbar=False,
                       vmin=-1, vmax=1, fmt = '',
                       cbar_ax=None)
            # print(i, j)
            # if j == 0:
            #     axes[k][j].set_ylabel((variable_list[k]))
            # if k == len(variable_list) - 1:
            #     axes[k][j].set_xlabel(variable_list[j])
            # axes[i][j].axis('off')
            axes[k][j].set_yticklabels([])
            axes[k][j].set_xticklabels([])

    plt.suptitle(categories+' ' + 'Correlation:' + ', '.join(variable_list))
    # plt.annotate('D stands for the detrend data', (0, 0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.figtext(0.99, 0.01, 'D stands for the detrended data', horizontalalignment='right')
    fig.savefig(figname, bbox_inches='tight')
    plt.close('all')


def detrend_corr(C):
    # This function caculates the detrend correlation between pairs of variables in C
    # C = np.column_stack([C, np.ones(C.shape[0])])
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            mask = C[:, i].mask | C[:, j].mask
            if len(C[:, i][~mask])>0:
                res_i = signal.detrend(C[:, i][~mask])
                res_j = signal.detrend(C[:, j][~mask])
                corr = np.ma.corrcoef(res_i, res_j)[0, 1]
            else:
                corr = -2
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr


def response_matrix_corr(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):

    hour_mask, hour_mod = [], []
    day_mask, day_mod = [], []
    month_mask, month_mod = [], []
    year_mask, year_mod = [], []
    scores_reponse_matrix = []

    for variable in variable_list:

        h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable)
        h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable)
        d_obs1, d_t_obs1, d_unit_obs1 = data_extract(d_o, variable)
        d_mod1, d_t_mod1, d_unit_mod1 = models_data_extract(d_obs1, d_models, variable)
        m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable)
        m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable)
        y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable)
        y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable)

        hour_mask.append(h_obs1.mask)
        day_mask.append(d_obs1.mask)
        month_mask.append(m_obs1.mask)
        year_mask.append(y_obs1.mask)

        h_mod1.insert(0, h_obs1)
        d_mod1.insert(0, d_obs1)
        m_mod1.insert(0, m_obs1)
        y_mod1.insert(0, y_obs1)

        hour_mod.append(np.asarray(h_mod1))
        day_mod.append(np.asarray(d_mod1))
        month_mod.append(np.asarray(m_mod1))
        year_mod.append(np.asarray(y_mod1))

    h_data = np.asarray(hour_mod)
    d_data = np.asarray(day_mod)
    m_data = np.asarray(month_mod)
    y_data = np.asarray(year_mod)

    h_data = np.ma.masked_where(h_data >= 1.0e+18, h_data)
    d_data = np.ma.masked_where(d_data >= 1.0e+18, d_data)
    m_data = np.ma.masked_where(m_data >= 1.0e+18, m_data)
    y_data = np.ma.masked_where(y_data >= 1.0e+18, y_data)

    h_data = np.ma.masked_invalid(h_data)
    d_data = np.ma.masked_invalid(d_data)
    m_data = np.ma.masked_invalid(m_data)
    y_data = np.ma.masked_invalid(y_data)

    # print(h_data.shape, d_data.shape, m_data.shape, y_data.shape)
    # hello
    h, d, m, y = [], [], [], []
    for i in range(len(h_mod1)):
        h.append(np.ma.masked_where(np.asarray(hour_mask), h_data[:, i, :, :]))
        d.append(np.ma.masked_where(np.asarray(day_mask), d_data[:, i, :, :]))
        m.append(np.ma.masked_where(np.asarray(month_mask), m_data[:, i, :, :]))
        y.append(np.ma.masked_where(np.asarray(year_mask), y_data[:, i, :, :]))

    all_corr_h, all_corr_d, all_corr_m, all_corr_y = [], [], [], []
    dall_corr_h, dall_corr_d, dall_corr_m, dall_corr_y = [], [], [], []

    for i in range(len(h_mod1)):
        corr_matrix_h,corr_matrix_d,corr_matrix_m,corr_matrix_y = [], [], [], []
        d_corr_matrix_h,d_corr_matrix_d,d_corr_matrix_m,d_corr_matrix_y = [], [], [], []

        for j in range(len(h_site_name_obs1)):
            corr_h, mask_h = correlation_matrix(h[i][:, j, :].T)
            corr_d, mask_d = correlation_matrix(d[i][:, j, :].T)
            corr_m, mask_m = correlation_matrix(m[i][:, j, :].T)
            corr_y, mask_y = correlation_matrix(y[i][:, j, :].T)

            detrend_corr_h = detrend_corr(h[i][:, j, :].T)
            detrend_corr_d = detrend_corr(d[i][:, j, :].T)
            detrend_corr_m = detrend_corr(m[i][:, j, :].T)
            detrend_corr_y = detrend_corr(y[i][:, j, :].T)

            corr_matrix_h.append(corr_h)
            corr_matrix_d.append(corr_d)
            corr_matrix_m.append(corr_m)
            corr_matrix_y.append(corr_y)

            d_corr_matrix_h.append(detrend_corr_h)
            d_corr_matrix_d.append(detrend_corr_d)
            d_corr_matrix_m.append(detrend_corr_m)
            d_corr_matrix_y.append(detrend_corr_y)

        all_corr_h.append(np.ma.masked_invalid(np.asarray(corr_matrix_h)))
        all_corr_d.append(np.ma.masked_invalid(np.asarray(corr_matrix_d)))
        all_corr_m.append(np.ma.masked_invalid(np.asarray(corr_matrix_m)))
        all_corr_y.append(np.ma.masked_invalid(np.asarray(corr_matrix_y)))
        dall_corr_h.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_h)))
        dall_corr_d.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_d)))
        dall_corr_m.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_m)))
        dall_corr_y.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_y)))

    corr_array_h = np.ma.masked_invalid(np.asarray(all_corr_h))
    corr_array_d = np.ma.masked_invalid(np.asarray(all_corr_d))
    corr_array_m = np.ma.masked_invalid(np.asarray(all_corr_m))
    corr_array_y = np.ma.masked_invalid(np.asarray(all_corr_y))
    d_corr_array_h = np.ma.masked_invalid(np.asarray(dall_corr_h))
    d_corr_array_d = np.ma.masked_invalid(np.asarray(dall_corr_d))
    d_corr_array_m = np.ma.masked_invalid(np.asarray(dall_corr_m))
    d_corr_array_y = np.ma.masked_invalid(np.asarray(dall_corr_y))

    directory = filedir + 'response_cc' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for j, site in enumerate(h_site_name_obs1):
        if h_site_name_obs1.mask[j]:
            continue

        print('Process on response_cc_' + site + '_No.' + str(j) + '!')
        # variable_list = ['LHF', 'ER', 'FSH', 'NEE', 'GPP']
        plot_variable_matrix_trend_and_detrend(corr_array_h[:, j, :, :], d_corr_array_h[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_hourly' + 'Observed' + '.png', categories='Hourly')
        plot_variable_matrix_trend_and_detrend(corr_array_d[:, j, :, :], d_corr_array_d[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_daily' + 'Observed' + '.png', categories='Daily')
        plot_variable_matrix_trend_and_detrend(corr_array_m[:, j, :, :], d_corr_array_m[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_monthly' + 'Observed' + '.png', categories='Monthly')
        plot_variable_matrix_trend_and_detrend(corr_array_y[:, j, :, :], d_corr_array_y[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_yearly' + 'Observed' + '.png', categories='Yearly')

    return scores_reponse_matrix


def seasonal_response_matrix_corr(filedir, h_site_name_obs1, h_o, d_o, m_o, y_o, h_models, d_models, m_models, y_models, variable_list):

    hour_mask, hour_mod = [], []
    day_mask, day_mod = [], []
    month_mask, month_mod = [], []
    year_mask, year_mod = [], []
    scores_reponse_matrix = []

    for variable in variable_list:

        # h_obs1, h_t_obs1, h_unit_obs1 = data_extract(h_o, variable)
        # h_mod1, h_t_mod1, h_unit_mod1 = models_data_extract(h_obs1, h_models, variable)
        d_obs11, d_t_obs11, d_unit_obs11 = data_extract(d_o, variable)
        d_mod11, d_t_mod11, d_unit_mod11 = models_data_extract(d_obs11, d_models, variable)
        # m_obs1, m_t_obs1, m_unit_obs1 = data_extract(m_o, variable)
        # m_mod1, m_t_mod1, m_unit_mod1 = models_data_extract(m_obs1, m_models, variable)
        # y_obs1, y_t_obs1, y_unit_obs1 = data_extract(y_o, variable)
        # y_mod1, y_t_mod1, y_unit_mod1 = models_data_extract(y_obs1, y_models, variable)

        h_obs1, d_obs1, m_obs1, y_obs1 = day_seasonly_process(d_obs11)
        h_mod1, d_mod1, m_mod1, y_mod1 = day_models_seasonly_process(d_mod11)


        hour_mask.append(h_obs1.mask)
        day_mask.append(d_obs1.mask)
        month_mask.append(m_obs1.mask)
        year_mask.append(y_obs1.mask)

        h_mod1.insert(0, h_obs1)
        d_mod1.insert(0, d_obs1)
        m_mod1.insert(0, m_obs1)
        y_mod1.insert(0, y_obs1)

        hour_mod.append(np.asarray(h_mod1))
        day_mod.append(np.asarray(d_mod1))
        month_mod.append(np.asarray(m_mod1))
        year_mod.append(np.asarray(y_mod1))

    h_data = np.asarray(hour_mod)
    d_data = np.asarray(day_mod)
    m_data = np.asarray(month_mod)
    y_data = np.asarray(year_mod)

    h_data = np.ma.masked_where(h_data >= 1.0e+18, h_data)
    d_data = np.ma.masked_where(d_data >= 1.0e+18, d_data)
    m_data = np.ma.masked_where(m_data >= 1.0e+18, m_data)
    y_data = np.ma.masked_where(y_data >= 1.0e+18, y_data)

    h_data = np.ma.masked_invalid(h_data)
    d_data = np.ma.masked_invalid(d_data)
    m_data = np.ma.masked_invalid(m_data)
    y_data = np.ma.masked_invalid(y_data)


    h, d, m, y = [], [], [], []
    for i in range(len(h_mod1)):
        h.append(np.ma.masked_where(np.asarray(hour_mask), h_data[:, i, :, :]))
        d.append(np.ma.masked_where(np.asarray(day_mask), d_data[:, i, :, :]))
        m.append(np.ma.masked_where(np.asarray(month_mask), m_data[:, i, :, :]))
        y.append(np.ma.masked_where(np.asarray(year_mask), y_data[:, i, :, :]))

    all_corr_h, all_corr_d, all_corr_m, all_corr_y = [], [], [], []
    dall_corr_h, dall_corr_d, dall_corr_m, dall_corr_y = [], [], [], []

    for i in range(len(h_mod1)):
        corr_matrix_h,corr_matrix_d,corr_matrix_m,corr_matrix_y = [], [], [], []
        d_corr_matrix_h,d_corr_matrix_d,d_corr_matrix_m,d_corr_matrix_y = [], [], [], []

        for j in range(len(h_site_name_obs1)):
            corr_h, mask_h = correlation_matrix(h[i][:, :, j].T)
            corr_d, mask_d = correlation_matrix(d[i][:, :, j].T)
            corr_m, mask_m = correlation_matrix(m[i][:, :, j].T)
            corr_y, mask_y = correlation_matrix(y[i][:, :, j].T)

            detrend_corr_h = detrend_corr(h[i][:, :, j].T)
            detrend_corr_d = detrend_corr(d[i][:, :, j].T)
            detrend_corr_m = detrend_corr(m[i][:, :, j].T)
            detrend_corr_y = detrend_corr(y[i][:, :, j].T)

            corr_matrix_h.append(corr_h)
            corr_matrix_d.append(corr_d)
            corr_matrix_m.append(corr_m)
            corr_matrix_y.append(corr_y)

            d_corr_matrix_h.append(detrend_corr_h)
            d_corr_matrix_d.append(detrend_corr_d)
            d_corr_matrix_m.append(detrend_corr_m)
            d_corr_matrix_y.append(detrend_corr_y)

        all_corr_h.append(np.ma.masked_invalid(np.asarray(corr_matrix_h)))
        all_corr_d.append(np.ma.masked_invalid(np.asarray(corr_matrix_d)))
        all_corr_m.append(np.ma.masked_invalid(np.asarray(corr_matrix_m)))
        all_corr_y.append(np.ma.masked_invalid(np.asarray(corr_matrix_y)))
        dall_corr_h.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_h)))
        dall_corr_d.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_d)))
        dall_corr_m.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_m)))
        dall_corr_y.append(np.ma.masked_invalid(np.asarray(d_corr_matrix_y)))

    corr_array_h = np.ma.masked_invalid(np.asarray(all_corr_h))
    corr_array_d = np.ma.masked_invalid(np.asarray(all_corr_d))
    corr_array_m = np.ma.masked_invalid(np.asarray(all_corr_m))
    corr_array_y = np.ma.masked_invalid(np.asarray(all_corr_y))
    d_corr_array_h = np.ma.masked_invalid(np.asarray(dall_corr_h))
    d_corr_array_d = np.ma.masked_invalid(np.asarray(dall_corr_d))
    d_corr_array_m = np.ma.masked_invalid(np.asarray(dall_corr_m))
    d_corr_array_y = np.ma.masked_invalid(np.asarray(dall_corr_y))

    directory = filedir + 'response_cc' + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for j, site in enumerate(h_site_name_obs1):
        if h_site_name_obs1.mask[j]:
            continue

        print('Process on response_cc_' + site + '_No.' + str(j) + '!')
        # variable_list = ['LHF', 'ER', 'FSH', 'NEE', 'GPP']
        plot_variable_matrix_trend_and_detrend(corr_array_h[:, j, :, :], d_corr_array_h[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_DJF' + 'Observed' + '.png', categories='DJF')
        plot_variable_matrix_trend_and_detrend(corr_array_d[:, j, :, :], d_corr_array_d[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_MAM' + 'Observed' + '.png', categories='MAM')
        plot_variable_matrix_trend_and_detrend(corr_array_m[:, j, :, :], d_corr_array_m[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_JJA' + 'Observed' + '.png', categories='JJA')
        plot_variable_matrix_trend_and_detrend(corr_array_y[:, j, :, :], d_corr_array_y[:, j, :, :], variable_list, filedir + 'response_cc' + '/' + site + '3d_Response_corr_matrix_SON' + 'Observed' + '.png', categories='SON')

    return scores_reponse_matrix










