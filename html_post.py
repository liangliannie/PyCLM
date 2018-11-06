class website_post(object):
    ''' A simple program to create an html file from given string,
    and call the default web browser to display the file. '''

    def __init__(self, web_dir, web_name):
        self.web_dir = web_dir
        self.web_name = web_name
        self.page_front = '''
       
        <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
    <html>
    <head>
      <meta content="text/html; charset=ISO-8859-1"
     http-equiv="content-type">
      <title>Hello</title>
    </head>
    <body>'''
        self.page_end ='''<script>
            function button_img(id) {
                var x = document.getElementById(id);
                if (x.style.display === "none") {
                    x.style.display = "block";
                } else {
                    x.style.display = "none";
                }
            }
            </script>
            </body> </html> '''
        self.body = ''


    def strToFile(self, text, filename):
        """Write a file with the given name and the given text."""
        output = open(self.web_dir + filename, "w")
        output.write(text)
        output.close()

    def browseLocal(self, webpageText):
        '''Start your webbrowser on a local file containing the text
        with given filename.'''
        import webbrowser, os.path
        self.strToFile(webpageText, self.web_name)
        webbrowser.open(os.path.abspath(self.web_dir + self.web_name))  # elaborated for Mac

    def add_title_h1(self, title):
        self.body += '<div style="text-align: center;"><h1>' + title + '</h1></div>'
        return self

    def add_title_h2(self, title):
        self.body += '<div style="text-align: center;"><h2>' + title + '</h2></div>'
        return self

    def add_text(self, text):
        self.body += text
        return self

    def add_figure(self, fig_dir, fig_name):
        import datetime
        img_id = str(datetime.datetime.now())
        '''Add a figure to the website '''
        data_uri = open(fig_dir + fig_name, 'rb').read().encode('base64').replace('\n', '')
        self.body += '<button onclick = button_img("%s")> Click to hide/view the picture </button>' % img_id[-6:]
        self.body +=  '<div id ="%s" style="text-align: center;">  <img src="data:image/png;base64,%s" > </div>' % (img_id[-6:], data_uri)
        # print('<div id ="%s" style="text-align: center;">  <img src="data:image/png;base64,%s"  height="1028" width="1028" > </div>' % (img_id[-6:], data_uri))
        return self

    def add_base_map(self,fig_dir, fig_name):
        ''' Add a basemap to the main website '''
        data_uri = open(fig_dir + fig_name, 'rb').read().encode('base64').replace('\n', '')
        self.body += '<div style="text-align: center;">  <img src="data:image/png;base64,%s" height="1800" width="3600" alt="Planets" usemap="#planetmap" > </div>' % data_uri
        #style="width:145px;height:126px;"
        return self

    def map_head(self):
        self.body += '<map name="planetmap">'
        return self

    def map_tail(self):
        self.body += '</map>'
        return

    def add_area(self, x, y, size, alt, href):
        self.body += '<area shape="circle" coords="%d,%d,%d" alt="%s" href="%s">' %(x, y, size, alt, href)
        return self

    def add_table(self, lol):
        '''Add a table to the website'''
        table =  '<div class="datagrid"> <table align ="center" border="1" ><caption> Table of score</caption>'
        table += '  <thead> <tr><th>'
        table += '  </th> <th>'.join(lol[0])
        table += '  </th> </tr> </thead>  <tbody>'

        for i, sublist in enumerate(lol):
            if i > 1:
                if i % 2 == 1:
                    table += '<tr class="alt" ><td>'
                else:
                    table += '<tr><td>'
                table += '  </td><td>'.join(sublist)
                table += '  </td></tr>'
        table += '</tbody> </table> </div>'
        self.body += table
        return self

    def add_link(self,link_dir, link_name, name):
        ''' Add a href to the website '''
        self.body += '''<a href =''' + link_dir + link_name + ''' >  ''' + name +'''   </a>'''
        return self



    def generate_post(self):
        ''' generate the html '''

        # variables = self.body
        # contents = self.page.format(**locals())
        contents = self.page_front + self.body + self.page_end
        self.browseLocal(contents)

# Table_name = ['Model', 'Mean', 'Bias', 'RMSE', 'Mean Score', 'Bias Score', 'RMSE Score', 'Cycles Score',
        #               'Frequency Score', 'Reponse Score', 'Overall Score']
        # m0 = ['Benchmark', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
        # m1 = ['m1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
        # m2 = ['m2', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
        # lol = []
        # lol.append(Table_name)
        # lol.append(m0)
        # lol.append(m1)
        # lol.append(m2)
        # h1.add_figure('/Users/lli51/Desktop/ornl/','summary_score.png')
        # h1.add_table(lol)
def plot_base_map(mainfiledir, lon, lat, site_name_obs):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
    # are the lat/lon values of the lower left and upper right corners
    # of the map.
    # resolution = 'c' means use crude resolution coastlines.
    fig8 = plt.figure(figsize=(27, 15))
    m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.fillcontinents(color='beige',lake_color='lightblue')
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,91.,30.),color='gray',dashes=[1,3],labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180.,181.,60.),color='gray',dashes=[1,3],labels=[0,0,0,1])
    m.drawmapboundary(fill_color='lightblue')
    plt.title('Site Analysis')
    # print(lon, lat)
    xpt1, ypt1 = [], []
    for i in range(len(site_name_obs)):
    # plot blue dot on Boulder, colorado and label it as such.
        xpt,ypt = m(lon[i],lat[i])
        lonpt, latpt = m(xpt,ypt,inverse=True)
        site = site_name_obs[i]
        m.plot(xpt,ypt,'o',label=site, markersize=15)  # plot a blue dot there
        xpt1.append(lon[i]), ypt1.append(lat[i])
        plt.text(xpt, ypt, site, fontsize=8)  # add link here
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    fig8.savefig(mainfiledir + 'site_plot' + '.png', dpi=100)
    return xpt1, ypt1, site_name_obs


def final_post(mainfiledir, lon, lat, site_name_obs, variable_list, variable_list1, variable_list2,len_models, variable_list_4d, variable_list_rr):
    import os
    if not os.path.exists('./output/websites/'):
        os.makedirs('./output/websites/')
    '''This post the final website'''

    for variable_name in variable_list:
        for j, site in enumerate(site_name_obs):
            if site_name_obs.mask[j]:
                continue
            print('Posting local HTML' + site +'_No.'+ str(j)+'!')

            h1 = website_post('./output/websites/', site+'time_series'+ variable_name + '.html')
            h2 = website_post('./output/websites/', site +'pdf'+ variable_name + '.html')
            h3 = website_post('./output/websites/', site +'wavelet'+ variable_name + '.html')
            h4 = website_post('./output/websites/', site +'spectrum'+ variable_name + '.html')
            h5 = website_post('./output/websites/', site +'imf'+ variable_name + '.html')
            h6 = website_post('./output/websites/', site +'cycle'+ variable_name + '.html')


            time_series = ['_time_basic_','_season_']
            time_titles = [ 'Times series', 'Seasonly cycles']

            pdf_cdf = ['_pdf_', '_cdf_','_season_pdf_', '_season_cdf_']
            pdf_cdf_titles = ['PDF', 'CDF','Seasonal PDF', 'Seasonal CDF']

            cycle= ['_day_', '_time_series_']
            cycle_titles = ['Diurnal cycle', 'Hourly, daily, monthly cycles']

            wavelet =['_wavelet_']
            wavelet_titles = ['Wavelet analysis for observations']

            imf  = ['_IMF_', '_Decompose_IMF_', '_deviation_IMF_', '_hht_IMF_observed']
            imf_titles = ['IMF analysis for observations ', 'Decomposed IMF', 'Instantaneous period of IMF(day)', 'HHT Spectrum Obs']

            spectrum =['_spectrum_']
            spectrum_title =['Spectrum analysis of observed data and models']

            taylor = ['_taylor_']
            taylor_titles = [ 'Taylor graph of observed data and models']

            for i, name in enumerate(time_series):
                # h1.add_title_h1(time_titles[i])
                h1.add_figure('./output/' + variable_name + '/', site + name + variable_name + '.png')
                h1.add_text(' <br />')
            h1.generate_post()

            for i, name in enumerate(pdf_cdf):
                # h2.add_title_h1(pdf_cdf_titles[i])
                h2.add_figure('./output/' + variable_name + '/', site + name + variable_name + '.png')
                h2.add_text(' <br />')
            h2.generate_post()

            for i, name in enumerate(cycle):
                # h6.add_title_h1(cycle_titles[i])
                h6.add_figure('./output/' + variable_name + '/', site + name + variable_name + '.png')
                h6.add_text(' <br />')
            h6.generate_post()


            # h3.add_title_h1(wavelet_titles[0])
            h3.add_figure('./output/' + variable_name + '/', site + wavelet[0] + variable_name + '.png')
            h3.add_text(' <br />')
            for m in range(len_models):
                # h3.add_title_h1('Errors between Observed data and Model ' + str(m + 1))
                h3.add_figure('./output/' + variable_name + '/',
                              site + 'model' + str(m) + '_wavelet_' + variable_name + '.png')
                h3.add_text(' <br />')
            h3.generate_post()

            # h4.add_title_h1(spectrum_title[0])
            h4.add_figure('./output/' + variable_name + '/', site + spectrum[0] + variable_name + '.png')
            h4.add_text(' <br />')
            h4.generate_post()

            for i, name in enumerate(imf):
                h5.add_title_h1(imf_titles[i])
                h5.add_figure('./output/' + variable_name + '/', site + name + variable_name + '.png')
                h5.add_text(' <br />')

            for m in range(len_models):
                h5.add_title_h1('Model ' + str(m + 1))
                h5.add_figure('./output/' + variable_name + '/', site + '_hht_IMF_model' + str(m + 1) + variable_name + '.png')
                h5.add_text(' <br />')

            h5.generate_post()


    for j, site in enumerate(site_name_obs):
        if site_name_obs.mask[j]:
            continue
        print('Posting response HTML' + site+ '_No.'+ str(j)+'!')
        for i in range(len(variable_list1)):
            variable1, variable2 = variable_list1[i], variable_list2[i]

            web_name = 'response'+site+variable1+'vs'+variable2+'.html'
            h7 = website_post('./output/websites/',web_name)

            # h7.add_title_h1(site + ':' + variable1 + ' vs ' + variable2 + 'Two variables response')
            h7.add_figure('./output/' + 'response' + '/' , site + '_' + variable1 + '_vs_' + variable2 + '_Response' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(site + '_' + variable1 + ' vs ' + variable2 + 'Corresponding response bin')
            h7.add_figure('./output/' + 'response' + '/' , site + '_' + variable1 + '_vs_' + variable2 + '_Response_Bin' + '.png')
            h7.add_text(' <br />')
            h7.generate_post()


    for j, site in enumerate(site_name_obs):
        if site_name_obs.mask[j]:
            continue
        print('Posting response3d HTML' + site + '_No.'+ str(j)+'!')
        for i in range(len(variable_list_4d)):
            variable1, variable2, variable3, variable4 = variable_list_4d[i]

            web_name = 'response3d'+ site+ variable1+ variable2+variable3+variable4+'.html'
            h7 = website_post('./output/websites/',web_name)

            # h7.add_title_h1(site + ':' + variable1 + ' vs ' + variable2 + ' vs ' + variable3 + ' vs ' + variable4 + 'Four variables\' response')
            h7.add_figure('./output/' + 'response_3d' + '/', site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'Observed' + '.png')
            h7.add_text(' <br />')
            h7.add_figure('./output/' + 'response_3d' + '/',  site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'seasonal' + '.png')
            h7.add_text(' <br />')
            h7.add_figure('./output/' + 'response_3d' + '/',  site + '3d_Response' + variable1 + variable2 + variable3 + variable4+'day_seasonal' + '.png')
            h7.add_text(' <br />')
            # for m in range(len_models):
            #     h7.add_title_h1(site + '_' + variable1 + '_vs_' + variable2 + '_vs_' + variable3 + '_vs_' + variable4 + '_Response 3d'+ 'model'+str(m+1))
            #     h7.add_figure('./output/' + 'response_3d' + '/' , site + '3d_Response' + "Model "+str(m+1) + '.png')
            #     h7.add_text(' <br />')
            h7.generate_post()

    for j, site in enumerate(site_name_obs):
        if site_name_obs.mask[j]:
            continue
        print('Posting partialcorr HTML' + site + '_No.'+ str(j)+'!')
        for i in range(len(variable_list_rr)):
            variable1, variable2, variable3, variable4 = variable_list_rr[i]

            web_name = 'partialcorr'+site+ variable1+ variable2+variable3+variable4+'.html'
            h7 = website_post('./output/websites/',web_name)

            # h7.add_title_h1(site + ':' + variable1 + ' vs ' + variable2 + ' vs ' + variable3 + ' vs ' + variable4 + 'Partial Correlation')
            h7.add_figure('./output/' + 'response_3d' + '/', site + variable1 + variable2 + variable3 + variable4 +'3d_Response_corr' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # for m in range(len_models):
            #     h7.add_title_h1(site + '_' + variable1 + '_vs_' + variable2 + '_vs_' + variable3 + '_vs_' + variable4 + '_Response 3d'+ 'model'+str(m+1))
            #     h7.add_figure('./output/' + 'response_3d' + '/' , site + '3d_Response' + "Model "+str(m+1) + '.png')
            #     h7.add_text(' <br />')
            h7.generate_post()
    for j, site in enumerate(site_name_obs):
        if site_name_obs.mask[j]:
            continue
        print('Posting variablecorr HTML' + site + '_No.' + str(j) + '!')
        for i in range(len(variable_list_rr)):
            variable1, variable2, variable3, variable4 = variable_list_rr[i]

            web_name = 'variablecorr' + site + '.html'
            h7 = website_post('./output/websites/', web_name)

            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Hourly')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_hourly' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Daily')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_daily' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Monthly')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_monthly' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Annually')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_yearly' + 'Observed' + '.png')
            h7.add_text(' <br />')

            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_DJF' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Daily')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_MAM' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Monthly')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_JJA' + 'Observed' + '.png')
            h7.add_text(' <br />')
            # h7.add_title_h1(
            #     site + ':' + 'All variables correlation: Annually')
            h7.add_figure('./output/' + 'response_cc' + '/',
                          site + '3d_Response_corr_matrix_SON' + 'Observed' + '.png')
            h7.add_text(' <br />')

            # h7.add_title_h1(
            #     site + ':' + 'Detrend all variables correlation: Hourly')
            # h7.add_figure('./output/' + 'response_cc' + '/',
            #               site + '3d_Response_corr_matrix_hourly_detrend' + 'Observed' + '.png')
            # h7.add_text(' <br />')
            # # h7.add_title_h1(
            # #     site + ':' + 'Detrend all variables correlation: Daily')
            # h7.add_figure('./output/' + 'response_cc' + '/',
            #               site + '3d_Response_corr_matrix_daily_detrend' + 'Observed' + '.png')
            # h7.add_text(' <br />')
            # # h7.add_title_h1(
            # #     site + ':' + 'Detrend all variables correlation: Monthly')
            # h7.add_figure('./output/' + 'response_cc' + '/',
            #               site + '3d_Response_corr_matrix_monthly_detrend' + 'Observed' + '.png')
            # h7.add_text(' <br />')
            # # h7.add_title_h1(
            # #     site + ':' + 'Detrend all variables correlation: Annually')
            # h7.add_figure('./output/' + 'response_cc' + '/',
            #               site + '3d_Response_corr_matrix_yearly_detrend' + 'Observed' + '.png')
            # h7.add_text(' <br />')


            h7.generate_post()

    from site_main_website_post import generate_post_site
    for j, site in enumerate(site_name_obs):
        if site_name_obs.mask[j]:
            continue
        print('Posting site HTML' + site + '_No.' + str(j) + '!')
        generate_post_site(site, variable_list, mainfiledir, variable_list1, variable_list2, variable_list_4d, variable_list_rr)

    print('Posting main HTML')
    xpt, ypt, site_names = plot_base_map(mainfiledir, lon, lat, site_name_obs)

    from main_website_post2 import generate_post
    generate_post('./', 'summary_score.png', site_names, xpt, ypt)

    # h2 = website_post(mainfiledir,'main.html')
    # h2.add_base_map(mainfiledir, 'site_plot.png')
    # h2.map_head()
    # for j, site in enumerate(site_name_obs):
    #     if site_name_obs.mask[j]:
    #         continue
    #     h2.add_area((xpt[j]+180)*10, (ypt[j]+90)*10, 500, site, mainfiledir+'main.html')
    # h2.map_tail()
    #
    # for site in site_name_obs:
    #     h2.add_text('Site Name:  ')
    #     h2.add_text(site)
    #     h2.add_text(' <br />')
    #     for variable_name in variable_list:
    #         link_name = site + variable_name +  'local_webpage.html'
    #         if variable_name == 'FSH_EFLX_LH_TOT':
    #             h2.add_link('./output/' + variable_name + '/', link_name, '    FSH\EFLX_LH_TOT')
    #         else:
    #             h2.add_link('./output/' + variable_name + '/', link_name, '    '+variable_name)
    #     h2.add_link('./output/' + 'response' + '/', site + 'local_webpage.html', 'Response')
    #     h2.add_text('<br />')
    #
    # h2.add_title_h1('A summary of all results')
    # h2.add_figure('./output/','summary.png')
    # h2.add_text(' <br />')
    # h2.generate_post()






