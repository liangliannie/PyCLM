import os

def strToFile(text, web_dir, web_name):
    """Write a file with the given name and the given text."""
    output = open(web_dir + web_name, "w")
    output.write(text)
    output.close()

def browseLocal(webpageText, web_dir, web_name):
    '''Start your webbrowser on a local file containing the text
    with given filename.'''
    import webbrowser, os.path
    strToFile(webpageText, web_dir, web_name)
    webbrowser.open(os.path.abspath(web_dir + web_name))  # elaborated for Mac

def add_figure_link(Fig_name, Fig_dir, Fig_alt, href, width=150, height=120):
    data_uri = open(Fig_dir, 'rb').read().encode('base64').replace('\n', '')

    return '<div class="imgContainer"> <figure> <figcaption> %s </figcaption> <a href = "%s" > <img border = "0" alt = "%s" src = "data:image/png;base64,%s" width = "%d" height = "%d" ></a><figure></div>' %(Fig_name, href, Fig_alt, data_uri, width, height)

def table_link(maindir, site, category, variable):

    if category == 'time_series':
        table_front = '''<p>
        <table>
      <tr>
        <th>Time series</th>
      </tr>'''
    elif category == 'pdf':
        table_front = '''<p>
          <table>
        <tr>
          <th>PDF and CDF</th>
        </tr>'''
    elif category == 'cycle':
        table_front = '''<p>
          <table>
        <tr>
          <th>Cycles</th>
        </tr>'''
    elif category == 'wavelet':
        table_front = '''<p>
          <table>
        <tr>
          <th>Wavelet</th>
        </tr>'''
    elif category == 'imf':
        table_front = '''<p>
          <table>
        <tr>
          <th>IMF</th>
        </tr>'''
    elif category == 'spectrum':
        table_front = '''<p>
          <table>
        <tr>
          <th>Spectrum</th>
        </tr>'''
    else:
        table_front = '''<p>
          <table>
        <tr>
          <th>Variable Name</th>
        </tr>'''


    for v in variable:
        s = './'+site+category+v+'.html'
        table_front += '''<tr><td><a href="%s"> %s </a></td></tr>''' %(s, v)

    table_end = '''</table></p>'''
    table = table_front + table_end
    return table

def table_link2(maindir, site, variable1, variable2):
    table_front = '''<p><table>
  <tr>
    <th>2-d response</th>
  </tr>'''
    for i in range(len(variable1)):
        s = './'+'response'+site+variable1[i]+'vs'+variable2[i]+'.html'
        table_front += '''<tr><td><a href="%s"> %s </a></td></tr>''' % (s, variable2[i] +' vs '+variable1[i])

    table_end = '''  </table></p>'''
    table = table_front + table_end
    return table

def table_link3(maindir, site, variable_list_4d):
    table_front = '''<p><table>
  <tr>
    <th>3-d response</th>
  </tr> 
                '''
    for i in range(len(variable_list_4d)):
        variable1, variable2, variable3, variable4 = variable_list_4d[i]
        s = './' + 'response3d' + site+ variable1+ variable2+variable3+variable4 + '.html'
        table_front += '''<tr><td><a href="%s"> %s </a></td></tr>''' % (s, variable_list_4d[i][3]+'('+variable_list_4d[i][0] + ', ' + variable_list_4d[i][1] + ', '+variable_list_4d[i][2] + ')')

    table_end = '''  </table></p>'''
    table = table_front + table_end
    return table

def table_link4(maindir, site, variable_list_4d):
    table_front = '''<p><table>
  <tr>
    <th>Partial Correlations</th>
  </tr> 
                '''
    for i in range(len(variable_list_4d)):
        variable1, variable2, variable3, variable4 = variable_list_4d[i]
        s = './' + 'partialcorr'+site+ variable1+ variable2+variable3+variable4+'.html'
        table_front += '''<tr><td><a href="%s"> %s </a></td></tr>''' % (s, variable_list_4d[i][0]+'('+variable_list_4d[i][1] + ', ' + variable_list_4d[i][2] + ', '+variable_list_4d[i][3] + ')')

    table_end = '''  </table></p>'''
    table = table_front + table_end
    return table

def table_link5(maindir, site):
    table_front = '''<p><table>
  <tr>
    <th>Variable Correlations</th>
  </tr>
                '''
    s = './' + 'variablecorr' + site + '.html'
    table_front += '''<tr><td><a href="%s"> Correlations of all variables </a></td></tr>''' % s

    table_end = '''  </table></p>'''
    table = table_front + table_end
    return table

def generate_post_site(site, variable, mainfiledir, variable1, variable2, variable_list_4d, variable_list_4d2):
    # < base href = "'''+mainfiledir+'''" > < / base >
    page_front = '''<!DOCTYPE html>
    <html>
        <head>
        <base href="./"></base>
    <style>
    * {
        box-sizing: border-box;
    }

          /* Optional: Makes the sample page fill the window. */
          html, body {
            height: 100%;
            margin: 20;
            padding: 0;
          }

    /* Create two equal columns that floats next to each other */
    .columnleft {
        float: left;
        width: 35%;
        /*height: 100%;  Should be removed. Only for demonstration */
    }

    .columnright {
        float: right;
        width: 60%;
        /*height: 100%;  Should be removed. Only for demonstration */
    }
    /* Clear floats after the columns */
    .row:after {
        content: "";
        display: table;
        clear: both;
    }
        .headertekst {
      text-align: center;
    }
    
    .imgContainer{
    float:left;
}

table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 50%;
}

td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {
    background-color: #dddddd;
}

    </style>
    </head>

    <body>'''
    page_body_left = '''   <div  class="columnleft"> <h2>Site Analysis </h2> <p> {instruction} </p> 
    '''

    page_body_right = ''' </div> <div class="columnright"> '''



    page_tail = '''</div>
    </body>
    </html>'''
    instruction = ''


    page_body_left = page_body_left.format(**locals())
    page_body_left  += add_figure_link('Score', 'output/score/'+site+'summary_score.png','scores', mainfiledir+'main_website.html',width=280, height=500)


    table_time = table_link('output/websites/', site, 'time_series', variable)
    page_body_right += add_figure_link('Time series Analysis', 'output/'+variable[0]+'/'+site+'_time_series_'+variable[0]+'.png', 'time_series', './'+site+'time_series'+variable[0]+'.html')
    page_body_right += table_time
    table_pdf = table_link('output/websites/', site, 'pdf', variable)
    page_body_right += add_figure_link('PDF and CDF Analysis', 'output/'+variable[0]+'/'+site+'_pdf_'+variable[0]+'.png', 'pdf', './'+site+'pdf'+variable[0]+'.html')
    page_body_right += table_pdf

    table_pdf = table_link('output/websites/', site, 'cycle', variable)
    page_body_right += add_figure_link('Cycles Analysis', 'output/'+variable[0]+'/'+site+'_time_series_'+variable[0]+'.png', 'pdf', './'+site+'cycle'+variable[0]+'.html')
    page_body_right += table_pdf


    table_frequency = table_link('output/websites/', site, 'wavelet', variable)
    page_body_right += add_figure_link('Wavelet Analysis', 'output/'+variable[0]+'/'+site+'_wavelet_'+variable[0]+'.png', 'wavelet', './'+site+'wavelet'+variable[0]+'.html')
    page_body_right += table_frequency

    table_frequency = table_link('output/websites/', site, 'imf', variable)
    page_body_right += add_figure_link('IMF Analysis', 'output/'+variable[0]+'/'+site+'_imf_'+variable[0]+'.png', 'imf', './'+site+'imf'+variable[0]+'.html')
    page_body_right += table_frequency

    table_frequency = table_link('output/websites/', site, 'spectrum', variable)
    page_body_right += add_figure_link('Spectrum Analysis', 'output/'+variable[0]+'/'+site+'_spectrum_'+variable[0]+'.png', 'spectrum', './'+site+'spectrum'+variable[0]+'.html')
    page_body_right += table_frequency

    table_response = table_link2('output/websites/', site, variable1, variable2)
    page_body_right += add_figure_link('Response Analysis(bin)', 'output/'+'response'+'/'+site+'_'+variable1[0]+'_vs_'+variable2[0]+'_Response.png', 'time_series', './'+'response'+site+variable1[0]+'vs'+variable2[0]+'.html')
    page_body_right += table_response


    table_response3d = table_link3('output/websites/', site, variable_list_4d)
    page_body_right += add_figure_link('Response Analysis(Four variables)', 'output/'+'response_3d'+'/'+site + '3d_Response' + variable_list_4d[0][0] + variable_list_4d[0][1] + variable_list_4d[0][2] + variable_list_4d[0][3]+'Observed' + '.png', 'time_series', './'+'response3d'+site+'.html')
    page_body_right += table_response3d

    table_responseco = table_link4('output/websites/', site, variable_list_4d2)
    page_body_right += add_figure_link('Response Analysis(Partial correlation)', 'output/'+'response_3d'+'/'+ site + variable_list_4d2[0][0] +variable_list_4d2[0][1] + variable_list_4d2[0][2] + variable_list_4d2[0][3] +'3d_Response_corr' + 'Observed' + '.png', 'time_series', './'+'partialcorr'+site+'.html')
    page_body_right += table_responseco


    table_responserr = table_link5('output/websites/', site)
    page_body_right += add_figure_link('Response Analysis(Variable correlations)', 'output/'+'response_cc'+'/'+ site + '3d_Response_corr_matrix_hourly' + 'Observed' + '.png', 'time_series', './' +'variablecorr' + site + '.html')
    page_body_right += table_responserr

    # page_body_right += add_figure_link('Response Bin'+table_response, 'output/'+'response'+'/'+site+'_'+variable1[0]+'_vs_'+variable2[0]+'_Response_Bin.png', 'time_series', './'+'response'+site+variable1[0]+'vs'+variable2[0]+'.html')
    contents = page_front + page_body_left + page_body_right + page_tail
    browseLocal(contents, 'output/websites/', site+'local.html')