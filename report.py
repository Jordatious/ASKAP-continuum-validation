import os
import collections
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from functions import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpld3
from matplotlib import colors
from mpld3 import plugins
from matplotlib.patches import Ellipse
import matplotlib.image as image
from astropy.io import fits as f
from astropy.io import votable
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from inspect import currentframe, getframeinfo

# ignore annoying astropy warnings and set my own obvious warning output
warnings.simplefilter('ignore', category=AstropyWarning)
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename


class report(object):

    def __init__(self, cat, main_dir, img=None, plot_to='html', css_style=None,
                 fig_font={'fontname': 'Serif', 'fontsize': 18}, fig_size={'figsize': (8, 8)},
                 label_size={'labelsize': 12}, markers={'s': 20, 'linewidth': 1,
                                                        'marker': 'o', 'color': 'b'},
                 colour_markers={'marker': 'o', 's': 30, 'linewidth': 0}, cmap='plasma',
                 cbins=20, arrows={'color': 'r', 'width': 0.04, 'scale': 20},
                 src_cnt_bins=50, redo=False, write=True, verbose=True):

        """Initialise a report object for writing a html report of the image and cross-matches,
        including plots.

        Arguments:
        ----------
        cat : catalogue
            Catalogue object with the data for plotting.
        main_dir : string
            Main directory that contains all the necessary files.

        Keyword arguments:
        ------------------
        img : radio_image
            Radio image object used to write report table. If None, report will not be written,
            but plots will be made.
        plot_to : string
            Where to show or write the plot. Options are:

                'html' - save as a html file using mpld3.

                'screen' - plot to screen [i.e. call plt.show()].

                'extn' - write file with this extension (e.g. 'pdf', 'eps', 'png', etc).

        css_style : string
            A css format to be inserted in <head>.
        fig_font : dict
            Dictionary of kwargs for font name and size for title and axis labels of matplotlib
            figure.
        fig_size : dict
            Dictionary of kwargs to pass into pyplot.figure.
        label_size : dict
            Dictionary of kwargs for tick params.
        markers : dict
            Dictionary of kwargs to pass into pyplot.figure.scatter, etc (when single colour used).
        colour_markers : dict
            Dictionary of kwargs to pass into pyplot.figure.scatter, etc (when colourmap used).
        arrows : dict
            Dictionary of kwargs to pass into pyplot.figure.quiver.
        redo: bool
            Produce all plots and save them, even if the files already exist.
        write : bool
            Write the source counts and figures to file. Input False to only write report.
        verbose : bool
            Verbose output.

        See Also:
        ---------
        matplotlib.pyplot
        mpld3"""

        self.cat = cat
        self.img = img
        self.plot_to = plot_to
        self.fig_font = fig_font
        self.fig_size = fig_size
        self.label_size = label_size
        self.markers = markers
        self.colour_markers = colour_markers
        self.arrows = arrows
        self.cmap = plt.get_cmap(cmap, cbins)
        self.src_cnt_bins = src_cnt_bins
        self.main_dir = main_dir
        self.redo = redo
        self.write = write
        self.verbose = verbose

        # set name of directory for figures and create if doesn't exist
        self.figDir = 'figures'
        if self.write and not os.path.exists(self.figDir):
            os.mkdir(self.figDir)

        # use css style passed in or default style for CASS web server below
        if css_style is not None:
            self.css_style = css_style
        else:
            self.css_style = """<?php include("base.inc"); ?>
            <meta name="DCTERMS.Creator" lang="en" content="personalName=Collier,Jordan" />
            <meta name="DC.Title" lang="en" content="Continuum Validation Report" />
            <meta name="DC.Description" lang="en" content="Continuum validation report
            summarising science readiness of data via several metrics" />
            <?php standard_head(); ?>
            <style>
                .reportTable {
                    border-collapse: collapse;
                    width: 100%;
                }

                .reportTable th, .reportTable td {
                    padding: 15px;
                    text-align: middle;
                    border-bottom: 1px solid #ddd;
                    vertical-align: top;
                }

                .reportTable tr {
                    text-align:center;
                    vertical-align:middle;
                }

                .reportTable tr:hover{background-color:#f5f5f5}

                #good {
                    background-color:#00FA9A;
                }

                #uncertain {
                    background-color:#FFA500;
                }

                #bad {
                    background-color:#FF6347;
                }

            </style>\n"""
            self.css_style += "<title>{0} Continuum Validation Report</title>\n""".format(
                self.cat.name)

        # filename of html report
        self.name = 'index.html'
        # Open file html file and write css style, title and heading
        self.write_html_head()
        # write table summary of observations and image if radio_image object passed in
        if img is not None:
            self.write_html_img_table(img)
            rms_map = f.open(img.rms_map)[0]
            solid_ang = 0
        # otherwise assume area based on catalogue RA/DEC limits
        else:
            rms_map = None
            solid_ang = self.cat.area*(np.pi/180)**2

        self.write_html_cat_table()

        # plot the int/peak flux as a function of peak flux
        self.int_peak_flux(usePeak=True)

        # write source counts to report using rms map to measure solid angle or approximate
        # solid angle
        if self.cat.name in list(self.cat.flux.keys()):
            self.source_counts(self.cat.flux[self.cat.name], self.cat.freq[self.cat.name],
                               rms_map=rms_map, solid_ang=solid_ang, write=self.write)
        else:
            self.sc_red_chi_sq = -1
        # write cross-match table header
        self.write_html_cross_match_table()

        # store dictionary of metrics, where they come from, how many matches they're
        # derived from, and their level (0,1 or 2) spectral index defaults to -99, as
        # there is a likelihood it will not be needed (if Taylor-term imaging is not done)
        # RA and DEC offsets used temporarily and then dropped before final metrics computed
        key_value_pairs = [('Flux Ratio', 0),
                           ('Flux Ratio Uncertainty', 0),
                           ('Positional Offset', 0),
                           ('Positional Offset Uncertainty', 0),
                           ('Resolved Fraction', self.cat.resolved_frac),
                           ('Spectral Index', 0),
                           ('RMS', self.cat.img_rms),
                           ('Source Counts Reduced Chi-squared', self.sc_red_chi_sq),
                           ('RA Offset', 0),
                           ('DEC Offset', 0)]

        self.metric_val = collections.OrderedDict(key_value_pairs)
        self.metric_source = self.metric_val.copy()
        self.metric_count = self.metric_val.copy()
        self.metric_level = self.metric_val.copy()
        
    def write_html_head(self):

        """Open the report html file and write the head."""

        self.html = open(self.name, 'w')
        self.html.write("""<!DOCTYPE HTML>
        <html lang="en">
        <head>
            {0}
        </head>
        <?php title_bar("atnf"); ?>
        <body>
            <h1 align="middle">{1} Continuum Data Validation Report</h1>""".format(self.css_style,
                                                                                   self.cat.name))
   
    def write_html_img_table(self, img):

        """Write an observations and image and catalogue report tables derived from fits image and header.

        Arguments:
        ----------
        img : radio_image
            A radio image object used to write values to the html table."""

        # generate link to confluence page for each project code
        project = img.project
        if project.startswith('AS'):
            project = self.add_html_link("https://confluence.csiro.au/display/askapsst/{0}+"
                                         "Data".format(img.project), img.project, file=False)

        # Write observations report table
        self.html.write("""
        <h2 align="middle">Observations</h2>
        <table class="reportTable">
            <tr>
                <th>SBID</th>
                <th>Project</th>
                <th>Date</th>
                <th>Duration<br>(hours)</th>
                <th>Field Centre</th>
                <th>Central Frequency<br>(MHz)</th>
            </tr>
            <tr>
                    <td>{0}</td>
                    <td>{1}</td>
                    <td>{2}</td>
                    <td>{3}</td>
                    <td>{4}</td>
                    <td>{5:.2f}</td>
                    </tr>
        </table>""".format(img.sbid,
                           project,
                           img.date,
                           img.duration,
                           img.centre,
                           img.freq))

        # Write image report table
        self.html.write("""
        <h2 align="middle">Image</h2>
        <h4 align="middle"><i>File: '{0}'</i></h3>
        <table class="reportTable">
            <tr>
                <th>ASKAPsoft<br>version</th>
                <th>Pipeline<br>version</th>
                <th>Synthesised Beam<br>(arcsec)</th>
                <th>Median r.m.s.<br>(uJy)</th>
                <th>Image peak<br>(Jy)</th>
                <th>Dynamic Range</th>
                <th>Sky Area<br>(deg<sup>2</sup>)</th>
            </tr>
            <tr>
                <td>{1}</td>
                <td>{2}</td>
                <td>{3:.1f} x {4:.1f}</td>
                <td>{5}</td>
                <td>{6:.2f}</td>
                <td>{7:.0E}</td>
                <td>{8:.2f}</td>
            </tr>
        </table>""".format(img.name,
                           img.soft_version,
                           img.pipeline_version,
                           img.bmaj,
                           img.bmin,
                           self.cat.img_rms,
                           self.cat.img_peak,
                           self.cat.dynamic_range,
                           self.cat.area))

    def write_html_cat_table(self):

        """Write an observations and image and catalogue report tables derived from fits
        image, header and catalogue."""

        flux_type = 'integrated'
        if self.cat.use_peak:
            flux_type = 'peak'
        if self.cat.med_si == -99:
            med_si = ''
        else:
            med_si = '{0:.2f}'.format(self.cat.med_si)

        # Write catalogue report table
        self.html.write("""
        <h2 align="middle">Catalogue</h2>
        <h4 align="middle"><i>File: '{0}'</i></h3>
        <table class="reportTable">
            <tr>
                <th>Source Finder</th>
                <th>Flux Type</th>
                <th>Number of<br>sources (&ge;{1}&sigma;)</th>
                <th>Multi-component<br>islands</th>
                <th>Sum of image flux vs.<br>sum of catalogue flux</th>
                <th>Median in-band spectral index</th>
                <th>Median int/peak flux</th>
                <th>Source Counts<br>&#967;<sub>red</sub><sup>2</sup></th>
            </tr>
            <tr>
                <td>{2}</td>
                <td>{3}</td>
                <td>{4}</td>
                <td>{5}</td>
                <td>{6:.1f} Jy vs. {7:.1f} Jy</td>
                <td>{8}</td>""".format(self.cat.filename,
                                       self.cat.SNR,
                                       self.cat.finder,
                                       flux_type,
                                       self.cat.initial_count,
                                       self.cat.blends,
                                       self.cat.img_flux,
                                       self.cat.cat_flux,
                                       med_si))

    def write_html_cross_match_table(self):

        """Write the header of the cross-matches table."""

        self.html.write("""
        <h2 align="middle">Cross-matches</h2>
        <table class="reportTable">
            <tr>
                <th>Survey</th>
                <th>Frequency<br>(MHz)</th>
                <th>Cross-matches</th>
                <th>Median offset<br>(arcsec)</th>
                <th>Median flux ratio</th>
                <th>Median spectral index</th> 
                </tr>""")
    
    def get_metric_level(self, good_condition, uncertain_condition):

        """Return metric level 1 (good), 2 (uncertain) or 3 (bad), according to the two input conditions.

        Arguments:
        ----------
        good_condition : bool
            Condition for metric being good.
        uncertain_condition : bool
            Condition for metric being uncertain."""

        if good_condition:
            return 1
        if uncertain_condition:
            return 2
        return 3

    def assign_metric_levels(self):

        """Assign level 1 (good), 2 (uncertain) or 3 (bad) to each metric, depending on specific tolerenace
        values. See https://confluence.csiro.au/display/askapsst/Continuum+validation+metrics"""

        for metric in list(self.metric_val.keys()):
            # Remove keys that don't have a valid value (value=-99 or -1111)
            if self.metric_val[metric] == -99 or self.metric_val[metric] == -111:
                self.metric_val.pop(metric)
                self.metric_source.pop(metric)
                self.metric_level.pop(metric)
            else:
                # flux ratio within 5/10%?
                if metric == 'Flux Ratio':
                    val = np.abs(self.metric_val[metric]-1)
                    good_condition = val < 0.05
                    uncertain_condition = val < 0.1
                    self.metric_source[metric] = 'Median flux density ratio [ASKAP / {0}]'.format(
                        self.metric_source[metric])
                # uncertainty on flux ratio less than 10/20%?
                elif metric == 'Flux Ratio Uncertainty':
                    good_condition = self.metric_val[metric] < 0.1
                    uncertain_condition = self.metric_val[metric] < 0.2
                    self.metric_source[metric] = 'R.M.S. of median flux density ratio '
                    '[ASKAP / {0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation '
                    'from median)'
                # positional offset < 1/5 arcsec
                elif metric == 'Positional Offset':
                    good_condition = self.metric_val[metric] < 1
                    uncertain_condition = self.metric_val[metric] < 5
                    self.metric_source[metric] = 'Median positional offset (arcsec) '
                    '[ASKAP-{0}]'.format(self.metric_source[metric])
                # uncertainty on positional offset < 1/5 arcsec
                elif metric == 'Positional Offset Uncertainty':
                    good_condition = self.metric_val[metric] < 5
                    uncertain_condition = self.metric_val[metric] < 10
                    self.metric_source[metric] = 'R.M.S. of median positional offset (arcsec) '
                    '[ASKAP-{0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation '
                    'from median)'
                # reduced chi-squared of source counts < 3/50?
                elif metric == 'Source Counts Reduced Chi-squared':
                    good_condition = self.metric_val[metric] < 3
                    uncertain_condition = self.metric_val[metric] < 50
                    self.metric_source[metric] = 'Reduced chi-squared of source counts'
                # resolved fraction of sources between 5-20%?
                elif metric == 'Resolved Fraction':
                    cond1 = self.metric_val[metric] > 0.05
                    cond2 = self.metric_val[metric] < 0.2
                    good_condition = cond1 and cond2
                    uncertain_condition = self.metric_val[metric] < 0.3
                    self.metric_source[metric] = 'Fraction of sources resolved according to ' 
                    'int/peak flux densities'
                # spectral index less than 0.2 away from -0.8?
                elif metric == 'Spectral Index':
                    val = np.abs(self.metric_val[metric]+0.8)
                    good_condition = val < 0.2
                    uncertain_condition = False
                    self.metric_source[metric] = 'Median in-band spectral index'
                elif metric == 'RMS':
                    good_condition = self.metric_val[metric] < 100
                    uncertain_condition = self.metric_val[metric] < 500
                    self.metric_source[metric] = 'Median image R.M.S. (uJy) from noise map'
                # if unknown metric, set it to 3 (bad)
                else:
                    good_condition = False
                    uncertain_condition = False

                # assign level to metric
                self.metric_level[metric] = self.get_metric_level(good_condition, 
                                                                  uncertain_condition)

        if self.img is not None:
            self.write_CASDA_xml()

    def write_pipeline_offset_params(self):

        """Write a txt file with offset params for ASKAPsoft pipeline for user to easily
        import into config file, and then drop them from metrics.
        See http://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/pipelines
        /ScienceFieldContinuumImaging.html?highlight=offset"""

        txt = open('offset_pipeline_params.txt', 'w')
        txt.write("DO_POSITION_OFFSET=true\n")
        txt.write("RA_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['RA Offset']))
        txt.write("DEC_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['DEC Offset']))
        txt.close()

        for metric in ['RA Offset', 'DEC Offset']:
            self.metric_val.pop(metric)
            self.metric_source.pop(metric)
            self.metric_level.pop(metric)
            self.metric_count.pop(metric)

    def write_CASDA_xml(self):

        """Write xml table with all metrics for CASDA."""

        tmp_table = Table([list(self.metric_val.keys()), list(self.metric_val.values()),
                           list(self.metric_level.values()), list(self.metric_source.values())],
                          names=['metric_name', 'metric_value', 'metric_status',
                                 'metric_description'],
                          dtype=[str, float, np.int32, str])
        vot = votable.from_table(tmp_table)
        vot.version = 1.3
        table = vot.get_first_table()
        table.params.extend([votable.tree.Param(vot, name="project", datatype="char",
                                                arraysize="*", value=self.img.project)])
        valuefield = table.fields[1]
        valuefield.precision = '2'
        prefix = ''
        if self.img.project != '':
            prefix = '{0}_'.format(self.img.project)
        xml_filename = '{0}CASDA_continuum_validation.xml'.format(prefix)
        votable.writeto(vot, xml_filename)

    def write_html_end(self):

        """Write the end of the html report file (including table of metrics) and close it."""

        # Close cross-matches table and write header of validation summary table
        self.html.write("""
                </td>
            </tr>
        </table>
        <h2 align="middle">{0} continuum validation metrics</h2>
        <table class="reportTable">
            <tr>
                <th>Flux Ratio<br>({0} / {1})</th>
                <th>Flux Ratio Uncertainty<br>({0} / {1})</th>
                <th>Positional Offset (arcsec)<br>({0} &mdash; {2})</th>
                <th>Positional Offset Uncertainty (arcsec)<br>({0} &mdash; {2})</th>
                <th>Resolved Fraction from int/peak Flux<br>({0})</th>
                <th>Source Counts &#967;<sub>red</sub><sup>2</sup><br>({0})</th>
                <th>r.m.s. (uJy)<br>({0})</th>
            """.format(self.cat.name, self.metric_source['Flux Ratio'],
                       self.metric_source['Positional Offset']))

        # assign levels to each metric
        self.assign_metric_levels()

        # flag if in-band spectral indices not derived
        spec_index = False
        if 'Spectral Index' in self.metric_val:
            spec_index = True

        if spec_index:
            self.html.write('<th>Median in-band<br>spectral index</th>')

        # Write table with values of metrics and colour them according to level
        self.html.write("""</tr>
        <tr>
            <td {0}>{1:.2f}</td>
            <td {2}>{3:.2f}</td>
            <td {4}>{5:.2f}</td>
            <td {6}>{7:.2f}</td>
            <td {8}>{9:.2f}</td>
            <td {10}>{11:.2f}</td>
            <td {12}>{13}</td>
        """.format(self.html_colour(self.metric_level['Flux Ratio']), self.metric_val['Flux Ratio'],
                        self.html_colour(self.metric_level['Flux Ratio Uncertainty']), 
                   self.metric_val['Flux Ratio Uncertainty'],
                        self.html_colour(self.metric_level['Positional Offset']), 
                   self.metric_val['Positional Offset'],
                        self.html_colour(self.metric_level['Positional Offset Uncertainty']), 
                   self.metric_val['Positional Offset Uncertainty'],
                        self.html_colour(self.metric_level['Resolved Fraction']),
                   self.metric_val['Resolved Fraction'],
                        self.html_colour(self.metric_level['Source Counts Reduced Chi-squared']), 
                   self.metric_val['Source Counts Reduced Chi-squared'],
                        self.html_colour(self.metric_level['RMS']), self.metric_val['RMS']))

        if spec_index:
            self.html.write('<td {0}>{1:.2f}</td>'.format(self.html_colour(
                self.metric_level['Spectral Index']), self.metric_val['Spectral Index']))

        by = ''
        if self.cat.name != 'ASKAP':
            by = """ by <a href="mailto:Jordan.Collier@csiro.au">Jordan Collier</a>"""
        # Close table, write time generated, and close html file
        self.html.write("""</tr>
            </table>
                <p><i>Generated at {0}{1}</i></p>
            <?php footer(); ?>
            </body>
        </html>""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), by))
        self.html.close()
        print("Continuum validation report written to '{0}'.".format(self.name))
        
    def add_html_link(self, target, link, file=True, newline=False):

        """Return the html for a link to a URL or file.

        Arguments:
        ----------
        target : string
            The name of the target (a file or URL).
        link : string
            The link to this file (thumbnail file name or string to list as link name).

        Keyword Arguments:
        ------------------
        file : bool
            The input link is a file (e.g. a thumbnail).
        newline : bool
            Write a newline / html break after the link.

        Returns:
        --------
        html : string
            The html link."""

        html = """<a href="{0}">""".format(target)
        if file:
            html += """<IMG SRC="{0}"></a>""".format(link)
        else:
            html += "{0}</a>".format(link)
        if newline:
            html += "<br>"
        return html
    
    
    def text_to_html(self, text):

        """Take a string of text that may include LaTeX, and return the html
        code that will generate it as LaTeX.

        Arguments:
        ----------
        text : string
            A string of text that may include LaTeX.

        Returns:
        --------
        html : string
            The same text readable as html."""

        # this will allow everything between $$ to be generated as LaTeX
        html = """
                    <script type="text/x-mathjax-config">
                      MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
                    </script>
                    <script type="text/javascript"
                      src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                    </script>
                    <br>

                    """

        # write a newline / break for each '\n' in string
        for line in text.split('\n'):
            html += line + '<br>'

        return html

    def html_colour(self, level):

        """Return a string representing green, yellow or red in html if level is 1, 2 or 3.

        Arguments:
        ----------
        level : int
            A validation level.

        Returns:
        --------
        colour : string
            The html for green, yellow or red."""

        if level == 1:
            colour = "id='good'"
        elif level == 2:
            colour = "id='uncertain'"
        else:
            colour = "id='bad'"
        return colour

    def int_peak_flux(self, usePeak=False):

        """Plot the int/peak fluxes as a function of peak flux.

        Keyword Arguments:
        ------------------
        usePeak : bool
            Use peak flux as x axis, instead of SNR."""

        ratioCol = '{0}_int_peak_ratio'.format(self.cat.name)
        self.cat.df[ratioCol] = self.cat.df[self.cat.flux_col] / self.cat.df[self.cat.peak_col]
        SNR = self.cat.df[self.cat.peak_col]/self.cat.df[self.cat.rms_val]
        ratio = self.cat.df[ratioCol]
        peak = self.cat.df[self.cat.peak_col]

        xaxis = SNR
        if usePeak:
            xaxis = peak

        # plot the int/peak flux ratio
        fig = plt.figure(**self.fig_size)
        title = "{0} int/peak flux ratio".format(self.cat.name)

        if self.plot_to == 'html':
            if usePeak:
                xlabel = 'Peak flux ({0})'.format(self.cat.flux_unit.replace('j', 'J'))
            else:
                xlabel = 'S/N'
            ylabel = 'Int / Peak Flux Ratio'
        else:
            xlabel = r'{\rm S_{peak}'
            if usePeak:
                xlabel += ' ({0})'.format(self.cat.flux_unit.replace('j', 'J'))
            else:
                xlabel += r'$\sigma_{rms}}$'
            ylabel = r'${\rm S_{int} / S_{peak}}$'

        if self.plot_to != 'screen':
            filename = '{0}/{1}_int_peak_ratio.{2}'.format(self.figDir, self.cat.name, 
                                                           self.plot_to)
        else:
            filename = ''

        # get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(xaxis, yaxis=ratio)
        plt.loglog()
        plt.gca().grid(b=True, which='minor', color='w', linewidth=0.5)

        # hack to overlay resolved sources in red
        xres, yres = xaxis[self.cat.resolved], ratio[self.cat.resolved]
        markers = self.markers.copy()
        markers['color'] = 'r'
        markers.pop('s')
        data, = plt.plot(xres, yres, 'o', zorder=50, **markers)
        leg_labels = ['Resolved', 'Unresolved']

        # derive the statistics of y and store in string
        ymed, ymean, ystd, yerr, ymad = get_stats(ratio)
        txt = r'$\widetilde{Ratio}$: %.2f\n' % ymed
        txt += r'$\overline{Ratio}$: %.2f\n' % ymean
        txt += r'$\sigma_{Ratio}$: %.2f\n' % ystd
        txt += r'$\sigma_{\overline{Ratio}}$: %.2f' % yerr

        # store median int/peak flux ratio and write to report table
        self.int_peak_ratio = ymed
        self.html.write('<td>{0:.2f}<br>'.format(ymed))

        # plot the int/peak flux ratio
        self.plot(x,
                  y=y,
                  c=c,
                  figure=fig,
                  line_funcs=[self.y1],
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  text=txt,
                  loc='tl',
                  axis_perc=0,
                  filename=filename,
                  leg_labels=leg_labels,
                  handles=[data],
                  redo=self.redo)


    def source_counts(self, fluxes, freq, rms_map=None, solid_ang=0, write=True):

        """Compute and plot the (differential euclidean) source counts based on the input flux densities.

        Arguments:
        ----------
        fluxes : list-like
            A list of fluxes in Jy.
        freq : float
            The frequency of these fluxes in MHz.

        Keyword arguments:
        ------------------
        rms_map : astropy.io.fits
            A fits image of the local rms in Jy.
        solid_ang : float
            A fixed solid angle over which the source counts are computed. Only used when
            rms_map is None.
        write : bool
            Write the source counts to file."""

        # derive file names based on user input
        filename = 'screen'
        counts_file = '{0}_source_counts.csv'.format(self.cat.basename)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_source_counts.{2}'.format(self.figDir, self.cat.name, self.plot_to)
        # read the log of the source counts from Norris+11 from same directory of this script
        df_Norris = pd.read_table('{0}/all_counts.txt'.format(self.main_dir), sep=' ')
        x = df_Norris['S']-3  # convert from log of flux in mJy to log of flux in Jy
        y = df_Norris['Counts']
        yerr = (df_Norris['ErrDown'], df_Norris['ErrUp'])

        # fit 6th degree polynomial to Norris+11 data
        deg = 6
        poly_paras = np.polyfit(x, y, deg)
        f = np.poly1d(poly_paras)
        xlin = np.linspace(min(x)*1.2, max(x)*1.2)
        ylin = f(xlin)

        # perform source counts if not already written to file or user specifies to re-do
        if not os.path.exists(counts_file) or self.redo:

            # warn user if they haven't input an rms map or fixed solid angle
            if rms_map is None and solid_ang == 0:
                warnings.warn_explicit("You must input a fixed solid angle or an rms map to"
                                       "compute the source counts!\n", UserWarning, WARN,
                                       cf.f_lineno)
                return

            # get the number of bins from the user
            nbins = self.src_cnt_bins
            print("Deriving source counts for {0} using {1} bins.".format(self.cat.name, nbins))

            # Normalise the fluxes to 1.4 GHz
            fluxes = flux_at_freq(1400, freq, fluxes, -0.8)

            # Correct for Eddington bias for every flux, assuming Hogg+98 model
            r = self.cat.df[self.cat.flux_col] / self.cat.df[self.cat.rms_val]
            slope = np.polyder(f)
            q = 1.5 - slope(fluxes)
            bias = 0.5 + 0.5*np.sqrt(1 - (4*q+4)/(r**2))

            # q is derived in log space, so correct for the bias in log space
            fluxes = 10**(np.log10(fluxes)/bias)

            if rms_map is not None:
                w = WCS(rms_map.header)
                if self.verbose:
                    print("Using rms map '{0}' to derive solid angle for each flux bin.".format(
                        self.img.rms_map))
                total_area = get_pixel_area(rms_map, flux=100, w=w)[0]
            else:
                total_area = 0

            # add one more bin and then discard it, since this is dominated by the
            # few brightest sources
            # we also add one more to the bins since there's one more bin edge
            # than number of bins
            edges = np.percentile(fluxes, np.linspace(0, 100, nbins+2))
            dN, edges, patches = plt.hist(fluxes, bins=edges)
            dN = dN[:-1]
            edges = edges[:-1]

            # derive the lower and upper edges and dS
            lower = edges[:-1]
            upper = edges[1:]
            dS = upper-lower
            S = np.zeros(len(dN))
            solid_angs = np.zeros(len(dN))

            for i in range(len(dN)):
                # derive the mean flux from all fluxes in current bin
                indices = (fluxes > lower[i]) & (fluxes < upper[i])
                S[i] = np.mean(fluxes[indices])

                # Get the pixels from the r.m.s. map where SNR*r.m.s. < flux
                if rms_map is not None:
                    solid_angs[i] = get_pixel_area(rms_map, flux=S[i]/self.cat.SNR, w=w)[1]

                # otherwise use the fixed value passed in
                else:
                    solid_angs[i] = solid_ang

            # compute the differential Euclidean source counts and uncertanties in linear space
            counts = (S**2.5)*dN/dS/solid_angs
            err = (S**2.5)*np.sqrt(dN)/dS/solid_angs

            # Store these and the log of these values in pandas data frame
            df = pd.DataFrame()
            df['dN'] = dN
            df['area'] = solid_angs/((np.pi/180)**2)
            df['S'] = S
            df['logS'] = np.log10(S)
            df['logCounts'] = np.log10(counts)
            df['logErrUp'] = np.log10(counts+err) - np.log10(counts)
            df['logErrDown'] = np.abs(np.log10(counts-err) - np.log10(counts))

            # remove all bins with less than 10% of total solid angle
            bad_bins = df['area'] / total_area < 0.1
            output = ['Solid angle for bin S={0:.2f} mJy less than 10% of total image. '
                      'Removing bin.'.format(S) for S in S[np.where(bad_bins)]*1e3]
            if self.verbose:
                for line in output:
                    print(line)
            df = df[~bad_bins]

            if write:
                if self.verbose:
                    print("Writing source counts to '{0}'.".format(counts_file))
                df.to_csv(counts_file, index=False)

        # otherwise simply read in source counts from file
        else:
            print("File '{0}' already exists. Reading source counts from this file.".format(
                counts_file))
            df = pd.read_csv(counts_file)

        # create a figure for the source counts
        plt.close()
        fig = plt.figure(**self.fig_size)
        title = '{0} 1.4 GHz source counts'.format(self.cat.name, self.cat.freq[self.cat.name])
        # write axes using unicode (for html) or LaTeX
        if self.plot_to == 'html':
            ylabel = "log\u2081\u2080 S\u00B2\u22C5\u2075 dN/dS [Jy\u00B9\u22C5\u2075 "
            "sr\u207B\u00B9]"
            xlabel = "log\u2081\u2080 S [Jy]"
        else:
            ylabel = r"$\log_{10}$ S$^{2.5}$ dN/dS [Jy$^{1.5}$ sr$^{-1}$]"
            xlabel = r"$\log_{10}$ S [Jy]"

        # for html plots, add labels for the bin centre, count and area for every data point
        labels = ['S: {0:.2f} mJy, dN: {1:.0f}, Area: {2:.2f} deg\u00B2'.format(bin, count, area)
                  for bin, count, area in zip(df['S']*1e3, df['dN'], df['area'])]

        # derive the square of the residuals (chi squared), and their sum
        # divided by the number of data points (reduced chi squared)
        chi = ((df['logCounts']-f(df['logS']))/df['logErrDown'])**2
        red_chi_sq = np.sum(chi)/len(df)

        # store reduced chi squared value
        self.sc_red_chi_sq = red_chi_sq

        # Plot Norris+11 data
        data = plt.errorbar(x, y, yerr=yerr, linestyle='none', marker='.', c='r')
        line, = plt.plot(xlin, ylin, c='black', linestyle='--', zorder=5)
        txt = ''
        if self.plot_to == 'html':
            txt += 'Data from <a href="http://adsabs.harvard.edu/abs/2011PASA...28..215N">'
            'Norris+11</a>'
            txt += ' (updated from <a href="http://adsabs.harvard.edu/abs/2003AJ....125..465H"'
            '>Hopkins+03</a>)\n'
        txt += r'$\chi^2_{red}$: %.2f' % red_chi_sq

        # Legend labels for the Norris data and line, and the ASKAP data
        xlab = 'Norris+11'
        leg_labels = [xlab, '{0}th degree polynomial fit to {1}'.format(deg, xlab), self.cat.name]

        # write reduced chi squared to report table
        self.html.write('</td><td>{0:.2f}<br>'.format(red_chi_sq))

        # Plot ASKAP data on top of Norris+11 data
        self.plot(df['logS'],
                  y=df['logCounts'],
                  yerr=(df['logErrDown'], df['logErrUp']),
                  figure=fig,
                  title=title,
                  labels=labels,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  axis_perc=0,
                  text=txt,
                  loc='br',
                  leg_labels=leg_labels,
                  handles=[data, line],
                  filename=filename,
                  redo=self.redo)

        self.html.write("""</td>
                        </tr>
                    </table>""")

    def x(self, x, y):

        """For given x and y data, return a line at y=x.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            The list of x values."""

        return x, x

    def y0(self, x, y):

        """For given x and y data, return a line at y=0.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            A list of zeros."""

        return x, x*0

    def y1(self, x, y):

        """For given x and y data, return a line at y=1.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            The same list of x values.
        y : list-like
            A list of ones."""

        return x, [1]*len(x)

    def x0(self, x, y):

        """For given x and y data, return a line at x=0.

        Arguments:
        ----------
        x : list-like
            A list of x values.
        y : list-like
            A list of y values.

        Returns:
        --------
        x : list-like
            A list of zeros.
        y : list-like
            The same list of y values."""

        return y*0, y

    def ratio_err_max(self, SNR, ratio):

        """For given x and y data (flux ratio as a function of S/N), return
        the maximum uncertainty in flux ratio.

        Arguments:
        ----------
        SNR : list-like
            A list of S/N ratios.
        ratio : list-like
            A list of flux ratios.

        Returns:
        --------
        SNR : list-like
            All S/N values > 0.
        ratio : list-like
            The maximum uncertainty in the flux ratio for S/N values > 0."""

        return SNR[SNR > 0], 1+3*np.sqrt(2)/SNR[SNR > 0]

    def ratio_err_min(self, SNR, ratio):

        """For given x and y data (flux ratio as a function of S/N), return the
        minimum uncertainty in flux ratio.

        Arguments:
        ----------
        SNR : list-like
            A list of S/N ratios.
        ratio : list-like
            A list of flux ratios.

        Returns:
        --------
        SNR : list-like
            All S/N values > 0.
        ratio : list-like
            The minimum uncertainty in the flux ratio for S/N values > 0."""

        return SNR[SNR > 0], 1-3*np.sqrt(2)/SNR[SNR > 0]
    
    def axis_to_np(self, axis):

        """Return a numpy array of the non-nan data from the input axis.

        Arguments:
        ----------
        axis : string or numpy.array or pandas.Series or list
            The data for a certain axis. String are interpreted as column names from catalogue
            object passed into constructor.

        Returns:
        --------
        axis : numpy.array
            All non-nan values of the data.

        See Also
        --------
        numpy.array
        pandas.Series"""

        # convert input to numpy array
        if type(axis) is str:
            axis = self.cat.df[axis].values
        elif axis is pd.Series:
            axis = axis.values

        return axis

    def shared_indices(self, xaxis, yaxis=None, caxis=None):

        """Return a list of non-nan indices shared between all used axes.

        Arguments:
        ----------
        xaxis : string or numpy.array or pandas.Series or list
            A list of the x axis data. String are interpreted as column names from catalogue
            object passed into constructor.
        yaxis : string or numpy.array or pandas.Series or list
            A list of the y axis data. String are interpreted as column names from catalogue object
            passed into constructor. If this is None, yaxis and caxis will be ignored.
        caxis : string or numpy.array or pandas.Series or list
            A list of the colour axis data. String are interpreted as column names from catalogue
            object passed into constructor. If this is None, caxis will be ignored.

        Returns:
        --------
        x : list
            The non-nan x data shared between all used axes.
        y : list
            The non-nan y data shared between all used axes. None returned if yaxis is None.
        c : list
            The non-nan colour data shared between all used axes. None returned if yaxis or
            caxis are None.
        indices : list
            The non-nan indices.

        See Also
        --------
        numpy.array
        pandas.Series"""

        # convert each axis to numpy array (or leave as None)
        x = self.axis_to_np(xaxis)
        y = self.axis_to_np(yaxis)
        c = self.axis_to_np(caxis)

        # get all shared indices from used axes that aren't nan
        if yaxis is None:
            indices = np.where(~np.isnan(x))[0]
            return x[indices], None, None, indices
        elif caxis is None:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
            return x[indices], y[indices], None, indices
        else:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(c)))[0]
            return x[indices], y[indices], c[indices], indices
    
    def plot(self, x, y=None, c=None, yerr=None, figure=None, arrows=None, line_funcs=None,
             title='', labels=None, text=None, reverse_x=False, xlabel='', ylabel='',
             clabel='', leg_labels='', handles=[], loc='bl', ellipses=None, axis_perc=10,
             filename='screen', redo=False):

        """Create and write a scatter plot of the data from an input x axis, and optionally,
        a y and colour axis. This function assumes shared_indices() has already been called
        and all input axes are equal in length and the same data type.

        Arguments:
        ----------
        x : numpy.array
            The data to plot on the x axis.

        Keyword arguments:
        ------------------
        y : numpy.array or pandas.Series
            The data to plot on the y axis. Use None to plot a histogram.
        c : numpy.array or pandas.Series
            The data to plot as the colour axis. Use None for no colour axis.
        yerr : numpy.array or pandas.Series
            The data to plot as the uncertainty on the y axis. Use None for no uncertainties.
        figure : pyplot.figure
            Use this matplotlib figure object.
        arrows : tuple
            A 2-element tuple with the lengths of the arrows to plot at x and y
            (usually a list) - i.e. (dx[],dy[])
        line_funcs : list-like
            A list of functions for drawing lines (e.g. [self.x0(), self.y1()]).
        title : string
            The title of the plot.
        lables : list
            A list of string labels to give each data point. Length must be the same as all
            used axes.
        text : string
            Annotate this text on the figure (written to bottom of page for html figures).
        reverse_x : bool
            Reverse the x-axis?
        xlabel : string
            The label of the x axis.
        ylabel : string
            The label of the y axis.
        clabel : string
            The label of the colour axis.
        leg_labels : list
            A list of labels to include as a legend.
        handles : list
            A list of pre-defined handles associated the legend labels.
        loc : string
            Location of the annotated text (not used for html plots). Options are 'bl',
            'br', 'tl' and 'tr'.
        ellipses : list of matplotlib.patches.Ellipse objects
            Draw these ellipses on the figure.
        axis_perc : float
            The percentage beyond which to calculate the axis limits. Use 0 for no limits.
        filename : string
            Write the plot to this file name. If string contains 'html', file will be written
            to html using mpld3.
            If it is 'screen', it will be shown on screen. Otherwise, it will attempt to
            write an image file.
        redo: bool
            Produce this plot and write it, even if the file already exists.

        See Also
        --------
        numpy.array
        pandas.Series
        matplotlib.patches.Ellipse"""

        # only write figure if user wants it
        if self.write:
            # derive name of thumbnail file
            thumb = '{0}_thumb.png'.format(filename[:-1-len(self.plot_to)])
            # don't produce plot if file exists and user didn't specify to re-do
            if os.path.exists(filename) and not redo:
                if self.verbose:
                    print('File already exists. Skipping plot.')
            else:
                # open html file for plot
                if 'html' in filename:
                    html_fig = open(filename, 'w')
                # use figure passed in or create new one
                if figure is not None:
                    fig = figure
                else:
                    fig = plt.figure(**self.fig_size)

                ax = plt.subplot(111)
                norm = None

                # plot histogram
                if y is None:
                    edges = np.linspace(-3, 2, 11)  # specific to spectral index
                    err_data = ax.hist(x, bins=edges)
                # plot scatter of data points with fixed colour
                elif c is None:
                    markers = self.markers.copy()
                    markers.pop('s')
                    ax.plot(x, y, 'o', zorder=20, alpha=0.0, **markers)
                    data, = ax.plot(x, y, 'o', **markers)
                    handles.append(data)
                # plot scatter of data points with colour axis
                else:
                    # normalise the colour bar so each bin contains equal number of data points
                    norm = colors.BoundaryNorm(np.percentile(c, np.linspace(0, 100,
                                                                            self.cmap.N+1)),
                                               self.cmap.N)
                    data = ax.scatter(x, y, c=c, cmap=self.cmap, norm=norm, **self.colour_markers)
                    cbar = plt.colorbar(data)
                    cbar.ax.tick_params(**self.label_size)
                    cbar.set_label(clabel, **self.fig_font)
                    data = ax.scatter(x, y, c=c, cmap=self.cmap, zorder=20, alpha=0.0, norm=norm,
                                      **self.colour_markers)  # same hack as above
                # plot error bars and add to list of handles
                if yerr is not None:
                    err_data = ax.errorbar(x, y, yerr=yerr, zorder=4, linestyle='none',
                                           marker=self.markers['marker'],
                                           color=self.markers['color'])
                    handles.append(err_data)

                # set default min and max axis limits, which may change
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                # derive limits of x and y axes axis_perc % beyond their current limit
                if axis_perc > 0:
                    xmin = axis_lim(x, min, perc=axis_perc)
                    xmax = axis_lim(x, max, perc=axis_perc)
                    ymin = axis_lim(y, min, perc=axis_perc)
                    ymax = axis_lim(y, max, perc=axis_perc)

                # plot each line according to the input functions
                if line_funcs is not None:
                    xlin = np.linspace(xmin, xmax, num=1000)
                    ylin = np.linspace(ymin, ymax, num=1000)

                    for func in line_funcs:
                        xline, yline = func(xlin, ylin)
                        # line = plt.plot(xline, yline, lw=2, color='black', linestyle='-'
                        # , zorder=12)

                # doing this here forces the lines in html plots to not increase the axis limits
                if reverse_x:
                    plt.xlim(xmax, xmin)
                else:
                    plt.xlim(xmin, xmax)

                plt.ylim(ymin, ymax)

                # overlay the title and labels according to given fonts and sizes
                plt.tick_params(**self.label_size)
                plt.title(title, **self.fig_font)
                plt.xlabel(xlabel, **self.fig_font)
                plt.ylabel(ylabel, **self.fig_font)

                # overlay arrows on each data point
                if arrows is not None:
                    if not (type(arrows) is tuple and len(arrows) == 2):
                        warnings.warn_explicit('Arrows not formatted correctly for plt.quiver(). '
                                               'Input a 2-element tuple.\n', UserWarning, WARN,
                                               cf.f_lineno)
                    elif c is None:
                        plt.quiver(x, y, arrows[0], arrows[1], units='x', **self.arrows)
                    else:
                        plt.quiver(x, y, arrows[0], arrows[1], c, units='x', cmap=self.cmap,
                                   norm=norm, **self.arrows)

                # annotate input text
                if text is not None and 'html' not in filename:
                    # write to given location on plot
                    kwargs = self.fig_font.copy()

                    if loc == 'tl':
                        args = (xmin, ymax, text)
                        kwargs.update({'horizontalalignment': 'left', 'verticalalignment': 'top'})
                    elif loc == 'tr':
                        args = (xmax, ymax, text)
                        kwargs.update({'horizontalalignment': 'right', 'verticalalignment': 'top'})
                    elif loc == 'br':
                        args = (xmax, ymin, text)
                        kwargs.update({'horizontalalignment': 'right', 'verticalalignment':
                                       'bottom'})
                    else:
                        args = (xmin, ymin, text)
                        kwargs.update({'horizontalalignment': 'left', 'verticalalignment':
                                       'bottom'})

                    plt.text(*args, **kwargs)

                # write a legend
                if len(leg_labels) > 0:
                    plt.legend(handles, leg_labels, fontsize=self.fig_font['fontsize']//1.5)

                # overlay ellipses on figure
                if ellipses is not None:
                    for e in ellipses:
                        ax.add_patch(e)

                if self.verbose:
                    print("Writing figure to '{0}'.".format(filename))

                # write thumbnail of this figure
                if filename != 'screen':
                    plt.savefig(thumb)
                    image.thumbnail(thumb, thumb, scale=0.05)

                # write html figure
                if 'html' in filename:
                    # include label for every datapoint
                    if labels is not None:
                        tooltip = plugins.PointHTMLTooltip(data, labels=labels)
                        plugins.connect(fig, tooltip)

                    # print coordinates of mouse as it moves across figure
                    plugins.connect(fig, plugins.MousePosition(fontsize=self.fig_font['fontsize']))
                    html_fig.write(mpld3.fig_to_html(fig))

                    # write annotations to end of html file if user wants html plots
                    if text is not None:
                        html_fig.write(self.text_to_html(text))

                # otherwise show figure on screen
                elif filename == 'screen':
                    plt.show()
                # otherwise write with given extension
                else:
                    plt.savefig(filename)

            # Add link and thumbnail to html report table
            self.html.write(self.add_html_link(filename, thumb))

        plt.close()
        
    def validate(self, name1, name2, redo=False):

        """Produce a validation report between two catalogues, and optionally produce plots.

        Arguments:
        ----------
        name1 : string
            The dictionary key / name of a catalogue from the main catalogue object used to
            compare other data.
        name2 : string
            The dictionary key / name of a catalogue from the main catalogue object used as
            a comparison.

        Keyword Arguments:
        ------------------
        redo: bool
            Produce this plot and write it, even if the file already exists.

        Returns:
        --------
        ratio_med : float
            The median flux density ratio. -1 if this is not derived.
        sep_med : float
            The median sky separation between the two catalogues.
        alpha_med : float
            The median spectral index. -1 if this is not derived."""

        print('Validating {0} with {1}...'.format(name1, name2))

        filename = 'screen'

        # write survey and number of matched to cross-matches report table
        self.html.write("""<tr>
                        <td>{0}</td>
                        <td>{1}</td>
                        <td>{2}""".format(name2, self.cat.freq[name2], self.cat.count[name2]))

        # plot the positional offsets
        fig = plt.figure(**self.fig_size)
        title = "{0} \u2014 {1} positional offsets".format(name1, name2)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry.{3}'.format(self.figDir, name1, name2, self.plot_to)

        # compute the S/N and its log based on main catalogue
        if name1 in list(self.cat.flux.keys()):
            self.cat.df['SNR'] = self.cat.flux[name1] / self.cat.flux_err[name1]
            self.cat.df['logSNR'] = np.log10(self.cat.df['SNR'])
            caxis = 'logSNR'
        else:
            caxis = None

        # get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(self.cat.dRA[name2],
                                               yaxis=self.cat.dDEC[name2], caxis=caxis)

        # derive the statistics of x and y and store in string to annotate on figure
        dRAmed, dRAmean, dRAstd, dRAerr, dRAmad = get_stats(x)
        dDECmed, dDECmean, dDECstd, dDECerr, dDECmad = get_stats(y)
        txt = r'$\widetilde{\Delta RA}$: %.2f\n' % dRAmed
        txt += r'$\overline{\Delta RA}$: %.2f\n' % dRAmean
        txt += r'$\sigma_{\Delta RA}$: %.2f\n' % dRAstd
        txt += r'$\sigma_{\overline{\Delta RA}}$: %.2f\n' % dRAerr
        txt += r'$\widetilde{\Delta DEC}$: %.2f\n' % dDECmed
        txt += r'$\overline{\Delta DEC}$: %.2f\n' % dDECmean
        txt += r'$\sigma_{\Delta DEC}$: %.2f\n' % dDECstd
        txt += r'$\sigma_{\overline{\Delta DEC}}$: %.2f' % dDECerr

        # create an ellipse at the position of the median with axes of standard deviation
        e1 = Ellipse((dRAmed, dDECmed), width=dRAstd, height=dDECstd, color='black', fill=False,
                     linewidth=3, zorder=10, alpha=0.9)

        # force axis limits of the search radius
        radius = max(self.cat.radius[name1], self.cat.radius[name2])
        plt.axis('equal')
        plt.xlim(-radius, radius)
        plt.ylim(-radius, radius)

        # create an ellipse at 0,0 with width 2 x search radius
        e2 = Ellipse((0, 0), width=radius*2, height=radius*2, color='grey', fill=False,
                     linewidth=3, linestyle='--', zorder=1, alpha=0.9)

        # format labels according to destination of figure
        if self.plot_to == 'html':
            xlabel = '\u0394RA (arcsec)'
            ylabel = '\u0394DEC (arcsec)'
            clabel = 'log\u2081\u2080 S/N'
        else:
            xlabel = r'$\Delta$RA (arcsec)'
            ylabel = r'$\Delta$DEC (arcsec)'
            clabel = r'$\log_{10}$ S/N'

        # for html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = ['S/N = {0:.2f}, separation = {1:.2f}\"'.format(cval, totSep) for cval,
                      totSep in zip(self.cat.df.loc[indices, 'SNR'], self.cat.sep[name2][indices])]
        else:
            labels = ['Separation = {0:.2f}\"'.format(cval) for cval in
                      self.cat.sep[name2][indices]]

        # get median separation in arcsec
        c1 = SkyCoord(ra=0, dec=0, unit='arcsec, arcsec')
        c2 = SkyCoord(ra=dRAmed, dec=dDECmed, unit='arcsec, arcsec')
        sep_med = c1.separation(c2).arcsec

        # get mad of separation in arcsec
        c1 = SkyCoord(ra=0, dec=0, unit='arcsec, arcsec')
        c2 = SkyCoord(ra=dRAmad, dec=dDECmad, unit='arcsec, arcsec')
        sep_mad = c1.separation(c2).arcsec

        # write the dRA and dDEC to html table
        self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} (RA)<br>{2:.2f} &plusmn {3:.2f} (Dec)
                        <br>""".format(dRAmed, dRAmad, dDECmed, dDECmad))

        # plot the positional offsets
        self.plot(x,
                  y=y,
                  c=c,
                  figure=fig,
                  line_funcs=(self.x0, self.y0),
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  clabel=clabel,
                  text=txt,
                  ellipses=(e1, e2),
                  axis_perc=0,
                  loc='tr',
                  filename=filename,
                  labels=labels,
                  redo=redo)

        # plot the positional offsets across the sky
        title += " by sky position"
        xlabel = 'RA (deg)'
        ylabel = 'DEC (deg)'
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry_sky.{3}'.format(self.figDir, name1, name2,
                                                               self.plot_to)

        # get non-nan data shared between each used axis as a numpy array
        x, y, c, indices = self.shared_indices(self.cat.ra[name2], yaxis=self.cat.dec[name2],
                                               caxis=caxis)

        # for html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = ['S/N = {0:.2f}, \u0394RA = {1:.2f}\", \u0394DEC = {2:.2f}\"'.format(cval, dra,
                                                                                          ddec)
                      for cval, dra, ddec in zip(self.cat.df.loc[indices, 'SNR'],
                                                 self.cat.dRA[name2][indices],
                                                 self.cat.dDEC[name2][indices])]
        else:
            labels = ['\u0394RA = {0:.2f}\", \u0394DEC = {1:.2f}\"'.format(dra, ddec) for dra, ddec
                      in zip(self.cat.dRA[name2][indices], self.cat.dDEC[name2][indices])]

        # plot the positional offsets across the sky
        self.plot(x=x,
                  y=y,
                  c=c,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  reverse_x=True,
                  arrows=(self.cat.dRA[name2][indices], self.cat.dDEC[name2][indices]),
                  clabel=clabel,
                  axis_perc=0,
                  filename=filename,
                  labels=labels,
                  redo=redo)

        # derive column names and check if they exist
        freq = int(round(self.cat.freq[name1]))
        fitted_flux_col = '{0}_extrapolated_{1}MHz_flux'.format(name2, freq)
        fitted_ratio_col = '{0}_extrapolated_{1}MHz_{2}_flux_ratio'.format(name2, freq, name1)
        ratio_col = '{0}_{1}_flux_ratio'.format(name2, name1)

        # only plot flux ratio if it was derived
        if ratio_col not in self.cat.df.columns and (fitted_ratio_col not in self.cat.df.columns
                                                     or np.all(np.isnan(
                                                         self.cat.df[fitted_ratio_col]))):
            print("Can't plot flux ratio since you haven't derived the fitted flux density "
                  "at this frequency.")
            ratio_med = -111
            ratio_mad = -111
            flux_ratio_type = ''
            self.html.write('<td>')
        else:
            # compute flux ratio based on which one exists and rename variable for figure title
            if ratio_col in self.cat.df.columns:
                ratio = self.cat.df[ratio_col]
                flux_ratio_type = name2
            elif fitted_ratio_col in self.cat.df.columns:
                ratio = self.cat.df[fitted_ratio_col]
                flux_ratio_type = '{0}-extrapolated'.format(name2)

            logRatio = np.log10(ratio)

            # plot the flux ratio as a function of S/N
            fig = plt.figure(**self.fig_size)

            title = "{0} / {1} flux ratio".format(name1, flux_ratio_type)
            xlabel = 'S / N'
            ylabel = 'Flux Density Ratio'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_ratio.{3}'.format(self.figDir, name1, name2, self.plot_to)

            # get non-nan data shared between each used axis as a numpy array
            x, y, c, indices = self.shared_indices('SNR', yaxis=ratio)
            plt.loglog()
            plt.gca().grid(b=True, which='minor', color='w', linewidth=0.5)

            # derive the ratio statistics and store in string to append to plot
            ratio_med, ratio_mean, ratio_std, ratio_err, ratio_mad = get_stats(y)
            txt = r'\widetilde{Ratio}$: %.2f\n' % ratio_med
            txt += r'\overline{Ratio}$: %.2f\n' % ratio_mean
            txt += r'\sigma_{Ratio}$: %.2f\n' % ratio_std
            txt += r'\sigma_{\overline{Ratio}}$: %.2f' % ratio_err
            print(txt)
            # for html plots, add flux labels for every data point
            if flux_ratio_type == name2:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1, flux1,
                                                                                  name2, flux2)
                          for flux1, flux2 in zip(self.cat.flux[name1][indices]*1e3,
                                                  self.cat.flux[name2][indices]*1e3)]
            else:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1, flux1,
                                                                                  flux_ratio_type,
                                                                                  flux2)
                          for flux1, flux2 in zip(self.cat.flux[name1][indices]*1e3,
                                                  self.cat.df[fitted_flux_col][indices]*1e3)]

            # write the ratio to html report table
            if flux_ratio_type == name2:
                type = 'measured'
            else:
                type = 'extrapolated'
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} ({2})<br>""".format(ratio_med, ratio_mad, type))

            # plot the flux ratio as a function of S/N
            self.plot(x,
                      y=y,
                      c=c,
                      figure=fig,
                      line_funcs=(self.y1, self.ratio_err_min, self.ratio_err_max),
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      text=txt,
                      loc='tr',
                      axis_perc=0,
                      filename=filename,
                      labels=labels,
                      redo=redo)
            # plot the flux ratio across the sky
            fig = plt.figure(**self.fig_size)
            title += " by sky position"
            xlabel = 'RA (deg)'
            ylabel = 'DEC (deg)'

            # format labels according to destination of figure
            if self.plot_to == 'html':
                clabel = 'log\u2081\u2080 Flux Ratio'
            else:
                clabel = r'$\log_{10}$ Flux Ratio'

            # for html plots, add flux ratio labels for every data point
            labels = ['{0} = {1:.2f}'.format('Flux Ratio', cval) for cval in ratio[indices]]

            # plot the flux ratio across the sky
            self.plot(x,
                      y=y,
                      c=c,
                      figure=fig,
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      clabel=clabel,
                      reverse_x=True,
                      axis_perc=0,
                      filename=filename,
                      labels=labels,
                      redo=redo)

        # derive spectral index column name and check if exists
        si_column = '{0}_{1}_alpha'.format(name1, name2)

        if si_column not in self.cat.df.columns:
            print("Can't plot spectral index between {0} and {1}, since it was not derived.".format(
                name1, name2))
            alpha_med = -111  # null flag
            self.html.write('<td>')
        else:
            # plot the spectral index
            fig = plt.figure(**self.fig_size)
            plt.xlim(-3, 2)
            title = "{0}-{1} Spectral Index".format(name1, name2)
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_spectal_index.{3}'.format(self.figDir, name1,
                                                                  name2, self.plot_to)

            # get non-nan data shared between each used axis as a numpy array
            x, y, c, indices = self.shared_indices(si_column)

            # format labels according to destination of figure
            freq1 = int(round(min(self.cat.freq[name1], self.cat.freq[name2])))
            freq2 = int(round(max(self.cat.freq[name1], self.cat.freq[name2])))
            if self.plot_to == 'html':
                xlabel = '\u03B1 [{0}-{1} MHz]'.format(freq1, freq2)
            else:
                xlabel = r'$\alpha_{%s}^{%s}$' % (freq1, freq2)

            # derive the statistics of x and store in string
            alpha_med, alpha_mean, alpha_std, alpha_err, alpha_mad = get_stats(x)
            txt = r'\widetilde{\\alpha}$: %.2f\n' % alpha_med
            txt += r'\overline{\\alpha}$: %.2f\n' % alpha_mean
            txt += r'\sigma_{\\alpha}$: %.2f\n' % alpha_std
            txt += r'\sigma_{\overline{\\alpha}}$: %.2f' % alpha_err

            # write the ratio to html report table
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f}<br>""".format(alpha_med, alpha_mad))

            # plot the spectral index
            self.plot(x,
                      figure=fig,
                      title=title,
                      xlabel=xlabel,
                      ylabel='N',
                      axis_perc=0,
                      filename=filename,
                      text=txt,
                      loc='tl',
                      redo=redo)

        # write the end of the html report table row
        self.html.write("""</td>
                    </tr>""")

        alpha_med = self.cat.med_si
        alpha_type = '{0}'.format(name1)

        # create dictionary of validation metrics and where they come from
        metric_val = {'Flux Ratio': ratio_med,
                      'Flux Ratio Uncertainty': ratio_mad,
                      'RA Offset': dRAmed,
                      'DEC Offset': dDECmed,
                      'Positional Offset': sep_med,
                      'Positional Offset Uncertainty': sep_mad,
                      'Spectral Index': alpha_med}

        metric_source = {'Flux Ratio': flux_ratio_type,
                         'Flux Ratio Uncertainty': flux_ratio_type,
                         'RA Offset': name2,
                         'DEC Offset': name2,
                         'Positional Offset': name2,
                         'Positional Offset Uncertainty': name2,
                         'Spectral Index': alpha_type}

        count = self.cat.count[name2]

        # overwrite values if they are valid and come from a larger catalogue
        for key in list(metric_val.keys()):
            if count > self.metric_count[key] and metric_val[key] != -111:
                self.metric_count[key] = count
                self.metric_val[key] = metric_val[key]
                self.metric_source[key] = metric_source[key]