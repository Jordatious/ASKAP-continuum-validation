#!/usr/bin/env python

"""Input an radio continuum image and produce a validation report (in html) in a directory named after the image, which summarises
several validation tests/metrics (e.g. astrometry, flux scale, source counts, etc) and whether the data passed or failed these tests.

Last updated: 27/06/2018

Usage:
  Radio_continuum_validation.py -h | --help
  Radio_continuum_validation.py [-I --fits=<img>] [-M --main=<main-cat>] [-N --noise=<map>] [-C --catalogues=<list>]
  [-F --filter=<config>] [-R --snr=<ratio>] [-v --verbose] [-f --refind] [-r --redo] [-p --peak-flux] [-w --write] [-x --no-write] [-m --SEDs=<models>]
  [-e --SEDfig=<extn>] [-t --telescope=<name>] [-d --main-dir=<path>] [-n --ncores=<num>] [-b --nbins=<num>] [-s --source=<src>] [-a --aegean-params]
  [-c --correct=<level>]

Required:
  -I --fits=<img>           A fits continuum image [default: None].
  AND/OR
  -M --main=<main-cat>     Use this catalogue config file (overwrites options -p and -t). Default is to run Aegean [default: None].

Options:
  -h --help                 Show this help message.
  -C --catalogues=<list>    A comma-separated list of filepaths to catalogue config files corresponding to catalogues to use (will look in --main-dir for
                            each file not found in given path) [default: NVSS_config.txt,SUMSS_config.txt,GLEAM_config.txt,TGSS_config.txt].
  -N --noise=<map>          Use this fits image of the local rms. Default is to run BANE [default: None].
  -F --filter=<config>      A config file for filtering the sources in the input fits file [default: None].
  -R --snr=<ratio>          The signal-to-noise ratio cut to apply to the input catalogue and the source counts (doesn't affect source finding) [default: 5.0].
  -v --verbose              Verbose output [default: False].
  -f --refind               Force source finding step when catalogue already exists (sets redo to True) [default: False].
  -r --redo                 Force every step again (except source finding), even when catalogues already exist [default: False].
  -p --peak-flux            Use the peak flux rather than the integrated flux of the input image (not used when -A used) [default: False].
  -w --write                Write intermediate files generated during processing (e.g. cross-matched and pre-filtered catalogues, etc).
                            This will save having to reprocess the cross-matches, etc when executing the script again. [default: False].
  -x --no-write             Don't write any files except the html report and any files output from BANE and Aegean. [default: False].
  -m --SEDs=<models>        A comma-separated list of SED models to fit to the radio spectra ('pow','SSA','FFA','curve',etc) [default: None].
  -e --SEDfig=<extn>        Write figures for each SED model with this file extension (may significantly slow down script) [default: None].
  -t --telescope=<name>     Unique name of the telescope or survey to give to the main catalogue (not used when -A used). [default: MeerKAT].
  -d --main-dir=<path>      The absolute path to the main directory where this script and other required files are located [default: $ACES/UserScripts/col52r].
  -n --ncores=<num>         The number of cores (per node) to use when running BANE and Aegean (using >=20 cores may result in memory error) [default: 8].
  -b --nbins=<num>          The number of bins to use when performing source counts [default: 50].
  -s --source=<src>         The format for writing plots (e.g. screen, html, eps, pdf, png, etc) [default: html].
  -a --aegean=<params>      A single string with any extra paramters to pass into Aegean (except cores, noise, background, and table) [default: --floodclip=3].
  -c --correct=<level>      Correct the input fits image, write to 'name_corrected.fits' and use this to run through 2nd iteration of validation.
                            Fits image is corrected according to input level (0: none, 1: positions, 2: positions + fluxes) [default: 0]."""

import os
import sys
import glob
import warnings
from functions import *
from radio_image import radio_image
from catalogue import catalogue
from report import report
from docopt import docopt
from datetime import datetime
from inspect import currentframe, getframeinfo

# Set my own obvious warning output
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename

# Raise error if user tries to pass in noise map and then run Aegean
try:
    args = docopt(__doc__)
    if args['--main'] == 'None':
        if args['--fits'] == 'None' or args['--noise'] != 'None':
            raise SyntaxError
except SyntaxError:
    warnings.warn_explicit("""You must pass in a fits image (option -I) and/or the main catalogue (option -M).\n""",UserWarning,WARN,cf.f_lineno)
    warnings.warn_explicit("""When no catalogue is passed in, you cannot input a noise map (option -N).\n""",UserWarning,WARN,cf.f_lineno)
    warnings.warn_explicit("""Use option -h to see usage.\n""",UserWarning,WARN,cf.f_lineno)
    sys.exit()

# don't use normal display environment unless user wants to view plots on screen
import matplotlib as mpl
if args['--source'] != 'screen':
    mpl.use('Agg')

# find directory that contains all the necessary files
main_dir = args['--main-dir']
if main_dir.startswith('$ACES') and 'ACES' in list(os.environ.keys()):
    ACES = os.environ['ACES']
    main_dir = main_dir.replace('$ACES',ACES)
if not os.path.exists('{0}/requirements.txt'.format(main_dir)):
    split = sys.argv[0].split('/')
    script_dir = '/'.join(split[:-1])
    print("Looking in '{0}' for necessary files.".format(script_dir))
    if 'Radio_continuum_validation' in split[-1]:
        main_dir = script_dir
    else:
        warnings.warn_explicit("Can't find necessary files in main directory - {0}.\n".format(main_dir),UserWarning,WARN,cf.f_lineno)

# Set paramaters passed in by user
img = parse_string(args['--fits'])
verbose = args['--verbose']
source = args['--source']
refind = args['--refind']
redo = args['--redo'] or refind #force redo when refind is True
use_peak = args['--peak-flux']
write_any = not args['--no-write']
write_all = args['--write']
aegean_params = args['--aegean']
scope = args['--telescope']
ncores = int(args['--ncores'])
nbins = int(args['--nbins'])
level = int(args['--correct'])
snr = float(args['--snr'])

if '*' in args['--catalogues']:
    config_files = glob.glob(args['--catalogues'])
    print(config_files)
else:
    config_files = args['--catalogues'].split(',')
SEDs = args['--SEDs'].split(',')
SEDextn = parse_string(args['--SEDfig'])
if args['--SEDs'] == 'None':
    SEDs = 'pow'
    fit_flux=False
else:
    fit_flux=True

# force write_all=False write_any=False
if not write_any:
    write_all = False

# add '../' to relative paths of these files, since
# we'll create and move into a directory for output files
filter_config = new_path(parse_string(args['--filter']))
main_cat_config = parse_string(args['--main'])
noise = new_path(parse_string(args['--noise']))

if __name__ == "__main__":

    # add S/N and peak/int to output directory/file names
    suffix = 'snr{0}_'.format(snr)
    if use_peak:
        suffix += 'peak'
    else:
        suffix += 'int'

    # Load in fits image
    if img is not None:
        changeDir(img,suffix,verbose=verbose)
        img = new_path(img)
        image = radio_image(img,verbose=verbose,rms_map=noise)

        # Run BANE if user hasn't input noise map
        if noise is None:
            image.run_BANE(ncores=ncores,redo=refind)

        # Run Aegean and create main catalogue object from its output
        if main_cat_config is None:
            image.run_Aegean(ncores=ncores,redo=refind,params=aegean_params,write=write_any)
            cat = catalogue(image.cat_comp,scope,finder='aegean',image=image,verbose=verbose,autoload=False,use_peak=use_peak)
    else:
        changeDir(main_cat_config,suffix,verbose=verbose)
        image = None

    if main_cat_config is not None:
        main_cat_config = new_path(main_cat_config)

        # Use input catalogue config file
        if verbose:
            print("Using config file '{0}' for main catalogue.".format(main_cat_config))
        main_cat_config = find_file(main_cat_config,main_dir,verbose=verbose)
        main_cat_config_dic = config2dic(main_cat_config,main_dir,verbose=verbose)
        main_cat_config_dic.update({'image' : image, 'verbose' : verbose, 'autoload' : False})
        cat = catalogue(**main_cat_config_dic)

    # Filter out sources below input SNR, and set key fields and create report object before
    # filtering catalogue further so source counts can be written for all sources above input SNR
    cat.filter_sources(SNR=snr,redo=redo,write=write_any,verbose=verbose,file_suffix='_snr{0}'.format(snr))
    cat.set_specs(image)
    myReport = report(cat,main_dir,img=image,verbose=verbose,plot_to=source,redo=redo,src_cnt_bins=nbins,write=write_any)

    # use config file for filtering sources if it exists
    if filter_config is not None:
        if verbose:
            print("Using config file '{0}' for filtering.".format(filter_config))
        filter_dic = config2dic(filter_config,main_dir,verbose=verbose)
        filter_dic.update({'redo' : redo, 'write' : write_all, 'verbose' : verbose})
        cat.filter_sources(**filter_dic)
    else:
        # Only use reliable point sources for comparison
        cat.filter_sources(flux_lim=1e-3, ratio_frac=1.4, reject_blends=True, flags=True, psf_tol=1.5, resid_tol=3, redo=redo, write=write_all, verbose=verbose)

    # process each catalogue object according to list of input catalogue config files
    # this will cut out a box, cross-match to this instance, and derive the spectral index.
    for config_file in config_files:
        if verbose:
            print("Using config file '{0}' for catalogue.".format(config_file))
        config_file = config_file.strip() # in case user put a space
        config_file = find_file(config_file, main_dir, verbose=verbose)
        cat.process_config_file(config_file, main_dir, redo=redo, verbose=verbose, write_all=write_all, write_any=write_any)

    # Derive spectral indices using all fluxes except from main catalogue, and derive the flux at this frequency
    if len(cat.cat_list) > 1:
        cat.fit_spectra(redo=redo, models=SEDs, GLEAM_subbands='int', GLEAM_nchans=None, cat_name=None, write=write_any, fit_flux=fit_flux, fig_extn=SEDextn)

    # Produce validation report for each cross-matched survey
    for cat_name in cat.cat_list[1:]:
        # print "Would validate {0}".format(cat_name)
        if cat.count[cat_name] > 1:
            myReport.validate(cat.name,cat_name,redo=redo)

    # write validation summary table and close html file
    myReport.write_html_end()

    # correct image
    flux_factor = 1.0
    if level == 2:
        flux_factor = myReport.metric_val['ratio']
    if level in (1,2):
        image.correct_img(myReport.metric_val['dRA'], myReport.metric_val['dDEC'], flux_factor=flux_factor)
