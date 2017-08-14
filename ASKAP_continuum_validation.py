#!/usr/bin/env python

"""Input an ASKAP continuum image and produce a validation report (in html) in a directory named after the image, which summarises
several validation tests/metrics (e.g. astrometry, flux scale, source counts, etc) and whether the data passed or failed these tests.

Last updated: 14/08/2017

Usage:
  ASKAP_continuum_validation.py -h | --help
  ASKAP_continuum_validation.py [-S --Selavy=<cat>] [-N --noise=<map>] [-C --catalogues=<list>] [-F --filter=<config>] [-R --snr=<ratio>]
  [-v --verbose] [-f --refind] [-r --redo] [-p --peak-flux] [-w --write] [-x --no-write] [-m --SEDs=<models>] [-e --SEDfig=<extn>]
  [-d --main-dir=<path>] [-n --ncores=<num>] [-b --nbins=<num>] [-s --source=<src>] [-a --aegean-params] <fits-file>

Arguments:
  <fits-file>               A fits continuum image from ASKAP.

Options:
  -h --help                 Show this help message.
  -S --Selavy=<cat>         Use this Selavy catalogue of the input ASKAP image. Default is to run Aegean [default: None].
  -N --noise=<map>          Use this fits image of the local rms. Default is to run BANE [default: None].
  -C --catalogues=<list>    A comma-separated list of filepaths to catalogue config files corresponding to catalogues to use
                            (will look in --main-dir for each file not found in given path) [default: NVSS_config.txt,SUMSS_config.txt].
  -F --filter=<config>      A config file for filtering the sources in the ASKAP catalogue [default: None].
  -R --snr=<ratio>          The signal-to-noise ratio cut to apply to the ASKAP catalogue and the source counts (doesn't affect source finding) [default: 5.0].
  -v --verbose              Verbose output [default: False].
  -f --refind               Force source finding step, even when catalogue already exists (sets --redo to True) [default: False].
  -r --redo                 Force every step again (except source finding), even when catalogues already exist [default: False].
  -p --peak-flux            Use the peak flux rather than the integrated flux of the ASKAP catalogue [default: False].
  -w --write                Write intermediate files generated during processing (e.g. cross-matched and pre-filtered catalogues, etc).
                            This will save having to reprocess the cross-matches, etc when executing the script again. [default: False].
  -x --no-write             Don't write any files except the html report (without figures) and any files output from BANE and Aegean. [default: False].
  -m --SEDs=<models>        A comma-separated list of SED models to fit to the radio spectra ('pow','SSA','FFA','curve',etc) [default: pow,powCIbreak,SSA].
  -e --SEDfig=<extn>        Write figures for each SED model with this file extension (will significantly slow down script) [default: None].
  -d --main-dir=<path>      The absolute path to the main directory where this script and other required files are located [default: $ACES/UserScripts/col52r].
  -n --ncores=<num>         The number of cores (per node) to use when running BANE and Aegean (using >=20 cores may result in memory error) [default: 8].
  -b --nbins=<num>          The number of bins to use when performing source counts [default: 50].
  -s --source=<src>         The format for writing plots (e.g. screen, html, eps, pdf, png, etc) [default: html].
  -a --aegean=<params>      A single string with any extra paramters to pass into Aegean (except cores, noise, background, and table) [default: --floodclip=3]."""

from __future__ import division
from docopt import docopt
import os
import sys
import glob
import collections
from datetime import datetime
import warnings
from inspect import currentframe, getframeinfo

from astropy.io import fits as f
from astropy.io import votable
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.coordinates as c
from astropy.wcs import WCS
from astropy.io.votable import parse_single_table
from astropy.utils.exceptions import AstropyWarning
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import scipy.special as special

#ignore annoying astropy warnings and set my own obvious warning output
warnings.simplefilter('ignore', category=AstropyWarning)
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename

#Raise error if user tries to pass in noise map and then run Aegean
try:
    args = docopt(__doc__)
    if args['--Selavy'] == 'None' and args['--noise'] != 'None':
        raise SyntaxError
except SyntaxError:
    warnings.warn_explicit("""When no Selavy catalogue is passed in (option -S), you cannot input a noise map (option -N).\n""",UserWarning,WARN,cf.f_lineno)
    sys.exit()

#don't use normal display environment unless user wants to view plots on screen
import matplotlib as mpl
if args['--source'] != 'screen':
    mpl.use('Agg')
import matplotlib.pyplot as plt, mpld3
from matplotlib import cm, ticker, colors
from mpld3 import plugins
from matplotlib.patches import Ellipse
import matplotlib.image as image
import seaborn

#find directory that contains all the necessary files
main_dir = args['--main-dir']
if main_dir.startswith('$ACES') and 'ACES' in os.environ.keys():
    ACES = os.environ['ACES']
    main_dir = main_dir.replace('$ACES',ACES)
elif not os.path.exists('{0}/requirements.txt'.format(main_dir)):
    split = sys.argv[0].split('/')
    script_dir = '/'.join(split[:-1])
    print "Looking in '{0}' for necessary files.".format(script_dir)
    if split[-1] == 'ASKAP_continuum_validation.py':
        main_dir = script_dir
    else:
        warnings.warn_explicit("Can't find necessary files in main directory - {0}.\n".format(main_dir),UserWarning,WARN,cf.f_lineno)


def changeDir(filepath,suffix,verbose=False):

    """Derive a directory name from an input file to store all output files, create it, and then change to it.

    Arguments:
    ----------
    filepath : string
        A path to a fits image or catalogue.
    suffix : string
        A suffix to append to the end of the created directory.

    Keyword arguments:
    ------------------
    verbose : bool
        Verbose output."""

    #derive directrory name for output files
    filename = filepath.split('/')[-1]
    basename = remove_extn(filename)
    dir = '{0}_continuum_validation_{1}'.format(basename,suffix)

    #create it if it doesn't exist
    if not os.path.exists(dir):
        if verbose:
            print "Making directory for output files - {0}.".format(dir)
        os.mkdir(dir)

    #move to that directory and update the filepath
    if verbose:
        print "Changing to directory for output files - '{0}'.".format(dir)
    os.chdir(dir)

###The following are radio SED models as a function of frequency and several fitted parmaters###

def powlaw(freq,S_norm,alpha):
    return S_norm*freq**alpha

def curve(freq, S_max, nu_m, alpha_thick, alpha_thin):
	return S_max/(1 -np.exp(-1))*((freq/nu_m)**alpha_thick)*(1 - np.exp(-(freq/nu_m)**(alpha_thin-alpha_thick)))

def pow_CIbreak(freq, S_norm, alpha, nu_br):
    return S_norm*(freq/nu_br)**(alpha+0.5 + 0.5*(1 + (freq/nu_br)**4)**-1)

def pow_CIbreak2(freq, S_norm, alpha, nu_br):
	alpha,freq = CIbreak(freq,alpha,nu_br)
	return S_norm*freq**alpha

def pow_JPbreak(freq, S_norm, alpha, nu_br):
    return S_norm*(freq**alpha)*JPbreak(freq,nu_br)

def SSA(freq,S_norm,beta,nu_m):
	return S_norm*((freq/nu_m)**(-(beta-1)/2))*(1-np.exp(-(freq/nu_m)**(-(beta+4)/2)))/((freq/nu_m)**(-(beta+4)/2))

def SSA_JPbreak(freq,S_norm,beta,nu_m,nu_br):
	return SSA(freq,S_norm,beta,nu_m)*JPbreak(freq,nu_br)

def SSA_CIbreak(freq,S_norm,beta,nu_m,nu_br):
    dummyalpha,dummyfreq = CIbreak(freq,beta,nu_br)
    return SSA(freq,S_norm,beta,nu_m)*dummyfreq**dummyalpha

def FFA(freq,S_norm,alpha,nu_m):
    return S_norm*(freq**(alpha))*np.exp(-(freq/nu_m)**(-2.1))

def Bic98_FFA(freq,S_norm,alpha,p,nu_m):
	return S_norm*(p+1)*((freq/nu_m)**(2.1*(p+1)+alpha))*special.gammainc((p+1),((freq/nu_m)**(-2.1)))*special.gamma(p+1)

def Bic98_FFA_CIbreak(freq,S_norm,alpha,p,nu_m,nu_br):
    dummyalpha,dummyfreq = CIbreak(freq,alpha,nu_br)
    return Bic98_FFA(freq,S_norm,alpha,p,nu_m)*dummyfreq**dummyalpha

def Bic98_FFA_JPbreak(freq,S_norm,alpha,p,nu_m,nu_br):
    return Bic98_FFA(freq,S_norm,alpha,p,nu_m)*JPbreak(freq,nu_br)

def CIbreak(freq,alpha,nu_br):
    alpha = np.where(freq <= nu_br, alpha, alpha-0.5)
    dummyfreq = freq / nu_br
    return alpha,dummyfreq

def JPbreak(freq,nu_br):
    return np.exp(-freq/nu_br)

def flux_at_freq(freq,known_freq,known_flux,alpha):

    """Get the flux of a source at a given frequency, according to a given power law.

    Arguments:
    ----------
    freq : float
        The frequency at which to measure the flux.
    known_freq : float
        A frequency at which the flux is known.
    known_flux : float
        The flux at the known frequency.
    alpha : float
        The spectral index.

    Returns:
    --------
    flux : float
        The flux at the given frequency."""

    return 10**(alpha*(np.log10(freq) - np.log10(known_freq)) + np.log10(known_flux))

def ticks_format_flux(value, index):

    """Return flux density ticks in mJy"""
    value = value*1e3
    return ticks_format(value, index)

def ticks_format_freq(value, index):

    """Return frequency ticks in GHz"""
    value = value/1e3
    return ticks_format(value, index)

def ticks_format(value, index):

    """Return matplotlib ticks in LaTeX format, getting the value as integer [0,99], a 1 digit float [0.1, 0.9], or otherwise n*10^m.

    Arguments:
    ----------
    value : float
        The value of the tick.
    index : float
        The index of the tick.

    Returns:
    --------
    tick : string
        The tick at that value, in LaTeX format."""

    #get the exponent and base
    exp = np.floor(np.log10(value))
    base = value/10**exp

    #format according to values
    if exp >= 0 and exp <= 3:
        return '${0:d}$'.format(int(value))
    elif exp <= -1:
        return '${0:.2f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def sig_figs(value, n=2):

    """Return a string of the input value with n significant figures.

    Arguments:
    ----------
    value : float
        The input value

    Keyword arguments:
    ------------------
    n : int
        The number of significant figures.

    Returns:
    --------
    value : string
        The value to n significant figures."""

    return ("{0:.%d}" % (n)).format(value)


def plot_spectra(freqs, fluxes, errs, models, names, params, param_errs, rcs, BICs, colours, labels, figname, annotate=True, model_selection='better'):

    """Plot a figure of the radio spectra of an individual source, according to the input data and models.

    Arguments:
    ----------
    freqs : list
        A list of frequencies in MHz.
    fluxes : list
        A list of fluxes in Jy.
    errs : list
        A list of flux uncertainties in Jy.
    models : list
        A list of functions corresponding to models of the radio spectrum.
    names : 2D list
        A list of fitted parameter names corresponding to each model above.
    params : 2D list
        A list of fitted parameter values corresponding to each model above.
    param_errs : 2D list
        A list of uncertainties on the fitted parameters corresponding to each model above.
    rcs : list
        A list of reduced chi squared values corresponding to each model above.
    BICs : list
        A list of Bayesian Information Criteria (BIC) values corresponding to each model above.
    colours : list
        A list of colours corresponding to each model above.
    labels : list
        A list of labels corresponding to each model above.
    figname : string
        The filename to give the figure when writing to file.

    Keyword arguments:
    ------------------
    annotate : bool
        Annotate fit info onto figure.
    model_selection : string
        How to select models for plotting, based on the BIC values. Options are:

            'best' - only plot the best model.

            'all' - plot all models.

            'better' - plot each model better than the previous, chronologically."""

    #create SEDs directory if doesn't already exist
    if not os.path.exists('SEDs'):
        os.mkdir('SEDs')

    fig=plt.figure()
    ax=plt.subplot()

    #plot frequency axis 20% beyond range of values
    xlin = np.linspace(min(freqs)*0.8,max(freqs)*1.2,num=5000)
    plt.ylabel(r'Flux Density $S$ (mJy)')
    plt.xlabel(r'Frequency $\nu$ (GHz)')
    plt.xscale('log')
    plt.yscale('log')

    #adjust the tick values and add grid lines at minor tick locations
    subs = [1.0, 2.0, 5.0]
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format_freq))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(ticks_format_flux))
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

    #plot flux measurements
    plt.errorbar(freqs,fluxes,yerr=errs,linestyle='none',marker='.',c='r',zorder=15)

    best_bic = 0
    dBIC = 3
    offset = 0
    plotted_models = 0

    #plot each model
    for i in range (len(models)):
        ylin = models[i](xlin,*params[i])
        txt = "{0}:\n   {1}".format(labels[i],r'$\chi^2_{\rm red} = %.1f$' % rcs[i])

        #compare BIC values
        bic = BICs[i]
        if i > 0:
            dBIC = best_bic - bic
            if model_selection != 'best':
                txt += ', {0}'.format(r'$\Delta{\rm BIC} = %.1f$' % (dBIC))
        if dBIC >= 3:
            best_bic = bic

        #plot model if selected according to input
        if model_selection == 'all' or (model_selection == 'better' and dBIC >= 3) or (model_selection == 'best' and BICs[i] == min(BICs)):

            plotted_models += 1
            plt.plot(xlin,ylin,c=colours[i],linestyle='--',zorder=i+1,label=labels[i])
            plt.legend(scatterpoints=1,fancybox=True,frameon=True,shadow=True)
            txt += '\n'

            #add each fitted parameter to string (in LaTeX format)
            for j,param in enumerate(names[i]):
                units = ''
                tokens = param.split('_')
                if len(tokens[0]) > 1:
                    tokens[0] = "\\" + tokens[0]
                if len(tokens) > 1:
                    param = r'%s_{\rm %s}' % (tokens[0],tokens[1])
                else:
                    param = tokens[0]
                val = params[i][j]
                err = param_errs[i][j]

                if param.startswith('S'):
                    units = 'Jy'
                    if val < 0.01:
                        val = val*1e3
                        err = err*1e3
                        units = 'mJy'
                elif 'nu' in param:
                    units = 'MHz'
                    if val > 100:
                        val = val/1e3
                        err = err/1e3
                        units = 'GHz'

                val = sig_figs(val)
                err = sig_figs(err)

                txt += '   ' + r'${0}$ = {1} $\pm$ {2} {3}'.format(param,val,err,units) + '\n'

            #annotate all fit info if it will fit on figure
            if annotate and plotted_models <= 3:
                plt.text(offset,0,txt,horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
                offset += 0.33

    #write figure and close
    plt.savefig('SEDs/{0}'.format(figname))
    plt.close()

def likelihood(ydata,ymodel,yerrs):

    """Return the likelihood for a given model of a single source.

    Arguments:
    ----------
    ydata : list
        The flux values at each frequency.
    ymodel : list
        The values of the model at each frequency.
    yerrs : list
        The uncertainty on the flux at each frequency.

    Returns:
    --------
    likelihood : float
        The likelihood."""

    return np.prod( ( 1 / (yerrs*np.sqrt(2*np.pi)) ) * np.exp( (-1/(2*yerrs**2)) * (ydata-ymodel)**2 ) )

def fit_info(ydata,ymodel,yerrs,deg):

    """Return the reduced chi squared and BIC values for a given model of a single source.

    Arguments:
    ----------
    ydata : list
        The flux values at each frequency.
    ymodel : list
        The values of the model at each frequency.
    yerrs : list
        The uncertainty on the flux at each frequency.
    deg : int
        The degrees of freedom.

    Returns:
    --------
    red_chi_sq : float
        The reduced chi squared value.
    BIC : float
        The Bayesian Information Criteria."""

    chi_sq=np.sum(((ydata-ymodel)/yerrs)**2)
    DOF=len(ydata)-deg
    red_chi_sq = chi_sq/DOF
    BIC = -2*np.log(likelihood(ydata,ymodel,yerrs)) + deg * np.log(len(ydata))
    return red_chi_sq,BIC

def two_freq_power_law(freq, freqs, fluxes, errs):

    """Derive a two-frequency spectral index, uncertainty and fitted flux at the input frequency.

    Arguments:
    ----------
    freq : float
        The frequency at which to calculate the flux.
    freqs : list
        A list of frequencies in the same units.
    fluxes : list
        A list of fluxes in the same units.
    errs : list
        A list of flux uncertainties in the same units.

    Returns:
    --------
    alpha : float
        The spectral index.
    alpha_err : float
        The uncertainty of the spectral index.
    flux : float
        The fitted flux at the input frequency."""

    #directly derive alpha and error from two fluxes
    alpha = np.log10(fluxes[0]/fluxes[1]) / np.log10(freqs[0]/freqs[1])
    alpha_err = np.sqrt((errs[0]/fluxes[0])**2 + (errs[1]/fluxes[1])**2)/np.log10(freqs[0]/freqs[1])
    flux = flux_at_freq(freq,freqs[0],fluxes[0],alpha)
    return alpha,alpha_err,flux

def SED(freq, freqs, fluxes, errs, models='pow', figname=None):

    """Fit SED models to an individual source and return the model params and errors along with the expected flux at a given frequency, for each input model.
    Lists must be the same length and contain at least two elements, all with the same units (ideally MHz and Jy).

    Arguments:
    ----------
    freq : float
        The frequency at which to calculate the flux.
    freqs : list
        A list of frequencies in the same units.
    fluxes : list
        A list of fluxes in the same units.
    errs : list
        A list of flux uncertainties in the same units.

    Keyword arguments:
    ------------------
    models : string or list
        A single model or list of models to fit (e.g. ['pow','FFA','SSA']).
    figname : string
        Write a figure of the radio spectra and model to file, using this filename. Use None to not write to file.

    Returns:
    --------
    fit_models : list
        A list of fitted models.
    names : 2D list
        A list of lists of names of fitted parameters, for each input model.
    params : 2D list
        A list of lists of fitted parameters, for each input model.
    errors : 2D list
        A list of lists of uncertainties on the fitted parameters, for each input model.
    fitted_fluxes : list
        A list of fitted fluxes at the input frequency, for each input model.
    rcs : list
        A list of reduced chi squared values, for each input model.
    BICs : list
        A list of Bayesian Information Criteria (BIC) values, for each input model."""

    #initial guesses of different params
    S_max = max(fluxes)
    nu_max = freqs[fluxes == S_max][0]
    alpha = -0.8
    beta = 1-2*alpha
    nu_br = np.mean(freqs)
    p = 0.5

    #initial guesses of different models
    params = { 'pow' : [S_max,alpha],
            'powcibreak' : [S_max, alpha, nu_br],
            'powjpbreak' : [S_max, alpha, nu_br],
            'curve' : [S_max, nu_max, 1, alpha],
            'ssa' : [S_max, beta, nu_max],
            'ssacibreak' : [S_max, beta, nu_max, nu_br],
            'ssajpbreak' : [S_max, beta, nu_max, nu_br],
            'ffa' : [S_max, alpha, nu_max],
            'bicffa' : [S_max, alpha, p, nu_max],
            'bicffacibreak' : [S_max, alpha, p, nu_max, nu_br],
            'bicffajpbreak' : [S_max, alpha, p, nu_max, nu_br]}

    #different SED models from functions above
    funcs = {   'pow' : powlaw,
                'powcibreak' : pow_CIbreak,
                'powjpbreak' : pow_JPbreak,
                'curve' : curve,
                'ssa' : SSA,
                'ssacibreak' : SSA_CIbreak,
                'ssajpbreak' : SSA_JPbreak,
                'ffa' : FFA,
                'bicffa' : Bic98_FFA,
                'bicffacibreak' : Bic98_FFA_CIbreak,
                'bicffajpbreak' : Bic98_FFA_JPbreak}

    #matplotlib colours
    colours = { 'pow' : 'black',
            'powcibreak' : 'b',
            'powjpbreak' : 'violet',
            'curve' : 'r',
            'ssa' : 'g',
            'ssacibreak' : 'r',
            'ssajpbreak' : 'g',
            'ffa' : 'o',
            'bicffa' : 'r',
            'bicffacibreak' : 'b',
            'bicffajpbreak' : 'r'}

    #labels
    labels = {  'pow' : 'Power law',
                'powcibreak' : 'Power law\n + CI break',
                'powjpbreak' : 'Power law\n + JP break',
                'curve' : 'Tschager+03 Curve',
                'ssa' : 'Single SSA',
                'ssacibreak' : 'Single SSA\n + CI break',
                'ssajpbreak' : 'Single SSA\n + JP break',
                'ffa' : 'Single FFA',
                'bicffa' : 'Bicknell+98 FFA',
                'bicffacibreak' : 'Bicknell+98 FFA\n + CI break',
                'bicffajpbreak' : 'Bicknell+98 FFA\n + JP break'}

    #store used models, fitted parameters and errors, fitted fluxes, reduced chi squared and BIC
    fit_models,fit_params,fit_param_errors,fitted_fluxes,rcs,BICs = [],[],[],[],[],np.array([])

    #convert single model to list
    if type(models) is str:
        models = [models]

    for model in models:
        model = model.lower()

        #fit model if DOF >= 1
        if len(freqs) >= len(params[model])+1:
            try:
                #perform a least squares fit
                popt, pcov = opt.curve_fit(funcs[model], freqs, fluxes, p0 = params[model], sigma = errs, maxfev = 10000)

                #add all fit info to lists
                fit_models.append(model)
                fit_params.append(popt)
                fit_param_errors.append(np.sqrt(np.diag(pcov)))
                RCS,bic = fit_info(fluxes,funcs[model](freqs,*popt),errs,len(popt))
                rcs.append(RCS)
                BICs = np.append(BICs,bic)
                fitted_fluxes.append(funcs[model](freq,*popt))
            except (ValueError,RuntimeError),e:
                print "Couldn't find good fit for {0} model.".format(model)
                print e

    #get lists of names, functions, colours and labels for all used models
    names = [funcs[model].func_code.co_varnames[1:funcs[model].func_code.co_argcount] for model in fit_models]
    funcs = [funcs[model] for model in fit_models]
    colours = [colours[model] for model in fit_models]
    labels = [labels[model] for model in fit_models]

    #write figure for this source
    if figname is not None and len(fit_models) > 0:
        plot_spectra(freqs,fluxes,errs,funcs,names,fit_params,fit_param_errors,rcs,BICs,colours,labels,figname,model_selection='all')

    return fit_models,names,fit_params,fit_param_errors,fitted_fluxes,rcs,BICs


def get_pixel_area(fits,flux=0,nans=False,ra_axis=0,dec_axis=1,w=None):

    """For a given image, get the area and solid angle of all non-nan pixels or all pixels below a certain flux (doesn't count pixels=0).
    The RA and DEC axes follow the WCS convention (i.e. starting from 0).

    Arguments:
    ----------
    fits : astropy.io.fits
        The primary axis of a fits image.

    Keyword arguments:
    ------------------
    flux : float
        The flux in Jy, below which pixels will be selected.
    nans : bool
        Derive the area and solid angle of all non-nan pixels.
    ra_axis : int
        The index of the RA axis (starting from 0).
    dec_axis : int
        The index of the DEC axis (starting from 0).
    w : astropy.wcs.WCS
        A wcs object to use for reading the pixel sizes.

    Returns:
    --------
    area : float
        The area in square degrees.
    solid_ang : float
        The solid angle in steradians.

    See Also:
    ---------
    astropy.io.fits
    astropy.wcs.WCS"""

    if w is None:
        w = WCS(fits.header)

    #count the pixels and derive area and solid angle of all these pixels
    if nans:
        count = fits.data[(~np.isnan(fits.data)) & (fits.data != 0)].shape[0]
    else:
        count = fits.data[(fits.data < flux) & (fits.data != 0)].shape[0]

    area = (count*np.abs(w.wcs.cdelt[ra_axis])*np.abs(w.wcs.cdelt[dec_axis]))
    solid_ang = area*(np.pi/180)**2
    return area,solid_ang


def axis_lim(data,func,perc=10):

    """Return an axis limit value a certain % beyond the min/max value of a dataset.

    Arguments:
    ----------
    data : list-like
        A list-like object input into min() or max(). Usually this will be a numpy array or pandas Series.
    func : function
        max or min.

    Keyword Arguments:
    ------------------
    perc : float
        The percentage beyond the limit of a dataset.

    Returns:
    --------
    lim : float
        A value the input % beyond the limit.

    See Also:
    --------
    numpy.array
    pandas.Series"""

    lim = func(data)

    if (lim < 0 and func is min) or (lim > 0 and func is max):
        lim *= (1 + (perc/100))
    else:
        lim *= (1 - (perc/100))

    return lim


def get_stats(data):

    """Return the median, mean, standard deviation, standard error and rms of the median absolute deviation (mad) of the non-nan values in a list.

    Arguments:
    ----------
    data : list-like (numpy.array or pandas.Series)
        The data used to calculate the statistics.

    Returns:
    --------
    med : float
        The median.
    mean : float
        The mean.
    std : float
        The standard deviation.
    err : float
        The standard error.
    rms_mad : float
        The rms of the mad

    See Also
    --------
    numpy.array
    pandas.Series"""

    #remove nan indices, as these affect the calculations
    values = data[~np.isnan(data)]

    med = np.median(values)
    mean = np.mean(values)
    std = np.std(values)
    sterr = std / np.sqrt(len(values))
    rms_mad = np.median(np.abs(values-np.median(values)))/0.6745

    return med,mean,std,sterr,rms_mad


def remove_extn(filename):

    """Return a file name without its extension.

    Arguments:
    ----------
    filename : string
        The file name.

    Returns:
    --------
    filename : string
        The file name without its extension."""

    #do this in case more than one '.' in file name
    return '.'.join(filename.split('.')[:-1])

def config2dic(filepath,verbose=False):

    """Read a configuration file and create an dictionary of arguments from its contents, which will usually be passed into a new object instance.

    Arguments:
    ----------
    filepath : string
        The absolute filepath of the config file.

    Keyword Arguments:
    ------------------
    verbose : bool
        Verbose output.

    Returns:
    --------
    args_dict : dict
        A dictionary of arguments, to be passed into some function, usually a new object instance."""

    #open file and read contents
    config_file = open(filepath)
    txt = config_file.read()
    args_dict = {}

    #set up dictionary of arguments based on their types
    for line in txt.split('\n'):
        if len(line) > 0 and line.replace(' ','')[0] != '#':
            #use '=' as delimiter and strip whitespace
            split = line.split('=')
            key = split[0].strip()
            val = split[1].strip()
            val = parse_string(val)

            #if parameter is filename, store the filepath
            if key == 'filename':
                val = find_file(val,verbose=verbose)

            args_dict.update({key : val})

    config_file.close()
    return args_dict


def parse_string(val):

    """Parse a string to another data type, based on its value.

    Arguments:
    ----------
    val : string
        The string to parse.

    Returns:
    --------
    val : string or NoneType or bool or float
        The parsed string."""

    if val.lower() == 'none':
        val = None
    elif val.lower() in ('true','false'):
        val = (val.lower() == 'true')
    elif val.replace('.','',1).replace('e','').replace('-','').isdigit():
        val = float(val)

    return val

def new_path(filepath):

    """For a given input filepath, return the path after having moved into a new directory. This will add '../' to the beginning of relative filepaths.

    Arguments:
    ----------
    filepath : string
        The filepath.

    Returns:
    --------
    filepath : string
        The updated filepath."""

    #add '../' to filepath if it's a relative filepath
    if filepath is not None and filepath[0] != '/':
        filepath = '../' + filepath

    return filepath

def find_file(filepath,verbose=True):

    """Look for a file in specific paths. Look one directory up if filepath is relative, otherwise look in main directory, otherwise raise exception.

    Arguments:
    ----------
    filepath : string
        An absolute or relative filepath.

    Returns:
    --------
    filepath : string
        The path to where the file was found."""

    #raise exception if file still not found
    if not (os.path.exists(filepath) or os.path.exists('{0}/{1}'.format(main_dir,filepath)) or os.path.exists(new_path(filepath))):
        raise Exception("Can't find file - {0}. Ensure this file is in input path or --main-dir.\n".format(filepath))
    #otherwise update path to where file exists
    elif not os.path.exists(filepath):
        #look in main directory if file doesn't exist in relative filepath
        if os.path.exists('{0}/{1}'.format(main_dir,filepath)):
            if verbose:
                print "Looking in '{0}' for '{1}'.".format(main_dir,filepath)
            filepath = '{0}/{1}'.format(main_dir,filepath)
        #update directory path if file is relative path
        else:
            filepath = new_path(filepath)

    return filepath



class radio_image(object):

    def __init__(self,filepath,aegean_suffix='_aegean',aegean_extn='fits',rms_map=None,SNR=5,verbose=False):

        """Initialise a radio image object.

        Arguments:
        ----------
        filepath : string
            The absolute path to a fits image (must have '.fits' extension).

        Keyword arguments:
        ------------------
        aegean_suffix : string
            A suffix to append to any Aegean files (don't include 'comp' or 'isle').
        aegean_extn : string
            The extension of the Aegean catalogue if source finding is performed.
        rms_map : string
            The filepath of a fits image of the local rms. If None is provided, a BANE map is used.
        SNR : float
            The signal-to-noise ratio, used to derive a search radius when cross-matching the catalogue of this image.
        verbose : bool
            Verbose output."""

        self.verbose = verbose
        if verbose:
            print "----------------------"
            print "| Reading fits image |"
            print "----------------------"
        if verbose:
            print "Initialising radio_image object using file '{0}'.".format(filepath.split('/')[-1])

        self.filepath = filepath
        self.name = filepath.split('/')[-1]
        self.rms_map = rms_map

        #Aegean format
        self.basename = remove_extn(self.name) + aegean_suffix
        self.bkg = '../{0}_bkg.fits'.format(self.basename)
        self.cat_name = '../{0}.{1}'.format(self.basename,aegean_extn)
        self.cat_comp = '../{0}_comp.{1}'.format(self.basename,aegean_extn)
        self.residual = '../{0}_residual.fits'.format(self.basename)
        self.model = '../{0}_model.fits'.format(self.basename)

        #open fits image and store header specs
        self.fits = f.open(filepath)[0] #HDU axis 0
        self.header_specs(self.fits,verbose=verbose)

        #expected positional error, given by FWHM/SNR (Condon, 1997)
        self.posErr = int(round(self.bmaj/SNR))


    def header_key(self,header,key,floatify=False):

        """Return the value of the key from a fits header. If key doesn't exist, '' will be returned.

        Arguments:
        ----------
        header : astropy.io.fits.header.Header
            A fits header object.
        key : string
            A key from the header.

        Keyword Arguments:
        ------------------
        floatify : bool
            Convert value to float.

        Returns:
        --------
        value : string
            '' if key doesn't exist, otherwise, the value of the key.

        See Also:
        ---------
        astropy.io.fits.header.Header"""

        if key not in header.keys():
            return ''
        elif floatify:
            return float(header[key])
        else:
            return header[key]


    def header_specs(self,fits,verbose=False):

        """Read the header of a fits file and set several fields for this instance, including the RA, DEC, BMIN, BMAJ, BPA, frequency, etc.

        Arguments:
        ----------
        fits : astropy.io.fits
            The primary axis of a fits image.

        Keyword arguments:
        ------------------
        verbose : bool
            Verbose output."""

        if verbose:
            print "Reading Bmaj, RA/DEC centre, frequency, etc from fits header."

        head = fits.header
        w = WCS(head)

        #Assume these keys exist
        self.bmaj = head['BMAJ']*3600
        self.bmin = head['BMIN']*3600
        self.bpa = head['BPA']

        #Set these to '' if they don't exist
        self.project = self.header_key(head,'PROJECT')
        self.sbid = self.header_key(head,'SBID')
        self.date = self.header_key(head,'DATE-OBS')
        self.duration = self.header_key(head,'DURATION',floatify=True) #seconds

        #get ASKAP soft version from history in header if it exists
        self.soft_version = ''
        self.pipeline_version = ''
        if 'HISTORY' in head.keys():
            for val in head['HISTORY']:
                if 'ASKAPsoft version' in val:
                    self.soft_version = val.split('/')[-1].split()[-1].replace(',','')
                if 'ASKAP pipeline version' in val:
                    self.pipeline_version = val.split()[-1].replace(',','')

        #derive duration in hours
        if self.duration != '':
            self.duration = '{0:.2f}'.format(self.duration/3600)

        #iterate through axes in header to find RA,DEC and frequency
        axis = 0
        while axis < w.naxis:
            chanType = w.wcs.ctype[axis]

            if(chanType.startswith('RA')):
                self.refRA = w.wcs.crval[axis]
                self.raPS = np.abs(w.wcs.cdelt[axis])*3600 #pixel size in arcsec
                self.ra_axis = axis
            elif(chanType.startswith('DEC')):
                self.refDEC = w.wcs.crval[axis]
                self.decPS = np.abs(w.wcs.cdelt[axis])*3600 #pixel size in arcsec
                self.dec_axis = axis
            elif(chanType.startswith('FREQ')):
                self.freq = w.wcs.crval[axis]/1e6 #freq in MHz
                w = w.dropaxis(axis)
                axis -= 1
            #drop all other axes from wcs object so only RA/DEC left
            else:
                w = w.dropaxis(axis)
                axis -= 1
            axis += 1

        #Get the area and solid angle from all non-nan pixels in this image
        self.area,self.solid_ang = get_pixel_area(fits, nans=True, ra_axis=self.ra_axis, dec_axis=self.dec_axis, w=w)

        #store the RA/DEC of the image as centre pixel and store image vertices
        naxis1 = int(head['NAXIS1'])
        naxis2 = int(head['NAXIS2'])
        pixcrd = np.array([[naxis1/2, naxis2/2]])
        centre = w.all_pix2world(pixcrd,1)
        self.ra = centre[0][0]
        self.dec = centre[0][1]
        self.centre = SkyCoord(ra=self.ra, dec=self.dec, unit="deg,deg").to_string(style='hmsdms',sep=':')
        self.vertices = w.calc_footprint()
        self.ra_bounds = min(self.vertices[:,:1])[0],max(self.vertices[:,:1])[0]
        self.dec_bounds = min(self.vertices[:,1:])[0],max(self.vertices[:,1:])[0]

        if verbose:
            print "Found psf axes {0:.2f} x {1:.2f} arcsec at PA {2}.".format(self.bmaj,self.bmin,self.bpa)
            print "Found a frequency of {0} MHz.".format(self.freq)
            print "Found a field centre of {0}.".format(self.centre)

    def run_BANE(self,ncores=8,redo=False):

        """Produce a noise and background map using BANE.

        Keyword arguments:
        ------------------
        ncores : int
            The number of cores to use (per node) when running BANE.
        redo : bool
            Reproduce the maps, even if they exist."""

        #Overwrite rms map input by user
        self.rms_map = '../{0}_rms.fits'.format(self.basename)

        if redo:
            print "Re-running BANE and overwriting background and rms maps."

        #Run BANE to create a map of the local rms
        if not os.path.exists(self.rms_map) or redo:

            print "----------------------------"
            print "| Running BANE for rms map |"
            print "----------------------------"

            command = "BANE --cores={0} --out=../{1} {2}".format(ncores,self.basename,self.filepath)
            print "Running BANE using following command:"
            print command
            os.system(command)
        else:
            print "'{0}' already exists. Skipping BANE.".format(self.rms_map)


    def run_Aegean(self,params='',ncores=8,write=True,redo=False):

        """Perform source finding on image using Aegean, producing just a component catalogue by default.

        Keyword arguments:
        ------------------
        params : string
            Any extra parameters to pass into Aegean (apart from cores, noise, background and table).
        ncores : int
            The number of cores to use (per node) when running BANE and Aegean.
        write : bool
            Write the fitted model and residual images.
        redo : bool
            Perform source finding, even if output catalogue(s) exist."""

        if redo:
            print "Re-doing source finding. Overwriting all Aegean and AeRes files."

        if not os.path.exists(self.cat_comp) or redo:

            print "--------------------------------"
            print "| Running Aegean for catalogue |"
            print "--------------------------------"

            #Run Aegean source finder to produce catalogue of image
            command = 'aegean --cores={0} --noise={1} --background={2} --table={3}'.format(ncores,self.rms_map,self.bkg,self.cat_name)

            #Also write ds9 region file and island fits file when used wants verbose output
            if self.verbose:
                command += ',{0}.reg'.format(remove_extn(self.cat_name))

            #Add any parameters used has input and file name
            command += " {0} {1}".format(params,self.filepath)
            print "Running Aegean with following command:"
            print command
            os.system(command)

            #Print error message when no sources are found and catalogue not created.
            if not os.path.exists(self.cat_comp):
                warnings.warn_explicit('Aegean catalogue not created. Check output from Aegean.\n',UserWarning,WARN,cf.f_lineno)
        else:
            print "'{0}' already exists. Skipping Aegean.".format(self.cat_comp)

        #Run AeRes when Aegean catalogue exists to produce fitted model and residual
        if write:
            if (not os.path.exists(self.residual) and os.path.exists(self.cat_comp)) or redo:
                print "----------------------------------------"
                print "| Running AeRes for model and residual |"
                print "----------------------------------------"

                command = 'AeRes -f {0} -c {1} -r {2} -m {3}'.format(self.filepath,self.cat_comp,self.residual,self.model)
                print "Running AeRes for residual and model images with following command:"
                print command
                os.system(command)
            else:
                print "'{0}' already exists. Skipping AeRes.".format(self.residual)


class catalogue(object):

    def __init__(self,filename,name,image=None,frequency=1400,search_rad=10.0,finder=None,ra_col='ra',dec_col='dec',ra_fmt='deg',dec_fmt='deg',
                 flux_col='int_flux',flux_err_col='err_int_flux',peak_col='peak_flux',peak_err_col='err_peak_flux',rms_val='local_rms',flux_unit='Jy',
                 use_peak=False,island_col='island',flag_col='flags',maj_col='a',SNR=5.0,col_suffix='',sep='\t',basename=None,autoload=True,verbose=False):

        """Initialise a catalogue object.

        Arguments:
        ----------
        filename : string
            The path to a fits, csv, xml or 'sep'-delimited catalogue.
        name : string
            A unique short-hand name for the catalogue (e.g. 'NVSS'). This will act as the key for
            a number of dictionaries of the key catalogue fields, including the positions and fluxes.

        Keyword arguments:
        ------------------
        image : radio_image
            A radio image object used to initialise certain fields.
        freq : float
            The frequency of this image in MHz.
        search_rad : float
            The search radius in arcsec to use when cross-matching this catalogue.
        finder : string
            The source finder that produced this catalogue (if known) ['Aegean' | 'Selavy' | 'pybdsm'].
            This sets all the column names (including rms) and flux units to their defaults for that source finder.
        ra_col : string
            The name of the RA column.
        dec_col : string
            The name of the DEC column.
        ra_fmt : string
            The format of the RA column, input to SkyCoord (e.g. 'deg' or 'hour').
        dec_fmt : string
            The format of the DEC column, input to SkyCoord.
        flux_col : string
            The name of the integrated flux column.
        flux_err_col : string
            The name of the integrated flux error column. Use None if this doesn't exist, and 10% errors will be assumed.
        peak_col : string
            The name of the peak flux column (if any). Use None if this doesn't exist and it won't be used.
        peak_err_col : string
            The name of the integrated flux error column. Use None if this doesn't exist, and 10% errors will be assumed.
        rms_val : string or float
            The name of the local rms column, or a fixed value across the whole catalogue. The units must be the same as the flux columns.
        flux_unit : string
            The (case-insensitive) units of all the flux columns ['Jy' | 'mJy' | 'uJy'].
        use_peak : bool
            Use the peak flux instead of the integrated flux.
        island_col : string
            The name of the island column (if any).
        flag_col : string
            The name of the flag column (if any).
        maj_col : string
            The name of the fitted major axis column (if any). This is assumed to be in arcsec.
        SNR : float
            The minimum signal-to-noise ratio of the input catalogue.
        col_suffix : string
            A suffix to add to the end of all column names (e.g. '_deep' for GLEAM).
        sep : string
            When reading in a delimited file, use this delimiter (not needed for csv file).
        basename : string
            The base of the name to use for all output catalogues. Use None to use the same as paramater 'name'.
        autoload : bool
            Look for files already processed and load these if they exist.
        verbose : bool
            Verbose output.

        See Also
        --------
        astropy.coordinates.SkyCoord
        pandas.DataFrame"""

        print "--------------------------"
        print "| Reading {0} catalogue |".format(name)
        print "--------------------------"

        self.verbose = verbose
        self.name = name
        self.filename = filename.split('/')[-1]
        self.image = image
        self.SNR = SNR

        #set basename
        self.basename = basename
        if self.basename is None:
            self.basename = name

        #set names of all output catalogues
        self.cutout_name = "{0}_cutout.csv".format(self.basename)
        self.filtered_name = "{0}_filtered.csv".format(self.basename)
        self.si_name = ''
        si_files = glob.glob("{0}*_si.csv".format(self.basename))
        if len(si_files) > 0:
            self.si_name = si_files[0] #this is a guess, but is updated later if doesn't exist

        #look for files already processed in order of preference
        fileFound = False
        if autoload:
            fileFound = True
            if os.path.exists(self.si_name):
                filename = self.si_name
                print "Spectral index catalogue already exists. Using file '{0}'".format(filename)
            elif os.path.exists(self.filtered_name):
                filename = self.filtered_name
                print "Filtered catalogue already exists. Using file '{0}'".format(filename)
            elif os.path.exists(self.cutout_name):
                filename = self.cutout_name
                print "Cutout catalogue already exists. Using file '{0}'".format(filename)
            else:
                fileFound = False

        #Convert file to pandas data frame
        self.df = self.cat2df(filename,sep,verbose=True)

        #Read frequency and search radius from image object if exists, otherwise from input paramaters
        if self.image is not None:
            self.frequency=self.image.freq
            self.search_rad=3*self.image.posErr
        else:
            self.frequency = frequency
            self.search_rad = search_rad

        #To ensure unique column names, append catalogue name to beginning of column names when not already done
        if not fileFound:
            self.df.columns = '{0}_'.format(self.name) + self.df.columns

        self.col_suffix = col_suffix
        if finder is None:
            self.finder = None
        else:
            self.finder = finder.lower()
        self.knownFinder = (self.finder in ['aegean','selavy','pybdsm'])

        if self.finder is not None and not self.knownFinder:
            warnings.warn_explicit("Unrecognised source finder: {0}. Use 'Aegean', 'Selavy' or 'pybdsm'\n".format(finder),UserWarning,WARN,cf.f_lineno)

        if self.finder is not None and self.finder == 'selavy':
            if self.verbose:
                print 'Using default configuration for Selavy.'

            #set default column names for Selavy, appending catalogue name to beginning
            self.flux_col=self.unique_col_name('flux_int')
            self.flux_err_col=self.unique_col_name('flux_int_err')
            self.peak_col=self.unique_col_name('flux_peak')
            self.peak_err_col=self.unique_col_name('flux_peak_err')
            self.rms_val=self.unique_col_name('rms_image')
            self.island_col=self.unique_col_name('island_id')
            self.flag_col=self.unique_col_name('fit_is_estimate')
            self.maj_col=self.unique_col_name('maj_axis')
            self.ra_col=self.unique_col_name('ra_deg_cont')
            self.dec_col=self.unique_col_name('dec_deg_cont')
            self.ra_fmt='deg'
            self.dec_fmt='deg'
            self.flux_unit='mjy'
            self.si_col=self.unique_col_name('spectral_index')

        elif self.finder is not None and self.finder == 'pybdsm':
            if self.verbose:
                print 'Using default configuration for pybdsm.'

            #set default column names for pybdsm, appending catalogue name to beginning
            self.flux_col=self.unique_col_name('Total_flux')
            self.flux_err_col=self.unique_col_name('E_Total_flux')
            self.peak_col=self.unique_col_name('Peak_flux')
            self.peak_err_col=self.unique_col_name('E_Peak_flux')
            self.rms_val=self.unique_col_name('Isl_rms')
            self.island_col=self.unique_col_name('Isl_id')
            self.flag_col=None
            self.maj_col=self.unique_col_name('DC_Maj')
            self.ra_col=self.unique_col_name('RA')
            self.dec_col=self.unique_col_name('DEC')
            self.ra_fmt='deg'
            self.dec_fmt='deg'
            self.flux_unit='jy'
            self.si_col=None

        else:
            if self.finder is not None and self.finder == 'aegean':
                if self.verbose:
                    print 'Using default configuration for Aegean.'

            #append catalogue name to beginning of all columns
            self.flux_col=self.unique_col_name(flux_col)
            self.flux_err_col=self.unique_col_name(flux_err_col)
            self.peak_col=self.unique_col_name(peak_col)
            self.peak_err_col=self.unique_col_name(peak_err_col)
            self.island_col=self.unique_col_name(island_col)
            self.flag_col=self.unique_col_name(flag_col)
            self.maj_col=self.unique_col_name(maj_col)
            if type(rms_val) is str:
                self.rms_val=self.unique_col_name(rms_val)
            else:
                self.rms_val = rms_val
            self.flux_unit = flux_unit.lower()

            #specific fix for GLEAM catalogue
            if finder is not None and finder.lower() == 'aegean':
                self.col_suffix = ''

            self.ra_col=self.unique_col_name(ra_col)
            self.dec_col=self.unique_col_name(dec_col)
            self.ra_fmt = ra_fmt
            self.dec_fmt = dec_fmt
            self.si_col=None

        self.use_peak = use_peak

        if self.flux_unit not in ('jy','mjy','ujy'):
            warnings.warn_explicit("Unrecognised flux unit '{0}'. Please use 'Jy', 'mJy' or 'uJy'.Assuming 'Jy'\n".format(flux_unit),UserWarning,WARN,cf.f_lineno)
            self.flux_unit = 'jy'

        #keep a running list of key values for all sources, as a dictionary with key 'name'
        self.cat_list = [self.name]
        self.freq = {self.name : self.frequency}
        self.radius = {self.name : self.search_rad}
        self.count = {self.name : len(self.df)}
        self.coords = {}
        self.ra = {}
        self.dec = {}
        self.flux = {}
        self.flux_err = {}
        self.rms = {}
        self.sep = {}
        self.dRAsec = {}
        self.dRA = {}
        self.dDEC = {}
        self.si = {}

        #Initialise all the key fields, including positions and fluxes.
        self.set_key_fields(set_coords=False)

    def unique_col_name(self,col):

        """Return a unique column name by appending the catalogue name to the beginning. If column is None, return None.

        Arguments:
        ----------
        col : string
            Column name.

        Returns:
        --------
        col : string
            Unique column name or None."""

        if col is not None:
            col = '{0}_{1}{2}'.format(self.name,col,self.col_suffix)
        return col

    def cat2df(self,filepath,sep,verbose=False):

        """Return a pandas dataframe of the provided catalogue.
        If a '.fits' or '.csv' file isn't given, a file delimited by 'sep' will be assumed.

        Arguments:
        ----------
        filepath : string
            The absolute path to the catalogue.

        Keyword arguments:
        ------------------
        sep : string
            Delimiter for delimited files.
        verbose : bool
            Verbose output.

        Returns:
        --------
        df : pandas.DataFrame
            A pandas dataframe of the catalogue.

        See Also
        --------
        pandas.DataFrame"""

        if verbose:
            print "Loading '{0}' catalogue into pandas.".format(filepath.split('/')[-1])

        #convert from fits or xml to dataframe
        extn = filepath.split('.')[-1].lower()
        if extn == 'fits' or extn == 'gz':
            table = f.open(filepath)[1]
            df = Table(data=table.data).to_pandas()
        elif extn == 'xml': #assumed to come from Selavy
            table = parse_single_table(filepath).to_table(use_names_over_ids=True)
            df = table.to_pandas()
        #otherwise, load straight into pandas as csv or 'sep' delimited file
        elif extn == 'csv':
            df = pd.read_csv(filepath)
        else:
            df = pd.read_table(filepath,sep=sep)

        if verbose:
            print "Catalogue contains {0} sources.".format(len(df))

        return df

    def set_specs(self,img):

        """Set the key fields of this catalogue using an input image. This must be done before the catalogue is filtered.

        img : radio_image
            A radio_image object corresponding to this catalogue, which is used to calculate various quantities."""

        #Get area of non-nan pixels of image
        self.area = img.area
        self.ra_bounds = img.ra_bounds
        self.dec_bounds = img.dec_bounds

        #Get dynamic range between image and rms map and sum of image flux vs. total catalogue flux
        rms_map = f.open(img.rms_map)[0]
        img_data = img.fits.data
        if self.finder == 'aegean':
            img_data = img_data[0][0]
        self.img_peak = np.max(img_data[~np.isnan(img_data)])
        self.rms_bounds = rms_map.data > 0
        self.img_rms = int(np.median(rms_map.data[self.rms_bounds])*1e6) #uJy
        self.img_peak_bounds = np.max(img_data[self.rms_bounds])
        self.img_peak_pos = np.where(img_data == self.img_peak_bounds)
        self.img_peak_rms = rms_map.data[self.img_peak_pos][0]
        self.dynamic_range = self.img_peak_bounds/self.img_peak_rms
        self.blends = len(np.where(self.df[self.island_col].value_counts() > 1)[0])
        self.img_flux = np.sum(img_data[~np.isnan(img_data)]) / (1.133*((img.bmaj * img.bmin) / (img.raPS * img.decPS))) #divide by beam area
        self.cat_flux = np.sum(self.flux[self.name])
        if self.name in self.si.keys():
            self.med_si = np.median(self.si[self.name])
        else:
            self.med_si = -99
        if self.verbose:
            print "Sum of flux in image is {0:.3f} Jy and sum of all fitted gaussians is {1:.3f} Jy.".format(self.img_flux,self.cat_flux)
            print "Image peak is {0:.2f} Jy.".format(self.img_peak)
            print "Dynamic range is {0:.0E}.".format(self.dynamic_range)
            print "Number of multi-component islands is {0}.".format(self.blends)
            if self.med_si != -99:
                print "Median spectral index is {0:.2f}.".format(self.med_si)

        #Store the initial length of the catalogue
        self.initial_count = self.count[self.name]

        #Derive the fraction of resolved sources as the fraction of sources with int flux > 3-sigma away from peak flux
        if self.name in self.flux.keys():
            sigma = np.abs(self.df[self.flux_col] - self.df[self.peak_col]) / np.sqrt(self.df[self.flux_err_col]**2 + self.df[self.peak_err_col]**2)
            self.resolved_frac = len(np.where(sigma > 3)[0]) / len(self.df)
        else:
            self.resolved_frac = -1


    def set_key_fields(self,indices=None,cat=None,set_coords=True):

        """Set the key fields, including the positions, frequency and fluxes. This must be run
        each time the dataframe is updated. Each field is a dictionary with key catalogue.name.

        Keyword Arguments
        -----------------
        indices : list
            A list of indices from this instance to subselect after removing rows. If indices is None, all indices of this instance will be used.
        cat : catalogue
            Another catalogue object used to initialise the fields. If None is provided, this instance will be used.
        set_coords : bool
            Create a list of SkyCoord objects for every source. As this is a time-consuming task, it is
            only recommended after a cutout has been applied. This only applies when cat is None.

        See Also
        --------
        astropy.coordinates.SkyCoord
        pandas.Series"""

        coords_set = False

        #after cross-match, add new coordinates, positions and fluxes, etc to dictionaries
        if cat is not None:

            #set easy names for columns
            prefix = '{0}_{1}_'.format(cat.name,self.name)
            sep = prefix + 'sep'
            dRAsec = prefix + 'dRAsec' #in seconds
            dRA = prefix + 'dRA' #in arcsec
            dDEC = prefix + 'dDEC'

            self.cat_list.append(cat.name)
            self.count[cat.name] = len(np.where(~np.isnan(cat.df[sep]))[0]) #length of indices that aren't nan
            self.freq[cat.name] = cat.freq[cat.name]
            self.radius[cat.name] = cat.radius[cat.name]
            self.ra[cat.name] = cat.ra[cat.name]
            self.dec[cat.name] = cat.dec[cat.name]

            #compute the positional offsets
            self.df[dRAsec] = (self.ra[self.name] - self.ra[cat.name])*3600 #in seconds
            self.df[dRA] = self.df[dRAsec]*np.cos(np.deg2rad((self.dec[self.name] + self.dec[cat.name])/2)) #in arcsec
            self.df[dDEC] = (self.dec[self.name] - self.dec[cat.name])*3600

            #store in dictionaries
            self.sep[cat.name] = cat.df[sep]
            self.dRAsec[cat.name] = self.df[dRAsec]
            self.dRA[cat.name] = self.df[dRA]
            self.dDEC[cat.name] = self.df[dDEC]

            if cat.name in cat.flux.keys():
                self.flux[cat.name] = cat.flux[cat.name]
                self.flux_err[cat.name] = cat.flux_err[cat.name]
                self.rms[cat.name] = cat.rms[cat.name]

                #write flux ratio if frequencies within 1%
                if self.name in self.flux.keys() and np.abs(cat.freq[cat.name]/self.freq[self.name]-1) < 0.01:
                    self.df[prefix + 'flux_ratio'] = self.flux[self.name]/self.flux[cat.name]

            if cat.si_col != None:
                self.si[cat.name] = cat.si[cat.name]

        #otherwise initialise or update dictionary for this instance
        else:
            if set_coords or (indices is not None and len(self.coords) == 0):

                #initialise SkyCoord object for all sky positions and create numpy array for RA and DEC in degrees
                self.coords[self.name] = SkyCoord(ra = self.df[self.ra_col], dec = self.df[self.dec_col],unit='{0},{1}'.format(self.ra_fmt,self.dec_fmt))
                self.ra[self.name] = self.coords[self.name].ra.deg
                self.dec[self.name] = self.coords[self.name].dec.deg
                coords_set = True

            #initliase fluxes
            if indices is None and not (self.flux_col is None and self.peak_col is None):

                #Create pandas series for peak or integrated flux and errors
                if self.use_peak and self.peak_col is None:
                    warnings.warn_explicit("Can't use peak flux since peak column name not specified. Using integrated flux.\n",UserWarning,WARN,cf.f_lineno)
                if self.use_peak and self.peak_col is not None:
                    self.flux[self.name] = self.df[self.peak_col].copy()

                    if self.peak_err_col != None:
                        self.flux_err[self.name] = self.df[self.peak_err_col].copy()
                    else:
                        self.flux_err[self.name] = self.flux[self.name]*0.1 #10% error
                else:
                    self.flux[self.name] = self.df[self.flux_col].copy()
                    if self.flux_err_col != None:
                        self.flux_err[self.name] = self.df[self.flux_err_col].copy()
                    else:
                        self.flux_err[self.name] = self.flux[self.name]*0.1 #10% error

                #set 10% errors where error is <=0
                if np.any(self.flux_err[self.name] <= 0):
                    i = np.where(self.flux_err[self.name] <= 0)[0]
                    self.flux_err[self.name][i] = self.flux[self.name][i]*0.1 #10% error

                #Set rms as pandas series or single value
                if type(self.rms_val) is str:
                    self.rms[self.name] = self.df[self.rms_val].copy()
                else:
                    self.rms[self.name] = self.rms_val

                #Force Jy units
                factor = 1
                if self.flux_unit == 'mjy':
                    factor = 1e-3
                elif self.flux_unit == 'ujy':
                    factor = 1e-6
                self.flux[self.name] *= factor
                self.flux_err[self.name] *= factor
                self.rms[self.name] *= factor

                if self.si_col != None:
                    self.si[self.name] = self.df[self.si_col]

            #otherwise just update this instance
            else:
                if not coords_set:
                    self.coords[self.name] = self.coords[self.name][indices]
                    self.ra[self.name] = self.ra[self.name][indices]
                    self.dec[self.name] = self.dec[self.name][indices]

                #reset indices of pandas series
                if self.name in self.flux.keys():
                    self.flux[self.name] = self.flux[self.name][indices].reset_index(drop=True)
                    self.flux_err[self.name] = self.flux_err[self.name][indices].reset_index(drop=True)
                    if type(self.rms_val) is str:
                        self.rms[self.name] = self.rms[self.name][indices].reset_index(drop=True)

                #only update these for cross-matched catalogues
                if self.name in self.sep.keys():
                    self.sep[self.name] = self.sep[self.name][indices].reset_index(drop=True)
                    self.dRAsec[self.name] = self.dRAsec[self.name][indices].reset_index(drop=True)
                    self.dRA[self.name] = self.dRA[self.name][indices].reset_index(drop=True)
                    self.dDEC[self.name] = self.dDEC[self.name][indices].reset_index(drop=True)

                if self.name is self.si.keys():
                    self.si[self.name] = self.si[self.name][indices].reset_index(drop=True)

        #reset indices and catalogue length after change has been made
        self.df = self.df.reset_index(drop=True)
        self.count[self.name] = len(self.df)

    def overwrite_df(self,catalogue,step='this',set_coords=True,verbose=True):

        """Overwrite self.df with another file or DataFrame. All other fields are assumed to stay the same.

        Arguments:
        ----------
        catalogue : string or pandas.DataFrame
            The filename of the catalogue (must be csv file), or a pandas data frame object.
            If a dataframe is input, it is assumed to have come from a catalogue object.

        Keyword arguments:
        ------------------
        verbose : bool
            Verbose output.

        See Also:
        ---------
        pandas.DataFrame"""

        #read from file if filename is provided
        if type(catalogue) is str:
            self.df = pd.read_csv(catalogue)

            if verbose:
                print "'{0}' already exists. Skipping {1} step and setting catalogue to this file.".format(catalogue,step)
                print "{0} catalogue now contains {1} sources.".format(self.name,len(self.df))

        else:
            self.df = catalogue

        #reset the key fields
        self.set_key_fields(set_coords=set_coords)

    def write_df(self,write,filename,verbose=True):

        """Write data frame to file.

        Arguments:
        ----------
        write : bool
            Write the file.
        filename : string
            The file name.

        Keyword Arguments:
        ------------------
        verbose : bool
            Verbose output."""

        if write:
            if verbose:
                print "Writing to '{0}'.".format(filename)
            self.df.to_csv(filename,index=False)

    def cutout_box(self,ra,dec,fov=10,redo=False,write=True,verbose=True):

        """Cut out a box of the catalogue, updating the catalogue to only contain sources within this box.
        Input a central RA and DEC and FOV or four vertices.

        Arguments:
        ----------
        ra : float or tuple
            The RA centre in degrees, or a tuple of two RA boundaries.
        dec : float
            The DEC centre in degrees, or a tuple of two DEC boundaries.

        Keyword arguments:
        ------------------
        fov : float
            The field of view in degrees (i.e. fov*fov degrees). Only used when RA & DEC are single values.
        redo : bool
            Cut out a box, even if a cutout file exists.
        write : bool
            Write the cutout catalogue to file.
        verbose : bool
            Verbose output."""

        filename = self.cutout_name

        #if column units aren't degrees, set sky coords and get RA/DEC in degrees
        if self.ra_fmt != 'deg' or self.dec_fmt != 'deg':
            if len(self.coords) == 0:
                self.set_key_fields(set_coords=True)
            RA = self.ra[self.name]
            DEC = self.dec[self.name]
        #otherwise just use column values
        else:
            RA = self.df[self.ra_col]
            DEC = self.df[self.dec_col]

        if (type(ra) is tuple and (type(dec) is not tuple or len(ra) != 2)) or \
        (type(dec) is tuple and (type(ra) is not tuple or len(dec) != 2)):
            warnings.warn_explicit('RA and DEC must both be single value or a tuple with two indices. Cutout not applied.\n',UserWarning,WARN,cf.f_lineno)

        elif redo or not os.path.exists(filename):
            if verbose:
                if redo:
                    print "Re-doing cutout step."
                    if write:
                        print "Overwriting '{0}'.".format(filename)
                print "Cutting out sources from {0}.".format(self.name)

            #cut out all rows outside RA and DEC boundaries
            if type(ra) is tuple:
                ra_min,ra_max,dec_min,dec_max = axis_lim(ra,min),axis_lim(ra,max),axis_lim(dec,min),axis_lim(dec,max)
                if verbose:
                    print "Selecting sources with {0} <= RA <= {1} and {2} <= DEC <= {3}.".format(ra_min,ra_max,dec_min,dec_max)
                self.df = self.df[(RA <= ra_max) & (RA >= ra_min) & (DEC <= dec_max) & (DEC >= dec_min)]

            #cut out all rows outside the FOV
            else:
                if verbose:
                    print "Using a {0}x{0} degree box centred at {1} deg, {2} deg.".format(fov,ra,dec)
                self.df = self.df[(DEC <= dec + fov/2) & (DEC >= dec - fov/2) &
                                    (RA <= ra + fov/2/np.cos(np.deg2rad(DEC))) &
                                    (RA >= ra - fov/2/np.cos(np.deg2rad(DEC)))]

            if verbose:
                print '{0} {1} sources within this region.'.format(len(self.df),self.name)

            #Drop the rejected rows, reset the key fields and write to file
            self.set_key_fields(indices=self.df.index.tolist())
            self.write_df(write,filename)

        #if file exists, simply read in catalogue
        else:
            self.overwrite_df(filename,step='cutout',set_coords=False)


    def filter_sources(self,flux_lim=0,SNR=0,ratio_frac=0,reject_blends=False,psf_tol=0,resid_tol=0,flags=False,
                        file_suffix='',redo=False,write=True,verbose=False):

        """Reject problematic sources according to several criteria. This will overwrite self.df.

        Keyword arguments:
        ------------------
        flux_lim : float
            The flux density limit in Jy, below which sources will be rejected.
        SNR : float
            The S/N ratio limit (where N is the rms), below which sources will be rejected. Use 0 to skip this step.
        ratio_frac : float
            The fraction given by the integrated flux divided by the peak flux, above which, sources will be rejected. Use 0 to skip this step.
        reject_blends : bool
            For Aegean and Selavy only. Reject multi-component sources.
        psf_tol : float
            For Aegean and Selavy only. Reject sources with fitted major axis this many times the psf major axis. Use 0 to skip this step.
        resid_tol : float
            For Aegean only. Reject sources with a std in the residual this many times the rms. Use 0 to skip this step.
        flags : bool
            Reject sources with flags != 0. Only performed if flag_col set in constructor of this instance.
        file_suffix : string
            Suffix to add to filename, for when doing several steps of filtering.
        redo : bool
            Perform the filtering, even if filtered file exists.
        write : bool
            Write the filtered catalogue to file.
        verbose : bool
            Verbose output."""

        if file_suffix != '':
            self.basename += file_suffix
            filename = '{0}.csv'.format(self.basename)
        else:
            filename = self.filtered_name

        if redo or not os.path.exists(filename):
            if verbose:
                if redo:
                    print "Re-doing filtering."
                    if write:
                        print "Overwriting '{0}'.".format(filename)
                print "Filtering sources in '{0}'...".format(filename)
                print "Initial number of sources: {0}.".format(len(self.df))

            #reject faint sources
            if flux_lim != 0:
                if self.flux_col != None:
                    self.df = self.df[self.flux[self.name] > flux_lim]
                    if verbose:
                        print "Rejected (faint) sources below {0} Jy.".format(flux_lim)
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("No int flux column given. Can't reject resolved sources based on flux.\n",UserWarning,WARN,cf.f_lineno)

            #reject low S/N sources
            if SNR != 0:
                if self.flux_col != None and self.rms_val != None:
                    #reindex key fields before comparison
                    self.set_key_fields(indices=self.df.index.tolist())
                    self.df = self.df[self.flux[self.name] > SNR*self.rms[self.name]]
                    if verbose:
                        print "Rejected (low-S/N) sources < {0} x r.m.s.".format(SNR)
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("rms or int flux column not given. Can't reject resolved sources based on SNR.\n",UserWarning,WARN,cf.f_lineno)

            #reject resolved sources based on flux ratio
            if ratio_frac != 0:
                if self.peak_col != None:
                    self.df = self.df[self.df[self.flux_col]/self.df[self.peak_col] <= ratio_frac]
                    if verbose:
                        print "Rejected (resolved) sources with total flux > {0} times the peak flux.".format(ratio_frac)
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("No peak flux column given. Can't reject resolved sources using flux ratio.\n",UserWarning,WARN,cf.f_lineno)

            #reject multi-component islands
            if reject_blends:
                if self.knownFinder:
                    island_counts = self.df[self.island_col].value_counts()
                    point_islands = island_counts[island_counts == 1].index
                    self.df = self.df[self.df[self.island_col].isin(point_islands)]
                    if verbose:
                        print "Rejected (resolved) sources belonging to a multi-component island."
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("Can't reject blends since finder isn't Aegean or Selavy or pyBDSM.\n",UserWarning,WARN,cf.f_lineno)

            #reject extended components based on component size
            if psf_tol != 0:
                if self.knownFinder:
                    if self.image is not None:
                        self.df = self.df[self.df[self.maj_col] <= psf_tol*self.image.bmaj]
                    elif self.finder == 'Aegean':
                        self.df = self.df[self.df['{0}_a'.format(self.name)] <= psf_tol*self.df['{0}_psf_a'.format(self.name)]]
                    else:
                        warnings.warn_explicit("Can't rejected resolved sources based on psf tolerance without\
                        inputting radio_image object to read psf.\n",UserWarning,WARN,cf.f_lineno)
                    if verbose:
                        print "Rejected (resolved) sources with fitted major axis > {0} times the psf major axis.".format(psf_tol)
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("Can't reject sources based on PSF since finder isn't Aegean or Selavy or pyBDSM.\n",UserWarning,WARN,cf.f_lineno)

            #reject sources with poor fit
            if resid_tol != 0:
                if self.finder == 'aegean':
                    #reindex key fields before comparison
                    self.set_key_fields(indices=self.df.index.tolist())
                    self.df = self.df[self.df['{0}_residual_std'.format(self.name)] <= resid_tol*self.rms[self.name]]
                    if verbose:
                        print "Rejected (poorly fit) sources with standard deviation in residual > {0} times the rms.".format(resid_tol)
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("Can't reject resolved sources based on residual since finder isn't Aegean.\n",UserWarning,WARN,cf.f_lineno)

            #reject sources with any flags
            if flags:
                if self.flag_col != None:
                    self.df = self.df[self.df[self.flag_col] == 0]
                    if verbose:
                        print "Rejecting (problematic) sources flagged as bad."
                        print "Number remaining: {0}.".format(len(self.df))
                else:
                    warnings.warn_explicit("Can't reject resolved sources based on flag since flag column not set.\n",UserWarning,WARN,cf.f_lineno)

            #Drop the rejected rows, reset the key fields and write to file
            self.set_key_fields(indices=self.df.index.tolist())
            self.write_df(write,filename)

        #if file exists, simply read in catalogue
        else:
            self.overwrite_df(filename,step='filtering',verbose=verbose)


    def cross_match(self,cat,radius='largest',join_type='1',redo=False,write=True):

        """Perform a nearest neighbour cross-match between this catalogue object and another catalogue object.
        This will set update this object's catalogue to the matched catalogue and add the key fields to the key field dictionaries.

        Arguments:
        ----------
        cat : catalogue
            A catalogue object to cross-match this instance to.

        Keyword arguments:
        ------------------
        radius : string or float
            The search radius in arcsec. Use 'largest' to use the larger of the two default radii.
        join_type : string
            The join type of the two catalogues. '1' to keep all rows from this instance, or '1and2' to keep only matched rows.
        redo : bool
            Perform the cross-matching, even if cross-matched file exists.
        write : bool
            Write the cross-matched catalogue to file."""

        if cat.name in self.cat_list:
            warnings.warn_explicit("You've already cross-matched to {0}. Catalogue unchanged.\n".format(cat.name),UserWarning,WARN,cf.f_lineno)
            return

        filename = '{0}_{1}.csv'.format(self.basename,cat.name)

        #Cross-match and create file if it doesn't exist, otherwise open existing file
        if redo or not os.path.exists(filename):
            if len(self.df) == 0 or len(cat.df) == 0:
                if self.verbose:
                    print 'No {0} sources to match. Catalogue unchanged.'.format(cat.name)
                return

            print "---------------------------------"
            print "| Cross-matching {0} and {1} |".format(self.name,cat.name)
            print "---------------------------------"

            if redo:
                print "Re-doing cross-match."
                if write:
                    print "Overwriting '{0}'.".format(filename)
            print "Cross-matching {0} {1} sources with {2} {3} sources.".format(len(self.df),self.name,len(cat.df),cat.name)

            #force coordinates to be set
            if len(self.coords) == 0:
                self.set_key_fields()
            if len(cat.coords) == 0:
                cat.set_key_fields()

            #get sky coordinates and find nearest match of every source
            #from this instance, independent of the search radius
            c1 = self.coords[self.name]
            c2 = cat.coords[cat.name]
            indices,sep,sep3d = c1.match_to_catalog_sky(c2)

            #take the maximum radius from the two
            if radius == 'largest':
                radius = max(self.search_rad,cat.search_rad)
                if self.verbose:
                    print 'Using the largest of the two search radii of {0} arcsec.'.format(radius)

            #only take matched rows from cat, so self.df and cat.df are parallel
            cat.df = cat.df.iloc[indices].reset_index(drop=True)

            #create pandas dataframe of the separations in arcsec
            sep_col = '{0}_{1}_sep'.format(cat.name,self.name)
            sepdf = pd.DataFrame(data = {sep_col : sep.arcsec})

            #only take matches within search radius
            indices = np.where(sep.arcsec < radius)[0]

            #only add cross-matched table when at least 1 match
            if len(indices) >= 1:
                print "Found {0} matches within {1} arcsec.".format(len(indices),radius)

                #don't reset indices so cat.df stays parallel with self.df
                cat.df = cat.df.iloc[indices]
                sepdf = sepdf.iloc[indices]

                #concatenate tables together according to match type
                if join_type == '1and2':
                    matched_df = pd.concat([self.df,cat.df,sepdf],axis=1,join='inner')
                elif join_type == '1':
                    matched_df = pd.concat([self.df,cat.df,sepdf],axis=1,join_axes=[self.df.index])

                #reset indices and overwrite data frame with matched one
                matched_df = matched_df.reset_index(drop=True)

                #set catalogue to matched df and write to file
                self.overwrite_df(matched_df)
                self.write_df(write,filename)

            else:
                print '{0} cross-matches between {1} and {2}. Catalogue unchanged.'.format(len(indices),self.name,cat.name)
                return

        #if file exists, simply read in catalogue
        else:
            print "'{0}' already exists. Skipping cross-matching step.".format(filename)
            print 'Setting catalogue to this file.'
            matched_df = pd.read_csv(filename)

        #update basename to this cross-matched catalogue
        self.basename = filename[:-4]

        #overwrite the df so unmatched rows (set to nan) are included
        cat.overwrite_df(matched_df)

        #add key fields from matched catalogue
        self.set_key_fields(cat=cat)


    def fit_spectra(self,cat_name=None,match_perc=0,models=['pow'],fig_extn=None,GLEAM_subbands=None,GLEAM_nchans=None,fit_flux=False,redo=False,write=True):

        """Derive radio spectra for this catalogue, using the input SED models. This will add new columns to the table, including the spectral index
        and error, and optionally, the fitted flux at the frequency of this instance and the ratio between this and the measured flux.

        Keyword arguments:
        ------------------
        cat_name : string
            Derive spectral indices between the catalogue given by this dictionary key and the main catalogue from this instance. If None is input,
            all data except the catalogue from this instance will be used, and the flux at its frequency will be derived. If 'all' is input, all data will be used.
        match_perc : float
            Don't use the fluxes from a catalogue if the number of sources is less than this percentage of the main catalogue.
            Only used when cat_name is None. Use 0 to accept all catalogues.
        models : list
            A list of strings corresponding to SED models to attempt to fit to the radio spectra.
        fig_extn : string
            Write figures of the SEDs with this extension. Use None to not write figures.
        GLEAM_subbands : string
            Use all GLEAM sub-band measurements, if GLEAM cross-matched. Input 'int' for integrated fluxes,
            'peak' for peak fluxes, and None to use none.
        GLEAM_nchans : int
            Average together this many 8 MHz GLEAM sub-bands. Use None for no averaging.
        fit_flux : bool
            Use all cross-matched catalogues to derive a fitted flux. If False, a typical spectral index of -0.8 will be assumed.
        redo : bool
            Derive spectral indices, even if the file exists and the spectral indices have been derived from these frequencies.
        write : bool
            Write the spectral index catalogue to file."""

        #update filename (usually after cross-match)
        if not self.basename.endswith('si') and cat_name is None:
            self.basename += '_si'
        self.si_name = "{0}.csv".format(self.basename)
        filename = self.si_name

        #if file exists, simply read in catalogue
        if not redo and self.basename.endswith('si') and os.path.exists(filename):
            self.overwrite_df(filename,step='spectral index')

        #when no cat name is given, use all available fluxes
        if cat_name in [None,'all']:

            print "-----------------------------"
            print "| Deriving spectral indices |"
            print "-----------------------------"

            #derive the number of frequencies used to measure spectral
            #indices, and store these for column names and for output
            num_cats,max_count = 0,0
            used_cats,max_cat = '',''

            #derive column names based on main catalogue
            freq = int(round(self.frequency))
            suffix = '_{0}MHz'.format(freq)
            fitted_flux_suffix = '_fitted{0}_flux'.format(suffix)
            fitted_ratio_suffix = '_fitted{0}_{1}_flux_ratio'.format(suffix,self.name)
            best_fitted_flux = 'best' + fitted_flux_suffix
            best_fitted_ratio = 'best' + fitted_ratio_suffix

            for cat in self.cat_list:
                count = self.count[cat]

                #Catalogue not considered useful if less than input % of sources have matches
                if cat != self.name and count > (match_perc/100)*self.count[self.name]:
                    num_cats += 1
                    used_cats += "{0}, ".format(cat)

                    #store the largest used catalogue
                    if count > max_count:
                        max_cat = cat
                        max_count = count

                    #derive extrapolated flux for each catalogue
                    fitted_flux = '{0}_extrapolated{1}_flux'.format(cat,suffix)
                    fitted_ratio = '{0}_extrapolated{1}_{2}_flux_ratio'.format(cat,suffix,self.name)
                    self.est_fitted_flux(fitted_flux,fitted_ratio,freq,cat)

            #don't derive spectral indices if there aren't 2+ catalogues to use,
            #but just derive flux at given frequency from a typical spectral index
            if (num_cats <= 1 or not fit_flux) and self.name in self.flux.keys() and (redo or best_fitted_flux not in self.df.columns):
                self.est_fitted_flux(best_fitted_flux,best_fitted_ratio,freq,max_cat)

            #otherwise, derive the spectral index and fitted flux using all available frequencies
            elif num_cats > 1 and (redo or best_fitted_flux not in self.df.columns):
                self.n_point_spectra(fitted_flux_suffix,fitted_ratio_suffix,best_fitted_flux,best_fitted_ratio,used_cats,freq,
                                     models=models,fig_extn=fig_extn,GLEAM_subbands=GLEAM_subbands,GLEAM_nchans=GLEAM_nchans,redo=redo)

        #otherwise derive the spectral index between this instance
        #and the given catalogue, if any cross-matches were found
        elif cat_name in self.cat_list:
            self.two_point_spectra(cat_name,redo)

        #write catalogue to file
        self.write_df(write,filename)

    def est_fitted_flux(self,fitted_flux_col,fitted_ratio_col,freq,cat,spec_index=-0.8):

        """Using a typical spectral index, derive the flux at the frequency of this
        catalogue instance, and the ratio between this and its measured value.

        Arguments:
        ----------
        fitted_flux_col : string
            The name of the fitted flux column to which to append the estimated flux.
        fitted_ratio_col : string
            The name of the fitted ratio column to which to append the estimated ratio.
        freq : string
            The frequency at which the flux is derived.
        cat : string
            The catalogue to use for extrapolating the spectral index.

        Keyword arguments:
        ------------------
        spec_index : float
            The assumed spectral index."""

        if self.verbose:
            print "Deriving the flux at {0} MHz asumming a typical spectral index of {1}.".format(freq,spec_index)

        self.df[fitted_flux_col] = flux_at_freq(self.freq[self.name],self.freq[cat],self.flux[cat],spec_index)
        self.df[fitted_ratio_col] = self.df[fitted_flux_col] / self.flux[self.name]

    def two_point_spectra(self,cat_name,redo=False):

        """Derive the spectral index, uncertainty and fitted flux between two frequencies.

        Arguments:
        ----------
        cat_name : string
            Derive spectral indices between the catalogue given by this dictionary key and the catalogue from this instance.

        Keyword arguments:
        ------------------
        redo : bool
            Derive spectral indices, even if the file exists and the spectral indices have been derived from these frequencies."""

        alpha_col = '{0}_{1}_alpha'.format(self.name,cat_name)
        alpha_err_col = '{0}_err'.format(alpha_col)

        if redo or alpha_col not in self.df.columns:
            #don't derive spectral indices if frequencies are <10% apart
            if np.abs(self.freq[cat_name]/self.freq[self.name]-1) <= 0.1:
                print "{0} and {1} are too close to derive spectral indices.".format(self.name,cat_name)
            else:
                print "Deriving spectral indices between {0} and {1}.".format(self.name,cat_name)
                self.df[alpha_col],self.df[alpha_err_col],flux = two_freq_power_law(self.freq[self.name],
                                                                                    [self.freq[self.name],self.freq[cat_name]],
                                                                                    [self.flux[self.name],self.flux[cat_name]],
                                                                                    [self.flux_err[self.name],self.flux_err[cat_name]])


    def n_point_spectra(self,fitted_flux_suffix,fitted_ratio_suffix,best_fitted_flux,best_fitted_ratio,used_cats,freq,
                        cat_name=None,models=['pow'],fig_extn=None,GLEAM_subbands=None,GLEAM_nchans=None,redo=False):

        """Derive the radio spectra from the input SED models, using the specified data, presumed to be >2 frequency measurements.

        Arguments:
        ----------
        fitted_flux_suffix : string
            The suffix for the column name to store the fitted flux for each model.
        fitted_ratio_suffix : string
            The suffix for the column name to store the fitted flux ratio for each model.
        best_fitted_flux : string
            The name of the column to store the best fitted flux.
        best_fitted_ratio : string
            The name of the column to store the best fitted flux ratio.
        used_cats : string
            The catalogues used to calculate the SEDs (for output).
        freq : string
            The frequency at which the flux is derived.

        Keyword arguments:
        ------------------
        cat_name : string
            If None is input, all data except the catalogue from this instance will be used, and the flux at the frequency
            given by this instance will be derived from these spectral indices. If 'all' is input, all data will be used.
        models : list
            A list of strings corresponding to SED models to fit to the radio spectra.
        fig_extn : string
            Write figures of the SEDs with this extension. Use None to not write figures.
        GLEAM_subbands : string
            Use all GLEAM sub-band measurements, if GLEAM cross-matched. Input 'int' for integrated fluxes,
            'peak' for peak fluxes, and None to use none.
        GLEAM_nchans : int
            Average together this many 8 MHz GLEAM sub-bands. Use None for no averaging.
        redo : bool
            Derive spectral indices, even if the file exists and the spectral indices have been derived from these frequencies."""

        print "Deriving SEDs using following catalogues: {0}.".format(used_cats[:-2])
        print "Deriving the flux for each model at {0} MHz.".format(freq)

        for col in [best_fitted_flux,best_fitted_ratio]:
            self.df[col] = np.full(len(self.df),np.nan)

        if fig_extn is not None and self.verbose:
            print "Writting SED plots to 'SEDs/'"

        #iterate through all sources and derive SED model where possible
        for i in range(len(self.df)):
            fluxes,errs,freqs = np.array([]),np.array([]),np.array([])

            #iterate through all catalogues and only take fluxes
            #that aren't nan and optionally don't include main catalogue
            for cat in self.flux.keys():
                flux = self.flux[cat].iloc[i]
                if not np.isnan(flux) and (cat != self.name or cat_name == 'all'):
                    fluxes=np.append(fluxes,flux)
                    errs=np.append(errs,self.flux_err[cat].iloc[i])
                    freqs=np.append(freqs,self.freq[cat])

            #append GLEAM sub-band measurements according to input type (int or peak)
            if GLEAM_subbands is not None and 'GLEAM' in self.cat_list:
                GLEAM_freqs,GLEAM_fluxes,GLEAM_errs = np.array([]),np.array([]),np.array([])
                for col in self.df.columns:
                    if col.startswith('GLEAM_{0}_flux_'.format(GLEAM_subbands)) and 'fit' not in col and 'wide' not in col and not np.isnan(self.df.loc[i,col]):
                        GLEAM_freq = col.split('_')[-1]
                        GLEAM_freqs = np.append(GLEAM_freqs,float(GLEAM_freq))
                        GLEAM_fluxes = np.append(GLEAM_fluxes,self.df.loc[i,'GLEAM_{0}_flux_{1}'.format(GLEAM_subbands,GLEAM_freq)])
                        GLEAM_errs = np.append(GLEAM_errs,self.df.loc[i,'GLEAM_err_{0}_flux_{1}'.format(GLEAM_subbands,GLEAM_freq)])

                #optionally average sub-bands together
                if GLEAM_nchans is not None:
                    index=0
                    used_index=0
                    while index < len(GLEAM_freqs) and GLEAM_freqs[index+GLEAM_nchans] <= 174:
                        GLEAM_freqs[used_index] = GLEAM_freqs[index:index+GLEAM_nchans].mean()
                        GLEAM_fluxes[used_index] = GLEAM_fluxes[index:index+GLEAM_nchans].mean()
                        GLEAM_errs[used_index] = np.sqrt(np.sum(GLEAM_errs[index:index+GLEAM_nchans]**2)) / GLEAM_nchans
                        index += GLEAM_nchans
                        used_index += 1

                    GLEAM_freqs=GLEAM_freqs[:used_index]
                    GLEAM_fluxes=GLEAM_fluxes[:used_index]
                    GLEAM_errs=GLEAM_errs[:used_index]

                #append GLEAM measurements
                freqs = np.append(freqs,GLEAM_freqs)
                fluxes = np.append(fluxes,GLEAM_fluxes)
                errs = np.append(errs,GLEAM_errs)

            #attempt to fit models if more than one frequency
            if len(freqs) > 1:
                figname = fig_extn
                #use island ID or otherwise row index for figure name
                if figname is not None:
                    if self.island_col is not None:
                        name = self.df.loc[i,self.island_col]
                    else:
                        name = i
                    figname = '{0}.{1}'.format(name,figname)
                #fit SED models
                mods,names,params,errors,fluxes,rcs,BICs = SED(self.freq[self.name],freqs,fluxes,errs,models,figname=figname)

                #append best fitted flux and ratio
                if len(mods) > 0:
                    best_flux = fluxes[np.where(BICs == min(BICs))[0][0]]
                    self.df.loc[i,best_fitted_flux] = best_flux
                    if self.name in self.flux.keys():
                        self.df.loc[i,best_fitted_ratio] = best_flux / self.flux[self.name][i]

                #iterate through each model and append fitted parameters
                for j,model in enumerate(mods):

                    fitted_flux_col = model + fitted_flux_suffix
                    fitted_ratio_col = model + fitted_ratio_suffix
                    rcs_col = model + '_rcs'
                    BIC_col = model + '_BIC'

                    for col in [fitted_flux_col,fitted_ratio_col,rcs_col,BIC_col]:
                        if col not in self.df.columns:
                            self.df[col] = np.full(len(self.df),np.nan)

                    self.df.loc[i,fitted_flux_col] = fluxes[j]
                    if self.name in self.flux.keys():
                        self.df.loc[i,fitted_ratio_col] = fluxes[j] / self.flux[self.name][i]
                    self.df.loc[i,rcs_col] = rcs[j]
                    self.df.loc[i,BIC_col] = BICs[j]

                    for k,name in enumerate(names[j]):
                        #derive column name for each parameter
                        para_col = '{0}_{1}'.format(model,name)
                        para_err_col = '{0}_err'.format(para_col)

                        if para_col not in self.df.columns:
                            #add new columns for each parameter and uncertainty, the fitted
                            #flux and the ratio between this and the measured flux
                            self.df[para_col] = np.full(len(self.df),np.nan)
                            self.df[para_err_col] = np.full(len(self.df),np.nan)

                        #store parameter value, error, fitted flux and ratio
                        self.df.loc[i,para_col] = params[j][k]
                        self.df.loc[i,para_err_col] = errors[j][k]


    def process_config_file(self,config_file,redo=False,write_all=True,write_any=True,verbose=False):

        """For a given catalogue config file, read the paramaters into a dictionary, pass it into a
        new catalogue object, cut out a box, cross-match to this instance, and derive the spectral index.

        Arguments:
        ----------
        config_file : string
            The filepath to a configuration file for a catalogue.

        Keyword arguments:
        ------------------
        redo : bool
            Re-do all processing, even if output files produced.
        write_all : bool
            Write all files during processing. If False, cutout file will still be written.
        write_any : bool
            Write any files whatsoever?"""

        #create dictionary of arguments, append verbose and create new catalogue instance
        config_dic = config2dic(config_file,verbose=verbose)
        config_dic.update({'verbose' : verbose})
        if redo:
            config_dic['autoload'] = False
        cat = catalogue(**config_dic)

        #Cut out a box in catalogues within boundaries of image, cross-match and derive spectral indices
        cat.cutout_box(self.ra_bounds,self.dec_bounds,redo=redo,verbose=verbose,write=write_any)
        self.cross_match(cat,redo=redo,write=write_all)
        if cat.name in self.cat_list and self.name in self.flux.keys():
            self.fit_spectra(cat_name=cat.name,redo=redo,write=write_all)


class report(object):

    def __init__(self,cat,img=None,plot_to='html',css_style=None,fig_font={'fontname':'Serif', 'fontsize' : 18},fig_size={'figsize' : (8,8)},
                 label_size={'labelsize' : 12},markers={'s' : 20, 'linewidth' : 1, 'marker' : 'o', 'color' : 'b'},
                 colour_markers={'marker' : 'o', 's' : 30, 'linewidth' : 0},cmap='plasma',cbins=20,
                 arrows={'color' : 'r', 'width' : 0.05, 'scale' : 10},src_cnt_bins=50,redo=False,write=True,verbose=True):

        """Initialise a report object for writing a html report of the image and cross-matches, including plots.

        Arguments:
        ----------
        cat : catalogue
            Catalogue object with the data for plotting.

        Keyword arguments:
        ------------------
        img : radio_image
            Radio image object used to write report table. If None, report will not be written, but plots will be made.
        plot_to : string
            Where to show or write the plot. Options are:

                'html' - save as a html file using mpld3.

                'screen' - plot to screen [i.e. call plt.show()].

                'extn' - write file with this extension (e.g. 'pdf', 'eps', 'png', etc).

        css_style : string
            A css format to be inserted in <head>.
        fig_font : dict
            Dictionary of kwargs for font name and size for title and axis labels of matplotlib figure.
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
        self.cmap = plt.get_cmap(cmap,cbins)
        self.src_cnt_bins = src_cnt_bins
        self.redo = redo
        self.write = write
        self.verbose = verbose

        #set name of directory for figures and create if doesn't exist
        self.figDir = 'figures'
        if self.write and not os.path.exists(self.figDir):
            os.mkdir(self.figDir)

        #use css style passed in or default style for CASS web server below
        if css_style is not None:
            self.css_style = css_style
        else:
            self.css_style = """<?php include("base.inc"); ?>
            <meta name="DCTERMS.Creator" lang="en" content="personalName=Collier,Jordan" />
            <meta name="DC.Title" lang="en" content="ASKAP Continuum Validation Report" />
            <meta name="DC.Description" lang="en" content="ASKAP continuum validation report summarising science readiness of data via several metrics" />
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

            </style>
            <title>ASKAP Continuum Validation Report</title>"""

        #filename of html report
        self.name = 'index.html'
        #Open file html file and write css style, title and heading
        self.write_html_head()
        #write table summary of observations and image if radio_image object passed in
        if img is not None:
            self.write_html_img_table(img)
            rms_map = f.open(img.rms_map)[0]
            solid_ang = 0
        #otherwise assume area based on catalogue RA/DEC limits
        else:
            rms_map = None
            solid_ang = self.cat.area*(np.pi/180)**2

        #write source counts to report using rms map to measure solid angle or approximate solid angle
        if self.cat.name in self.cat.flux.keys():
            self.source_counts(self.cat.flux[self.cat.name],self.cat.freq[self.cat.name],rms_map=rms_map,solid_ang=solid_ang,write=self.write)
        else:
            self.sc_red_chi_sq = -99
        #write cross-match table header
        self.write_html_cross_match_table()

        #store dictionary of metrics, where they come from, how many matches they're derived from, and their level (0,1 or 2)
        #spectral index defaults to -99, as there is a likelihood it will not be needed (if Taylor-term imaging is not done)
        #RA and DEC offsets used temporarily and then dropped before final metrics computed
        key_value_pairs = [ ('Flux Ratio' , 0),
                            ('Flux Ratio Uncertainty' , 0),
                            ('Positional Offset' , 0),
                            ('Positional Offset Uncertainty' , 0),
                            ('Resolved Fraction' , self.cat.resolved_frac),
                            ('Spectral Index' , -99),
                            ('Source Counts Reduced Chi-squared' , self.sc_red_chi_sq),
                            ('RA Offset' , 0),
                            ('DEC Offset' , 0)]

        self.metric_val = collections.OrderedDict(key_value_pairs)
        self.metric_source = self.metric_val.copy()
        self.metric_count = self.metric_val.copy()
        self.metric_level = self.metric_val.copy()


    def write_html_head(self):

        """Open the report html file and write the head."""

        self.html = open(self.name,'w')
        self.html.write("""<!DOCTYPE HTML>
        <html lang="en">
        <head>
            {0}
        </head>
        <?php title_bar("atnf"); ?>
        <body>
            <h1 align="middle">ASKAP Continuum Data Validation Report</h1>""".format(self.css_style))

    def write_html_img_table(self,img):

        """Write an observations, image and catalogue report tables derived from fits image, header and catalogue.

        Arguments:
        ----------
        img : radio_image
            A radio image object used to write values to the html table."""

        #generate link to confluence page for each project code
        project = self.add_html_link("https://confluence.csiro.au/display/askapsst/{0}+Data".format(img.project),img.project,file=False)
        flux_type = 'integrated'
        if self.cat.use_peak:
            flux_type = 'peak'
        if self.cat.med_si == -99:
            med_si = ''
        else:
            med_si = '{0:.2f}'.format(self.cat.med_si)

        #Write observations report table
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
        </table>""".format( img.sbid,
                            project,
                            img.date,
                            img.duration,
                            img.centre,
                            img.freq))

        #Write image report table
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
        </table>""".format( img.name,
                            img.soft_version,
                            img.pipeline_version,
                            img.bmaj,
                            img.bmin,
                            self.cat.img_rms,
                            self.cat.img_peak,
                            self.cat.dynamic_range,
                            self.cat.area))

        #Write catalogue report table
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
                <th>Median spectral index</th>
                <th>Source Counts<br>&#967;<sub>red</sub><sup>2</sup></th>
            </tr>
            <tr>
                <td>{2}</td>
                <td>{3}</td>
                <td>{4}</td>
                <td>{5}</td>
                <td>{6:.1f} Jy vs. {7:.1f} Jy</td>
                <td>{8}</td>""".format( self.cat.filename,
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

        self.html.write("""</td>
            </tr>
        </table>
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


    def get_metric_level(self,good_condition,uncertain_condition):

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

        """Assign level 1 (good), 2 (uncertain) or 3 (bad) to each metric, depending on specific tolerenace values.
        See https://confluence.csiro.au/display/askapsst/Continuum+validation+metrics"""

        for metric in self.metric_val.keys():
            # Remove keys that don't have a valid value (value=-99)
            if self.metric_val[metric] == -99:
                self.metric_val.pop(metric)
                self.metric_source.pop(metric)
                self.metric_level.pop(metric)
            else:
                #flux ratio within 5/10%?
                if metric == 'Flux Ratio':
                    val = np.abs(self.metric_val[metric]-1)
                    good_condition = val < 0.05
                    uncertain_condition = val < 0.1
                    self.metric_source[metric] = 'Median flux density ratio [ASKAP / {0}]'.format(self.metric_source[metric])
                #uncertainty on flux ratio less than 10/20%?
                elif metric == 'Flux Ratio Uncertainty':
                    good_condition = self.metric_val[metric] < 0.1
                    uncertain_condition = self.metric_val[metric] < 0.2
                    self.metric_source[metric] = 'R.M.S. of median flux density ratio [ASKAP / {0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation from median)'
                #positional offset < 1/5 arcsec
                elif metric == 'Positional Offset':
                    good_condition = self.metric_val[metric] < 1
                    uncertain_condition = self.metric_val[metric] < 5
                    self.metric_source[metric] = 'Median positional offset (arcsec) [ASKAP-{0}]'.format(self.metric_source[metric])
                #uncertainty on positional offset < 1/5 arcsec
                elif metric == 'Positional Offset Uncertainty':
                    good_condition = self.metric_val[metric] < 5
                    uncertain_condition = self.metric_val[metric] < 10
                    self.metric_source[metric] = 'R.M.S. of median positional offset (arcsec) [ASKAP-{0}]'.format(self.metric_source[metric])
                    self.metric_source[metric] += ' (estimated from median absolute deviation from median)'
                #reduced chi-squared of source counts < 3/50?
                elif metric == 'Source Counts Reduced Chi-squared':
                    good_condition = self.metric_val[metric] < 3
                    uncertain_condition = self.metric_val[metric] < 50
                    self.metric_source[metric] = 'Reduced chi-squared of source counts'
                #resolved fraction of sources between 5-20%?
                elif metric == 'Resolved Fraction':
                    good_condition = self.metric_val[metric] > 0.05 and self.metric_val[metric] < 0.2
                    uncertain_condition = False
                    self.metric_source[metric] = 'Fraction of sources resolved according to int/peak flux densities'
                #spectral index less than 0.2 away from -0.8?
                elif metric == 'Spectral Index':
                    val = np.abs(self.metric_val[metric]+0.8)
                    good_condition = val < 0.2
                    uncertain_condition = False
                    self.metric_source[metric] = 'Median catalogued spectral index [{0}]'.format(self.metric_source['Spectral Index'])
                #if unknown metric, set it to 3 (bad)
                else:
                    good_condition = False
                    uncertain_condition = False

                #assign level to metric
                self.metric_level[metric] = self.get_metric_level(good_condition,uncertain_condition)

        self.write_CASDA_xml()

    def write_pipeline_offset_params(self):

        """Write a txt file with offset params for ASKAPsoft pipeline for user to easily import into config file, and then drop them from metrics.
        See http://www.atnf.csiro.au/computing/software/askapsoft/sdp/docs/current/pipelines/ScienceFieldContinuumImaging.html?highlight=offset"""

        txt = open('offset_pipeline_params.txt','w')
        txt.write("DO_POSITION_OFFSET=true\n")
        txt.write("RA_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['RA Offset']))
        txt.write("DEC_POSITION_OFFSET={0:.2f}\n".format(-self.metric_val['DEC Offset']))
        txt.close()

        for metric in ['RA Offset','DEC Offset']:
            self.metric_val.pop(metric)
            self.metric_source.pop(metric)
            self.metric_level.pop(metric)
            self.metric_count.pop(metric)

    def write_CASDA_xml(self):

        """Write xml table with all metrics for CASDA."""

        tmp_table = Table(  [self.metric_val.keys(),self.metric_val.values(),self.metric_level.values(),self.metric_source.values()],
                            names=['metric_name','metric_value','metric_status','metric_description'],
                            dtype=[str,float,np.int32,str])
        vot = votable.from_table(tmp_table)
        vot.version = 1.3
        table = vot.get_first_table()
        table.params.extend([votable.tree.Param(vot, name="project", datatype="char", arraysize="*", value=self.img.project)])
        valuefield=table.fields[1]
        valuefield.precision='2'
        prefix = ''
        if self.img.project != '':
            prefix = '{0}_'.format(self.img.project)
        xml_filename = '{0}CASDA_continuum_validation.xml'.format(prefix)
        votable.writeto(vot, xml_filename)

    def write_html_end(self):

        """Write the end of the html report file (including table of metrics) and close it."""

        #Close cross-matches table and write header of validation summary table
        self.html.write("""
                </td>
            </tr>
        </table>
        <h2 align="middle">{0} continuum validation metrics</h2>
        <table class="reportTable">
            <tr>
                <th>Flux Ratio<br>({0} / {1})</th>
                <th>Flux Ratio Deviation<br>({0} / {1})</th>
                <th>Positional Offset (arcsec)<br>({0} &mdash; {2})</th>
                <th>Positional Offset Deviation (arcsec)<br>({0} &mdash; {2})</th>
                <th>Resolved Fraction from int/peak Flux<br>({0})</th>
                <th>Spectral Index<br>({3})</th>
                <th>Source Counts &#967;<sub>red</sub><sup>2</sup><br>({0})</th>
            </tr>""".format(self.cat.name,self.metric_source['Flux Ratio'],self.metric_source['Positional Offset'],self.metric_source['Spectral Index']))

        #assign levels to each metric
        self.assign_metric_levels()

        #Write table with values of metrics and colour them according to level
        self.html.write("""
        <tr>
            <td {0}>{1:.2f}</td>
            <td {2}>{3:.2f}</td>
            <td {4}>{5:.2f}</td>
            <td {6}>{7:.2f}</td>
            <td {8}>{9:.2f}</td>
            <td {10}>{11:.2f}</td>
            <td {12}>{13:.2f}</td>
        </tr>""".format(self.html_colour(self.metric_level['Flux Ratio']),self.metric_val['Flux Ratio'],
                        self.html_colour(self.metric_level['Flux Ratio Uncertainty']),self.metric_val['Flux Ratio Uncertainty'],
                        self.html_colour(self.metric_level['Positional Offset']),self.metric_val['Positional Offset'],
                        self.html_colour(self.metric_level['Positional Offset Uncertainty']),self.metric_val['Positional Offset Uncertainty'],
                        self.html_colour(self.metric_level['Resolved Fraction']),self.metric_val['Resolved Fraction'],
                        self.html_colour(self.metric_level['Spectral Index']),self.metric_val['Spectral Index'],
                        self.html_colour(self.metric_level['Source Counts Reduced Chi-squared']),self.metric_val['Source Counts Reduced Chi-squared']))

        #Close table, write time generated, and close html file
        self.html.write("""</table>
                <p><i>Generated at {0}</i></p>
            <?php footer(); ?>
            </body>
        </html>""".format(datetime.now()))
        self.html.close()
        print "Continuum validation report written to '{0}'.".format(self.name)


    def add_html_link(self,target,link,file=True,newline=False):

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


    def text_to_html(self,text):

        """Take a string of text that may include LaTeX, and return the html code that will generate it as LaTeX.

        Arguments:
        ----------
        text : string
            A string of text that may include LaTeX.

        Returns:
        --------
        html : string
            The same text readable as html."""

        #this will allow everything between $$ to be generated as LaTeX
        html = """
                    <script type="text/x-mathjax-config">
                      MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
                    </script>
                    <script type="text/javascript"
                      src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                    </script>
                    <br>

                    """

        #write a newline / break for each '\n' in string
        for line in text.split('\n'):
            html += line + '<br>'

        return html

    def html_colour(self,level):

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

    def source_counts(self,fluxes,freq,rms_map=None,solid_ang=0,write=True):

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
            A fixed solid angle over which the source counts are computed. Only used when rms_map is None.
        write : bool
            Write the source counts to file."""

        #derive file names based on user input
        filename = 'screen'
        counts_file = '{0}_source_counts.csv'.format(self.cat.basename)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_source_counts.{2}'.format(self.figDir,self.cat.name,self.plot_to)

        #perform source counts if not already written to file or user specifies to re-do
        if not os.path.exists(counts_file) or self.redo:

            #warn user if they haven't input an rms map or fixed solid angle
            if rms_map is None and solid_ang == 0:
                warnings.warn_explicit("You must input a fixed solid angle or an rms map to compute the source counts!\n",UserWarning,WARN,cf.f_lineno)
                return

            #get the number of bins from the user
            nbins = self.src_cnt_bins
            print "Deriving source counts for {0} using {1} bins.".format(self.cat.name,nbins)

            #Normalise the fluxes to 1.4 GHz
            fluxes = flux_at_freq(1400,freq,fluxes,-0.8)

            if rms_map is not None:
                w = WCS(rms_map.header)
                if self.verbose:
                    print "Using rms map '{0}' to derive solid angle for each flux bin.".format(self.img.rms_map)

            #add one more bin and then discard it, since this is dominated by the few brightest sources
            #we also add one more to the bins since there's one more bin edge than number of bins
            edges = np.percentile(fluxes,np.linspace(0,100,nbins+2))
            dN,edges,patches=plt.hist(fluxes,bins=edges)
            dN = dN[:-1]
            edges = edges[:-1]

            #derive the lower and upper edges and dS
            lower = edges[:-1]
            upper = edges[1:]
            dS = upper-lower
            S = np.zeros(len(dN))
            solid_angs = np.zeros(len(dN))

            for i in range(len(dN)):
                #derive the mean flux from all fluxes in current bin
                S[i] = np.mean(fluxes[(fluxes > lower[i]) & (fluxes < upper[i])])

                #Get the pixels from the r.m.s. map where SNR*r.m.s. < flux
                if rms_map is not None:
                    solid_angs[i] = get_pixel_area(rms_map, flux=S[i]/self.cat.SNR, w=w)[1]
                #otherwise use the fixed value passed in
                else:
                    solid_angs[i] = solid_ang

                    #if no pixels found, assume solid angle of one beam per source in bin
                    if solid_angs[i] == 0:
                        solid_angs[i] = dN[i]*(self.img.bmaj/3600)*(self.img.bmin/3600)*np.pi*(np.pi/180)**2

            #compute the differential Euclidean source counts and uncertanties in linear space
            counts = (S**2.5)*dN/dS/solid_angs
            err = (S**2.5)*np.sqrt(dN)/dS/solid_angs

            #Store these and the log of these values in pandas data frame
            df = pd.DataFrame()
            df['dN'] = dN
            df['area'] = solid_angs/((np.pi/180)**2)
            df['S'] = S
            df['logS'] = np.log10(S)
            df['logCounts'] = np.log10(counts)
            df['logErrUp'] = np.log10(counts+err) - np.log10(counts)
            df['logErrDown'] = np.abs(np.log10(counts-err) - np.log10(counts))

            if write:
                if self.verbose:
                    print "Writing source counts to '{0}'.".format(counts_file)
                df.to_csv(counts_file,index=False)

        #otherwise simply read in source counts from file
        else:
            print "File '{0}' already exists. Reading source counts from this file.".format(counts_file)
            df = pd.read_csv(counts_file)

        #create a figure for the source counts
        plt.close()
        fig=plt.figure(**self.fig_size)
        title = '{0} 1.4 GHz source counts'.format(self.cat.name,self.cat.freq[self.cat.name])
        #write axes using unicode (for html) or LaTeX
        if self.plot_to == 'html':
            ylabel = u"log\u2081\u2080 S\u00B2\u22C5\u2075 dN/dS [Jy\u00B9\u22C5\u2075 sr\u207B\u00B9]"
            xlabel = u"log\u2081\u2080 S [Jy]"
        else:
            ylabel = r"$\log_{10}$ S$^{2.5}$ dN/dS [Jy$^{1.5}$ sr$^{-1}$]"
            xlabel = r"$\log_{10}$ S [Jy]"

        #for html plots, add labels for the bin centre, count and area for every data point
        labels = [u'S: {0:.2f} mJy, dN: {1:.0f}, Area: {2:.2f} deg\u00B2'.format(bin,count,area) for bin,count,area in zip(df['S']*1e3,df['dN'],df['area'])]

        #read the log of the source counts from Norris+11 from same directory of this script
        df_Norris = pd.read_table('{0}/all_counts.txt'.format(main_dir),sep=' ')
        x = df_Norris['S']-3 #convert from log of flux in mJy to log of flux in Jy
        y = df_Norris['Counts']
        yerr = (df_Norris['ErrDown'],df_Norris['ErrUp'])

        #fit 6th degree polynomial to Norris+11 data
        deg = 6
        poly_paras = np.polyfit(x,y,deg)
        f = np.poly1d(poly_paras)
        xlin = np.linspace(min(x)*1.2,max(x)*1.2)
        ylin = f(xlin)

        #derive the square of the residuals (chi squared), and their sum
        #divided by the number of data points (reduced chi squared)
        chi = ((df['logCounts']-f(df['logS']))/df['logErrDown'])**2
        red_chi_sq = np.sum(chi)/len(df)

        #store reduced chi squared value
        self.sc_red_chi_sq = red_chi_sq

        #Plot Norris+11 data
        data = plt.errorbar(x,y,yerr=yerr,linestyle='none',marker='.',c='r')
        line, = plt.plot(xlin,ylin,c='black',linestyle='--',zorder=5)
        txt = ''
        if self.plot_to == 'html':
            txt += 'Data from <a href="http://adsabs.harvard.edu/abs/2011PASA...28..215N">Norris+11</a>'
            txt += ' (updated from <a href="http://adsabs.harvard.edu/abs/2003AJ....125..465H">Hopkins+03</a>)\n'
        txt += '$\chi^2_{red}$: %.2f' % red_chi_sq

        #Legend labels for the Norris data and line, and the ASKAP data
        xlab = 'Norris+11'
        leg_labels = [xlab,'{0}th degree polynomial fit to {1}'.format(deg,xlab),self.cat.name]

        #write reduced chi squared to report table
        self.html.write('<td>{0:.2f}<br>'.format(red_chi_sq))

        #Plot ASKAP data on top of Norris+11 data
        self.plot(df['logS'],
                  y=df['logCounts'],
                  yerr=(df['logErrDown'],df['logErrUp']),
                  figure=fig,
                  title=title,
                  labels=labels,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  axis_perc=0,
                  text=txt,
                  loc='br',
                  leg_labels=leg_labels,
                  handles=[data,line],
                  filename=filename)

    def y0(self,x,y):

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

        return x,x*0

    def y1(self,x,y):

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

        return x,[1]*len(x)

    def x0(self,x,y):

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

        return y*0,y

    def ratio_err_max(self,SNR,ratio):

        """For given x and y data (flux ratio as a function of S/N), return the maximum uncertainty in flux ratio.

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

        return SNR[SNR > 0],1+3*np.sqrt(2)/SNR[SNR > 0]

    def ratio_err_min(self,SNR,ratio):

        """For given x and y data (flux ratio as a function of S/N), return the minimum uncertainty in flux ratio.

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

        return SNR[SNR > 0],1-3*np.sqrt(2)/SNR[SNR > 0]


    def axis_to_np(self,axis):

        """Return a numpy array of the non-nan data from the input axis.

        Arguments:
        ----------
        axis : string or numpy.array or pandas.Series or list
            The data for a certain axis. String are interpreted as column names from catalogue object passed into constructor.

        Returns:
        --------
        axis : numpy.array
            All non-nan values of the data.

        See Also
        --------
        numpy.array
        pandas.Series"""

        #convert input to numpy array
        if type(axis) is str:
            axis = self.cat.df[axis].values
        elif axis is pd.Series:
            axis = axis.values

        return axis

    def shared_indices(self,xaxis,yaxis=None,caxis=None):

        """Return a list of non-nan indices shared between all used axes.

        Arguments:
        ----------
        xaxis : string or numpy.array or pandas.Series or list
            A list of the x axis data. String are interpreted as column names from catalogue object passed into constructor.
        yaxis : string or numpy.array or pandas.Series or list
            A list of the y axis data. String are interpreted as column names from catalogue object passed into constructor.
            If this is None, yaxis and caxis will be ignored.
        caxis : string or numpy.array or pandas.Series or list
            A list of the colour axis data. String are interpreted as column names from catalogue object passed into constructor.
            If this is None, caxis will be ignored.

        Returns:
        --------
        x : list
            The non-nan x data shared between all used axes.
        y : list
            The non-nan y data shared between all used axes. None returned if yaxis is None.
        c : list
            The non-nan colour data shared between all used axes. None returned if yaxis or caxis are None.
        indices : list
            The non-nan indices.

        See Also
        --------
        numpy.array
        pandas.Series"""

        #convert each axis to numpy array (or leave as None)
        x = self.axis_to_np(xaxis)
        y = self.axis_to_np(yaxis)
        c = self.axis_to_np(caxis)

        #get all shared indices from used axes that aren't nan
        if yaxis is None:
            indices = np.where(~np.isnan(x))[0]
            return x[indices],None,None,indices
        elif caxis is None:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)))[0]
            return x[indices],y[indices],None,indices
        else:
            indices = np.where((~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(c)))[0]
            return x[indices],y[indices],c[indices],indices


    def plot(self,x,y=None,c=None,yerr=None,figure=None,arrows=None,line_funcs=None,title='',labels=None,text=None,
             xlabel='',ylabel='',clabel='',leg_labels='',handles=[],loc='bl',ellipses=None,axis_perc=10,filename='screen',redo=False):

        """Create and write a scatter plot of the data from an input x axis, and optionally, a y and colour axis.
        This function assumes shared_indices() has already been called and all input axes are equal in length and the same data type.

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
            A 2-element tuple with the lengths of the arrows to plot at x and y (usually a list) - i.e. (dx[],dy[])
        line_funcs : list-like
            A list of functions for drawing lines (e.g. [self.x0(), self.y1()]).
        title : string
            The title of the plot.
        lables : list
            A list of string labels to give each data point. Length must be the same as all used axes.
        text : string
            Annotate this text on the figure (written to bottom of page for html figures).
        xlabel : string
            The label of the x axis.
        ylabel : string
            The label of the y axis.
        clabel : string
            The label of the colour axis.
        leg_labels : list
            A list of labels to include as a legend.
        handles : list
            A list of pre-defined handles associated the legend labels (must be same length or length-1 when yerr is not None).
        loc : string
            Location of the annotated text (not used for html plots). Options are 'bl', 'br', 'tl' and 'tr'.
        ellipses : list of matplotlib.patches.Ellipse objects
            Draw these ellipses on the figure.
        axis_perc : float
            The percentage beyond which to calculate the axis limits. Use 0 for no limits.
        filename : string
            Write the plot to this file name. If string contains 'html', file will be written to html using mpld3.
            If it is 'screen', it will be shown on screen. Otherwise, it will attempt to write an image file.
        redo: bool
            Produce this plot and write it, even if the file already exists.

        See Also
        --------
        numpy.array
        pandas.Series
        matplotlib.patches.Ellipse"""

        #only write figure if user wants it
        if self.write:
            #derive name of thumbnail file
            thumb = '{0}_thumb.png'.format(filename[:-1-len(self.plot_to)])

            #don't produce plot if file exists and user didn't specify to re-do
            if os.path.exists(filename) and not self.redo:
                if self.verbose:
                    print 'File already exists. Skipping plot.'
            else:
                #open html file for plot
                if 'html' in filename:
                    html_fig = open(filename,'w')
                #use figure passed in or create new one
                if figure is not None:
                    fig = figure
                else:
                    fig = plt.figure(**self.fig_size)

                ax = plt.subplot(111)
                norm = None

                #plot histogram
                if y is None:
                    edges = np.linspace(-3,2,11) #specific to spectral index
                    ax.hist(x,bins=edges)
                #plot scatter of data points with fixed colour
                elif c is None:
                    #hack to display html labels when line or ellipse overlaid
                    data = ax.scatter(x,y,zorder=20,alpha=0.0,**self.markers)
                    ax.scatter(x,y,**self.markers)
                #plot scatter of data points with colour axis
                else:
                    #normalise the colour bar so each bin contains equal number of data points
                    norm = colors.BoundaryNorm(np.percentile(c,np.linspace(0,100,self.cmap.N+1)),self.cmap.N)
                    data = ax.scatter(x,y,c=c,cmap=self.cmap,norm=norm,**self.colour_markers)
                    cbar = plt.colorbar(data)
                    cbar.ax.tick_params(**self.label_size)
                    cbar.set_label(clabel,**self.fig_font)
                    data = ax.scatter(x,y,c=c,cmap=self.cmap,zorder=20,alpha=0.0,norm=norm,**self.colour_markers) #same hack as above
                #plot error bars and add to list of handles
                if yerr is not None:
                    err_data = ax.errorbar(x,y,yerr=yerr,zorder=4,linestyle='none',marker=self.markers['marker'],color=self.markers['color'])
                    handles.append(err_data)

                #set default min and max axis limits, which may change
                xmin,xmax = ax.get_xlim()
                ymin,ymax = ax.get_ylim()

                #derive limits of x and y axes axis_perc % beyond their current limit
                if axis_perc > 0:
                    xmin = axis_lim(x,min,perc=axis_perc)
                    xmax = axis_lim(x,max,perc=axis_perc)
                    ymin = axis_lim(y,min,perc=axis_perc)
                    ymax = axis_lim(y,max,perc=axis_perc)

                #plot each line according to the input functions
                if line_funcs is not None:
                    xlin = np.linspace(xmin,xmax,num=1000)
                    ylin = np.linspace(ymin,ymax,num=1000)

                    for func in line_funcs:
                        xline,yline = func(xlin,ylin)
                        line = plt.plot(xline,yline,lw=2,color='black',linestyle='-',zorder=12)

                #doing this here forces the lines in html plots to not increase the axis limits
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)

                #overlay the title and labels according to given fonts and sizes
                plt.tick_params(**self.label_size)
                plt.title(title,**self.fig_font)
                plt.xlabel(xlabel,**self.fig_font)
                plt.ylabel(ylabel,**self.fig_font)

                #overlay arrows on each data point
                if arrows is not None:
                    if not (type(arrows) is tuple and len(arrows) == 2):
                        warnings.warn_explicit('Arrows not formatted correctly for plt.quiver(). Input a 2-element tuple.\n',UserWarning,WARN,cf.f_lineno)
                    elif c is None:
                        plt.quiver(x,y,arrows[0],arrows[1],units='x',**self.arrows)
                    else:
                        plt.quiver(x,y,arrows[0],arrows[1],c,units='x',cmap=self.cmap,norm=norm,**self.arrows)

                #annotate input text
                if text is not None and 'html' not in filename:
                    #write to given location on plot
                    kwargs = self.fig_font.copy()

                    if loc == 'tl':
                        args = (xmin,ymax,text)
                        kwargs.update({'horizontalalignment' : 'left', 'verticalalignment' : 'top'})
                    elif loc == 'tr':
                        args = (xmax,ymax,text)
                        kwargs.update({'horizontalalignment' : 'right', 'verticalalignment' : 'top'})
                    elif loc == 'br':
                        args = (xmax,ymin,text)
                        kwargs.update({'horizontalalignment' : 'right', 'verticalalignment' : 'bottom'})
                    else:
                        args = (xmin,ymin,text)
                        kwargs.update({'horizontalalignment' : 'left', 'verticalalignment' : 'bottom'})

                    plt.text(*args,**kwargs)

                #write a legend
                if len(handles) > 0:
                    plt.legend(handles,leg_labels,fontsize=self.fig_font['fontsize']//1.5)
                    #BELOW NOT CURRENTLY WORKING WELL
                    #if 'html' in filename:
                        #interactive_legend = plugins.InteractiveLegendPlugin(handles,leg_labels)
                        #plugins.connect(fig, interactive_legend)

                #overlay ellipses on figure
                if ellipses is not None:
                    for e in ellipses:
                        ax.add_patch(e)

                if self.verbose:
                    print "Writing figure to '{0}'.".format(filename)

                #write thumbnail of this figure
                if filename != 'screen':
                    plt.savefig(thumb)
                    image.thumbnail(thumb,thumb,scale=0.05)

                #write html figure
                if 'html' in filename:
                    #include label for every datapoint
                    if labels is not None:
                        tooltip = plugins.PointHTMLTooltip(data, labels=labels)
                        plugins.connect(fig, tooltip)

                    #print coordinates of mouse as it moves across figure
                    plugins.connect(fig, plugins.MousePosition(fontsize=self.fig_font['fontsize']))
                    html_fig.write(mpld3.fig_to_html(fig))

                    #write annotations to end of html file if user wants html plots
                    if text is not None:
                        html_fig.write(self.text_to_html(text))

                #otherwise show figure on screen
                elif filename == 'screen':
                    plt.show()
                #otherwise write with given extension
                else:
                    plt.savefig(filename)

            #Add link and thumbnail to html report table
            self.html.write(self.add_html_link(filename,thumb))

        plt.close()


    def validate(self,name1,name2,int_flux_ratio=False,useAplpy=False):

        """Produce a validation report between two catalogues, and optionally produce plots.

        Arguments:
        ----------
        name1 : string
            The dictionary key / name of a catalogue from the main catalogue object used to compare other data.
        name2 : string
            The dictionary key / name of a catalogue from the main catalogue object used as a comparison.

        Keyword arguments:
        ------------------
        int_flux_ratio : bool
            Plot the integrated to peak flux ratio.
        useAplpy : bool
            For sky plots, use aplpy instead of matplotlib. NOT CURRENTLY USED.

        Returns:
        --------
        ratio_med : float
            The median flux density ratio. -99 if this is not derived.
        sep_med : float
            The median sky separation between the two catalogues.
        alpha_med : float
            The median spectral index. -99 if this is not derived.

        See Also:
        ---------
        aplpy"""

        print 'Validating {0} with {1}...'.format(name1,name2)

        filename = 'screen'

        #write survey and number of matched to cross-matches report table
        self.html.write("""<tr>
                        <td>{0}</td>
                        <td>{1}</td>
                        <td>{2}""".format(name2,self.cat.freq[name2],self.cat.count[name2]))

        #plot the positional offsets
        fig = plt.figure(**self.fig_size)
        title = u"{0} \u2014 {1} positional offsets".format(name1,name2)
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry.{3}'.format(self.figDir,name1,name2,self.plot_to)

        #compute the S/N and its log based on main catalogue
        if name1 in self.cat.flux.keys():
            self.cat.df['SNR'] = self.cat.flux[name1] / self.cat.flux_err[name1]
            self.cat.df['logSNR'] = np.log10(self.cat.df['SNR'])
            caxis = 'logSNR'
        else:
            caxis = None

        #get non-nan data shared between each used axis as a numpy array
        x,y,c,indices = self.shared_indices(self.cat.dRA[name2],yaxis=self.cat.dDEC[name2],caxis=caxis)

        #derive the statistics of x and y and store in string to annotate on figure
        dRAmed,dRAmean,dRAstd,dRAerr,dRAmad = get_stats(x)
        dDECmed,dDECmean,dDECstd,dDECerr,dDECmad = get_stats(y)
        txt = '$\widetilde{\Delta RA}$: %.2f\n' % dRAmed
        txt += '$\overline{\Delta RA}$: %.2f\n' % dRAmean
        txt += '$\sigma_{\Delta RA}$: %.2f\n' % dRAstd
        txt += '$\sigma_{\overline{\Delta RA}}$: %.2f\n' % dRAerr
        txt += '$\widetilde{\Delta DEC}$: %.2f\n' % dDECmed
        txt += '$\overline{\Delta DEC}$: %.2f\n' % dDECmean
        txt += '$\sigma_{\Delta DEC}$: %.2f\n' % dDECstd
        txt += '$\sigma_{\overline{\Delta DEC}}$: %.2f' % dDECerr

        #create an ellipse at the position of the median with axes of standard deviation
        e1 = Ellipse((dRAmed,dDECmed),width=dRAstd,height=dDECstd,color='black',fill=False,linewidth=3,zorder=10,alpha=0.9)

        #force axis limits of the search radius
        radius = max(self.cat.radius[name1],self.cat.radius[name2])
        plt.axis('equal')
        plt.xlim(-radius,radius)
        plt.ylim(-radius,radius)

        #create an ellipse at 0,0 with width 2 x search radius
        e2 = Ellipse((0,0),width=radius*2,height=radius*2,color='grey',fill=False,linewidth=3,linestyle='--',zorder=1,alpha=0.9)

        #format labels according to destination of figure
        if self.plot_to == 'html':
            xlabel =  u'\u0394RA (arcsec)'
            ylabel = u'\u0394DEC (arcsec)'
            clabel = u'log\u2081\u2080 S/N'
        else:
            xlabel =  '$\Delta$RA (arcsec)'
            ylabel = '$\Delta$DEC (arcsec)'
            clabel = r'$\log_{10}$ S/N'

        #for html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = ['S/N = {0:.2f}, separation = {1:.2f}\"'.format(cval,totSep)\
                  for cval,totSep in zip(self.cat.df.loc[indices,'SNR'],self.cat.sep[name2][indices])]
        else:
            labels = ['Separation = {0:.2f}\"'.format(cval) for cval in self.cat.sep[name2][indices]]

        #get median separation in arcsec
        c1 = SkyCoord(ra=0,dec=0,unit='arcsec,arcsec')
        c2 = SkyCoord(ra=dRAmed,dec=dDECmed,unit='arcsec,arcsec')
        sep_med = c1.separation(c2).arcsec

        #get mad of separation in arcsec
        c1 = SkyCoord(ra=0,dec=0,unit='arcsec,arcsec')
        c2 = SkyCoord(ra=dRAmad,dec=dDECmad,unit='arcsec,arcsec')
        sep_mad = c1.separation(c2).arcsec

        #write the dRA and dDEC to html table
        self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} (RA)<br>{2:.2f} &plusmn {3:.2f} (Dec)<br>""".format(dRAmed,dRAmad,dDECmed,dDECmad))

        #plot the positional offsets
        self.plot(x,
                  y=y,
                  c=c,
                  figure=fig,
                  line_funcs=(self.x0,self.y0),
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  clabel=clabel,
                  text=txt,
                  ellipses=(e1,e2),
                  axis_perc=0,
                  loc='tr',
                  filename=filename,
                  labels=labels)

        #plot the positional offsets across the sky
        title += " by sky position"
        xlabel = 'RA (deg)'
        ylabel = 'DEC (deg)'
        if self.plot_to != 'screen':
            filename = '{0}/{1}_{2}_astrometry_sky.{3}'.format(self.figDir,name1,name2,self.plot_to)

        #get non-nan data shared between each used axis as a numpy array
        x,y,c,indices = self.shared_indices(self.cat.ra[name2],yaxis=self.cat.dec[name2],caxis=caxis)

        #for html plots, add S/N and separation labels for every data point
        if caxis is not None:
            labels = [u'S/N = {0:.2f}, \u0394RA = {1:.2f}\", \u0394DEC = {2:.2f}\"'.format(cval,dra,ddec) for cval,dra,ddec\
                  in zip(self.cat.df.loc[indices,'SNR'],self.cat.dRA[name2][indices],self.cat.dDEC[name2][indices])]
        else:
            labels = [u'\u0394RA = {0:.2f}\", \u0394DEC = {1:.2f}\"'.format(dra,ddec) for dra,ddec\
                  in zip(self.cat.dRA[name2][indices],self.cat.dDEC[name2][indices])]

        #plot the positional offsets across the sky
        self.plot(x,
                  y=y,
                  c=c,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  arrows=(self.cat.dRA[name2][indices],self.cat.dDEC[name2][indices]),
                  clabel=clabel,
                  axis_perc=0,
                  filename=filename,
                  labels=labels)

        #derive column names and check if they exist
        freq = int(round(self.cat.freq[name1]))
        fitted_flux_col = '{0}_extrapolated_{1}MHz_flux'.format(name2,freq)
        fitted_ratio_col = '{0}_extrapolated_{1}MHz_{2}_flux_ratio'.format(name2,freq,name1)
        ratio_col = '{0}_{1}_flux_ratio'.format(name2,name1)

        #only plot flux ratio if it was derived
        if ratio_col not in self.cat.df.columns and (fitted_ratio_col not in self.cat.df.columns or np.all(np.isnan(self.cat.df[fitted_ratio_col]))):
            print "Can't plot flux ratio since you haven't derived the fitted flux density at this frequency."
            ratio_med = -99
            ratio_mad = -99
            flux_ratio_type = ''
            self.html.write('<td>')
        else:
            #compute flux ratio based on which one exists and rename variable for figure title
            if ratio_col in self.cat.df.columns:
                ratio = self.cat.df[ratio_col]
                flux_ratio_type = name2
            elif fitted_ratio_col in self.cat.df.columns:
                ratio = self.cat.df[fitted_ratio_col]
                flux_ratio_type = '{0} extrapolation'.format(name2)

            logRatio = np.log10(ratio)

            #plot the flux ratio as a function of S/N
            fig = plt.figure(**self.fig_size)

            title = "{0} / {1} {2} MHz flux ratio".format(name1,flux_ratio_type,freq)
            xlabel = 'S/N'
            ylabel = 'Flux Density Ratio'
            clabel = 'Declination'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_ratio.{3}'.format(self.figDir,name1,name2,self.plot_to)

            #get non-nan data shared between each used axis as a numpy array
            x,y,c,indices = self.shared_indices('SNR',yaxis=ratio,caxis=self.cat.dec[name1])
            plt.xlim(-20,axis_lim(x,max,perc=10))
            plt.ylim(axis_lim(y,min),axis_lim(y,max))

            #format scale according to destination of figure
            if self.plot_to != 'html':
                plt.yscale('log')
                plt.gca().grid(b=True, which='minor', color='w', linewidth=0.5)

            #derive the ratio statistics and store in string to append to plot
            ratio_med,ratio_mean,ratio_std,ratio_err,ratio_mad = get_stats(y)
            txt = '$\widetilde{Ratio}$: %.2f\n' % ratio_med
            txt += '$\overline{Ratio}$: %.2f\n' % ratio_mean
            txt += '$\sigma_{Ratio}$: %.2f\n' % ratio_std
            txt += '$\sigma_{\overline{Ratio}}$: %.2f' % ratio_err

            #for html plots, add flux labels for every data point
            if flux_ratio_type == name2:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1,flux1,name2,flux2)\
                          for flux1,flux2 in zip(self.cat.flux[name1][indices]*1e3,self.cat.flux[name2][indices]*1e3)]
            else:
                labels = ['{0} flux = {1:.2f} mJy, {2} flux = {3:.2f} mJy'.format(name1,flux1,flux_ratio_type,flux2)\
                          for flux1,flux2 in zip(self.cat.flux[name1][indices]*1e3,self.cat.df[fitted_flux_col][indices]*1e3)]

            #write the ratio to html report table
            if flux_ratio_type == name2:
                type = 'measured'
            else:
                type = 'extrapolated'
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f} ({2})<br>""".format(ratio_med,ratio_mad,type))

            #plot the flux ratio as a function of S/N
            self.plot(x,
                      y=y,
                      c=c,
                      figure=fig,
                      line_funcs=(self.y1,self.ratio_err_min,self.ratio_err_max),
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      clabel=clabel,
                      text=txt,
                      loc='tr',
                      axis_perc=0,
                      filename=filename,
                      labels=labels)


            #plot the flux ratio across the sky
            fig = plt.figure(**self.fig_size)
            title += " by sky position"
            xlabel = 'RA (deg)'
            ylabel = 'DEC (deg)'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_ratio_sky.{3}'.format(self.figDir,name1,name2,self.plot_to)

            #get non-nan data shared between each used axis as a numpy array
            x,y,c,indices = self.shared_indices(self.cat.ra[name2],yaxis=self.cat.dec[name2],caxis=logRatio)

            #format labels according to destination of figure
            if self.plot_to == 'html':
                clabel = u'log\u2081\u2080 Flux Ratio'
            else:
                clabel = r'$\log_{10}$ Flux Ratio'

            #for html plots, add flux ratio labels for every data point
            labels = [u'{0} = {1:.2f}'.format('Flux Ratio',cval) for cval in ratio[indices]]

            #plot the flux ratio across the sky
            self.plot(x,
                      y=y,
                      c=c,
                      figure=fig,
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      clabel=clabel,
                      axis_perc=0,
                      filename=filename,
                      labels=labels)

        if int_flux_ratio:

            self.cat.df['int_peak_ratio'] = self.cat.df[self.cat.flux_col]/self.cat.df[self.cat.peak_col]

            #plot the int/peak flux ratio
            fig = plt.figure(**self.fig_size)

            title = "{0} int/peak flux ratio".format(name1)
            xlabel = 'S/N'
            ylabel = 'Int/Peak Flux Ratio'
            clabel = 'Declination'
            if self.plot_to != 'screen':
                filename = '{0}/{1}_int_peak_ratio.{2}'.format(self.figDir,name1,self.plot_to)

            #get non-nan data shared between each used axis as a numpy array
            x,y,c,indices = self.shared_indices('SNR',yaxis='int_peak_ratio',caxis=self.cat.dec[name1])
            plt.xlim(-20,axis_lim(x,max,perc=10))
            plt.ylim(axis_lim(y,min),axis_lim(y,max))

            #derive the statistics of y and store in string
            ymed,ymean,ystd,yerr,ymad = get_stats(y)
            txt = '$\widetilde{Ratio}$: %.2f\n' % ymed
            txt += '$\overline{Ratio}$: %.2f\n' % ymean
            txt += '$\sigma_{Ratio}$: %.2f\n' % ystd
            txt += '$\sigma_{\overline{Ratio}}$: %.2f' % yerr

            #for html plots, add flux labels for every data point
            labels = ['Int flux = {0:.2f} mJy, Peak flux = {1:.2f} mJy'.format(int_flux,peak_flux) for int_flux,peak_flux in\
                         zip(self.cat.df[self.cat.flux_col][indices]*1e3,self.cat.df[self.cat.peak_col][indices]*1e3)]

            #plot the int/peak flux ratio
            self.plot(x,
                      y=y,
                      c=c,
                      figure=fig,
                      line_funcs=[self.y1],
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      clabel=clabel,
                      text=txt,
                      loc='tr',
                      axis_perc=0,
                      filename=filename,
                      labels=labels)

        #derive spectral index column name and check if exists
        si_column = '{0}_{1}_alpha'.format(name1,name2)

        if not si_column in self.cat.df.columns:
            print "Can't plot spectral index between {0} and {1}, since it was not derived.".format(name1,name2)
            alpha_med = -99 #null flag
            self.html.write('<td>')
        else:
            #plot the spectral index
            fig = plt.figure(**self.fig_size)
            plt.xlim(-3,2)
            title = "{0}-{1} Spectral Index".format(name1,name2)
            if self.plot_to != 'screen':
                filename = '{0}/{1}_{2}_spectal_index.{3}'.format(self.figDir,name1,name2,self.plot_to)

            #get non-nan data shared between each used axis as a numpy array
            x,y,c,indices = self.shared_indices(si_column)

            #format labels according to destination of figure
            freq1 = int(round(min(self.cat.freq[name1],self.cat.freq[name2])))
            freq2 = int(round(max(self.cat.freq[name1],self.cat.freq[name2])))
            if self.plot_to == 'html':
                xlabel = u'\u03B1 [{0}-{1} MHz]'.format(freq1,freq2)
            else:
                xlabel = r'$\alpha_{%s}^{%s}$' % (freq1,freq2)

            #derive the statistics of x and store in string
            alpha_med,alpha_mean,alpha_std,alpha_err,alpha_mad = get_stats(x)
            txt = '$\widetilde{\\alpha}$: %.2f\n' % alpha_med
            txt += '$\overline{\\alpha}$: %.2f\n' % alpha_mean
            txt += '$\sigma_{\\alpha}$: %.2f\n' % alpha_std
            txt += '$\sigma_{\overline{\\alpha}}$: %.2f' % alpha_err

            #write the ratio to html report table
            self.html.write("""</td>
                        <td>{0:.2f} &plusmn {1:.2f}<br>""".format(alpha_med,alpha_mad))

            #plot the spectral index
            self.plot(x,
                      figure=fig,
                      title=title,
                      xlabel=xlabel,
                      ylabel='N',
                      axis_perc=0,
                      filename=filename,
                      text=txt,
                      loc='tl')

        #write the end of the html report table row
        self.html.write("""</td>
                    </tr>""")

        if self.cat.med_si != -99:
            alpha_med = self.cat.med_si
            alpha_type = '{0} in-band'.format(name1)
        else:
            alpha_type = '{0}-{1}'.format(name1,name2)
        if alpha_med < -90.:
            alpha_med=-99

        #create dictionary of validation metrics and where they come from
        metric_val = {  'Flux Ratio' : ratio_med,
                        'Flux Ratio Uncertainty' : ratio_mad,
                        'RA Offset' : dRAmed,
                        'DEC Offset' : dDECmed,
                        'Positional Offset' : sep_med,
                        'Positional Offset Uncertainty' : sep_mad,
                        'Spectral Index' : alpha_med}

        metric_source = {'Flux Ratio' : flux_ratio_type,
                        'Flux Ratio Uncertainty' : flux_ratio_type,
                        'RA Offset' : name2,
                        'DEC Offset' : name2,
                        'Positional Offset' : name2,
                        'Positional Offset Uncertainty' : name2,
                        'Spectral Index' : alpha_type}

        count = self.cat.count[name2]

        #overwrite values if they are valid and come from a larger catalogue
        for key in metric_val.keys():
            if count > self.metric_count[key] and metric_val[key] != -99:
                self.metric_count[key] = count
                self.metric_val[key] = metric_val[key]
                self.metric_source[key] = metric_source[key]



#Set paramaters passed in by user
img = find_file(parse_string(args['<fits-file>']))
verbose = args['--verbose']
source = args['--source']
refind = args['--refind']
redo = args['--redo'] or refind #force redo when refind is True
use_peak = args['--peak-flux']
write_any = not args['--no-write']
write_all = args['--write']
aegean_params = args['--aegean']
ncores = int(args['--ncores'])
nbins = int(args['--nbins'])
snr = float(args['--snr'])
config_files = args['--catalogues'].split(',')
SEDs = args['--SEDs'].split(',')
SEDextn = parse_string(args['--SEDfig'])

#force write_all=False write_any=False
if not write_any:
    write_all = False

#add '../' to relative paths of these files, since
#we'll create and move into a directory for output files
filter_config = new_path(parse_string(args['--filter']))
selavy_cat = new_path(parse_string(args['--Selavy']))
noise = new_path(parse_string(args['--noise']))

if __name__ == "__main__":

    #Run Aegean for source finding or use input Selavy catalogue
    if selavy_cat is not None:
        main_cat = selavy_cat
        finder = 'selavy'
    else:
        finder = 'aegean'

    #add S/N and peak/int to output directory/file names
    suffix = '{0}_snr{1}_'.format(finder,snr)
    if use_peak:
        suffix += 'peak'
    else:
        suffix += 'int'

    #Load in ASKAP image
    changeDir(img,suffix,verbose=verbose)
    img = new_path(img)
    AK = radio_image(img,verbose=verbose,rms_map=noise,SNR=snr)

    #Run BANE if user hasn't input noise map
    if noise is None:
        AK.run_BANE(ncores=ncores,redo=refind)

    #Run Aegean if user didn't pass in Selavy catalogue
    if finder == 'aegean':
        main_cat = AK.cat_comp
        AK.run_Aegean(ncores=ncores,redo=refind,params=aegean_params,write=write_any)

    #Create ASKAP catalogue object
    AKcat = catalogue(main_cat,'ASKAP',finder=finder,image=AK,SNR=snr,verbose=verbose,autoload=False,use_peak=use_peak)

    #Filter out sources below input SNR, set ASKAP specs and create report object before filtering
    #catalogue further so specs and source counts can be written for all sources above input SNR
    AKcat.filter_sources(SNR=snr,redo=redo,write=write_any,verbose=verbose,file_suffix='_snr{0}'.format(snr))
    AKcat.set_specs(AK)
    AKreport = report(AKcat,img=AK,verbose=verbose,plot_to=source,redo=redo,src_cnt_bins=nbins,write=write_any)

    #use config file for filtering sources if it exists
    if filter_config is not None:
        if verbose:
            print "Using config file '{0}' for filtering.".format(filter_config)
        filter_dic = config2dic(filter_config,verbose=verbose)
        filter_dic.update({'redo' : redo, 'write' : write_all, 'verbose' : verbose})
        AKcat.filter_sources(**filter_dic)
    else:
        #otherwise use default criteria, selecting reliable point sources for comparison
        AKcat.filter_sources(flux_lim=1e-3,ratio_frac=1.4,reject_blends=True,flags=True,psf_tol=1.5,resid_tol=3,redo=redo,write=write_all,verbose=verbose)

    #process each catalogue object according to list of input catalogue config files
    #this will cut out a box, cross-match to this instance, and derive the spectral indices.
    for config_file in config_files:
        if verbose:
            print "Using config file '{0}' for catalogue.".format(config_file)
        config_file = config_file.strip() #in case user put a space
        config_file = find_file(config_file,verbose=verbose)
        AKcat.process_config_file(config_file,redo=redo,verbose=verbose,write_all=write_all,write_any=write_any)

    #Fit radio SED models using all fluxes except
    #ASKAP, and derive the flux at ASKAP frequency
    if len(AKcat.cat_list) > 1:
        AKcat.fit_spectra(redo=redo,models=SEDs,GLEAM_subbands='int',GLEAM_nchans=4,cat_name=None,write=write_any,fig_extn=SEDextn)

    print "----------------------------"
    print "| Running validation tests |"
    print "----------------------------"

    #Produce validation report for each cross-matched survey
    for cat_name in AKcat.cat_list[1:]:
        AKreport.validate(AKcat.name,cat_name,int_flux_ratio=False)

    #write file with RA/DEC offsets for ASKAPsoft pipeline
    #and append validation metrics to html file and then close it
    AKreport.write_pipeline_offset_params()
    AKreport.write_html_end()
