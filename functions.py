import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.special as special
from astropy.wcs import WCS
from matplotlib import ticker


def changeDir(filepath, suffix, verbose=False):
    """Derive a directory name from an input file to store all output files, create it,
       and then change to it.

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
    # derive directrory name for output files
    filename = filepath.split('/')[-1]
    basename = remove_extn(filename)
    dir = '{0}_continuum_validation_{1}'.format(basename, suffix)
    # create it if it doesn't exist
    if not os.path.exists(dir):
        if verbose:
            print("Making directory for output files - {0}.".format(dir))
        os.mkdir(dir)
    # move to that directory and update the filepath
    if verbose:
        print("Changing to directory for output files - '{0}'.".format(dir))
    os.chdir(dir)

# The following are radio SED models as a function of frequency and several fitted parmaters


def powlaw(freq, S_norm, alpha):
    return S_norm*freq**alpha


def curve(freq, S_max, nu_m, alpha_thick, alpha_thin):
    num1 = (1 - np.exp(-1))*((freq / nu_m)**alpha_thick)
    num2 = (1 - np.exp(-(freq/nu_m)**(alpha_thin-alpha_thick)))
    return S_max / (num1*num2)


def pow_CIbreak(freq, S_norm, alpha, nu_br):
    return S_norm*(freq/nu_br)**(alpha+0.5 + 0.5*(1 + (freq/nu_br)**4)**-1)


def pow_CIbreak2(freq, S_norm, alpha, nu_br):
    alpha, freq = CIbreak(freq, alpha, nu_br)
    return S_norm*freq**alpha


def pow_JPbreak(freq, S_norm, alpha, nu_br):
    return S_norm*(freq**alpha)*JPbreak(freq, nu_br)


def SSA(freq, S_norm, beta, nu_m):
    factor1 = ((freq/nu_m)**(-(beta-1)/2))
    factor2 = (1-np.exp(-(freq/nu_m)**(-(beta+4)/2)))/((freq/nu_m)**(-(beta+4)/2))
    return S_norm*factor1*factor2


def SSA_JPbreak(freq, S_norm, beta, nu_m, nu_br):
    return SSA(freq, S_norm, beta, nu_m)*JPbreak(freq, nu_br)


def SSA_CIbreak(freq, S_norm, beta, nu_m, nu_br):
    dummyalpha, dummyfreq = CIbreak(freq, beta, nu_br)
    return SSA(freq, S_norm, beta, nu_m)*dummyfreq**dummyalpha


def FFA(freq, S_norm, alpha, nu_m):
    return S_norm*(freq**(alpha))*np.exp(-(freq/nu_m)**(-2.1))


def Bic98_FFA(freq, S_norm, alpha, p, nu_m):
    factor1 = ((freq/nu_m)**(2.1*(p+1)+alpha))
    factor2 = special.gammainc((p+1), ((freq/nu_m)**(-2.1)))*special.gamma(p+1)
    return S_norm*(p+1)*factor1*factor2


def Bic98_FFA_CIbreak(freq, S_norm, alpha, p, nu_m, nu_br):
    dummyalpha, dummyfreq = CIbreak(freq, alpha, nu_br)
    return Bic98_FFA(freq, S_norm, alpha, p, nu_m)*dummyfreq**dummyalpha


def Bic98_FFA_JPbreak(freq, S_norm, alpha, p, nu_m, nu_br):
    return Bic98_FFA(freq, S_norm, alpha, p, nu_m)*JPbreak(freq, nu_br)


def CIbreak(freq, alpha, nu_br):
    alpha = np.where(freq <= nu_br, alpha, alpha-0.5)
    dummyfreq = freq / nu_br
    return alpha, dummyfreq


def JPbreak(freq, nu_br):
    return np.exp(-freq / nu_br)


def flux_at_freq(freq, known_freq, known_flux, alpha):

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
    """Return matplotlib ticks in LaTeX format, getting the value as integer [0,99],
    a 1 digit float [0.1, 0.9], or otherwise n*10^m.

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

    # get the exponent and base
    exp = np.floor(np.log10(value))
    base = value/10**exp

    # format according to values
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


def plot_spectra(freqs, fluxes, errs, models, names, params, param_errs, rcs,
                 BICs, colours, labels, figname, annotate=True,
                 model_selection='better'):
    """Plot a figure of the radio spectra of an individual source,
       according to the input data and models.

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

    # create SEDs directory if doesn't already exist
    if not os.path.exists('SEDs'):
        os.mkdir('SEDs')

    # fig=plt.figure()
    ax = plt.subplot()

    # plot frequency axis 20% beyond range of values
    xlin = np.linspace(min(freqs)*0.8, max(freqs)*1.2, num=5000)
    plt.ylabel(r'Flux Density $S$ (mJy)')
    plt.xlabel(r'Frequency $\nu$ (GHz)')
    plt.xscale('log')
    plt.yscale('log')

    # adjust the tick values and add grid lines at minor tick locations
    subs = [1.0, 2.0, 5.0]
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format_freq))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(ticks_format_flux))
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

    # plot flux measurements
    plt.errorbar(freqs, fluxes, yerr=errs, linestyle='none', marker='.', c='r', zorder=15)

    best_bic = 0
    dBIC = 3
    offset = 0
    plotted_models = 0

    # plot each model
    for i in range(len(models)):
        ylin = models[i](xlin, *params[i])
        txt = "{0}:\n   {1}".format(labels[i], r'$\chi^2_{\rm red} = %.1f$' % rcs[i])

        # compare BIC values
        bic = BICs[i]
        if i > 0:
            dBIC = best_bic - bic
            if model_selection != 'best':
                txt += ', {0}'.format(r'$\Delta{\rm BIC} = %.1f$' % (dBIC))
        if dBIC >= 3:
            best_bic = bic

        # plot model if selected according to input
        if model_selection == 'all' or (model_selection == 'better' and
                                        dBIC >= 3) or (model_selection == 'best' and
                                                       BICs[i] == min(BICs)):

            plotted_models += 1
            plt.plot(xlin, ylin, c=colours[i], linestyle='--', zorder=i+1, label=labels[i])
            plt.legend(scatterpoints=1, fancybox=True, frameon=True, shadow=True)
            txt += '\n'

            # add each fitted parameter to string (in LaTeX format)
            for j, param in enumerate(names[i]):
                units = ''
                tokens = param.split('_')
                if len(tokens[0]) > 1:
                    tokens[0] = "\\" + tokens[0]
                if len(tokens) > 1:
                    param = r'%s_{\rm %s}' % (tokens[0], tokens[1])
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

                txt += '   ' + r'${0}$ = {1} $\pm$ {2} {3}'.format(param, val, err, units) + '\n'

            # annotate all fit info if it will fit on figure
            if annotate and plotted_models <= 3:
                plt.text(offset, 0, txt, horizontalalignment='left', verticalalignment='bottom',
                         transform=ax.transAxes)
                offset += 0.33

    # write figure and close
    plt.savefig('SEDs/{0}'.format(figname))
    plt.close()
    return


def likelihood(ydata, ymodel, yerrs):
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

    return np.prod((1 / (yerrs*np.sqrt(2*np.pi))) * np.exp((-1/(2*yerrs**2)) * (ydata-ymodel)**2))


def fit_info(ydata, ymodel, yerrs, deg):
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

    chi_sq = np.sum(((ydata-ymodel)/yerrs)**2)
    DOF = len(ydata) - deg
    red_chi_sq = chi_sq/DOF
    BIC = -2*np.log(likelihood(ydata, ymodel, yerrs)) + deg * np.log(len(ydata))
    return red_chi_sq, BIC


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

    # directly derive alpha and error from two fluxes
    alpha = np.log10(fluxes[0]/fluxes[1]) / np.log10(freqs[0]/freqs[1])
    alpha_err = np.sqrt((errs[0]/fluxes[0])**2 + (errs[1]/fluxes[1])**2)/np.log10(freqs[0]/freqs[1])
    flux = flux_at_freq(freq, freqs[0], fluxes[0], alpha)
    return alpha, alpha_err, flux


def SED(freq, freqs, fluxes, errs, models='pow', figname=None):
    """Fit SED models to an individual source and return the model params and
    errors along with the expected flux at a given frequency, for each input model.
    Lists must be the same length and contain at least two elements, all with
    the same units (ideally MHz and Jy).

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
        Write a figure of the radio spectra and model to file, using this filename.
        Use None to not write to file.

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

    # initial guesses of different params
    S_max = max(fluxes)
    nu_max = freqs[fluxes == S_max][0]
    alpha = -0.8
    beta = 1-2*alpha
    nu_br = np.mean(freqs)
    p = 0.5

    # initial guesses of different models
    params = {'pow': [S_max, alpha],
              'powcibreak': [S_max, alpha, nu_br],
              'powjpbreak': [S_max, alpha, nu_br],
              'curve': [S_max, nu_max, 1, alpha],
              'ssa': [S_max, beta, nu_max],
              'ssacibreak': [S_max, beta, nu_max, nu_br],
              'ssajpbreak': [S_max, beta, nu_max, nu_br],
              'ffa': [S_max, alpha, nu_max],
              'bicffa': [S_max, alpha, p, nu_max],
              'bicffacibreak': [S_max, alpha, p, nu_max, nu_br],
              'bicffajpbreak': [S_max, alpha, p, nu_max, nu_br]}

    # different SED models from functions above
    funcs = {'pow': powlaw,
             'powcibreak': pow_CIbreak,
             'powjpbreak': pow_JPbreak,
             'curve': curve,
             'ssa': SSA,
             'ssacibreak': SSA_CIbreak,
             'ssajpbreak': SSA_JPbreak,
             'ffa': FFA,
             'bicffa': Bic98_FFA,
             'bicffacibreak': Bic98_FFA_CIbreak,
             'bicffajpbreak': Bic98_FFA_JPbreak}

    # matplotlib colours
    colours = {'pow': 'black',
               'powcibreak': 'b',
               'powjpbreak': 'violet',
               'curve': 'r',
               'ssa': 'g',
               'ssacibreak': 'r',
               'ssajpbreak': 'g',
               'ffa': 'orange',
               'bicffa': 'r',
               'bicffacibreak': 'b',
               'bicffajpbreak': 'r'}

    # labels
    labels = {'pow': 'Power law',
              'powcibreak': 'Power law\n + CI break',
              'powjpbreak': 'Power law\n + JP break',
              'curve': 'Tschager+03 Curve',
              'ssa': 'Single SSA',
              'ssacibreak': 'Single SSA\n + CI break',
              'ssajpbreak': 'Single SSA\n + JP break',
              'ffa': 'Single FFA',
              'bicffa': 'Bicknell+98 FFA',
              'bicffacibreak': 'Bicknell+98 FFA\n + CI break',
              'bicffajpbreak': 'Bicknell+98 FFA\n + JP break'}

    # store used models, fitted parameters and errors, fitted fluxes, reduced chi squared and BIC
    fit_models, fit_params, fit_param_errors, fitted_fluxes, rcs = [], [], [], [], []
    BICs = np.array([])

    # convert single model to list
    if type(models) is str:
        models = [models]

    for model in models:
        model = model.lower()

        # fit model if DOF >= 1
        if len(freqs) >= len(params[model])+1:
            try:
                # perform a least squares fit
                popt, pcov = opt.curve_fit(funcs[model], freqs, fluxes, p0=params[model],
                                           sigma=errs, maxfev=10000)

                # add all fit info to lists
                fit_models.append(model)
                fit_params.append(popt)
                fit_param_errors.append(np.sqrt(np.diag(pcov)))
                RCS, bic = fit_info(fluxes, funcs[model](freqs, *popt), errs, len(popt))
                rcs.append(RCS)
                BICs = np.append(BICs, bic)
                fitted_fluxes.append(funcs[model](freq, *popt))
            except (ValueError, RuntimeError) as e:
                print("Couldn't find good fit for {0} model.".format(model))
                print(e)

    # get lists of names, functions, colours and labels for all used models
    names = [funcs[model].__code__.co_varnames[1:funcs[model].__code__.co_argcount]
             for model in fit_models]
    funcs = [funcs[model] for model in fit_models]
    colours = [colours[model] for model in fit_models]
    labels = [labels[model] for model in fit_models]

    # write figure for this source
    if figname is not None and len(fit_models) > 0:
        plot_spectra(freqs, fluxes, errs, funcs, names, fit_params, fit_param_errors,
                     rcs, BICs, colours, labels, figname, model_selection='all')

    return fit_models, names, fit_params, fit_param_errors, fitted_fluxes, rcs, BICs


def get_pixel_area(fits, flux=0, nans=False, ra_axis=0, dec_axis=1, w=None):
    """For a given image, get the area and solid angle of all non-nan pixels or
    all pixels below a certain flux (doesn't count pixels=0). The RA and DEC axes
    follow the WCS convention (i.e. starting from 0).

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

    # count the pixels and derive area and solid angle of all these pixels
    if nans:
        count = fits.data[(~np.isnan(fits.data)) & (fits.data != 0)].shape[0]
    else:
        count = fits.data[(fits.data < flux) & (fits.data != 0)].shape[0]

    area = (count*np.abs(w.wcs.cdelt[ra_axis])*np.abs(w.wcs.cdelt[dec_axis]))
    solid_ang = area*(np.pi/180)**2
    return area, solid_ang


def axis_lim(data, func, perc=10):
    """Return an axis limit value a certain % beyond the min/max value of a dataset.

    Arguments:
    ----------
    data : list-like
        A list-like object input into min() or max(). Usually this will be a numpy array or
        pandas Series.
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
    """Return the median, mean, standard deviation, standard error and rms of
    the median absolute deviation (mad) of the non-nan values in a list.

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

    # remove nan indices, as these affect the calculations
    values = data[~np.isnan(data)]

    med = np.median(values)
    mean = np.mean(values)
    std = np.std(values)
    sterr = std / np.sqrt(len(values))
    rms_mad = np.median(np.abs(values-np.median(values)))/0.6745

    return med, mean, std, sterr, rms_mad


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

    # do this in case more than one '.' in file name
    return '.'.join(filename.split('.')[:-1])


def config2dic(filepath, main_dir, verbose=False):
    """Read a configuration file and create an dictionary of arguments
    from its contents, which will usually be passed into a new object instance.

    Arguments:
    ----------
    filepath : string
        The absolute filepath of the config file.
    main_dir : string
        Main directory that contains all the necessary files.

    Keyword Arguments:
    ------------------
    verbose : bool
        Verbose output.

    Returns:
    --------
    args_dict : dict
        A dictionary of arguments, to be passed into some function, usually a new object
        instance."""

    # open file and read contents
    config_file = open(filepath)
    txt = config_file.read()
    args_dict = {}

    # set up dictionary of arguments based on their types
    for line in txt.split('\n'):
        if len(line) > 0 and line.replace(' ', '')[0] != '#':
            # use '=' as delimiter and strip whitespace
            split = line.split('=')
            key = split[0].strip()
            val = split[1].strip()
            val = parse_string(val)

            # if parameter is filename, store the filepath
            if key == 'filename':
                val = find_file(val, main_dir, verbose=verbose)

            args_dict.update({key: val})

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
    elif val.lower() in ('true', 'false'):
        val = (val.lower() == 'true')
    elif val.replace('.', '', 1).replace('e', '').replace('-', '').isdigit():
        val = float(val)

    return val


def new_path(filepath):
    """For a given input filepath, return the path after having moved into a new
    directory. This will add '../' to the beginning of relative filepaths.

    Arguments:
    ----------
    filepath : string
        The filepath.

    Returns:
    --------
    filepath : string
        The updated filepath."""

    # add '../' to filepath if it's a relative filepath
    if filepath is not None and filepath[0] != '/':
        filepath = '../' + filepath

    return filepath


def find_file(filepath, main_dir, verbose=True):

    """Look for a file in specific paths. Look one directory up if filepath is relative,
    otherwise look in main directory, otherwise raise exception.

    Arguments:
    ----------
    filepath : string
        An absolute or relative filepath.
    main_dir : string
        Main directory that contains all the necessary files.

    Returns:
    --------
    filepath : string
        The path to where the file was found."""

    # raise exception if file still not found
    if not (os.path.exists(filepath) or os.path.exists('{0}/{1}'.format(main_dir, filepath))
            or os.path.exists(new_path(filepath))):
        raise Exception("Can't find file - {0}. Ensure this file is in input "
                        "path or --main-dir.\n".format(filepath))
    # otherwise update path to where file exists
    elif not os.path.exists(filepath):
        # look in main directory if file doesn't exist in relative filepath
        if os.path.exists('{0}/{1}'.format(main_dir, filepath)):
            if verbose:
                print("Looking in '{0}' for '{1}'.".format(main_dir, filepath))
            filepath = '{0}/{1}'.format(main_dir, filepath)
        # update directory path if file is relative path
        else:
            filepath = new_path(filepath)

    return filepath
