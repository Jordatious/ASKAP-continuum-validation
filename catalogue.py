from __future__ import division
from functions import *
import os
import glob
import numpy as np
import pandas as pd

from astropy.io import fits as f
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table
from astropy.utils.exceptions import AstropyWarning

import warnings
from inspect import currentframe, getframeinfo

#ignore annoying astropy warnings and set my own obvious warning output
warnings.simplefilter('ignore', category=AstropyWarning)
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename


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

            #specific fix for GLEAM catalogue
            if finder is not None and finder.lower() == 'aegean':
                self.col_suffix = ''
            self.flux_unit = flux_unit

            self.ra_col=self.unique_col_name(ra_col)
            self.dec_col=self.unique_col_name(dec_col)
            self.ra_fmt = ra_fmt
            self.dec_fmt = dec_fmt
            self.si_col=None

        self.use_peak = use_peak

        self.flux_unit = self.flux_unit.lower()
        if self.flux_unit not in ('jy','mjy','ujy'):
            warnings.warn_explicit("Unrecognised flux unit '{0}'. Use 'Jy', 'mJy' or 'uJy'. Assuming 'Jy'\n".format(flux_unit),UserWarning,WARN,cf.f_lineno)
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

        if img is not None:
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
            self.img_flux = np.sum(img_data[~np.isnan(img_data)]) / (1.133*((img.bmaj * img.bmin) / (img.raPS * img.decPS))) #divide by beam area

        #Get the approximate area from catalogue
        else:
            if self.name not in self.coords.keys():
                self.set_key_fields()
            self.ra_bounds = (max(self.ra[self.name]),min(self.ra[self.name]))
            dRA = (self.ra_bounds[0] - self.ra_bounds[1])*np.cos(np.deg2rad(np.mean(self.dec[self.name])))
            self.dec_bounds = (max(self.dec[self.name]),min(self.dec[self.name]))
            dDEC = abs(self.dec_bounds[0] - self.dec_bounds[1])
            self.area = dRA*dDEC
            self.img_peak = np.max(self.flux[self.name])
            self.img_rms = int(np.median(self.rms[self.name])*1e6) #uJy
            self.dynamic_range = self.img_peak/self.img_rms
            self.img_flux = np.nan

        self.blends = len(np.where(self.df[self.island_col].value_counts() > 1)[0])
        self.cat_flux = np.sum(self.flux[self.name])

        #get median spectral index if column exists
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
            #Josh's method
            self.uncertainty = np.sqrt(self.df[self.flux_err_col]**2 + self.df[self.peak_err_col]**2)
            if self.finder == 'selavy':
                self.uncertainty += np.sqrt(self.df[self.rms_val]**2)
            self.sigma = (self.df[self.flux_col] - self.df[self.peak_col]) / self.uncertainty
            self.resolved = np.where(self.sigma > 3)[0]
            self.resolved_frac = len(self.resolved) / len(self.df)

            #Tom's method
            # self.R = np.log(self.df[self.flux_col] / self.df[self.peak_col])
            # self.uncertainty = np.sqrt((self.df[self.flux_err_col]/self.df[self.flux_col])**2 + (self.df[self.peak_err_col]/self.df[self.peak_col])**2)
            # if self.finder == 'selavy':
            #     self.uncertainty += np.sqrt(self.df[self.rms_val]**2)
            # self.resolved = np.where(self.R > 2*self.uncertainty)[0]
            # self.resolved_frac = len(self.resolved) / len(self.df)
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

                if self.name in self.si.keys():
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


    def filter_sources(self,flux_lim=0,SNR=0,ratio_frac=0,ratio_sigma=0,reject_blends=False,psf_tol=0,
                        resid_tol=0,flags=False,file_suffix='',redo=False,write=True,verbose=False):

        """Reject problematic sources according to several criteria. This will overwrite self.df.

        Keyword arguments:
        ------------------
        flux_lim : float
            The flux density limit in Jy, below which sources will be rejected.
        SNR : float
            The S/N ratio limit (where N is the rms), below which sources will be rejected. Use 0 to skip this step.
        ratio_frac : float
            The fraction given by the integrated flux divided by the peak flux, above which, sources will be rejected. Use 0 to skip this step.
        ratio_sigma : float
            Reject sources based on the validation metric for resolved sources, based on their int/peak, above this sigma value.
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

            #reject resolved sources based on flux ratio metric
            if ratio_sigma != 0:
                if self.peak_col != None:
                    uncertainty = np.sqrt(self.df[self.flux_err_col]**2 + self.df[self.peak_err_col]**2)
                    if self.finder == 'selavy':
                        uncertainty += np.sqrt(self.df[self.rms_val]**2)
                    resolved = self.df[self.flux_col] - self.df[self.peak_col] > ratio_sigma * uncertainty
                    self.df = self.df[~resolved]
                    if verbose:
                        print "Rejected (resolved) sources according to int/peak metric, above {0} sigma.".format(ratio_sigma)
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
            self.set_key_fields(indices=self.df.index.tolist(),set_coords=False)
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
        self.overwrite_df(matched_df)

        #add key fields from matched catalogue
        self.set_key_fields(cat=cat)


    def fit_spectra(self,cat_name=None,match_perc=0,models=['pow'],fig_extn=None,GLEAM_subbands=None,GLEAM_nchans=None,fit_flux=False,redo=False,write=True):

        """Derive radio spectra for this catalogue, using the input SED models. This will add new columns to the table, including the spectral index
        and error, and optionally, the fitted flux at the frequency of this instance and the ratio between this and the measured flux.

        Keyword arguments:
        ------------------
        cat_name : string
            Derive spectral indices between the catalogue given by this dictionary key and the main catalogue from this instance. If None is input, all data
            except the catalogue from this instance will be used, and the flux at its frequency will be derived. If 'all' is input, all data will be used.
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
            print "Deriving the flux at {0} MHz assuming a typical spectral index of {1}.".format(freq,spec_index)

        self.df[fitted_flux_col] = flux_at_freq(self.freq[self.name],self.freq[cat],self.flux[cat],spec_index)
        self.df[fitted_ratio_col] = self.flux[self.name] / self.df[fitted_flux_col]

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


    def process_config_file(self,config_file,main_dir,redo=False,write_all=True,write_any=True,verbose=False):

        """For a given catalogue config file, read the paramaters into a dictionary, pass it into a
        new catalogue object, cut out a box, cross-match to this instance, and derive the spectral index.

        Arguments:
        ----------
        config_file : string
            The filepath to a configuration file for a catalogue.
        main_dir : string
            Main directory that contains all the necessary files.

        Keyword arguments:
        ------------------
        redo : bool
            Re-do all processing, even if output files produced.
        write_all : bool
            Write all files during processing. If False, cutout file will still be written.
        write_any : bool
            Write any files whatsoever?"""

        #create dictionary of arguments, append verbose and create new catalogue instance
        config_dic = config2dic(config_file,main_dir,verbose=verbose)
        config_dic.update({'verbose' : verbose})
        if redo:
            config_dic['autoload'] = False
        cat = catalogue(**config_dic)

        #Cut out a box in catalogues within boundaries of image, cross-match and derive spectral indices
        cat.cutout_box(self.ra_bounds,self.dec_bounds,redo=redo,verbose=verbose,write=write_any)
        self.cross_match(cat,redo=redo,write=write_all)
        if cat.name in self.cat_list and self.name in self.flux.keys():
            self.fit_spectra(cat_name=cat.name,redo=redo,write=write_all)
