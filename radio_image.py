from __future__ import division
from functions import *
import os
import numpy as np

from astropy.io import fits as f
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning

import warnings
from inspect import currentframe, getframeinfo

#ignore annoying astropy warnings and set my own obvious warning output
warnings.simplefilter('ignore', category=AstropyWarning)
cf = currentframe()
WARN = '\n\033[91mWARNING: \033[0m' + getframeinfo(cf).filename

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

    def correct_img(self,dRA,dDEC,flux_factor=1.0):

        """Correct the header of the fits image from this instance, by shifting the reference positions,
        and optionally multiplying the pixels by a factor. Write the image to 'name_corrected.fits'.

        Arguments:
        ----------
        dRA : float
            The RA offset in SECONDS. NOTE: This is not in arcsec.
        dDEC : float
            The DEC offset in arcsec.

        Keyword Arguments:
        ------------------
        flux_factor : float
            The factor by which to mutiply all pixels."""

        filename = '{0}_corrected.fits'.format(self.basename)
        print "Correcting header of fits image and writing to '{0}'".format(filename)
        print "Shifting RA by {0} seconds and DEC by {1} arcsec".format(dRA,dDEC)

        if flux_factor != 1.0:
            print "Multiplying image by {0}".format(flux_factor)

        #Shift the central RA/DEC in degrees, and multiply the image by the flux factor (x1 by default)
        #WCS axes start at 0 but fits header axes start at 1
        self.fits.header['CRVAL' + str(self.ra_axis+1)] += dRA/3600
        self.fits.header['CRVAL' + str(self.dec_axis+1)] += dDEC/3600
        self.fits.data[0][0] *= flux_factor
        self.fits.writeto(filename,clobber=True)
