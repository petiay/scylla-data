# coding: utf-8
"""
Make quality cuts on the Scylla photometry, save new catalog, and optionally plot data spatially and in a CMD.
Petia YMJ, Mar 2021
"""
import argparse
import os
import time
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.ion()


def make_cuts(catalog_in='data/smc_6-15891_SMC-3956ne-9632/proc_default/15891_SMC-3956ne-9632.st.fits',
              catalog_out=None,
              filters=['F475W', 'F814W'],
              crowdcut=[0.3],
              sharpcut=[0.15],
              roundcut=[0.6],
              errcut=None,
              plot=False):
    """
    Make quality cuts on the input photometry catalog file and record them in a new catalog file.

    Parameters
    ----------
    catalog_in : str
        catalog file to which to apply photometry quality cuts
    catalog_out : str
        catalog file to which to record new photometry
    filters : str or a list of strings
        HST filters to which to apply quality cuts. Example: ['F336W','F475W','F814W']
    crowdcut : float or a list of floats
        If single value, cut is applied to all filters. If multiple values, the number of values has to match the
        number of filters, and the cuts are applied in order based on the filter order.
    sharpcut : float or a list of floats
        If single value, cut is applied to all filters. If multiple values, the number of values has to match the
        number of filters, and the cuts are applied in order based on the filter order.
    roundcut : float or a list of floats
        If single value, cut is applied to all filters. If multiple values, the number of values has to match the
        number of filters, and the cuts are applied in order based on the filter order.
    errcut : float or a list of floats
        If single value, cut is applied to all filters. If multiple values, the number of values has to match the
        number of filters, and the cuts are applied in order based on the filter order.
    plot : Boolean
        If True, generate two plots as a check and a visualization of the data with quality cuts:
        1) spatial plot
        2) CMDs: if more than one filter is provided, plot a CMD with color=filter1-filter2 and mag=filter1
    """

    field_id = catalog_in.split('.st')[0].split('/15891_')[1]
    print('Culling catalog for %s ' % field_id)

    if catalog_out is None:
        catalog_out = catalog_in.replace('.st', '.vgst')

    if errcut:
        catalog_out = catalog_out.replace('.fits', '_errcut.fits')
        print(catalog_out)
        if len(errcut) < len(filters):
            errcut = np.linspace(errcut, errcut, len(filters))

    # turn single-valued into multi-valued arrays for proper referencing in the for loop below
    if len(crowdcut) < len(filters):
        crowdcut = np.linspace(crowdcut, crowdcut, len(filters))
    if len(sharpcut) < len(filters):
        sharpcut = np.linspace(sharpcut, sharpcut, len(filters))
    if len(roundcut) < len(filters):
        roundcut = np.linspace(roundcut, roundcut, len(filters))

    # Create a copy of the catalog file to work with
    os.system("cp " + catalog_in + " " + catalog_out)

    t = Table.read(catalog_out)
    start_time_cull = time.time()

    # Set the following entries outside the desired cut range to 99.999, or 99 ('FLAG')
    # VEGA, STD, ERR, CHI, SNR, SHARP, ROUND, CROWD, FLAG
    for i in range(len(filters)):
        if filters[i] == 'F336W':
            continue
        inds_to_cut = np.where((t[filters[i] + '_CROWD'] > crowdcut[i]) |
                               (t[filters[i] + '_SHARP'] * t[filters[i] + '_SHARP'] > sharpcut[i]) |
                               (t[filters[i] + '_ROUND'] > roundcut[i]))[0]
        if errcut:
            errinds_to_cut = np.where(t[filters[i] + '_ERR'] > errcut)[0]
            inds_to_cut = np.unique(np.concatenate((inds_to_cut, errinds_to_cut), 0))

        t[filters[i] + '_VEGA'][inds_to_cut] = 99.999
        t[filters[i] + '_STD'][inds_to_cut] = 99.999
        t[filters[i] + '_ERR'][inds_to_cut] = 99.999
        t[filters[i] + '_CHI'][inds_to_cut] = 99.999
        t[filters[i] + '_SNR'][inds_to_cut] = 99.999
        t[filters[i] + '_SHARP'][inds_to_cut] = 99.999
        t[filters[i] + '_ROUND'][inds_to_cut] = 99.999
        t[filters[i] + '_CROWD'][inds_to_cut] = 99.999
        t[filters[i] + '_FLAG'][inds_to_cut] = 99

    t.write(catalog_out, overwrite=True)
    print('Done culling in %.2f sec.' % round((time.time()-start_time_cull), 2))
    print('New catalog:' % catalog_out)

    if plot:

        inds_noncut = np.where((t[filters[0] + '_FLAG'] < 99) & (t[filters[1] + '_FLAG'] < 99))[0]
        ra = t['RA'][inds_noncut]
        dec = t['DEC'][inds_noncut]

        if errcut:
            errcutstr = '_errcut'
        else:
            errcutstr = ''

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 4.5))
        fig.suptitle('Scylla %s%s (vgst catalog)' % (field_id, errcutstr))
        ax1.plot(ra, dec, ',', ls='', label='N = %s' % len(ra))
        ax1.set_xlabel('RA', fontsize=15)
        ax1.set_ylabel('DEC', fontsize=15)
        ax1.legend(loc='upper right')

        # only plot CMDs if multiple filters are provided
        if len(filters) > 1:
            inds_non99 = np.where((t[filters[0] + '_VEGA'] < 99) & (t[filters[1] + '_VEGA'] < 99))[0]
            inds_cmd = np.intersect1d(inds_noncut, inds_non99)
            mag1 = t[filters[0] + '_VEGA'][inds_cmd]
            mag2 = t[filters[1] + '_VEGA'][inds_cmd]

            # Set up CMD Kernel Density Estimation (KDE) scatter plot to see the density of sources
            print('Setting up KDE scatter plot...')
            start_time_kde = time.time()
            xy = np.vstack([(mag1-mag2), mag1])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = (mag1-mag2)[idx], mag1[idx], z[idx]
            print('Scatter plot set up in %.2f sec.' % round((time.time()-start_time_kde), 2))
            ax2.scatter(x, y, c=z, s=3, cmap='GnBu_r', label='N = %s' % len(mag1))

            ax2.set_xlabel('%s - %s' % (filters[0], filters[1]), fontsize=15)
            ax2.set_ylabel('%s' % filters[0], fontsize=15)
            ax2.set_xlim(-2, 5)
            ax2.set_ylim(32.5, min(mag1)-0.5)
            ax2.legend(loc='upper right')

        plt.savefig('scylla-%s%s.png' % (field_id, errcutstr))
        print('Plot saved as scylla-%s%s.png' % (field_id, errcutstr))
        plt.close()


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog_in", type=str, help="File name of the input catalog",
    )
    parser.add_argument(
        "--catalog_out", default=None, type=str, help="desired name of output catalog. If none given, '.st' in the"
                                                      "input catalog is changed to '.vgst' by default"
    )
    parser.add_argument(
        "--filters", default=['F475W', 'F814W'], type=float, help="filters to which to apply cuts"
    )
    parser.add_argument(
        "--crowdcut", default=[0.3], type=float, help="desired crowding cut(s)"
    )
    parser.add_argument(
        "--sharpcut", default=[0.15], type=float, help="desired sharpness squared cut(s)"
    )
    parser.add_argument(
        "--roundcut", default=[0.6], type=float, help="desired rounding cut(s)"
    )
    parser.add_argument(
        "--errcut", default=None, type=float, help="desired error cut(s)"
    )
    parser.add_argument(
        "--plot", default=False, type=bool, help="make two plots: spatial and CMD after the cuts are applied"
    )
    args = parser.parse_args()

    make_cuts(args.catalog_in, catalog_out=args.catalog_out, filters=args.filters, crowdcut=args.crowdcut,
              sharpcut=args.sharpcut, roundcut=args.roundcut, errcut=args.errcut, plot=args.plot)
