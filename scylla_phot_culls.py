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
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
plt.ion()


def make_cuts(catalog_in,
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
    print('New catalog: \'%s\'' % catalog_out)

    if plot:

        inds_noncut = np.where((t[filters[0] + '_FLAG'] < 99) & (t[filters[1] + '_FLAG'] < 99))[0]
        inds_cut = np.where((t[filters[0] + '_FLAG'] >= 99) | (t[filters[1] + '_FLAG'] >= 99))[0]
        ra = t['RA'][inds_noncut]
        dec = t['DEC'][inds_noncut]

        # read in original catalog
        t_st = Table.read(catalog_in)
        ra_st = t_st['RA']
        dec_st = t_st['DEC']

        if errcut:
            errcutstr = '_errcut'
        else:
            errcutstr = ''

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey='row', figsize=(8, 7))
        fig.suptitle('Scylla %s%s (vgst catalog cuts)' % (field_id, errcutstr))

        # spatial plot of original st catalog
        ax1.plot(ra_st, dec_st, ',', ls='')
        ax1.set_xlabel('RA', fontsize=15)
        ax1.set_ylabel('DEC', fontsize=15)
        ax1.legend(handles=[Line2D([0], [0], color='#126D94', marker='o', markersize=3, ls='',
                                   label='st catalog, N = %s' % len(ra_st))],
                   loc='upper right', fontsize='small')

        # spatial plot of the culled vgst catalog
        ax2.plot(ra, dec, ',', ls='')
        ax2.set_xlabel('RA', fontsize=15)
        ax2.legend(handles=[Line2D([0], [0], color='#126D94', marker='o', markersize=3, ls='',
                                   label='vgst catalog, N = %s' % len(ra))],
                   loc='upper right', fontsize='small')

        if len(filters) > 1:
            inds_vega_non99 = np.where((t[filters[0] + '_VEGA'] < 99) & (t[filters[1] + '_VEGA'] < 99))[0]
            inds_cmd = np.intersect1d(inds_noncut, inds_vega_non99)
            mag1 = t[filters[0] + '_VEGA'][inds_cmd]
            mag2 = t[filters[1] + '_VEGA'][inds_cmd]

            mag1_cut = t_st[filters[0] + '_VEGA'][inds_cut]
            mag2_cut = t_st[filters[1] + '_VEGA'][inds_cut]

            # Set up CMD Kernel Density Estimation (KDE) scatter plot to see the density of removed sources
            print('Setting up KDE scatter plot for bad (removed) sources...')
            start_time_kde = time.time()
            xy_cut = np.vstack([(mag1_cut-mag2_cut), mag1_cut])
            z_cut = gaussian_kde(xy_cut)(xy_cut)
            idx_cut = z_cut.argsort()
            x_cut, y_cut, z_cut = (mag1_cut-mag2_cut)[idx_cut], mag1_cut[idx_cut], z_cut[idx_cut]
            print('Scatter plot set up in %.2f sec.' % round((time.time() - start_time_kde), 2))

            # CMD of sources removed from the st catalog
            ax3.scatter(x_cut, y_cut, c=z_cut, s=3, cmap='RdPu_r')
            ax3.set_xlabel('%s - %s' % (filters[0], filters[1]), fontsize=15)
            ax3.set_ylabel('%s' % filters[0], fontsize=15)
            ax3.set_xlim(-6, 8)
            ax3.set_ylim(32.5, min(mag1_cut)-1.5)
            ax3.legend(handles=[Line2D([0], [0], color='#A91864', marker='o', markersize=3, ls='',
                                       label='Removed, N = %s' % len(mag1_cut))],
                       loc='upper right', fontsize='small')

            # Set up KDE for remaining oved sources
            print('Setting up KDE scatter plot for good (kept) sources...')
            start_time_kde = time.time()
            xy = np.vstack([(mag1-mag2), mag1])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = (mag1-mag2)[idx], mag1[idx], z[idx]
            print('Scatter plot set up in %.2f sec.' % round((time.time() - start_time_kde), 2))

            # CMD of sources remaining in the vgst catalog
            ax4.scatter(x, y, c=z, s=3, cmap='GnBu_r')
            ax4.set_xlabel('%s - %s' % (filters[0], filters[1]), fontsize=15)
            ax4.set_xlim(-6, 8)
            ax4.set_ylim(32.5, min(mag1)-1.5)
            ax4.legend(handles=[Line2D([0], [0], color='#35C0D6', marker='o', markersize=3, ls='',
                                       label='Kept (vgst), N = %s' % len(mag1))],
                       loc='upper right', fontsize='small')

        plt.savefig(catalog_out.replace('vgst.fits', '%spng' % errcutstr))
        print('Plot saved as \'%s\'' % catalog_out.replace("vgst.fits", "%spng" % errcutstr))
        plt.close()


if __name__ == "__main__":

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
