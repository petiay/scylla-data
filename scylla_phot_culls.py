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


def make_cuts(
    catalog_in,
    catalog_out=None,
    filters=["F475W", "F814W"],
    crowdcut=[0.3],
    sharpcut=[0.15],
    roundcut=[0.6],
    errcut=None,
    ast_apply=False,
    plot=False,
):
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
    ast_apply: boolean (default=False)
        If True, do not remove cut sources from catalog, just flag them by introducing new flag ("CUT_FLAG").
        If False (default), remove cut sources from catalog.
    plot : bool
        Generates spatial plot of the original and the trimmed catalogs, and CMDs of the removed and remaining sources.
    """

    field_id = catalog_in.split(".st")[0].split("_")[-1]
    print("Culling catalog for %s " % field_id)

    # import st catalog
    t_st = Table.read(catalog_in)

    # get all filters present in a catalog
    filters_list = []
    for i in range(len(t_st.colnames)):
        if "_" in t_st.colnames[i]:
            filt_str = t_st.colnames[i].split("_")[0]
            if filt_str in filters_list:
                continue
            elif (filt_str != "RA") & (filt_str != "DEC"):
                filters_list.append(filt_str)

    # check that all the required filters are present
    for i in range(len(filters)):
        if filters[i] not in filters_list:
            print("Error: Filter '{0}' missing in catalog".format(filters[i]))
            exit()

    if catalog_out is None:
        catalog_out = catalog_in.replace(".st", ".vgst")

    if errcut:
        catalog_out = catalog_out.replace(".fits", "_errcut.fits")
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

    # check what the RA/DEC naming convention in the catalog is, define keywords to be used
    if "RA" in t.colnames:
        ra_key = "RA"
        dec_key = "DEC"
    else:
        ra_key = "RA_J2000"
        dec_key = "DEC_J2000"

    # Setting FLAG=99 for crowd, sharp and round cuts
    for i in range(len(filters)):
        if filters[i] not in ("F475W", "F814W"):
            continue

        inds_to_cut = np.where(
            (t[filters[i] + "_CROWD"] > crowdcut[i])
            | (t[filters[i] + "_SHARP"] * t[filters[i] + "_SHARP"] > sharpcut[i])
            | (t[filters[i] + "_ROUND"] > roundcut[i])
        )[0]

        if errcut:
            errinds_to_cut = np.where(t[filters[i] + "_ERR"] > errcut)[0]
            inds_to_cut = np.unique(np.concatenate((inds_to_cut, errinds_to_cut), 0))

        # flag sources not passing quality cuts
        t[filters[i] + "_FLAG"][inds_to_cut] = 99

    # get all filters present in a catalog
    filters_list = [x.split('_')[0] for x in t.colnames if "VEGA" in x]

    # Remove flux=0, flag!=0 or flag!=2 sources
    bad_srcs_list = []
    for i in range(len(filters_list)):
        flux_zero = np.where(t[filters_list[i] + "_RATE"] == 0.0)[0]
        flag_not_0_or_2 = np.where(
            (t[filters_list[i] + "_FLAG"] == 1)
            | (t[filters_list[i] + "_FLAG"] == 3)
            | (t[filters_list[i] + "_FLAG"] == 99)
        )[0]

        bad_srcs = np.unique(np.concatenate((flux_zero, flag_not_0_or_2), 0))
        bad_srcs_list.append(bad_srcs)

    bad_srcs_flat_list = np.unique(
        [item for sublist in bad_srcs_list for item in sublist]
    )
    if ast_apply:
        t["CUT_FLAG"] = np.zeros(len(t[ra_key]), dtype=int)
        t["CUT_FLAG"][bad_srcs_flat_list] = 1
    else:
        t.remove_rows(bad_srcs_flat_list)
    t.write(catalog_out, overwrite=True)

    print("%s sources removed from catalog." % len(bad_srcs_flat_list))
    print("%s sources remaining in catalog." % len(t[ra_key]))
    print("New catalog: '%s'" % catalog_out.split("/")[-1])

    if plot:
        # read in original st catalog
        t_st_cov = Table.read(catalog_in)
        t_st_phot = Table.read(catalog_in)

        if errcut:
            errcutstr = "_errcut"
        else:
            errcutstr = ""

        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharey="row", figsize=(15, 7))
        fig.suptitle("Scylla %s%s (vgst catalog cuts)" % (field_id, errcutstr))

        # spatial plot of st catalog
        ax1.plot(t_st[ra_key], t_st[dec_key], ",", ls="")
        ax1.set_xlabel("RA", fontsize=15)
        ax1.set_ylabel("DEC", fontsize=15)
        ax1.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#126D94",
                    marker="o",
                    markersize=3,
                    ls="",
                    label="st catalog, N = %s" % len(t_st[ra_key]),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        # spatial plot of st catalog w/o flux in all bands
        not_full_cov_list = []
        for i in range(len(filters_list)):
            flux_zero = np.where(t_st_cov[filters_list[i] + "_RATE"] == 0.0)[0]
            not_full_cov_list.append(flux_zero)

        not_full_cov_list = np.unique(np.concatenate(not_full_cov_list))
        t_st_cov = t_st_cov[not_full_cov_list]

        ax2.plot(t_st_cov[ra_key], t_st_cov[dec_key], ",", ls="")
        ax2.set_xlabel("RA", fontsize=15)
        ax2.set_ylabel("DEC", fontsize=15)
        ax2.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#126D94",
                    marker="o",
                    markersize=3,
                    ls="",
                    label="st catalog not in all bands, N = %s" % len(t_st_cov[ra_key]),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        # spatial plot of st catalog that don't pass photometry cuts

        # Setting FLAG=99 for crowd, sharp and round cuts
        for i in range(len(filters)):
            if filters[i] not in ("F475W", "F814W"):
                continue

            inds_to_cut = np.where(
                (t_st_phot[filters[i] + "_CROWD"] > crowdcut[i])
                | (t_st_phot[filters[i] + "_SHARP"] * t_st_phot[filters[i] + "_SHARP"] > sharpcut[i])
                | (t_st_phot[filters[i] + "_ROUND"] > roundcut[i])
            )[0]

            if errcut:
                errinds_to_cut = np.where(t_st_phot[filters[i] + "_ERR"] > errcut)[0]
                inds_to_cut = np.unique(np.concatenate((inds_to_cut, errinds_to_cut), 0))

            # flag sources not passing quality cuts
            t_st_phot[filters[i] + "_FLAG"][inds_to_cut] = 99

      # Remove flux=0, flag!=0 or flag!=2 sources
        not_phot_list = []
        for i in range(len(filters_list)):
            flag_not_0_or_2 = np.where(
                (t_st_phot[filters_list[i] + "_FLAG"] == 1)
                | (t_st_phot[filters_list[i] + "_FLAG"] == 3)
                | (t_st_phot[filters_list[i] + "_FLAG"] == 99)
            )[0]

            not_phot_list.append(flag_not_0_or_2)

        not_phot_list = np.unique(np.concatenate(not_phot_list))
        t_st_phot = t_st_phot[not_phot_list]

        ax3.plot(t_st_phot[ra_key], t_st_phot[dec_key], ",", ls="")
        ax3.set_xlabel("RA", fontsize=15)
        ax3.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#126D94",
                    marker="o",
                    markersize=3,
                    ls="",
                    label="don't pass photometry cuts, N = %s" % len(t_st_phot[ra_key]),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        # spatial plot of the culled vgst catalog

        ax4.plot(t[ra_key], t[dec_key], ",", ls="")
        ax4.set_xlabel("RA", fontsize=15)
        ax4.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#126D94",
                    marker="o",
                    markersize=3,
                    ls="",
                    label="vgst catalog, N = %s" % len(t[ra_key]),
                )
            ],
            loc="upper right",
            fontsize="small",
        )


        mag1 = t[filters[0] + "_VEGA"]
        if len(filters) > 1:
            mag2 = t[filters[1] + "_VEGA"]
        else:
            mag2 = t["F475W_VEGA"]

        mag1_all = t_st[filters[0] + "_VEGA"]
        mag2_all = t_st[filters[1] + "_VEGA"]

        mag1_cut = t_st[filters[0] + "_VEGA"][bad_srcs_flat_list]
        mag2_cut = t_st[filters[1] + "_VEGA"][bad_srcs_flat_list]

        xlim = (-6, 8)
        ylim = (32.5, min(mag1) - 1.5)

        # limit the axes ranges to exclude 99s for other filters
        bad = np.where(
            ((mag1_cut - mag2_cut) > -15.0)
            & ((mag1_cut - mag2_cut) < 15)
            & (mag1_cut < 40)
        )[0]

        # CMD Kernel Density Estimation (KDE) scatter plot plotting the density of sources
        x_cut, y_cut, z_cut = kde_scatterplot_args(mag1_all, mag2_all)
        # CMD of sources removed from st catalog
        ax5.scatter(x_cut, y_cut, c=z_cut, s=3, cmap="RdPu")
        ax5.set_xlabel("%s - %s" % (filters[0], filters[1]), fontsize=15)
        ax5.set_ylabel("%s" % filters[0], fontsize=15)
        ax5.set_xlim(xlim)
        ax5.set_ylim(ylim)
        ax5.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#A91864",
                    ls="",
                    label="All sources in st catalog, N = %s" % len(mag1_all),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        mag1_cov = t_st_cov[filters[0] + "_VEGA"]
        mag2_cov = t_st_cov[filters[1] + "_VEGA"]

        # CMD Kernel Density Estimation (KDE) scatter plot plotting the density of sources
        x_cut, y_cut, z_cut = kde_scatterplot_args(mag1_cov, mag2_cov)
        # CMD of sources removed from st catalog
        ax6.scatter(x_cut, y_cut, c=z_cut, s=3, cmap="RdPu")
        ax6.set_xlabel("%s - %s" % (filters[0], filters[1]), fontsize=15)
        ax6.set_ylabel("%s" % filters[0], fontsize=15)
        ax6.set_xlim(xlim)
        ax6.set_ylim(ylim)
        ax6.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#A91864",
                    ls="",
                    label="Removed, N = %s" % len(mag1_cov),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        mag1_phot = t_st_phot[filters[0] + "_VEGA"]
        mag2_phot = t_st_phot[filters[1] + "_VEGA"]

        # CMD Kernel Density Estimation (KDE) scatter plot plotting the density of sources
        x_cut, y_cut, z_cut = kde_scatterplot_args(mag1_phot, mag2_phot)
        # CMD of sources removed from st catalog
        ax7.scatter(x_cut, y_cut, c=z_cut, s=3, cmap="RdPu")
        ax7.set_xlabel("%s - %s" % (filters[0], filters[1]), fontsize=15)
        ax7.set_ylabel("%s" % filters[0], fontsize=15)
        ax7.set_xlim(xlim)
        ax7.set_ylim(ylim)
        ax7.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#A91864",
                    ls="",
                    label="Removed, N = %s" % len(mag1_phot),
                )
            ],
            loc="upper right",
            fontsize="small",
        )

        good = np.where(((mag1 - mag2) > -15.0) & ((mag1 - mag2) < 15) & (mag1 < 40))[0]

        x, y, z = kde_scatterplot_args(mag1[good], mag2[good])
        # CMD of remaining sources (vgst catalog)
        ax8.scatter(x, y, c=z, s=3, cmap="GnBu_r")
        ax8.set_xlabel("%s - %s" % (filters[0], filters[1]), fontsize=15)
        ax8.set_xlim(xlim)
        ax8.set_ylim(ylim)
        ax8.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#35C0D6",
                    ls="",
                    label="Kept (vgst), N = %s" % len(mag1),
                )
            ],
            loc="upper right",
            fontsize="small",
        )
        # print(catalog_out.replace(".fits", ".png"))
        plt.show()
        plt.savefig(catalog_out.replace(".fits", ".png"))
        print(
            "Plot saved as '%s'" % catalog_out.split("/")[-1].replace(".fits", ".png")
        )
        plt.close()


def kde_scatterplot_args(mag1, mag2):
    """
    Perform a Gaussian kernel density estimate of the PDF of color-mag values to enable a scatter-density CMD plot.
    Parameters
    ----------
    mag1: astropy table column array
        An array of mag1 values
    mag2: astropy table column array
        An array of mag2 values
    Returns
    -------
    x : astropy table column
        color values
    y : astropy table column
        the y-axis values
    z : ndarray
        The Gaussian KDE estimating the PDF of a color/mag input; Used as a sequence of numbers to be
        mapped to colors using a colormap.
    """

    print("Setting up CMD scatter density plot...")
    start_time_kde = time.time()
    xy = np.vstack([(mag1 - mag2), mag1])
    z = gaussian_kde(xy)(xy)

    # sorting the points by density to ensure densest are plotted on top
    idx = z.argsort()
    x, y, z = (mag1 - mag2)[idx], mag1[idx], z[idx]
    print("The KDE took %.2f sec." % round((time.time() - start_time_kde), 2))

    return x, y, z


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog_in",
        type=str,
        help="File name of the input catalog",
    )
    parser.add_argument(
        "--catalog_out",
        default=None,
        type=str,
        help="desired name of output catalog. If none given, '.st' in the"
        "input catalog is changed to '.vgst' by default",
    )
    parser.add_argument(
        "--filters",
        default=["F475W", "F814W"],
        nargs="*",
        type=str,
        help="filters to which to apply cuts. Enter as,"
        "e.g., '--filters 'F336W' 'F475W' '.",
    )
    parser.add_argument(
        "--crowdcut",
        default=[0.3],
        nargs="*",
        type=float,
        help="desired crowding cut(s). Enter as, e.g., " "'--crowdcut 0.15 0.2 '.",
    )
    parser.add_argument(
        "--sharpcut",
        default=[0.15],
        nargs="*",
        type=float,
        help="desired sharpness squared cut(s)",
    )
    parser.add_argument(
        "--roundcut",
        default=[0.6],
        nargs="*",
        type=float,
        help="desired rounding cut(s)",
    )
    parser.add_argument(
        "--errcut", default=None, nargs="*", type=float, help="desired error cut(s)"
    )
    parser.add_argument(
	"--ast_apply",
	default=False,
	type=bool,
	help="if True, do not remove cut sources from catalog, just flag them by introducing new flag (CUT_FLAG)"
    )
    parser.add_argument(
        "--plot",
        default=False,
        type=bool,
        help="make two plots: spatial and CMD after the cuts are applied",
    )
    args = parser.parse_args()

    make_cuts(
        args.catalog_in,
        catalog_out=args.catalog_out,
        filters=args.filters,
        crowdcut=args.crowdcut,
        sharpcut=args.sharpcut,
        roundcut=args.roundcut,
        errcut=args.errcut,
     	ast_apply=args.ast_apply,
        plot=args.plot,
    )
