"""
Generate input ASTs from Scylla vgst files, including diagnostic plots
Christina Lindberg, November 2021
"""

from glob import glob
import os
import beast_ast_inputs
import plot_source_density_map
import argparse

from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt

def make_input_asts(
    field_name,
    plot=True
):
    #retrieve reference image
    # find ref image with F475W filter
    ref_image = glob("./data/{0}/proc_default/{0}_F475W_drz.chip1.fits".format(field_name))

    # if it doesn't exist, try F814W
    if len(ref_image) == 0:
        ref_image = glob("./data/{0}/proc_default/{0}_F814W_drz.chip1.fits".format(field_name))

    # if that one also doesn't exist, try F275W
    if len(ref_image) == 0:
        ref_image = glob("./data/{0}/proc_default/{0}_F275W_drz.chip1.fits".format(field_name))

    # if that one also doesn't exist, try F336W
    if len(ref_image) == 0:
        ref_image = glob("./data/{0}/proc_default/{0}_F336W_drz.chip1.fits".format(field_name))

    if len(ref_image) == 0:
        raise IOError("Reference image not found.")

    print(field_name)
    print(ref_image)

    beast_ast_inputs.beast_ast_inputs(
        field_name=field_name,
        ref_image=ref_image[0],
        galaxy = field_name.split("-")[0].split("_")[-1],
    )

    if plot:
        beast_file = glob("./data/{0}/proc_default/beast*".format(field_name))
        sd_image_file = glob("./data/{0}/proc_default/{0}.vgst_source_den_image.fits".format(field_name))

        plot_source_density_map.plot_source_density_map(sd_image_file[0], beast_file[0])

        plot_ast_positions(field_name)


def plot_ast_positions(field_name):
    ast_data = pd.read_csv("./{0}/{0}_inputAST.txt".format(field_name), delim_whitespace=True)
    gst_data = Table.read("./data/{0}/proc_default/{0}.vgst_with_sourceden.fits".format(field_name))

    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharex=True, sharey=True)

    im = ax[0].scatter(gst_data["X"], gst_data["Y"], s=10, alpha=1, c=gst_data["SourceDensity"], cmap='viridis', label="Source Density")

    ax[1].scatter(ast_data["X"], ast_data["Y"], s=0.1, alpha=1, label="ASTs")

    ax[0].set_title("GSTs")
    ax[1].set_title("ASTs")

    ax[0].set_xlabel("X")
    ax[1].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].set_ylabel("Y")


    cbar = plt.colorbar(im, fraction=0.05, pad=0.01, ax=ax[0])#, ticks=sd_bins)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Source Density', rotation=270)

    plt.tight_layout()
    plt.savefig("./{0}/{0}_inputAST_positions.pdf".format(field_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
      "field_name",
      type=str,
      help="Field name with the /data folder",
    )
    parser.add_argument(
        "--plot",
        default=True,
        type=bool,
        help="make two plots: AST spatial locations and the source density map",
    )

    args = parser.parse_args()

    make_input_asts(
    args.field_name,
    plot=args.plot,
    )
