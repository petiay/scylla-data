#!/usr/bin/env python
# coding: utf-8

# In[2]:


from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clip

from matplotlib.colors import LogNorm
import matplotlib.pylab as plt
import cmocean

import numpy as np
import os
import types

from scipy.signal import medfilt

from beast.tools import (
    beast_settings,
    create_background_density_map)
from beast.tools.density_map import BinnedDensityMap
from beast.plotting import plot_mag_hist
from astropy.table import Table
from beast.plotting import plot_toothpick_details
import beast.plotting.plot_toothpick_details
import importlib

importlib.reload(plot_toothpick_details)

fontsize = 13
font = {"size": fontsize}
plt.rc("font", **font)

# ## Array for grid of results; only set at the beginning of a param grid search

def find_max_compl_to_den_ratio(cr_range=[0.2, 2.25], rnd_range, sh_range, kernel_size_range, den_thr_range, n_grid_pts,
                                plot=False, savefig=False):
    """
    Photometric parameters ranges
    Kernel size range
    phot_cat_src_den_thr_range
    fake_cat_src_den_thr_range
    n_grid_pts: number of points into which to split each param range

    Returns
    -------

    """

grid_results = []
global_maximum = []
npsrcden = []
phot_cuts_frac_photcat = []
ds_cuts_frac_photcat = []
zero_flux_frac_photcat = []
phot_cuts_frac_fakecat = []
ds_cuts_frac_fakecat = []
zero_flux_frac_fakecat = []


# Number of combinations of parameters; needs to be updated every time
trials = 1
plot = True

# Do we want to generate a new catalog containing the cut sources which are flagged with CUT_FLAG=1?
# Since all catalogs now need cut_flag, then all that's needed is a check; if no cut_flag column, then add it.
add_cutflag = True
savefig = True

# ## Quality thresholds

# In[6]:


crowd_th = 0.75
sharp_th = 0.2
round_th = 2
bright_th = 20  # mag

# ## Get observations catalog

# In[7]:


target = 'SMC-3956ne-9632'
inDir = '/Users/pyanchulova/Documents/scylla/'
outDir = '/Users/pyanchulova/Documents/scylla/'
if not os.path.exists(outDir):
    os.mkdir(outDir)

# In[9]:


stfile = inDir + '15891_%s.st.fits' % target
gstfile = inDir + '15891_%s.gst.fits' % target

stcat = fits.getdata(stfile)
gstcat = fits.getdata(gstfile)

# confirm catalog length
print('st/gst catalogs for %s have %s/%s entries' % (target, len(stcat['RA']), len(gstcat['RA'])))
t = Table.read(stfile)
filters = [x.split('_')[0] for x in t.colnames if "VEGA" in x]
print(filters)

# ### A check on F336W_IN (~9000 NaNs)

# In[10]:


# Only an issue if reading in fake stars catalog
fake_cat = False

if fake_cat:
    f336nan = np.isnan(stcat['F336W_IN'])
    print('N (F336W_IN = NaN)', len(np.where(f336nan == True)[0]))

# Designate these with 99s farther down when recording the new catalog


# In[41]:


t.colnames
if 'RA' in t.colnames:
    ra_key = "RA"
    dec_key = "DEC"
else:
    ra_key = "RA_J2000"
    dec_key = "DEC_J2000"

print(ra_key, dec_key)

# In[12]:


two_key = ['F475W', 'F814W']

# ## Test quality cuts

# In[13]:


print('Total # sources in st cat:', len(stcat))

# Stars that did not pass the cuts in each filter have mag=99.999
print('Remaining sources:')
default_cut = (gstcat[two_key[0] + '_VEGA'] < 99) & (gstcat[two_key[1] + '_VEGA'] < 99)
print('default gst cut (has err and snr cuts):', len(gstcat[default_cut]))

# Cut on DOLPHOT flag. Keep only sources with flag=0 or 2
flag_cut = ((stcat[two_key[0] + '_FLAG'] == 0) | (stcat[two_key[0] + '_FLAG'] == 2)) & (
            (stcat[two_key[1] + '_FLAG'] == 0) | (stcat[two_key[1] + '_FLAG'] == 2))
print('FLAG remaining:', len(stcat[flag_cut]))

# Cut on RATE. Keep only sources with rate != 0.0 in ALL bands
rate_cut = np.all([stcat[x + '_VEGA'] != 0 for x in filters], axis=0).tolist()
print('RATE remaining:', len(stcat[rate_cut]))

# Cut on CROWD. Keep only sources with CROWD < crowd_threshold (GST default=2.25)
crowd_cut = (stcat[two_key[0] + '_CROWD'] < crowd_th) & (stcat[two_key[1] + '_CROWD'] < crowd_th)
print('CROWD remaining:', len(stcat[crowd_cut]))

# Cut on SHARP. Keep only sources with sharp**2 < sharp_threshold (GST default=0.2)
sharp_cut = (stcat[two_key[0] + '_SHARP'] ** 2 < sharp_th) & (stcat[two_key[1] + '_SHARP'] ** 2 < sharp_th)
print('SHARP remaining:', len(stcat[sharp_cut]))

# Cut on ROUND. Keep only sources with round < round_threshold (no ROUND cut in the default cut)
round_cut = (stcat[two_key[0] + '_ROUND'] < round_th) & (stcat[two_key[1] + '_ROUND'] < round_th)
print('ROUND remaining:', len(stcat[round_cut]))

# Cut on bright sources. Keep sources with F475W_VEGA <= bright_threshold
bright_cut = (stcat[two_key[0] + '_VEGA'] <= bright_th)
print('Bright stars (f475w_vega <= %s): %s' % (bright_th, len(stcat[bright_cut])))

qual_cut = crowd_cut & sharp_cut & round_cut
my_cut = crowd_cut & sharp_cut & round_cut & flag_cut & rate_cut | bright_cut
len_my_cut = len(stcat[~my_cut])
print('Total sources removed by quality cuts:', len_my_cut)

my_cut_no_bright = crowd_cut & sharp_cut & round_cut & flag_cut & rate_cut
len_my_cut_no_bright = len(stcat[~my_cut_no_bright])
print('Total BRIGHT sources removed by quality cuts:', len_my_cut_no_bright - len_my_cut)

print('My cut (# of srcs remaining): %s (%s %%)' % (
len(stcat[my_cut]), np.around(len(stcat[my_cut]) / len(stcat), decimals=2) * 100))

# In[15]:


if plot:
    plt.figure(figsize=(15, 13))

    plt.subplot(321)
    h = plt.hist(stcat[two_key[0] + '_VEGA'], bins=100, alpha=0.5, log=True)
    h = plt.hist(stcat[two_key[0] + '_VEGA'][my_cut], bins=100)
    h = plt.hist(gstcat[two_key[0] + '_VEGA'][default_cut], bins=100, alpha=0.5)
    plt.xlabel(two_key[0] + '_VEGA')
    plt.legend(['All ST', 'My cut', 'Default GST'])

    plt.subplot(322)
    h = plt.hist(stcat[two_key[0] + '_CROWD'], bins=100, alpha=0.5, log=True)
    h = plt.hist(stcat[two_key[0] + '_CROWD'][my_cut], bins=100)
    h = plt.hist(gstcat[two_key[0] + '_CROWD'][default_cut], bins=100, alpha=0.5)
    plt.xlabel('CROWD')
    plt.legend(['All ST', 'My cut', 'Default GST'])

    plt.subplot(323)
    h = plt.hist(stcat[two_key[0] + '_SHARP'], bins=100, alpha=0.5, log=True)
    h = plt.hist(stcat[two_key[0] + '_SHARP'][my_cut], bins=100, log=True)
    h = plt.hist(gstcat[two_key[0] + '_SHARP'][default_cut], bins=100, alpha=0.5)
    plt.xlabel('SHARP')
    plt.legend(['All ST', 'My cut', 'Default GST'])

    plt.subplot(324)
    h = plt.hist(stcat[two_key[0] + '_ROUND'], bins=100, alpha=0.5, log=True)
    h = plt.hist(stcat[two_key[0] + '_ROUND'][my_cut], bins=100, log=True)
    h = plt.hist(gstcat[two_key[0] + '_ROUND'][default_cut], bins=100, alpha=0.5)
    plt.xlabel('ROUND')
    plt.legend(['All ST', 'My cut', 'Default GST'])

    plt.savefig(outDir + 'histograms_crwd_%s_sh_%s_rnd_%s.png' % (crowd_th, sharp_th, round_th))

# ## Plot default gst cuts

# In[16]:


nbins = 200
if plot is False:
    h_map = plt.hist2d(stcat[ra_key], stcat[dec_key], bins=nbins, norm=LogNorm(), cmap=cmocean.cm.deep);
    h = plt.hist2d(stcat[ra_key][default_cut], stcat[dec_key][default_cut],
                   bins=nbins, norm=LogNorm(vmax=h_map[0].max()),
                   range=((h_map[1].min(), h_map[1].max()), (h_map[2].min(), h_map[2].max())), cmap=cmocean.cm.deep);

plt.set_cmap('bone_r')

if plot:
    fig = plt.figure(figsize=(17, 16))

    plt.subplot(221)
    h_map = plt.hist2d(stcat[ra_key], stcat[dec_key], bins=nbins, norm=LogNorm(), cmap=cmocean.cm.deep)
    plt.xlabel(ra_key, fontsize=15)
    plt.ylabel(dec_key, fontsize=15)
    plt.title('%d ST objects' % (len(stcat[ra_key])))

    plt.subplot(222)
    h = plt.hist2d(stcat[ra_key][default_cut], stcat[dec_key][default_cut],
                   bins=nbins, norm=LogNorm(vmax=h_map[0].max()),
                   range=((h_map[1].min(), h_map[1].max()), (h_map[2].min(), h_map[2].max())), cmap=cmocean.cm.deep)
    plt.xlabel(ra_key, fontsize=15)
    plt.ylabel(dec_key, fontsize=15)
    plt.title('%d GST objects' % (len(stcat[default_cut])))

    cmd_st_sources = stcat[two_key[0] + '_VEGA'] < 99
    len_cmd_st_sources = len(stcat[cmd_st_sources])

    plt.subplot(223)
    h_all = plt.hist2d(stcat[two_key[0] + '_VEGA'] - stcat[two_key[1] + '_VEGA'],
                       stcat[two_key[0] + '_VEGA'], bins=200, norm=LogNorm(),
                       range=((-2, 6), (18, 32)), cmap=cmocean.cm.thermal)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel(two_key[0] + '-' + two_key[1], fontsize=15)
    plt.ylabel(two_key[0], fontsize=15)
    plt.title('%d ST objects' % (len_cmd_st_sources))

    cmd_gst_sources = gstcat[two_key[0] + '_VEGA'] < 99
    len_cmd_gst_sources = len(gstcat[cmd_gst_sources])

    plt.subplot(224)
    h_def = plt.hist2d(stcat[two_key[0] + '_VEGA'][default_cut] -
                       stcat[two_key[1] + '_VEGA'][default_cut],
                       stcat[two_key[0] + '_VEGA'][default_cut], bins=200,
                       norm=LogNorm(vmax=h_all[0].max()),
                       range=((-2, 6), (18, 32)), cmap=cmocean.cm.thermal)
    # plt.ylim(32,18)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel(two_key[0] + '-' + two_key[1], fontsize=15)
    plt.ylabel(two_key[0], fontsize=15)
    plt.title('%d GST objects' % (len_cmd_gst_sources))
    plt.savefig(outDir + 'gst_cuts_spatial_cmd_%s.png' % target)

# # Try to remove objects associated with diffraction spikes using kernel filter
#

# In[17]:


# h_map = plt.hist2d(stcat[ra_key], stcat[dec_key], bins=nbins, norm=LogNorm(), cmap=cmocean.cm.deep);
kernel_size = 5
median_map = medfilt(h_map[0], kernel_size=kernel_size)

if plot:
    fig = plt.figure(figsize=(17, 14))
    plt.suptitle('Remove spikes: ks=%s, crowd=%s, round=%s, sharp^2=%s' % (kernel_size, crowd_th, round_th, sharp_th))

    plt.subplot(221)
    plt.imshow(h_map[0].T - median_map.T, origin='lower', aspect='auto', cmap=plt.cm.rainbow,
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()], vmax=30)
    cb = plt.colorbar()
    cb.set_label('Star Count - Median Star Count')
    plt.title('Star Count (st) - Median Star Count')

    plt.subplot(222)
    h = plt.hist((h_map[0].T - median_map.T).ravel(), bins=150, log=True)
    plt.xlabel('Star Count - Median Star Count')

    plt.subplot(223)
    plt.imshow(median_map.T, origin='lower', aspect='auto', cmap=plt.cm.rainbow,
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()])
    cb = plt.colorbar()
    cb.set_label('Median Filtered Star Count')
    plt.title('Median Star Count, ks = %s' % kernel_size)

    plt.subplot(224)
    h = plt.hist(median_map.T.ravel(), bins=150, log=True)
    plt.xlabel('Median Star Count')

    plt.savefig('%sstars-medianstars_%s_ks_%s_crwd_%s_rnd_%s.png' % (outDir, target, kernel_size, crowd_th, round_th),
                dpi=300)

# ## Select a Median star count threshold

# In[18]:


import matplotlib.colors as colors

sd_thre = 20

if plot:
    plt.figure(figsize=(16, 7))

    plt.subplot(121)
    plt.imshow(h_map[0].T, origin='lower', norm=LogNorm(), aspect='auto',
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()],
               cmap=cmocean.cm.deep)
    plt.title('ST catalog')

# Remove sources which, after the median-filtered results are removed, still surpass a threshold.
high_sd = h_map[0] - median_map > sd_thre
len_high_sd = len(np.where(high_sd == True)[0])
print('%s regions with source density above the %d threshold' % (len_high_sd, sd_thre))

if plot:
    plt.subplot(122)
    plt.imshow(h_map[0].T * high_sd.T, origin='lower', norm=colors.LogNorm(vmin=h_map[1].min(), vmax=h_map[1].max()),
               cmap=cmocean.cm.deep, aspect='auto',
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()])
    plt.title('High-density sources, Threshold: %s' % sd_thre)

    plt.savefig(outDir + 'threshold_map_th_' + str(sd_thre) + '_ks_' + str(kernel_size) + '.png')

# ## Identify sources associated with high src density regions;
# ## Place a bright sources exception

# In[20]:


indices = []
nrow, ncol = h_map[0].shape
# indices_diff_spikes = []
for ira in range(nrow):
    for jdec in range(ncol):
        if high_sd[ira, jdec]:
            # original selection: only above threshold
            stars, = np.where((stcat[ra_key] >= h_map[1][ira]) & (stcat[ra_key] < h_map[1][ira + 1]) & (
                        stcat[dec_key] >= h_map[2][jdec]) & (stcat[dec_key] < h_map[2][jdec + 1]))

            #             dim_stars, = np.where((stcat['F336W_VEGA'][stars] > 20))
            #             indices_diff_spikes.append(dim_stars.tolist())

            indices.append(stars.tolist())

indices = np.concatenate(indices).astype('int')
# indices_diff_spikes = np.concatenate(indices_diff_spikes).astype('int')

# remove only sources fainter than 20 mag
# inds_no_bright = np.where(stcat['F475W_VEGA'][indices_diff_spikes] > 20)[0]
# print('Only faint diffraction spikes', min(stcat['F475W_VEGA'][inds_no_bright]))

print('Sources removed as diffraction spikes: %s (%s %%)' % (
len(indices), np.around((len(indices) / len(stcat[ra_key])), decimals=2)))

# ## Combine DS and quality cuts

# In[21]:


diff_flag = np.ones(len(stcat), dtype=bool)
# diff_flag[indices_diff_spikes] = False
diff_flag[indices] = False
high_sd_true_nonfake = np.where(high_sd_fake is True)[0]
print(len(high_sd_true))

final_cut = my_cut & diff_flag

# ## Calculate pre- and post- bright-star stats

# In[22]:


lt_16_st = (stcat['F336W_VEGA'] < 16.);
nlt16st = len(stcat[lt_16_st])
lt_17_st = (stcat['F336W_VEGA'] < 17.);
nlt17st = len(stcat[lt_17_st])
lt_18_st = (stcat['F336W_VEGA'] < 18.);
nlt18st = len(stcat[lt_18_st])
lt_19_st = (stcat['F336W_VEGA'] < 19.);
nlt19st = len(stcat[lt_19_st])
lt_20_st = (stcat['F336W_VEGA'] < 20.);
nlt20st = len(stcat[lt_20_st])
all_lt_20_st = nlt16st + nlt17st + nlt18st + nlt19st + nlt20st
print(nlt16st, nlt17st, nlt18st, nlt19st, nlt20st)

lt_16_final = final_cut & (stcat['F336W_VEGA'] < 16.);
nlt16fin = len(stcat[lt_16_final])
lt_17_final = final_cut & (stcat['F336W_VEGA'] < 17.);
nlt17fin = len(stcat[lt_17_final])
lt_18_final = final_cut & (stcat['F336W_VEGA'] < 18.);
nlt18fin = len(stcat[lt_18_final])
lt_19_final = final_cut & (stcat['F336W_VEGA'] < 19.);
nlt19fin = len(stcat[lt_19_final])
lt_20_final = final_cut & (stcat['F336W_VEGA'] < 20.);
nlt20fin = len(stcat[lt_20_final])
all_lt_20_final = nlt16fin + nlt17fin + nlt18fin + nlt19fin + nlt20fin
print(nlt16fin, nlt17fin, nlt18fin, nlt19fin, nlt20fin)

print('All st stars with F336W_VEGA <= 20 mag: ', all_lt_20_st)
print('All final stars with F336W_VEGA <= 20 mag: ', all_lt_20_final)

# ## Plot final stats

# In[23]:


if plot:
    fig = plt.figure(figsize=(15, 21))

    plt.suptitle('Quality stats: ks=%s, th=%s, crowd=%s, round=%s, sharp^2=%s' % (
    kernel_size, sd_thre, crowd_th, round_th, sharp_th))

    plt.subplot(321)
    plt.imshow(h_map[0].T, origin='lower', norm=LogNorm(vmin=1, vmax=h_map[0].max()),
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()],
               aspect='auto')
    plt.plot(stcat[ra_key][indices], stcat[dec_key][indices], '.b', ms=3)
    plt.xlabel(ra_key)
    plt.ylabel(dec_key)
    plt.title('All %d ST objects' % (len(stcat[ra_key])))
    plt.legend(['Diff spikes, N=%s' % len(indices)], loc=2)

    plt.subplot(322)
    plt.contour(h_all[0].T, extent=[-2, 6, 18, 32], colors='k', alpha=0.2)
    h = plt.hist2d(stcat[two_key[0] + '_VEGA'][~diff_flag] - stcat[two_key[1] + '_VEGA'][~diff_flag],
                   stcat[two_key[0] + '_VEGA'][~diff_flag], bins=nbins,
                   norm=LogNorm(vmin=1, vmax=h_all[0].max()),
                   range=((-2, 6), (18, 32)), cmap=plt.cm.jet)
    plt.ylim(32, 18)
    plt.xlabel(two_key[0] + '-' + two_key[1])
    plt.ylabel(two_key[0])
    plt.title('Objects associated w/ diff. spikes')

    plt.subplot(323)
    # h = plt.hist2d(stcat['RA'][default_cut], stcat['DEC'][default_cut],
    #                bins=nbins, norm=LogNorm(vmin=1,vmax=h_map[0].max()),
    #                range=((h_map[1].min(),h_map[1].max()),(h_map[2].min(), h_map[2].max())))
    # plt.xlabel('RA',fontsize=15)
    # plt.ylabel('Dec',fontsize=15)
    # plt.title('%d objects with the default GST cut' % (len(stcat[default_cut])))
    h = plt.hist2d(stcat[ra_key][final_cut], stcat[dec_key][final_cut],
                   bins=nbins, norm=LogNorm(vmin=1, vmax=h_map[0].max()),
                   range=((h_map[1].min(), h_map[1].max()), (h_map[2].min(), h_map[2].max())))
    plt.xlabel(ra_key, fontsize=15)
    plt.ylabel(dec_key, fontsize=15)
    plt.title('%d objects remain w/ final cut (cr=%s, rn=%s)' % (len(stcat[final_cut]), crowd_th, round_th))

    plt.subplot(324)
    # h = plt.hist2d(stcat['RA'][~default_cut], stcat['DEC'][~default_cut],
    #                bins=nbins, norm=LogNorm(vmin=1,vmax=h_map[0].max()),
    #                range=((h_map[1].min(),h_map[1].max()),(h_map[2].min(),h_map[2].max())))
    # plt.xlabel('RA',fontsize=15)
    # plt.ylabel('Dec',fontsize=15)
    # plt.title('%d objects did not make the default GST cut' % (len(stcat[~default_cut])))
    h = plt.hist2d(stcat[ra_key][~final_cut], stcat[dec_key][~final_cut],
                   bins=nbins, norm=LogNorm(vmin=1, vmax=h_map[0].max()),
                   range=((h_map[1].min(), h_map[1].max()), (h_map[2].min(), h_map[2].max())))
    plt.xlabel(ra_key, fontsize=15)
    plt.ylabel(dec_key, fontsize=15)
    plt.title('%d objects removed w/ final cut' % (len(stcat[~final_cut])))

    plt.subplot(325)
    # Show original stats on spatial plot
    plt.imshow(h_map[0].T, origin='lower', norm=LogNorm(vmin=1, vmax=h_map[0].max()),
               extent=[h_map[1].min(), h_map[1].max(), h_map[2].min(), h_map[2].max()],
               aspect='auto')
    plt.plot(stcat[ra_key][lt_20_st], stcat[dec_key][lt_20_st], '.', c='gold', ms=5, label='F336<=20, N=%s' % nlt20st)
    plt.plot(stcat[ra_key][lt_19_st], stcat[dec_key][lt_19_st], '.', c='turquoise', ms=5,
             label='F336<=19, N=%s' % nlt19st)
    plt.plot(stcat[ra_key][lt_18_st], stcat[dec_key][lt_18_st], '.', c='red', ms=5, label='F336<=18, N=%s' % nlt18st)
    plt.plot(stcat[ra_key][lt_17_st], stcat[dec_key][lt_17_st], '.', c='green', ms=5, label='F336<=17, N=%s' % nlt17st)
    plt.plot(stcat[ra_key][lt_16_st], stcat[dec_key][lt_16_st], '.', c='dodgerblue', ms=5,
             label='F336<=16, N=%s' % nlt16st)
    plt.xlabel(ra_key)
    plt.ylabel(dec_key)
    plt.title('Bright stars (F336W<20mag) in ST catalog, N=%s' % all_lt_20_st)
    plt.legend()

    plt.subplot(326)
    # Show resulting stats on spatial plot
    h = plt.hist2d(stcat[ra_key][final_cut], stcat[dec_key][final_cut],
                   bins=nbins, norm=LogNorm(vmin=1, vmax=h_map[0].max()),
                   range=((h_map[1].min(), h_map[1].max()), (h_map[2].min(), h_map[2].max())))
    plt.plot(stcat[ra_key][lt_20_final], stcat[dec_key][lt_20_final], '.', c='gold', ms=7,
             label='F336<=20, N=%s' % nlt20fin)
    plt.plot(stcat[ra_key][lt_19_final], stcat[dec_key][lt_19_final], '.', c='turquoise', ms=7,
             label='F336<=19, N=%s' % nlt19fin)
    plt.plot(stcat[ra_key][lt_18_final], stcat[dec_key][lt_18_final], '.', c='red', ms=7,
             label='F336<=18, N=%s' % nlt18fin)
    plt.plot(stcat[ra_key][lt_17_final], stcat[dec_key][lt_17_final], '.', c='green', ms=7,
             label='F336<=17, N=%s' % nlt17fin)
    plt.plot(stcat[ra_key][lt_16_final], stcat[dec_key][lt_16_final], '.', c='dodgerblue', ms=7,
             label='F336<=16, N=%s' % nlt16fin)
    plt.xlabel(ra_key)
    plt.ylabel(dec_key)
    plt.title('Bright stars (F336W<20mag) in FINAL catalog, N=%s' % all_lt_20_final)
    plt.legend()

    plt.savefig(outDir + 'clean_catalog_%s_ks_%s_th_%s_crwd_%s_rnd_%s_bright_%s.png' % (
    target, kernel_size, sd_thre, crowd_th, round_th, all_lt_20_final), dpi=300)

# ## Add 'CUT_FLAG' columns; turn F336W_IN NaNs into 99s

# In[24]:


# open catalog as a table to add columns
# t = Table.read(inDir+'15891_%s.st.fake.fits' % target)
# t_vgst = Table.read(inDir+'15891_%s.vgst.fake.fits' % target)

# t = Table.read(stfile)

# Designate F336W_IN = NaN with 99s (otherwise error)
if fake_cat:
    f336nan = np.isnan(stcat['F336W_IN'])
    print('N (F336W_IN = NaN)', len(np.where(f336nan == True)[0]))  # N=9574
    t["F336W_IN"][f336nan] = 99

# add RA/DEC columns; not needed.
# t.add_column(np.zeros(len(t[ra_key])), name="RA", index=-1)
# t.add_column(np.zeros(len(t[dec_key])), name="DEC", index=-1)
# t["RA"] = t[ra_key]
# t["DEC"] = t[dec_key]

# keep bad sources in catalog and flag them with a new column, CUT_FLAG = 1
add_cutflag = True
if add_cutflag:
    t.add_column(np.zeros(len(t), dtype=int), name="CUT_FLAG", index=-1)
    t["CUT_FLAG"][~final_cut] = 1
    t["CUT_FLAG"][final_cut] = 0

# ## Write new catalog

# In[25]:


# hdu = fits.BinTableHDU(data=stcat[final_cut])
# hdu.writeto(inDir+'15275_%s.vgst.fits' % (target), overwrite=True)

hdu = fits.BinTableHDU(data=t)
outcat = inDir + '15891_%s_%s_%s_%s_%s_with_cutflag.est.fits' % (target, kernel_size, sd_thre, crowd_th, round_th)
hdu.writeto(outcat, overwrite=True)

# ### Make a source density map

# In[26]:


flag_filter = ["F475W"]
ref_filter = ["F475W"]
pixsize = 5
print('Confirm resulting catalog:', outcat)

peak_mags = plot_mag_hist.plot_mag_hist(outcat, stars_per_bin=70, max_bins=75)

# Try to make sure the pixel size matches the kernel size
sourceden_args = types.SimpleNamespace(
    subcommand="sourceden",
    catfile=outcat,
    erode_boundary=0.1,
    pixsize=pixsize,
    npix=None,
    mag_name=ref_filter[0] + "_VEGA",
    mag_cut=[16, peak_mags[ref_filter[0]] - 0.5],
    flag_name=flag_filter[0] + "_FLAG",
)
create_background_density_map.main_make_map(sourceden_args)

# ### Group tiles by density bin (optional)

# In[27]:


map_file = outcat.replace(".fits", "_sourceden_map.hd5")
sd_binmode = "custom"
# sd_custom = [0.5, 1, 2, 2.5, 3, 3.5, 4, 5, 10]
# sd_custom = [0.2,0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6]

bdm = BinnedDensityMap.create(
    map_file,
    bin_mode="linear",
    N_bins=None,
    bin_width=None,
    custom_bins=None,
)

# ## Reading in the new catalog, to ensure correctness

# In[28]:


cat_t = Table.read(outcat)
scylla_t = cat_t
print('Number of sources in trimmed catalog (%s): %s' % (outcat, len(scylla_t)))

# In[29]:


bin_foreach_source = np.zeros(len(scylla_t), dtype=int)

# Find the density bin into which each ra, dec source fits
for i in range(len(cat_t)):
    bin_foreach_source[i] = bdm.bin_for_position(
        scylla_t[ra_key][i], scylla_t[dec_key][i]
    )
print('len and max bin_foreach_source', len(bin_foreach_source), max(bin_foreach_source[:]))
print('shape bin_foreach_source', np.shape(bin_foreach_source))

# Find the RA/DEC indices for each bin
binnrs = np.unique(bin_foreach_source)
bin_idxs = []
for b in binnrs:
    sources_for_bin = bin_foreach_source == b
    bin_idxs.append(sources_for_bin)

n_srcs_in_bin = []
for k in range(len(binnrs)):
    cat = cat_t[bin_idxs[k]]
    #     print(binnrs[k], np.shape(cat['RA']))
    n_srcs_in_bin.append(np.shape(cat['RA'])[0])
# print(n_srcs_in_bin, sum(n_srcs_in_bin))
print('max n_srcs_in_bin', max(n_srcs_in_bin))

# ## Plot the source density map and the density bin of each ra/dec

# In[42]:


plot_ast_cat = True

# if plotting the photometry st catalog stats
if plot_ast_cat is False:
    outcat_src_den_img = outcat.replace(".fits", "_source_den_image.fits")
    src_den_img = fits.open(outcat_src_den_img)[0].data
    # print('max src_den_img', max(src_den_img))

    plt.figure(figsize=(16, 16))
    plt.suptitle('Source Density: ks=%s, th=%s, crwd=%s, rnd=%s, bright=%s, pixsize=%s' % (
    kernel_size, sd_thre, crowd_th, round_th, all_lt_20_final, pixsize))

    # Plot the source density image
    plt.subplot(221)
    im = plt.imshow(src_den_img, origin="lower")
    plt.colorbar(im, label='N / 1 arcsec$^2$ (total N per %sx%s=%s*colorbar value)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("BEAST: Source density per %s arcsec$^2$" % pixsize)

    # Histogram of number of sources in each source density bin
    plt.subplot(222)
    plt.bar(binnrs, n_srcs_in_bin)
    plt.title('BEAST: N sources in each src density bin')

    nbins = [47, 44]  # [47,44] for 5" pixsize
    # 2D histogram for the stcat
    # h_st, xedges_st, yedges_st = np.histogram2d(stcat['RA'], stcat['DEC'], bins=nbins)
    # plt.subplot(223)
    # h2d_st = plt.imshow(h_st.T, origin='lower')
    # plt.colorbar(h2d_st)
    # plt.title('st catalog')

    # 2D hist for the new vgst catalog
    cuts = (scylla_t['CUT_FLAG'] == 0)

    h = plt.hist2d(scylla_t[ra_key][cuts], scylla_t[dec_key][cuts],
                   bins=nbins, norm=LogNorm())

    #     h, xedges, yedges = np.histogram2d(scylla_t['RA'], scylla_t['DEC'], bins=nbins)
    plt.subplot(223)
    h2d = plt.imshow(h.T, origin='lower')
    plt.colorbar(h2d)
    plt.title('np.hist2d: Src den 47 x 44 bins')

    # histogram of number of sources per bin
    plt.subplot(224)
    plt.hist((h[1].T), bins=10, log=True)
    plt.xlabel('Histogram: N sources per bin in y')

if plot_ast_cat:
    cuts = (scylla_t['CUT_FLAG'] == 0)
    print(len(np.where(cuts == True)[0]))

    fig = plt.figure(figsize=(16, 7))

    plt.suptitle('Source Density: ks=%s, th=%s, crwd=%s, rnd=%s, bright=%s, pixsize=%s' % (
    kernel_size, sd_thre, crowd_th, round_th, all_lt_20_final, pixsize))

    nbins = [47, 44]  # [47,44] for 5" pixsize
    # 2D hist for the new vgst catalog
    plt.subplot(121)
    #     h_plt = plt.hist2d(scylla_t[ra_key], scylla_t[dec_key], bins=nbins, norm=LogNorm());
    #     fig.colorbar(h[3])#, ax=ax)
    h, xedges, yedges = np.histogram2d(scylla_t[ra_key][cuts], scylla_t[dec_key][cuts],
                                       bins=nbins)  # , normed=LogNorm())

    cb = plt.imshow(h.T, origin='lower')  # ,norm=LogNorm())
    cbar = plt.colorbar(cb)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label='N', size=16)
    plt.xlabel('X (N bins)')
    plt.ylabel('Y (N bins)')

    plt.title('Source density of culled phot catalog')

    # histogram of number of sources per bin
    plt.subplot(122)
    #     plt.hist(h[0], bins=10, log=True)
    plt.hist(h.T, bins=10, log=True)
    plt.xlabel('N sources per bin')

plt.savefig(outDir + 'source_density_bins_%s_ks_%s_th_%s_crwd_%s_rnd_%s_bright_%s.png' % (
target, kernel_size, sd_thre, crowd_th, round_th, all_lt_20_final), dpi=300)

# ## Range of source densities for culled photometry catalog

# In[31]:


src_den_range = h.max() - h.min()
print('h min; h max', h.min(), h.max())
print('The source density range:', src_den_range)

# ## Apply est quality cuts to st.fake

# In[125]:


scylla_st_asts = 'scylla/15891_SMC-3956ne-9632.st.fake.fits'

st_fake_t = Table.read(scylla_st_asts)
len(st_fake_t)

# Cut on DOLPHOT flag. Keep only sources with flag=0 or 2
flag_cut_fake = ((st_fake_t[two_key[0] + '_FLAG'] == 0) | (st_fake_t[two_key[0] + '_FLAG'] == 2)) & (
            (st_fake_t[two_key[1] + '_FLAG'] == 0) | (st_fake_t[two_key[1] + '_FLAG'] == 2))
print('FLAG remaining:', len(st_fake_t[flag_cut_fake]))

# Cut on RATE. Keep only sources with rate != 0.0 in ALL bands
rate_cut_fake = np.all([st_fake_t[x + '_VEGA'] != 0 for x in filters], axis=0).tolist()
print('RATE remaining:', len(st_fake_t[rate_cut_fake]))

# Cut on CROWD. Keep only sources with CROWD < crowd_threshold (GST default=2.25)
crowd_cut_fake = (st_fake_t[two_key[0] + '_CROWD'] < crowd_th) & (st_fake_t[two_key[1] + '_CROWD'] < crowd_th)
print('CROWD remaining:', len(st_fake_t[crowd_cut_fake]))

# Cut on SHARP. Keep only sources with sharp**2 < sharp_threshold (GST default=0.2)
sharp_cut_fake = (st_fake_t[two_key[0] + '_SHARP'] ** 2 < sharp_th) & (st_fake_t[two_key[1] + '_SHARP'] ** 2 < sharp_th)
print('SHARP remaining:', len(st_fake_t[sharp_cut_fake]))

# Cut on ROUND. Keep only sources with round < round_threshold (no ROUND cut in the default cut)
round_cut_fake = (st_fake_t[two_key[0] + '_ROUND'] < round_th) & (st_fake_t[two_key[1] + '_ROUND'] < round_th)
print('ROUND remaining:', len(st_fake_t[round_cut_fake]))

# Cut on bright sources. Keep sources with F475W_VEGA <= bright_threshold
bright_cut_fake = (st_fake_t[two_key[0] + '_VEGA'] <= bright_th)
print('Bright stars (f475w_vega <= %s): %s' % (bright_th, len(st_fake_t[bright_cut_fake])))

qual_cut_st_fake = crowd_cut_fake & sharp_cut_fake & round_cut_fake
my_cut_fake = crowd_cut_fake & sharp_cut_fake & round_cut_fake & flag_cut_fake & rate_cut_fake | bright_cut_fake
len_my_cut_fake_removed = len(st_fake_t[~my_cut_fake])
print('Total sources removed by quality cuts:', len_my_cut_fake_removed)
print('Total sources remaining:', len(st_fake_t[my_cut_fake]))

my_cut_no_bright_fake = crowd_cut_fake & sharp_cut_fake & round_cut_fake & flag_cut_fake & rate_cut_fake

# ## Apply est median filter cuts to st.fake - Part I: Identify diffraction spikes

# In[78]:


if 'RA' in st_fake_t.colnames:
    ra_key_f = "RA"
    dec_key_f = "DEC"
else:
    ra_key_f = "RA_J2000"
    dec_key_f = "DEC_J2000"
print(ra_key_f, dec_key_f)

# Plot a histogram of the st catalog
plt.figure(figsize=(9, 7))

# plt.subplot(121)
h_map_fake = plt.hist2d(st_fake_t[ra_key_f], st_fake_t[dec_key_f], bins=nbins, norm=LogNorm(), cmap=cmocean.cm.deep)
cb = plt.colorbar()
cb.set_label('Source density')
plt.xlabel(ra_key_f, fontsize=15)
plt.ylabel(dec_key_f, fontsize=15)
plt.title('%d Fake ST objects (no cuts)' % (len(st_fake_t)))

# In[79]:


# Use same kernel size as the one applied on the photometry catalog
median_map_fake = medfilt(h_map_fake[0], kernel_size=kernel_size)

if plot:
    fig = plt.figure(figsize=(17, 14))
    plt.suptitle('Remove spikes: ks=%s, crowd=%s, round=%s, sharp^2=%s' % (kernel_size, crowd_th, round_th, sharp_th))

    plt.subplot(221)
    plt.imshow(h_map_fake[0].T - median_map_fake.T, origin='lower', aspect='auto', cmap=plt.cm.rainbow_r,
               extent=[h_map_fake[1].min(), h_map_fake[1].max(), h_map_fake[2].min(), h_map_fake[2].max()], vmax=30)
    cb = plt.colorbar()
    cb.set_label('Star Count - Median Star Count')
    plt.title('Star Count (st) - Median Star Count')

    plt.subplot(222)
    h_fake = plt.hist((h_map_fake[0].T - median_map_fake.T).ravel(), bins=150, log=True)
    plt.xlabel('Star Count - Median Star Count')
    plt.xlim(-700, 4000)

    plt.subplot(223)
    plt.imshow(median_map_fake.T, origin='lower', aspect='auto', cmap=plt.cm.rainbow_r,
               extent=[h_map_fake[1].min(), h_map_fake[1].max(), h_map_fake[2].min(), h_map_fake[2].max()])
    cb = plt.colorbar()
    cb.set_label('Median Filtered Star Count')
    plt.title('Median Star Count, ks = %s' % kernel_size)

    plt.subplot(224)
    h_fake = plt.hist(median_map_fake.T.ravel(), bins=150, log=True)
    plt.xlabel('Median Star Count')

#     plt.savefig('%sstars-medianstars_%s_ks_%s_crwd_%s_rnd_%s.png' % \
#                 (outDir, target, kernel_size, crowd_th, round_th), dpi=300)


# ## Apply est median filter cuts to st.fake - Part II: Place a source density threshold

# In[146]:


sd_thre_fake = 700

if plot:
    plt.figure(figsize=(16, 7))

    plt.subplot(121)
    plt.imshow(h_map_fake[0].T, origin='lower', norm=LogNorm(), aspect='auto',
               extent=[h_map_fake[1].min(), h_map_fake[1].max(), h_map_fake[2].min(), h_map_fake[2].max()],
               cmap=cmocean.cm.deep)
    plt.title('ST catalog')

# Remove sources which, after the median-filtered results are removed, still surpass a threshold.
high_sd_fake = h_map_fake[0] - median_map_fake > sd_thre_fake
len_high_sd_fake = len(np.where(high_sd_fake == True)[0])
print('%s regions with source density above the %d threshold' % (len_high_sd_fake, sd_thre_fake))

if plot:
    plt.subplot(122)
    plt.imshow(h_map_fake[0].T * high_sd_fake.T, origin='lower',
               norm=colors.LogNorm(vmin=h_map_fake[1].min(), vmax=h_map[1].max()), cmap=cmocean.cm.deep, aspect='auto',
               extent=[h_map_fake[1].min(), h_map_fake[1].max(), h_map_fake[2].min(), h_map_fake[2].max()])
    plt.title('High-density sources threshold: %s' % sd_thre_fake)

#     plt.savefig(outDir + 'threshold_map_th_' + str(sd_thre) + '_ks_' + str(kernel_size) + '.png')


# ### Combine diffraction spikes sources to cut

# In[147]:


inds_ds_fake = []
nrow_f, ncol_f = h_map_fake[0].shape
# high_sd_fake = h_map_fake[0] - median_map_fake > sd_thre_fake
print(high_sd_fake.shape)

for ira in range(nrow_f):
    for jdec in range(ncol_f):
        # If this ra/dec value of the high_sd_fake 2d hist is True (above the set threshold), then proceed
        if high_sd_fake[ira, jdec]:
            # original selection: only above threshold
            stars_fake, = np.where(
                (st_fake_t[ra_key_f] >= h_map_fake[1][ira]) & (st_fake_t[ra_key_f] < h_map_fake[1][ira + 1]) & (
                            st_fake_t[dec_key_f] >= h_map_fake[2][jdec]) & (
                            st_fake_t[dec_key_f] < h_map_fake[2][jdec + 1]))

            inds_ds_fake.append(stars_fake.tolist())

inds_ds_fake = np.concatenate(inds_ds_fake).astype('int')

diff_flag_fake = np.ones(len(st_fake_t), dtype=bool)
diff_flag_fake[inds_ds_fake] = False

final_cut_fake = my_cut_fake & diff_flag_fake

len_diff_flag_fake = len(st_fake_t[diff_flag_fake])
len_final_cut_fake = len(st_fake_t[final_cut_fake])
print('N sources removed as diffraction spikes onlyfrom st.fake: %s (%s %%)' % (
len_diff_flag_fake, np.around((len_diff_flag_fake / len(st_fake_t)), decimals=2) * 100))
print('N sources removed by phot quality cuts only from st.fake:', len_my_cut_fake_removed,
      '(%s %%)' % (np.around(len_my_cut_fake_removed / len(st_fake_t), decimals=3) * 100))
print('N total sources removed from st fake: %s (%s %%)' % (
len(st_fake_t[~final_cut_fake]), np.around(len(st_fake_t[~final_cut_fake]) / len(st_fake_t), decimals=2) * 100))
print('N total sources remaining in st.fake:', len_final_cut_fake,
      '(%s %%)' % (np.around(len_final_cut_fake / len(st_fake_t), decimals=2) * 100))

# ## Plot st.fake cuts

# In[148]:


import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 100000

# len_diff_flag_fake = len(st_fake_t[diff_flag_fake])
# len_final_cut_fake = len(st_fake_t[final_cut_fake])

fig = plt.figure(figsize=(15, 21))

# removed sources from photometric quality cuts on the st.fake catalog
plt.subplot(321)
h_fake = plt.hist2d(st_fake_t[ra_key_f], st_fake_t[dec_key_f], bins=nbins, norm=LogNorm())  # , cmap=cmocean.cm.deep)
h_qual_cut = plt.plot(st_fake_t[ra_key_f][~my_cut_fake], st_fake_t[dec_key_f][~my_cut_fake], '.', c='magenta', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.ylabel(dec_key_f, fontsize=15)
plt.title('%d st.fake phot quality sources removed' % len_my_cut_fake_removed)

# remaining sources from photometric quality cuts
plt.subplot(322)
h_qual = plt.plot(st_fake_t[ra_key_f][my_cut_fake], st_fake_t[dec_key_f][my_cut_fake], ',', c='purple', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.title('%d st.fake sources remain after phot cuts' % len(st_fake_t[my_cut_fake]))

# removed sources from diffraction spikes cuts on the st.fake catalog
plt.subplot(323)
plt.hist2d(st_fake_t[ra_key_f], st_fake_t[dec_key_f], bins=nbins, norm=LogNorm())  # , cmap=cmocean.cm.deep)
h_ds_cut = plt.plot(st_fake_t[ra_key_f][~diff_flag_fake], st_fake_t[dec_key_f][~diff_flag_fake], '.', c='lime', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.ylabel(dec_key_f, fontsize=15)
plt.title('%d st.fake diffraction spikes removed' % len(st_fake_t[~diff_flag_fake]))

# remaining sources from diffraction spikes cut
plt.subplot(324)
h_ds = plt.plot(st_fake_t[ra_key_f][diff_flag_fake], st_fake_t[dec_key_f][diff_flag_fake], ',', c='seagreen', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.title('%d st.fake sources remain after DS cuts' % len_diff_flag_fake)

# remaining sources from photometric quality cuts
plt.subplot(325)
plt.hist2d(st_fake_t[ra_key_f], st_fake_t[dec_key_f], bins=nbins, norm=LogNorm())  # , cmap=cmocean.cm.deep)
h_qual = plt.plot(st_fake_t[ra_key_f][~final_cut_fake], st_fake_t[dec_key_f][~final_cut_fake], '.', c='aqua', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.ylabel(dec_key_f, fontsize=15)
plt.title('%d Total sources removed from st.fake' % len(st_fake_t[~final_cut_fake]))

# remaining sources from diffraction spikes cut
plt.subplot(326)
h_ds = plt.plot(st_fake_t[ra_key_f][diff_flag_fake], st_fake_t[dec_key_f][diff_flag_fake], ',', c='deepskyblue', ls='')
plt.xlabel(ra_key_f, fontsize=15)
plt.title('%d Total sources remaining in st.fake' % len_final_cut_fake)

if savefig:
    figname = outDir + 'clean_st_3_catalog_%s_ks_%s_th_%s_crwd_%s_rnd_%s.png' % (
    target, kernel_size, sd_thre_fake, crowd_th, round_th)
    plt.savefig(figname, dpi=200)
    print('st.fake cleaned catalogs saved as', figname)

cmd_st_sources = st_fake_t[two_key[0] + '_VEGA'] < 99
len_cmd_st_sources = len(st_fake_t[cmd_st_sources])

# ## Record st.fake with cuts ('est.fake'); correct NaNs

# In[149]:


f336nan_fake = np.isnan(st_fake_t['F336W_IN'])
print('N (F336W_IN = NaN):', len(st_fake_t[f336nan_fake]))  # N=9574
st_fake_t["F336W_IN"][f336nan_fake] = 99

# Check if st.fake has CUT_FLAG colum. If not, create it.
if "CUT_FLAG" not in st_fake_t.colnames:
    print("Adding a CUT_FLAG")
    st_fake_t.add_column(np.zeros(len(st_fake_t), dtype=int), name="CUT_FLAG", index=-1)

    # making sure cut_flag=0 if multiple tests record 0s and 1s in catalog
st_fake_t["CUT_FLAG"] = 0

# test: generate a catalog with only photometric cuts
st_fake_t_no_ds_cut = st_fake_t
st_fake_t_no_ds_cut["CUT_FLAG"][my_cut_fake] = 0
st_fake_t_no_ds_cut["CUT_FLAG"][~my_cut_fake] = 1
print('diff_flag_fake len', len(st_fake_t[~diff_flag_fake]))

# designate removed sources with cut_flag of 1
st_fake_t["CUT_FLAG"][final_cut_fake] = 0
st_fake_t["CUT_FLAG"][~final_cut_fake] = 1

# check
print('first # should be smaller than second # since it shows N srs remaining after all cuts:')
print(len(st_fake_t[final_cut_fake]))
print(len(st_fake_t_no_ds_cut[diff_flag_fake]))

# record all sources
outcat_fake = inDir + '15891_%s_%s_%s_%s_%s.est.fake.fits' % (target, kernel_size, sd_thre_fake, crowd_th, round_th)
outcat_fake_no_ds_cut = inDir + '15891_%s_%s_%s.est.fake.fits' % (target, crowd_th, round_th)

hdu_stfake = fits.BinTableHDU(data=st_fake_t)
hdu_stfake.writeto(outcat_fake, overwrite=True)

hdu_stfake_no_ds_cut = fits.BinTableHDU(data=st_fake_t_no_ds_cut)
hdu_stfake_no_ds_cut.writeto(outcat_fake_no_ds_cut, overwrite=True)

# Read table in again as it is written to fits file; make checks
incat_fake = Table.read(outcat_fake)
incat_fake_no_ds_cut = Table.read(outcat_fake_no_ds_cut)
print(np.mean(incat_fake["CUT_FLAG"]))  # 0.6 for smc-6.st.fake
print(np.mean(st_fake_t["CUT_FLAG"]))  # 0.6 as well. check passed
print(np.mean(incat_fake_no_ds_cut["CUT_FLAG"]))

# ## Calculate the area under the completeness curve

# In[142]:


scylla_sed_grid = 'scylla/15891_SMC-3956ne-9632_seds.grid.hd5'

# cut sources should be flagged and read by the BEAST


# In[150]:

# Need to tweak plot_toothpick details so it returns fluxes and completeness arrays
fluxes, compls = plot_toothpick_details.plot_toothpick_details(outcat_fake, scylla_sed_grid, savefig='png')

# ## Calculate completeness curve area; source density to completeness ratio

# In[151]:


print(grid_results)

# In[153]:


print('Source density range:', src_den_range)
compls_sums = [sum(x) for x in compls]
src_den_to_comp_sum = [x / src_den_range for x in compls_sums]

# array to hold all parameters and ratios
param_results = np.empty((len(filters), 6), dtype=object)
for i in range(len(filters)):
    param_results[i, 0] = src_den_to_comp_sum[i]

param_results[:, 1] = kernel_size
param_results[:, 2] = sd_thre_fake
param_results[:, 3] = crowd_th
param_results[:, 4] = round_th
param_results[:, 5] = sharp_th

print(param_results)

# uncomment for new results
grid_results.append(param_results)

grid_results_arr = np.asarray(grid_results, dtype=float)

# In[154]:


print('Completeness sum in each filter:', compls_sums)
print('Completeness to source density ratio', src_den_to_comp_sum)

# ## Check if global minimum; if so, replace previous minimum

# In[65]:


if trials == 1:
    global_maximum = [0.0, 0.0, 0.0]

for i in range(len(filters)):
    if src_den_to_comp_sum[i] > global_maximum[i]:
        print('There is a new global maximum for %s!' % filters[i])
        global_maximum[i] = src_den_to_comp_sum[i]

# ## Kernel Size vs Density Threshold

# In[157]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
cmap = 'inferno'

# Plot results for each filter
for j in range(trials):
    for i, ax in enumerate(axes.flat):
        sc = ax.scatter(grid_results_arr[j, i, 1], grid_results_arr[j, i, 2], c=grid_results_arr[j, i, 0], marker='s',
                        s=170, cmap=cmap)
        ax.set_xlabel('Kernel size', fontsize=16)
        if i == 0:
            ax.set_ylabel('Density threshold', fontsize=16)
        ax.set_title('%s' % filters[i])

fig.subplots_adjust(top=0.7)  # to make space for the cbar
cbar_ax = fig.add_axes([0.12, 0.85, 0.77, 0.04])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=14)
cb.set_label(label='Completeness-to-source density ratio', size=16)

# ## Crowding vs Roundness

# In[158]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
cmap = 'inferno'

# Plot results for each filter
for j in range(trials):
    for i, ax in enumerate(axes.flat):
        sc = ax.scatter(grid_results_arr[j, i, 3], grid_results_arr[j, i, 4], c=grid_results_arr[j, i, 0], marker='s',
                        s=170, cmap=cmap)
        ax.set_xlabel('Crowding', fontsize=16)
        if i == 0:
            ax.set_ylabel('Roundness', fontsize=16)
        ax.set_title('%s' % filters[i])

fig.subplots_adjust(top=0.7)  # to make space for the cbar
cbar_ax = fig.add_axes([0.12, 0.85, 0.77, 0.04])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=14)
cb.set_label(label='Completeness-to-source density ratio', size=16)

# ## Global Maximum:

# In[159]:


print(global_maximum)

# ## Grid Results

# In[160]:


print(grid_results)
