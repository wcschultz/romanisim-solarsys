import asdf
import numpy as np
from astropy import table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from romanisim.parameters import pixel_scale
import os
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.signal import convolve2d
from skimage import measure

def overview_plot(base_file_path):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')

    file_prefix = base_file_path.split('/')[-1]
    if not os.path.exists(f'./overview_plots/{file_prefix[:-1]}/'):
        os.mkdir(f'./overview_plots/{file_prefix[:-1]}/')

    catalog_table = l1_file['romanisim']['moving_bodies_catalog']

    exp_time = l2_file['roman']['meta']['exposure']['exposure_time']

    extra_offset = 20
    for mobj in catalog_table:
        mb_x, mb_y = mobj['initial_position']
        rad_angle = mobj['direction'] * np.pi / 180
        x_offset = extra_offset * np.sign(np.cos(rad_angle))
        y_offset = extra_offset * np.sign(np.sin(rad_angle))

        # if sin or cos == 0, ensure there is non-zero width
        if x_offset == 0:
            x_offset = extra_offset
        if y_offset == 0:
            y_offset = extra_offset

        start_x = int(mb_x - x_offset)
        start_y = int(mb_y - y_offset)

        pix_speed = mobj['angular_speed'] / 1000 / pixel_scale
        width = pix_speed * exp_time

        end_x = int(mb_x + width * np.cos(rad_angle) + x_offset)
        end_y = int(mb_y + width * np.sin(rad_angle) + y_offset)

        end_x = min(end_x, 4088)
        end_y = min(end_y, 4088)

        slice = np.s_[start_y:end_y, start_x:end_x]

        x_for_mesh = np.arange(start_x, end_x) - mb_x
        y_for_mesh = np.arange(start_y, end_y) - mb_y

        xmesh, ymesh = np.meshgrid(x_for_mesh, y_for_mesh)

        last_res_trimmed = l1_file['roman']['data'][-1,4:-4,4:-4][slice]
        l2_img = l2_file['roman']['data'][slice]
        jump_mask = np.bitwise_and(l2_file['roman']['dq'][slice], 4, casting='safe') > 0

        if np.sum(jump_mask) > 0:
            max_count = np.max(last_res_trimmed[jump_mask])
        else:
            max_count = np.max(last_res_trimmed)

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        pcm = axs[0].pcolormesh(xmesh, ymesh, last_res_trimmed, norm=LogNorm(vmax=max_count), shading='nearest')
        fig.colorbar(pcm, ax=axs[0], label='Counts [DN]', shrink=0.6)
        axs[0].set_title('Final Resultant from L1')
        axs[0].set_ylabel('Relative Pixels from Start')
    
        pcm = axs[1].pcolormesh(xmesh, ymesh, l2_img, shading='nearest', vmin=0) #norm=LogNorm(vmin=1e-2, vmax=1e2)
        fig.colorbar(pcm, ax=axs[1], label='Count Rate [DN/s]', shrink=0.6)
        axs[1].set_title(f'{mobj['magnitude']} mag object moving at {mobj['angular_speed']} mas/s in a {int(exp_time)} s ramp\n\nL2 Slope Image')
        axs[1].set_xlabel('Relative Pixels from Start')

        pcm = axs[2].pcolormesh(xmesh, ymesh, jump_mask, shading='nearest')
        axs[2].set_title('L2 Jump Detected Flag')
        fig.colorbar(pcm, ax=axs[2], label='1 = Jump Detected', shrink=0.6)

        for ax in axs:
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'./overview_plots/{file_prefix[:-1]}/{file_prefix}overview_m{mobj['magnitude']:.2f}_as{mobj['angular_speed']:.2f}.png')
        plt.close(fig)

#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_catalog_MA3_F146_WFI01_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_catalog_MA11_F146_WFI01_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_default_catalog_MA3_F146_WFI01_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_default_catalog_MA6_F146_WFI01_')

#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_w0_catalog_MA3_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_w0_catalog_MA6_F146_WFI01_')

#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_MA3_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_MA6_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_MA8_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_MA11_F146_WFI01_')

#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA3_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA4_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA6_F146_WFI01_')
#overview_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA11_F146_WFI01_')

#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA5_F146_WFI01_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA9_F146_WFI01_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA11_F146_WFI01_')

#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA4_F146_WFI11_')
#overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA7_F146_WFI11_')

################################
################################
################################

def check_aperture_photometry(base_file_path):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')

    file_prefix = base_file_path.split('/')[-1]

    star_positions = [(star[0]+4, star[1]+4) for star in l1_file['romanisim']['simcatobj']]
    mb_positions = [(mb['initial_position'][0]+3,mb['initial_position'][1]+3)  for mb in l1_file['romanisim']['moving_bodies_catalog']]
    mb_mags = [mb['magnitude'] for mb in l1_file['romanisim']['moving_bodies_catalog']]
    mb_speeds = [mb['angular_speed'] for mb in l1_file['romanisim']['moving_bodies_catalog']]

    star_apertures = CircularAperture(star_positions, 4)
    mb_apertures = CircularAperture(mb_positions, 4)

    star_aps = {}
    mb_aps = {}

    plt.figure()
    plt.imshow(l1_file['roman']['data'][-1,:,:], origin='lower')
    star_apertures.plot(plt.gca())
    mb_apertures.plot(plt.gca())

    for rind in range(l1_file['roman']['data'].shape[0]):
        med_val = np.median(l1_file['roman']['data'][rind,:,:])
        star_ap_phot = aperture_photometry(l1_file['roman']['data'][rind,:,:] - med_val, star_apertures)
        mb_ap_phot = aperture_photometry(l1_file['roman']['data'][rind,:,:] - med_val, mb_apertures)

        if rind == 0:
            for ap in star_ap_phot:
                star_aps[ap['id']] = [ap['aperture_sum']]
                #print(ap['aperture_sum'])
            for ap in mb_ap_phot:
                mb_aps[ap['id']] = [ap['aperture_sum']]
        else:
            for ap in star_ap_phot:
                star_aps[ap['id']].append(ap['aperture_sum'] - star_aps[ap['id']][0])
            for ap in mb_ap_phot:
                mb_aps[ap['id']].append(ap['aperture_sum'] - mb_aps[ap['id']][0])

    cmap = plt.get_cmap('tab20', len(star_aps.keys()))

    fig, axs = plt.subplots(1,4, figsize=(16,4))
    for i, key in enumerate(star_aps):
        axs[0].plot(star_aps[key][1:], 's', color=cmap(i), label=f'mag={mb_mags[i]}, speed={mb_speeds[i]}')
        axs[0].plot(mb_aps[key][1:], 'o', color=cmap(i))
        axs[1].plot(np.array(star_aps[key][1:]) - np.array(mb_aps[key][1:]), color=cmap(i))
        axs[2].plot(abs(np.array(mb_aps[key][1:]) - np.array(star_aps[key][1:])) / np.array(star_aps[key][1:]), color=cmap(i))

        # calculate relative uncertainty
        ma = np.array(mb_aps[key][1:])
        sa = np.array(star_aps[key][1:])
        del_ma = np.sqrt(ma)
        del_sa = np.sqrt(sa)
        del_sa_ma = np.sqrt(del_ma**2. + del_sa**2.)
        sigma = (ma + sa) / sa * np.sqrt((del_sa_ma / (sa + ma))**2. + (del_sa / sa)**2.)

        axs[2].plot(sigma, color=cmap(i), linestyle=":")
        
        axs[3].plot(np.array(mb_aps[key][1:]) / np.array(star_aps[key][1:]), color=cmap(i))

    axs[0].legend(ncol=2, loc='lower right', fontsize=6)
    axs[0].set_yscale('log')
    #axs[1].set_yscale('log')
    axs[1].set_yscale('symlog', linthresh=10)
    axs[1].set_xlabel('Resultant number - 1')
    axs[0].set_ylabel('Counts in aperture')
    axs[1].set_ylabel('Star photometry - MB photometry')
    axs[2].set_ylabel('|Star photometry - MB photometry| / Star photometry')
    axs[2].set_yscale('log')
    axs[3].set_ylabel('MB photometry / Star photometry')
    axs[3].set_yscale('linear')
    axs[3].set_ylim(0.75, 1.25)
    axs[2].set_ylim(1e-2, 10)
    axs[1].set_ylim(-1e5, 1e5)
    axs[0].set_ylim(1, 1e6)
    #axs[3].axhline(1, color='k', linestyle=':')
    plt.tight_layout()
    plt.savefig(f'test_plots/{file_prefix}_ap_phot_test.png')


#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA3_F146_WFI01_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA4_F146_WFI01_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA6_F146_WFI01_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA11_F146_WFI01_')

#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_no_pn_MA3_F146_WFI01_')

#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_no_pn_MA5_F146_WFI01_')

#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA5_F146_WFI01_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA5_F146_WFI02_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA5_F146_WFI05_')
#check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_catalog_new_imp_MA8_F146_WFI05_')
#for det_i in range(18):
#    det_n = det_i + 1
    #check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_det_sweep_MA4_F146_WFI{det_n:02}_')
    #check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_det_sweep_noflat_MA4_F146_WFI{det_n:02}_')
#    check_aperture_photometry(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_det_sweep_noflat_persist_MA4_F146_WFI{det_n:02}_')

################################
################################
################################

def num_jump_detected_plot(base_file_path):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')

    file_prefix = base_file_path.split('/')[-1]
    if not os.path.exists('./num_jump_det_pix_plots/'):
        os.mkdir('./num_jump_det_pix_plots/')

    catalog_table = l1_file['romanisim']['moving_bodies_catalog']

    exp_time = l2_file['roman']['meta']['exposure']['exposure_time']

    extra_offset = 20
    num_jump_dets = np.zeros(len(catalog_table))

    for i,mobj in enumerate(catalog_table):
        mb_x, mb_y = mobj['initial_position']
        rad_angle = mobj['direction'] * np.pi / 180
        x_offset = extra_offset * np.sign(np.cos(rad_angle))
        y_offset = extra_offset * np.sign(np.sin(rad_angle))

        # if sin or cos == 0, ensure there is non-zero width
        if x_offset == 0:
            x_offset = extra_offset
        if y_offset == 0:
            y_offset = extra_offset

        start_x = int(mb_x - x_offset)
        start_y = int(mb_y - y_offset)

        pix_speed = mobj['angular_speed'] / 1000 / pixel_scale
        width = pix_speed * exp_time

        end_x = int(mb_x + width * np.cos(rad_angle) + x_offset)
        end_y = int(mb_y + width * np.sin(rad_angle) + y_offset)

        end_x = min(end_x, 4088)
        end_y = min(end_y, 4088)

        slice = np.s_[start_y:end_y, start_x:end_x]

        num_jump_dets[i] = np.sum(np.bitwise_and(l2_file['roman']['dq'][slice], 4, casting='safe') > 0)

    catalog_table['num_jump_dets'] = num_jump_dets

    # time to make the plot
    unique_mags = np.unique(catalog_table['magnitude'].data)
    unique_speeds = np.unique(catalog_table['angular_speed'].data)

    fig, axs = plt.subplots(1,2, figsize=(12,6))

    mag_cmap = plt.get_cmap('jet', len(unique_mags))
    for i, mag in enumerate(unique_mags):
        mag_mask = catalog_table['magnitude'] == mag
        mag_cat = catalog_table[mag_mask]
        axs[0].plot(mag_cat['angular_speed'], mag_cat['num_jump_dets'], color=mag_cmap(i), marker='.', label=f'mag={mag}')
        
    axs[0].set_ylabel('Number of Jump Detections in Box')
    axs[0].set_xlabel('Angular Speed [mas/s]')
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].legend(bbox_to_anchor=(0, 1.02), ncol=2, loc='lower left')

    speed_cmap = plt.get_cmap('jet', len(unique_speeds))

    for i, speed in enumerate(unique_speeds):
        speed_mask = catalog_table['angular_speed'] == speed
        speed_cat = catalog_table[speed_mask]
        axs[1].plot(speed_cat['magnitude'], speed_cat['num_jump_dets'], color=speed_cmap(i), marker='.', label=f'ang. speeed ={speed:.2f}')
        
    axs[1].set_ylabel('Number of Jump Detections in Box')
    axs[1].set_xlabel('AB Magnitude')
    axs[1].set_yscale('log')
    axs[1].legend(bbox_to_anchor=(0, 1.02), ncol=2, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'./num_jump_det_pix_plots/{file_prefix[:-1]}.png')
 
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_default_catalog_MA3_F146_WFI01_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_full_default_catalog_MA6_F146_WFI01_')
#for i in range(3, 12):
#    num_jump_detected_plot(f'/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA{i}_F146_WFI01_')

#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA3_F146_WFI01_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA4_F146_WFI01_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA5_F146_WFI01_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA9_F146_WFI01_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA11_F146_WFI01_')

#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA4_F146_WFI11_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA7_F146_WFI11_')
#num_jump_detected_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA11_F146_WFI11_')

################################
################################
################################


def correlation_identification(base_file_path, det_threshold=4, det_radius=50):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')

    file_prefix = base_file_path.split('/')[-1]
    if not os.path.exists('./num_jump_det_pix_plots/'):
        os.mkdir('./num_jump_det_pix_plots/')

    catalog_table = l1_file['romanisim']['moving_bodies_catalog']

    exp_time = l2_file['roman']['meta']['exposure']['exposure_time']

    jump_dq = np.bitwise_and(l2_file['roman']['dq'], 4, casting='safe') > 0
    filter = np.ones((3,3))
    
    print('performing convolution')
    convolution = convolve2d(jump_dq, filter, mode='same')
    detections = convolution > det_threshold
    print('labelling')
    labeled_dets = measure.label(detections, connectivity=2)
    detection_inds = [np.where(labeled_dets == label) for label in np.unique(labeled_dets) if label]
    mean_det_inds = [[np.mean(dets[0]), np.mean(dets[1])] for dets in detection_inds]
    
    cat_xs = [pos[0] for pos in catalog_table['initial_position']]
    cat_ys = [pos[1] for pos in catalog_table['initial_position']]

    det_cat_inds = []
    false_pos = []
    for det_ind in mean_det_inds:
        dists = np.sqrt((det_ind[1]-cat_xs)**2. + (det_ind[0]-cat_ys)**2.)
        min_dist_ind = np.argmin(dists)
        if (dists[min_dist_ind] < det_radius):
            if (min_dist_ind not in det_cat_inds):
                det_cat_inds.append(min_dist_ind)
        else:
            false_pos.append(det_ind)
        
    ang_s = np.unique(catalog_table['angular_speed'])
    fit_line = -2*(np.log10(ang_s)-np.log10(2))**2. + 26

    plt.plot(catalog_table['angular_speed'], catalog_table['magnitude'], 'k.')
    plt.plot(catalog_table['angular_speed'][det_cat_inds], catalog_table['magnitude'][det_cat_inds], 'g.')
    plt.plot(ang_s, fit_line, 'k--')
    plt.ylabel("Magnitude (AB)")
    plt.xlabel("Angular speed [mas/s]")
    plt.title(f'{exp_time}s exposure, threshold={det_threshold}, {len(false_pos)} false positives')
    #plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.ylim(28, 18)
    plt.tight_layout()
    plt.savefig(f'./corr_id_plots/{file_prefix[:-1]}_thresh_{det_threshold}.png')
    plt.close()

#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA4_F146_WFI01_')
#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA4_F146_WFI01_', det_threshold=3)
#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA4_F146_WFI01_', det_threshold=2)
#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA4_F146_WFI01_', det_threshold=1)
#for i in range(4):
    #correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_default_no_cr_catalog_MA3_F146_WFI01_', det_threshold={i})
    #correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA4_F146_WFI11_', det_threshold=(i+1))
    #correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA7_F146_WFI11_', det_threshold=(i+1))
    #correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA11_F146_WFI11_', det_threshold=(i+1))

#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA4_F146_WFI11_', det_threshold=2)
#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA7_F146_WFI11_', det_threshold=2)
#correlation_identification('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA11_F146_WFI11_', det_threshold=2)

 ################################
 #################################
 ################################

def plot_jumps(base_file_path, catalog_table):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')

    read_pattern = l1_file['roman']['meta']['exposure']['read_pattern']
    frame_time = l1_file['roman']['meta']['exposure']['frame_time']

    resultant_times = [np.mean(np.array(res) * frame_time) for res in read_pattern]

    exp_time = l2_file['roman']['meta']['exposure']['exposure_time']

    extra_offset = 20
    for mobj in catalog_table:
        mb_x, mb_y = mobj['start_position']
        rad_angle = mobj['direction'] * np.pi / 180
        x_offset = extra_offset * np.sign(np.cos(rad_angle))
        y_offset = extra_offset * np.sign(np.sin(rad_angle))

        # if sin or cos == 0, ensure there is non-zero width
        if x_offset == 0:
            x_offset = extra_offset
        if y_offset == 0:
            y_offset = extra_offset

        start_x = int(mb_x - x_offset)
        start_y = int(mb_y - y_offset)

        pix_speed = mobj['angular_speed'] / 1000 / pixel_scale
        width = pix_speed * exp_time

        end_x = int(mb_x + width * np.cos(rad_angle) + x_offset)
        end_y = int(mb_y + width * np.sin(rad_angle) + y_offset)

        slice_2d = np.s_[start_y:end_y, start_x:end_x]
        slice_3d = np.s_[:,start_y:end_y, start_x:end_x]

        l1_trimmed = l1_file['roman']['data'][:,4:-4,4:-4].value[slice_3d]
        jump_mask = np.bitwise_and(l2_file['roman']['dq'][slice_2d], 4, casting='safe') > 0

        count_diffs = np.diff(l1_trimmed, axis=0)
        time_diffs = np.diff(resultant_times)
        diff_times = 0.5 * (np.array(resultant_times[1:]) + np.array(resultant_times[:-1]))

        fig, axs = plt.subplots(1,3, sharey=True, figsize=(12,4))
        for i in range(jump_mask.shape[0]):
            for j in range(jump_mask.shape[1]):
                slopes = count_diffs[:,i,j] / time_diffs
                max_slope = max(slopes)
                max_slope_time = diff_times[slopes == max_slope][0] #[0] handles repeatds of max
                if jump_mask[i,j]:
                    axs[0].plot(resultant_times, l1_trimmed[:,i,j], color='r', alpha=0.1)
                    axs[0].set_title('Jump Detected')
                    axs[2].plot(max_slope_time, max_slope, '.', color='r', zorder=0)
                else:
                    axs[1].plot(resultant_times, l1_trimmed[:,i,j], color='b', alpha=0.1)
                    axs[1].set_title('No Jump Detected')
                    axs[2].plot(max_slope_time, max_slope, '.', color='b', zorder=-1)
        for ax in axs:
            ax.set_yscale('log')
        
        axs[0].set_ylabel('Counts')
        axs[0].set_xlabel('Resultant Times [s]')
        plt.savefig(f'./overview_plots/l1_ramps_m{mobj['magnitude']}_as{mobj['angular_speed']}.png')

"""test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [10] # milliarcsec/sec
test_cat["direction"] = [0]
plot_jumps('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p01_test_WFI01_', test_cat) 

test_cat = table.Table()
test_cat["magnitude"] = [20]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [5] # milliarcsec/sec
test_cat["direction"] = [0]
plot_jumps('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p005_m20_test_WFI01_', test_cat)"""


def resultant_dq_plot(base_file_path):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')
    resdq_file = asdf.open(base_file_path + 'resdq.asdf', 'r')

    file_prefix = base_file_path.split('/')[-1]
    if not os.path.exists(f'./res_dq_plots/{file_prefix[:-1]}/'):
        os.mkdir(f'./res_dq_plots/{file_prefix[:-1]}/')

    catalog_table = l1_file['romanisim']['moving_bodies_catalog']

    exp_time = l2_file['roman']['meta']['exposure']['exposure_time']

    extra_offset = 20
    for mobj in catalog_table:
        mb_x, mb_y = mobj['initial_position']
        rad_angle = mobj['direction'] * np.pi / 180
        x_offset = extra_offset * np.sign(np.cos(rad_angle))
        y_offset = extra_offset * np.sign(np.sin(rad_angle))

        # if sin or cos == 0, ensure there is non-zero width
        if x_offset == 0:
            x_offset = extra_offset
        if y_offset == 0:
            y_offset = extra_offset

        start_x = int(mb_x - x_offset)
        start_y = int(mb_y - y_offset)

        pix_speed = mobj['angular_speed'] / 1000 / pixel_scale
        width = pix_speed * exp_time

        end_x = int(mb_x + width * np.cos(rad_angle) + x_offset)
        end_y = int(mb_y + width * np.sin(rad_angle) + y_offset)

        end_x = min(end_x, 4088)
        end_y = min(end_y, 4088)

        x_for_mesh = np.arange(start_x, end_x) - mb_x
        y_for_mesh = np.arange(start_y, end_y) - mb_y
        res_for_mesh = np.arange(resdq_file['resultantdq'].shape[0]) + 1 

        xmesh, ymesh = np.meshgrid(x_for_mesh, y_for_mesh)
        resx_mesh, xres_mesh = np.meshgrid(res_for_mesh, x_for_mesh)
        resy_mesh, yres_mesh = np.meshgrid(res_for_mesh, y_for_mesh)

        slice = np.s_[:,start_y:end_y, start_x:end_x]

        res_dq_slice = resdq_file['resultantdq'][:,4:-4, 4:-4][slice]
        

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(res_inds, y_inds, x_inds)
        #ax.set_aspect('equal')

        def logical_or(x):
            return int(np.sum(x) > 0)

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].pcolormesh(xmesh, ymesh, np.apply_along_axis(logical_or, 0, res_dq_slice), shading='nearest')
        axs[0].set_title('All flagged pixels in L1')
        axs[0].set_ylabel('Y')
        axs[0].set_xlabel('X')

        axs[1].pcolormesh(resy_mesh, yres_mesh, np.apply_along_axis(logical_or, 2, res_dq_slice).transpose(), shading='nearest')
        axs[1].set_title('Y Separation')
        axs[1].set_ylabel('Y')
        axs[1].set_xlabel('Resultant')

        axs[2].pcolormesh(resx_mesh, xres_mesh, np.apply_along_axis(logical_or, 1, res_dq_slice).transpose(), shading='nearest')
        axs[2].set_title('X Separation')
        axs[2].set_ylabel('X')
        axs[2].set_xlabel('Resultant')
    
        for ax in axs:
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'./res_dq_plots/{file_prefix[:-1]}/{file_prefix}resdq_m{mobj['magnitude']:.2f}_as{mobj['angular_speed']:.2f}.png')
        plt.close(fig)

#resultant_dq_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_w_star_no_cr_vis_MA4_F146_WFI01_')
resultant_dq_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_proposal__MA7_F146_WFI11_')

