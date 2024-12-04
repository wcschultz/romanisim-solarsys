import asdf
import numpy as np
from astropy import table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from romanisim.parameters import pixel_scale

def overview_plot(base_file_path, catalog_table):
    l1_file = asdf.open(base_file_path + 'uncal.asdf', 'r')
    l2_file = asdf.open(base_file_path + 'cal.asdf', 'r')

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

        slice = np.s_[start_y:end_y, start_x:end_x]

        x_for_mesh = np.arange(start_x, end_x) - mb_x
        y_for_mesh = np.arange(start_y, end_y) - mb_y

        xmesh, ymesh = np.meshgrid(x_for_mesh, y_for_mesh)

        last_res_trimmed = l1_file['roman']['data'][-1,4:-4,4:-4].value[slice]
        l2_img = l2_file['roman']['data'].value[slice]
        jump_mask = np.bitwise_and(l2_file['roman']['dq'][slice], 4, casting='safe') > 0

        max_count = np.max(last_res_trimmed[jump_mask])

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        pcm = axs[0].pcolormesh(xmesh, ymesh, last_res_trimmed, norm=LogNorm(vmax=max_count), shading='nearest')
        fig.colorbar(pcm, ax=axs[0], label='Counts [DN]', shrink=0.6)
        axs[0].set_title('Final Resultant from L1')
        axs[0].set_ylabel('Relative Pixels from Start')
    
        pcm = axs[1].pcolormesh(xmesh, ymesh, l2_img, shading='nearest', vmin=0) #norm=LogNorm(vmin=1e-2, vmax=1e2)
        fig.colorbar(pcm, ax=axs[1], label='Count Rate [DN/s]', shrink=0.6)
        axs[1].set_title('L2 Slope Image')
        axs[1].set_xlabel('Relative Pixels from Start')

        pcm = axs[2].pcolormesh(xmesh, ymesh, jump_mask, shading='nearest')
        axs[2].set_title('L2 Jump Detected Flag')
        fig.colorbar(pcm, ax=axs[2], label='1 = Jump Detected', shrink=0.6)

        for ax in axs:
            ax.set_aspect('equal')

        plt.savefig(f'./overview_plots/l1_l2_overview_m{mobj['magnitude']}_as{mobj['angular_speed']}.png')


"""test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [100] # milliarcsec/sec
test_cat["direction"] = [45]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p1_test_WFI01_', test_cat)


test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [50] # milliarcsec/sec
test_cat["direction"] = [0]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p05_test_WFI01_', test_cat)

test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [10] # milliarcsec/sec
test_cat["direction"] = [0]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p01_test_WFI01_', test_cat) 

test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [1] # milliarcsec/sec
test_cat["direction"] = [0]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p001_test_WFI01_', test_cat)

test_cat = table.Table()
test_cat["magnitude"] = [15]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [5] # milliarcsec/sec
test_cat["direction"] = [0]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p005_test_WFI01_', test_cat)

test_cat = table.Table()
test_cat["magnitude"] = [20]
test_cat["start_position"] = [(100,100)] # in pixels
test_cat["angular_radius"] = [-1] #arcsec
test_cat["angular_speed"] = [5] # milliarcsec/sec
test_cat["direction"] = [0]
overview_plot('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p005_m20_test_WFI01_', test_cat)"""


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


test_cat = table.Table()
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
plot_jumps('/Users/wschultz/Roman_Solar_System/romanisim-solarsys/scripts/mb_0p005_m20_test_WFI01_', test_cat)