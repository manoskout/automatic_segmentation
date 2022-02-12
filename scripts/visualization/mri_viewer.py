# %matplotlib widget
import matplotlib.pyplot as plt

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(cvt_s_slice.pixel_array, 'gray', interpolation='none')
# plt.subplot(1,2,2)
# plt.imshow(cvt_s_slice.pixel_array, 'gray', interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.5)
# plt.show()


def multi_slice_viewer(mask_volume, mri_volume,title_mask_volume, title_mri_volume, no_axis=False):
    remove_keymap_conflicts({'j', 'k'})
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
    
    ax1.volume = mri_volume
    ax1.index = len(mri_volume) // 2
    
    ax2.volume = mask_volume
    ax2.index = len(mask_volume) // 2

    ax3.index = ax2.index
    ax3.volume = combined_volume = [(i * 0.001) + (m * 0.7) for i, m in zip(mri_volume, mask_volume)]
    
    ax1.imshow(mri_volume[ax1.index], cmap="gray", interpolation="none")
    ax1.set_title(title_mri_volume)
    ax2.imshow(mask_volume[ax1.index],cmap="jet", interpolation="none")
    ax2.set_title(title_mask_volume)
    ax3.imshow(combined_volume[ax3.index], cmap="gray", interpolation="none")
    ax3.set_title("Combined")

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()
            
def process_key(event):
    # Process key_press events
    fig = event.canvas.figure
    ax1 = fig.axes[0]
    ax2 = fig.axes[1]
    ax3 = fig.axes[2]

    if event.key == 'j':
        previous_slice(ax1)
        previous_slice(ax2)
        previous_slice(ax3)

    elif event.key == 'k':
        next_slice(ax1) 
        next_slice(ax2) 
        next_slice(ax3) 

    fig.canvas.draw()

def previous_slice(ax):
    # Go to the previous slice
    volume = ax.volume
    ax.index = (ax.index-1) % len(volume)
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    # Go to the next slice
    volume = ax.volume
    ax.index = (ax.index+1) % len(volume)
    ax.images[0].set_array(volume[ax.index])