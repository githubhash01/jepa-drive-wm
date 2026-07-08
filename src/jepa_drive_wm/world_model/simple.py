from jepa_drive_wm.data.kitti import KITTISequence
from jepa_drive_wm.utils.vjepa_wrapper import VJEPA21Wrapper, VJEPA21Size
from jepa_drive_wm.utils.visualiser import Visualiser, FVVisualizer
import matplotlib.pyplot as plt

# visualise some kitti
seq = KITTISequence(0)
# grab the first few frames and plot them 

images = [seq.load_image(i) for i in range(5)]

# plot the images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axs):
    ax.imshow(images[i])
    ax.axis('off')  
plt.show()

# Per-frame features (image mode): one (Hp*Wp, D) grid per frame.
vjepa_wrapper = VJEPA21Wrapper(size=VJEPA21Size.BASE)

# Joint clip features (video mode).
clip_len = 4
clip_tokens = vjepa_wrapper.encode_clip(seq.left_images[:clip_len])
print("clip tokens:", tuple(clip_tokens.shape))
print("as grid:", tuple(vjepa_wrapper.unflatten(clip_tokens, clip_len).shape))

# Per-frame features (image mode): one (Hp*Wp, D) grid per frame.
per_frame = vjepa_wrapper.encode_images(seq.left_images[:1])
print("per-frame tokens:", tuple(per_frame.shape))

# Use the visualiser to plot the features using PCA 
fv_vis = FVVisualizer(vjepa_wrapper)

fv_vis.display_vjepa_embedding_pca(features=per_frame[0])