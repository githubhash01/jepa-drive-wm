# Approach to Dense Decoding


## Data Interface: 

- We have 21 KITTI sequences, from the KITTI odometry dataset
- We generate pseudolabels for Depth using FoundationStereo, and Semantic Segmentation using OneFormer

- Our datasets therefore must be split up by sequence


## Decoder Architecture

- We tried using a linear probe on ViTB VJEPA2.1 embeddings, even with a more complex convolutional upscaling, the results were quite poor. 
- So we just feed in 4 identical VJEPA2.1 final features into the DINOV3 DPT architecture, and use the utils for converting features to depth
- We use the same architecture for semantic segmentation, just with differnet losses 