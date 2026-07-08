import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pathlib


SEQUENCES_DIR = pathlib.Path("/home/hashim/Desktop/Datasets/KITTI/data_odometry_color/dataset/sequences")
GT_POSES_DIR = pathlib.Path("/home/hashim/Desktop/Datasets/KITTI/data_odometry_poses/dataset/poses")


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Inverse of a (4, 4) rigid transform, using R^T instead of a general inverse."""
    R = T[:3, :3]
    t = T[:3, 3:4]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.transpose(-1, -2)
    T_inv[:3, 3:4] = -R.transpose(-1, -2) @ t
    return T_inv
 
 
def translation(t: torch.Tensor) -> torch.Tensor:
    """Pure-translation (4, 4) transform from a length-3 vector."""
    T = torch.eye(4, dtype=t.dtype)
    T[:3, 3] = t
    return T
 

"""
KITTICalibration: turn KITTI projection matrices P2/P3 into explicit, named
intrinsics and frame transforms. Deliberately boring and literal.

Frames (T_A_B maps a point from frame B into frame A: p_A = T_A_B @ p_B):

    c0 : rectified reference camera. P0 has zero translation, so c0 is the origin
         of the projection matrices AND the frame KITTI odometry poses live in.
    c2 : left  colour camera (image_2). Depth and images live here.
    c3 : right colour camera (image_3).

After rectification all KITTI cameras share orientation, so c0<->c2 and c0<->c3
are pure x-translations. The depth you backproject with K2 is in c2, but the
poses are in c0 -- so you must convert c2 <-> c0 explicitly.
"""


class KITTICalibration:

    def __init__(self, P2_rect: torch.Tensor, P3_rect: torch.Tensor) -> None:
        self.P2_rect = torch.as_tensor(P2_rect, dtype=torch.float64)
        self.P3_rect = torch.as_tensor(P3_rect, dtype=torch.float64)

        # Intrinsics: P = K [I | t] after rectification, so the left 3x3 block is K.
        self.K2 = self.P2_rect[:3, :3].clone()
        self.K3 = self.P3_rect[:3, :3].clone()

        # Camera centre in c0 coords: C = -M^{-1} p4  (general; reduces to
        # [-P[0,3]/fx, 0, 0] for rectified KITTI where p4 = [P[0,3], 0, 0]).
        self.C_c2_in_c0 = self._camera_centre(self.P2_rect)
        self.C_c3_in_c0 = self._camera_centre(self.P3_rect)

        # p_c0 = p_c2 + C_c2_in_c0  =>  T_c0_c2 is translation by C_c2_in_c0.
        self.T_c0_c2 = translation(self.C_c2_in_c0)
        self.T_c2_c0 = se3_inverse(self.T_c0_c2)
        self.T_c0_c3 = translation(self.C_c3_in_c0)
        self.T_c3_c0 = se3_inverse(self.T_c0_c3)

        # Stereo baseline for image_2/image_3 depth: distance between the two centres.
        self.baseline_c2_c3 = float(torch.abs(self.C_c3_in_c0[0] - self.C_c2_in_c0[0]))

    @staticmethod
    def _camera_centre(P: torch.Tensor) -> torch.Tensor:
        M = P[:3, :3]
        p4 = P[:3, 3]
        return -torch.linalg.solve(M, p4)

    def sanity_check(self) -> None:
        """Cheap invariants. Raises if the calibration is internally inconsistent."""
        assert torch.allclose(self.T_c2_c0 @ self.T_c0_c2, torch.eye(4, dtype=torch.float64), atol=1e-9)
        print("K2 =\n", self.K2.numpy())
        print("C_c2_in_c0 =", self.C_c2_in_c0.tolist())
        print("C_c3_in_c0 =", self.C_c3_in_c0.tolist())
        print(f"baseline_c2_c3 = {self.baseline_c2_c3:.6f} m  (expect ~0.54 for KITTI)")



"""
KITTI Sequence Object:

- times:             (N,)      timestamps t0..t_{N-1}
- left_images:       list[str] paths to image_2 frames  (frame c2)
- right_images:      list[str] paths to image_3 frames  (frame c3)
- depths:            list[str] paths to depth maps       (in c2, FoundationStereo)
- gt_poses:          (N, 4, 4) T_w_c0[i]  -- poses of the REFERENCE camera c0, not c2
- P2_rect:           (3, 4)    projection matrix for image_2 (left colour)
- P3_rect:           (3, 4)    projection matrix for image_3 (right colour)
- calib:             KITTICalibration -- K2/K3, camera centres, c0<->c2/c3 transforms, baseline

Frame convention (see geometry.py): T_A_B maps a point from frame B into frame A.
KEY FACT: images/depth live in c2; poses live in c0. Convert c2 <-> c0 explicitly.

"""


class KITTISequence:

    def __init__(self, sequence_nr: int) -> None:
        self.sequence_nr: int = sequence_nr
        self.sequence_folder = SEQUENCES_DIR / f"{sequence_nr:02d}"

        self.times: np.ndarray = self._load_times()
        self.left_images: list[str] = self._load_left_images()
        self.right_images: list[str] = [p.replace("image_2", "image_3") for p in self.left_images]
        self.depths: list[str] = [p.replace("image_2", "depth") for p in self.left_images]

        self.gt_poses: torch.Tensor = self._load_gt_poses()        # T_w_c0[i]

        # Raw projection matrices, then everything derived lives on the calib object.
        self.P2_rect, self.P3_rect = self._load_calibration()
        self.calib = KITTICalibration(self.P2_rect, self.P3_rect)

    def load_image(self, i: int) -> np.ndarray:
        """Load the left colour image at frame i as a numpy array (H, W, 3) in [0, 1]."""
        path = self.left_images[i]
        img = plt.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return img
    
    def load_depth(self, i: int) -> np.ndarray:
        """Load the depth map at frame i as a numpy array (H, W) in metres."""
        path = self.depths[i]
        if path.endswith(".npy"):
            depth = np.load(path)
        else:
            # Matplotlib normalizes 16-bit PNGs to [0, 1], losing the metric scale.
            # PIL preserves the stored uint16 values so KITTI-style depth / 256 works.
            depth = np.asarray(Image.open(path))

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 256.0
        return depth.astype(np.float32, copy=False)

    def _load_times(self) -> np.ndarray:
        times_path = self.sequence_folder / "times.txt"
        # check if the file exists
        if not times_path.exists():
            # get the number of frames from the image_2 folder
            image_dir = self.sequence_folder / "image_2"
            num_frames = len(list(image_dir.glob("*.png")))
            self.number_frames = num_frames
            self.duration = num_frames * 0.1  # assuming 10 Hz
            return np.arange(0, num_frames * 0.1, 0.1)  # default to 10 frames at 10 Hz
        with open(times_path, "r") as f:
            times = [float(line) for line in f if line.strip()]
        self.number_frames = len(times)
        self.duration = times[-1] - times[0]
        return np.array(times, dtype=np.float64)

    def _load_calibration(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the raw P2 (left colour) and P3 (right colour) projection matrices."""
        calib_path = self.sequence_folder / "calib.txt"
        P2 = P3 = None
        with open(calib_path, "r") as f:
            for line in f:
                if line.startswith("P2:"):
                    P2 = torch.tensor(list(map(float, line.split()[1:])), dtype=torch.float64).reshape(3, 4)
                elif line.startswith("P3:"):
                    P3 = torch.tensor(list(map(float, line.split()[1:])), dtype=torch.float64).reshape(3, 4)
        if P2 is None or P3 is None:
            raise ValueError(f"Missing P2/P3 in {calib_path}")
        return P2, P3

    def _load_left_images(self) -> list[str]:
        image_dir = self.sequence_folder / "image_2"
        return [str(image_dir / f"{i:06d}.png") for i in range(self.number_frames)]

    def _load_gt_poses(self) -> torch.Tensor:
        """Load KITTI poses file -> (N, 4, 4) SE(3) tensor T_w_c0[i] in float64."""
        poses_path = GT_POSES_DIR / f"{self.sequence_nr:02d}.txt"

        # if the poses file does not exist, return identity poses
        if not poses_path.exists():
            print(f"Warning: poses file {poses_path} does not exist. Using identity poses.")
            return torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(self.number_frames, 1, 1)
        
        rows = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, 3, 4)   # (N, 3, 4)

        T = np.tile(np.eye(4, dtype=np.float64), (rows.shape[0], 1, 1))     # (N, 4, 4)
        T[:, :3, :4] = rows

        if rows.shape[0] != self.number_frames:
            raise ValueError(
                f"Pose count ({rows.shape[0]}) does not match frame count ({self.number_frames})"
            )
        return torch.from_numpy(T)

    def __len__(self) -> int:
        return self.number_frames

    # --- explicit, named pose access ----------------------------------------

    def pose_cam0_to_world(self, i: int) -> torch.Tensor:
        """T_w_c0[i]: maps a point in the reference camera at frame i into world."""
        return self.gt_poses[i]

    def cam2_point_to_world(self, p_c2: torch.Tensor, i: int) -> torch.Tensor:
        """p_w = T_w_c0[i] @ T_c0_c2 @ p_c2. p_c2 homogeneous, shape (..., 4)."""
        T_w_c2 = self.gt_poses[i] @ self.calib.T_c0_c2
        return (T_w_c2 @ p_c2.unsqueeze(-1)).squeeze(-1)

    def relative_left_color_pose(self, i: int, j: int) -> torch.Tensor:
        """
        T_c2j_c2i: maps a point in the image_2 frame at time i into the image_2
        frame at time j. This is the transform to warp image_2 from i to j.
        """
        c = self.calib
        T_w_c0_i = self.gt_poses[i]
        T_w_c0_j = self.gt_poses[j]
        return c.T_c2_c0 @ se3_inverse(T_w_c0_j) @ T_w_c0_i @ c.T_c0_c2

    # --- exports -------------------------------------------------------------

    def write_foundationstereo_intrinsics(self) -> str:
        """
        Write FoundationStereo intrinsic file.

        Format expected by scripts/run_demo.py:
        line 1: flattened 3x3 K matrix (K2, left colour)
        line 2: stereo baseline in metres (|C_c3 - C_c2|)
        """
        K = self.calib.K2.detach().cpu().numpy()
        baseline = self.calib.baseline_c2_c3

        out_path = self.sequence_folder / "K.txt"
        with open(out_path, "w") as f:
            f.write(" ".join(map(str, K.reshape(-1))) + "\n")
            f.write(str(baseline) + "\n")

        print(f"Wrote FoundationStereo intrinsics: {out_path}")
        print("K2:\n", K)
        print(f"baseline: {baseline:.6f} m")
        return str(out_path)


def main():
    seq_10 = KITTISequence(10)
    print(f"Sequence 10 has {len(seq_10)} frames, duration {seq_10.duration:.2f} seconds.")
    seq_10.calib.sanity_check()

    # Identity relative transform must be the identity.
    rel = seq_10.relative_left_color_pose(5, 5)
    print("relative_left_color_pose(5, 5) ~ I:",
          torch.allclose(rel, torch.eye(4, dtype=torch.float64), atol=1e-9))

    seq_10.write_foundationstereo_intrinsics()
    # write out the intrinsics for all sequences
    for seq_nr in range(22):
        seq = KITTISequence(seq_nr)
        seq.write_foundationstereo_intrinsics()

    # test loading an image and depth
    # seq_10 = KITTISequence(0)
    # img = seq_10.load_image(0)
    # depth = seq_10.load_depth(0)
    # print(f"Loaded image shape: {img.shape}, depth shape: {depth.shape}")

    # # plot the image and depth
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.title("Left Color Image (image_2)")
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # plt.imshow(depth, cmap="plasma")
    # plt.title("Depth Map")
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    main()
