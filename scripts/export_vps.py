
import os
import glob
import numpy as np
import scipy.io as sio

YUD_DIR = "../data/YorkUrbanDB"
OUTPUT_2D = "../data/vps2d"
OUTPUT_3D = "../data/vps3d"

os.makedirs(OUTPUT_2D, exist_ok=True)
os.makedirs(OUTPUT_3D, exist_ok=True)

camera = sio.loadmat(os.path.join(YUD_DIR, "cameraParameters.mat"))

f = float(camera["focal"][0, 0])
ps = float(camera["pixelSize"][0, 0])
pp = camera["pp"][0, :]

K = np.array([
    [f / ps, 0, pp[0]],
    [0, f / ps, pp[1]],
    [0, 0, 1]
], dtype=np.float64)

mat_files = sorted(
    glob.glob(os.path.join(YUD_DIR, "*", "*GroundTruthVP_CamParams.mat"))
)

print(f"Found {len(mat_files)} VP annotation files.")

success = 0
failed = 0

for mat_path in mat_files:
    try:
        image_id = os.path.basename(os.path.dirname(mat_path))

        data = sio.loadmat(mat_path)

        vds = np.asarray(data["vp"], dtype=np.float64).copy()

        vds[1, :] *= -1
        vps_h = K @ vds

        w = vps_h[2, :]
        if np.any(np.abs(w) < 1e-12):
            raise ValueError("A VP has a near-zero homogeneous coordinate.")

        vps2d = (vps_h[:2, :] / w[None, :]).T

        vds_rows = vds.T
        norms = np.linalg.norm(vds_rows, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("A 3D VP direction has near-zero norm.")
        vds_rows = vds_rows / norms

        np.savetxt(
            os.path.join(OUTPUT_2D, image_id + ".txt"),
            vps2d,
            fmt="%.10f",
            header="x y",
            comments=""
        )

        np.savetxt(
            os.path.join(OUTPUT_3D, image_id + ".txt"),
            vds_rows,
            fmt="%.12f",
            header="vx vy vz",
            comments=""
        )

        success += 1
        print(f"{image_id}: exported {len(vds_rows)} VPs")

    except Exception as e:
        failed += 1
        print(f"ERROR: {mat_path}: {e}")

print(f"\nFinished. Success: {success}, failed: {failed}")
