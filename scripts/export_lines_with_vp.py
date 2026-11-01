import os
import glob
import numpy as np
import scipy.io as sio


YUD_DIR = "../data/YorkUrbanDB"
OUTPUT_DIR = "../data/lines"

os.makedirs(OUTPUT_DIR, exist_ok=True)
mat_files = sorted(glob.glob(os.path.join(YUD_DIR, "*", "*LinesAndVP.mat")))
print(f"Found {len(mat_files)} annotation files.")

success = 0
failed = 0
for mat_path in mat_files:
    try:
        filename = os.path.basename(mat_path)
        img_id = filename.replace("LinesAndVP.mat", "")
        output_path = os.path.join(OUTPUT_DIR, f"{img_id}_yud_vp.txt")
    
        data = sio.loadmat(mat_path)
        lines = data["lines"]
        vp_association = data["vp_association"].flatten()
        line_segments = lines.reshape(-1, 2, 2)

        if len(line_segments) != len(vp_association):
            raise ValueError(
                f"{len(line_segments)} segments but "
                f"{len(vp_association)} VP associations"
            )

        output = []
        for segment, vp_id in zip(line_segments, vp_association):
            (x1, y1), (x2, y2) = segment
            output.append([x1, y1, x2, y2, int(vp_id)])

        np.savetxt(
            output_path,
            output,
            fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%d"],
            header="x1 y1 x2 y2 vp_id",
            comments=""
        )

        success += 1
        print(
            f"[{success + failed}/{len(mat_files)}] "
            f"{img_id}: {len(line_segments)} lines -> "
            f"{output_path}"
        )

    except Exception as e:
        failed += 1
        print(f"ERROR processing {mat_path}: {e}")

print("\nFinished.")
print(f"Successfully processed: {success}")
print(f"Failed: {failed}")
print(f"Output directory: {OUTPUT_DIR}")
