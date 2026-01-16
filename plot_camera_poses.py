import os, json
import numpy as np
import matplotlib.pyplot as plt
from preprocess import RESULT_DIR

def main():
    extr_path = os.path.join(RESULT_DIR, "all-extrinsic.json")
    if not os.path.exists(extr_path):
        raise FileNotFoundError(f"not found: {extr_path} (先运行: python sfm.py --dataset mini-temple [--ba])")

    with open(extr_path, "r") as f:
        all_extr = json.load(f)

    ids = list(all_extr.keys())
    centers = []
    for image_id in ids:
        extr = np.array(all_extr[image_id], dtype=float)  # 3x4
        R = extr[:, :3]
        t = extr[:, 3]
        C = -R.T @ t
        centers.append(C)
    centers = np.stack(centers, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:,0], centers[:,1], centers[:,2], "-o", markersize=2)
    ax.set_title("Camera Centers (from all-extrinsic.json)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    out_path = os.path.join(RESULT_DIR, "camera_poses.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("saved:", out_path)

if __name__ == "__main__":
    main()