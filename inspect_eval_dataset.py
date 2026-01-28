import numpy as np

def inspect_eval_dataset(path="eval_dataset.npz"):
    data = np.load(path)

    print("=== Keys ===")
    for k in data.files:
        print(f"  {k}")

    print("\n=== Shapes ===")
    for k in data.files:
        arr = data[k]
        print(f"{k:12s}: shape={arr.shape}, dtype={arr.dtype}")

    # Optional: quick semantic checks
    pos = data["pos"]
    normal = data["normal"]
    dirs = data["dirs"]
    radiance = data["radiance_gt"]

    print("\n=== Sanity checks ===")
    print("Positions min/max:", pos.min(axis=0), pos.max(axis=0))
    print("Normals norm (mean):", np.linalg.norm(normal, axis=1).mean())
    print("Dirs norm (mean):", np.linalg.norm(dirs, axis=1).mean())
    print("Radiance stats: min / mean / max =",
          radiance.min(), radiance.mean(), radiance.max())


if __name__ == "__main__":
    inspect_eval_dataset("eval_dataset.npz")
