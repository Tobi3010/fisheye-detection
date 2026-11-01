import pandas as pd

df = pd.read_csv("vp_detection_per_image.txt", sep="\t")
rows = []

for focal in sorted(df["focal_length"].unique(), reverse=True):
    subset = df[df["focal_length"] == focal]
    all_lines = subset[subset["configuration"] == "all_lines"][["image", "recall"]]
    toward = subset[subset["configuration"] == "toward"][["image", "recall"]]

    merged = pd.merge(all_lines, toward, on="image", suffixes=("_all", "_toward"))
    delta = (merged["recall_toward"] - merged["recall_all"])

    rows.append({
        "focal": focal,
        "images": len(merged),
        "toward_better": int((delta > 0).sum()),
        "toward_worse": int((delta < 0).sum()),
        "same": int((delta == 0).sum())
    })

results = pd.DataFrame(rows)
results.to_csv("toward_vs_all_summary.txt", sep="\t", index=False)
print(results)