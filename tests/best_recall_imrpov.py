import pandas as pd

FOCAL = 100
TOP_K = 6

df = pd.read_csv(
    "vp_detection_per_image.txt",
    sep="\t"
)

df = df[df["focal_length"] == FOCAL]

all_lines_df = (
    df[df["configuration"] == "all_lines"]
    [["image", "recall"]]
    .rename(columns={"recall": "all_lines_recall"})
)

toward_df = (
    df[df["configuration"] == "toward"]
    [["image", "recall"]]
    .rename(columns={"recall": "toward_recall"})
)

merged = pd.merge(
    all_lines_df,
    toward_df,
    on="image",
    how="inner"
)

merged["recall_improvement"] = (
    merged["toward_recall"]
    - merged["all_lines_recall"]
)

best = merged.sort_values(
    "recall_improvement",
    ascending=False
).head(TOP_K)

print(best[
    [
        "image",
        "all_lines_recall",
        "toward_recall",
        "recall_improvement"
    ]
].to_string(index=False))