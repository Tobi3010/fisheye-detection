import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vp_detection_results.txt", sep="\t")

for metric in ["precision", "recall", "AA_5deg", "mean_angular_error"]:

    plt.figure(figsize=(9, 5))

    for configuration in df["configuration"].unique():
        subset = df[df["configuration"] == configuration]

        plt.plot(
            subset["focal_length"],
            subset[metric],
            marker="o",
            label=configuration
        )

    plt.xlabel("Focal length (pixels)")
    plt.ylabel(metric)
    plt.title(f"{metric} across different distortion levels")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300)
    plt.show()