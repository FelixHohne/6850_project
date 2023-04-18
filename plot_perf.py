import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import sys


if __name__ == "__main__":
    result_dfs = []
    dataset_name = sys.argv[1]
    print(dataset_name)
    for filename in sys.argv[2:]:
        result_dfs.append(pd.read_csv(filename))
    print(len(result_dfs))

    for i, df in enumerate(result_dfs):
        sns.lineplot(x=df['epoch'], y=df['test'], label=sys.argv[i + 2].split(".")[0])
    plt.title(f"Test performance for {dataset_name} across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.savefig("plot_perf.png", dpi=600, bbox_inches="tight")
        

    

