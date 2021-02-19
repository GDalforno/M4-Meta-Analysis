import sys
import warnings
import numpy as np
import pandas as pd

sys.path.append("../pymfe/")
from tsmfe import TSMFE

warnings.simplefilter("ignore")


data = {
    "Yearly":pd.read_csv("../Dataset/Train/Yearly-train.csv"),
    "Quarterly":pd.read_csv("../Dataset/Train/Quarterly-train.csv"),
    "Monthly":pd.read_csv("../Dataset/Train/Monthly-train.csv")
}

for period in ("Yearly", "Quarterly", "Monthly"):
    print(f"Extracting from {period} subset")
    print("-----------------------------")
    for group in ("landmarking", "general", "global-stat",
            "local-stat", "model-based", "info-theory", 'stat-tests', 
            "autocorr", "randomize", "freq-domain"):
        print(f"\tGroup : {group}")

        with open(f"./groups/{group}.txt", "r") as f:
            columns = f.read().split(",")

        meta_features = []
        extractor = TSMFE(groups=group, suppress_warnings=True)

        for i in range(data[period].shape[0]):
            print(f"\t\t{i}")
            try:
                ts = data[period].iloc[i, 1:].dropna().values
                extractor.fit(ts, suppress_warnings=True)
                mf = extractor.extract(suppress_warnings=True)
                meta_features.append(mf[1])
            except:
                meta_features.append(np.repeat(np.NaN, len(columns)))


        meta_features = pd.DataFrame(meta_features, columns=columns)
        meta_features.to_csv(f"./{period}/{group}.csv")

