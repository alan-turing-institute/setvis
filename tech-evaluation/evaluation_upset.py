import yaml
import csv
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from utils import generate_pattern, eval_data

# load config file
config_yaml = open("config.yaml")
config = yaml.load(config_yaml, Loader=yaml.FullLoader)
# extract parameters for evaluation
num_rows = config["dataset"]["num_rows"]
num_cols = config["dataset"]["num_cols"]
patterns = config["patterns"]
filename = config["output"]["filename"]
package = "upset"
output_file = f"{filename}_{package}.csv"


# prepare output file
with open(output_file, "w", newline="\n") as csvfile:
    try:

        w = csv.writer(csvfile, delimiter=",")
        w.writerow(
            [
                "Package",
                "Pattern",
                "Num_rows",
                "Num_cols",
                "Stage",
                "Tims (s)",
                "RAM",
            ]
        )

        # run evaluation
        for pattern in patterns:
            for row in num_rows:
                for col in num_cols:
                    # step 1: generate data
                    df = generate_pattern(pattern, row, col)
                    # step 2: evaluate data
                    results = eval_data(df, package, pattern, row, col)
                    # step 3: write result to file
                    w.writerows(results)

    except:
        raise
