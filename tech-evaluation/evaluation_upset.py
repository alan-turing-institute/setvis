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
num_int = config["dataset"]["intersections"]  # number of intersections
data_type = config["dataset"]["datatype"]
seed = config["seed"][0]
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
                "Num_intersections",
                "Stage",
                "Tims (s)",
                "RAM",
            ]
        )

        # run evaluation
        for pattern in patterns:
            for type in data_type:
                for inter in num_int:
                    for row in num_rows:
                        for col in num_cols:
                            # step 1: generate data
                            df = generate_pattern(
                                pattern, row, col, inter, type, seed
                            )
                            # step 2: evaluate data
                            results = eval_data(
                                df, package, pattern, row, col, inter, type,
                            )
                            # step 3: write result to file
                            w.writerows(results)

    except:
        raise
