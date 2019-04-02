import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from utilities import open_json

file_name = "MULTI AGENT - PROPER A2C.csv"
folder_path = "results"
partial_df = open_json("{}\{}".format(folder_path, file_name))
print(partial_df)