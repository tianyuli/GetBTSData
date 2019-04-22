# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:06:26 2019

@author: TL
"""

import pandas as pd
import math

INPUT_PATH = "output.csv"
OUTPUT_PATH = "sampled.csv"
CHUNK_SIZE = 1000000
SAMPLE_RATE = 0.1
RANDOM_STATE = 1

def main():
    n = 1
    random_state = RANDOM_STATE
    
    for chunk in pd.read_csv(INPUT_PATH, index_col = 0, chunksize = CHUNK_SIZE):
        count = len(chunk.index);
        sample_size = math.floor(count / 10)
        sample = chunk.sample(frac = SAMPLE_RATE, random_state = random_state)
       
        print(f"round {n}, {sample_size} out of {count} rows are sampled")
       
        sample.to_csv(OUTPUT_PATH, mode = "a")
        n = n + 1
        random_state = random_state + 1
main()