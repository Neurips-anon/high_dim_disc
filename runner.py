import numpy as np
from SEMScore import *
from fges import *
import time
import sys

import pickle
import argparse

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 0)

def main(dataset="test.tmp",save_name="fges_results.tmp", sparsity=5):
    #args = parser.parse_args()
    dataset = load_file(dataset)
    score = SEMBicScore(sparsity, dataset=dataset) # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score,
                filename=dataset,
                checkpoint_frequency=False,
                save_name=save_name)
    start_time = time.time()
    result = fges.search()

    print("--- %s seconds ---" % (time.time() - start_time))
    with open(save_name + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
