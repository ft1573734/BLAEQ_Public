from re import I
import BLAEQ
import numpy as np
import cupy as cp
import pickle
import os
import time
from tqdm import tqdm
import random
import pathlib


def main():


    '''=== Configuration ==='''
    ALG = BLAEQ.BLAEQ()
    # Whether to execute BLAEQ on GPU or CPU. If ON_GPU = False, BLAEQ will be executed on CPU.
    ALG.ON_GPU = True
    
    # The name of the data set, including suboff_xxx, yupeng_xxx, kvlcc2_xxx.
    # All the data sets are stored in Python pickle file format for efficient loading.
    case = "suboff_small"
    
    # Manually set K here. If not set i.e. K = -1, BLAEQ will use the automatically selected K.
    K = -1
    
    # A small variable for eliminating 0s within BLAEQ, since 0 value is sensitive in sparse linear algebra.
    epsilon = 0.001
    
    # Whether one needs to print the detailed query result i.e. the ids of each node.
    PRINT_QUERY_RESULTS = False


    '''=== Loading Data ==='''
    
    DataDir = "Data\\"
    print("Loading case "+ case)
    
    pickle_file = open(DataDir + case,'rb')
    Parameters = pickle.load(pickle_file)
    pickle_file.close()
    mesh = np.row_stack([Parameters.X,Parameters.Y,Parameters.Z])

    
    # The ultimate objective is to store an ndarray into the BLAEQ.Original Mesh. One can implement their own loading functions to process data of various types.
    # See the descrpiton of BLAEQ.Load_Mesh() for detail.
    ALG.Load_Mesh(mesh)
    print("Mesh size = "+str(len(ALG.OriginalMesh[0])))
    # Automatic choice of K. 
    if K == -1:
        K = np.sqrt(2*ALG.N)


    ''' === Generating Queries === '''
    #Query Generator
    sub_queries = []
    print('K = ' + str(K))
    #subquery generator:
    for d in range(0, ALG.D):
        [min_d, max_d] = ALG.MIN_MAX[d]
        one_third_range = (max_d - min_d)/3
        top_third = [min_d, min_d + one_third_range]
        mid_third = [min_d + one_third_range, max_d - one_third_range]
        bot_third = [max_d - one_third_range, max_d]
        sub_queries.append([top_third, mid_third, bot_third])

    Qs = []

    # Generating Range Queries
    for first_component in sub_queries[0]:
        #first_component = sub_queries[0][i]
        for second_component in sub_queries[1]:
            #second_component = sub_queries[1][j]
            for third_component in sub_queries[2]:
                #third_component = sub_queries[2][k]
                Qs.append([first_component,second_component,third_component])
                

    ''' === Experimenting ==='''
    print("=== Experimenting BLAEQ Multigrid ===")
    start = time.time()
    ALG.Multi_Layer_Multigrid_Generator(K)
    # perform a pseudo query to warm up the GPU
    # ALG.Multi_Layer_Multigrid_Query(0,100,0,100,0,100)
    end = time.time()
    print("Multigrid Generation Time: "+ str(end-start))

    counter = 0
    Q_times = []
    if ALG.ON_GPU:
        Qs = cp.asarray(Qs)
    for Q in Qs:
        counter += 1
        # Query on each dimension
        start = time.time()
        result_BLAEQ = ALG.Multi_Layer_Multigrid_Query(Q[0][0],Q[0][1],Q[1][0],Q[1][1],Q[2][0],Q[2][1])
        end = time.time()

        result_count = cp.sum(result_BLAEQ)
        print("Query "+str(counter)+" completed, time: "+ str(end-start)+ ", discovered points count = "+str(result_count))
        
        if PRINT_QUERY_RESULTS:
            print("Results Point IDs: \n" )
            print(cp.arange(0,ALG.N)[cp.nonzero(result_BLAEQ)])
        Q_times.append(end-start)
            

if __name__ =="__main__":
    main()
    

