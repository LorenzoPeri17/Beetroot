import multiprocessing as mp
import numpy as np
from Beetroot.signal import get_Signal, get_Signal_Loss

mp.freeze_support()

# ~ For best performace, we should use a chunksize >1
# ~ however, we need to make sure to split the load equally
# ~ between the processes
# ~ Cmputational time becomes MUCH LARGER for larger de

# def _reorder_arguments(args, ncores):
#     reodered_args = []
#     nargs = len(args)
#     j = 0
#     crossovers = 0
#     for i in range(nargs):
#         new_index = j * ncores + crossovers
        
#         if new_index >= nargs:
#             crossovers += 1
#             j=0
#             new_index = crossovers

#         reodered_args.append( args[j * ncores + crossovers])
#         j+=1

#     return reodered_args

# def _reorder_map(Map, de):
#     return np.array([Map[i] for i in np.argsort(de)])

# ! Here for backwards compatibility
# ! please use get_map instead
def get_map_single(N : int, eps : np.ndarray, de:np.ndarray, 
                Gamma:float, omega:float, kt:float, 
                toll:float = 1e-4) -> np.ndarray:
    return get_map(N, eps, de, Gamma, omega, kt, toll)

def get_map(N : int, eps : np.ndarray, de:np.ndarray, 
                Gamma:float, omega:float, kt:float, 
                toll:float = 1e-4) -> np.ndarray:

    Map = []

    args = [(N, eps, de0, Gamma, omega, kt, toll) for de0 in de]

    ncores = mp.cpu_count()

    with mp.Pool(processes=ncores) as pool:
        for res in pool.starmap(get_Signal, args, chunksize= 1): #len(args)//ncores):

            Map.append(res)
        
        return np.array(Map)

def get_map_loss(N : int, eps : np.ndarray, de:np.ndarray, 
                Gamma:float, omega:float, kt:float, g :float, kappa:float, 
                toll:float = 1e-4, maxiter_MIB :int = 30) -> np.ndarray:

    Map = []

    args = [(N, eps, de0, Gamma, omega, kt, g, kappa, toll, maxiter_MIB) for de0 in de]
    ncores = mp.cpu_count()

    with mp.Pool(processes=ncores) as pool:
        for res in pool.starmap(get_Signal_Loss, args, chunksize= 1): #len(args)//ncores):

            Map.append(res)
        
        return np.array(Map)
    