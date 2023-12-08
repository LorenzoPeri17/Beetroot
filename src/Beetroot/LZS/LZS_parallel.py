from LZS_Signal import get_LZS_Signal
import numpy as np
import multiprocessing as mp

mp.freeze_support()


def get_LZS_map_MW(N : int, eps : np.ndarray, 
                de_MW:np.ndarray, de_rf:float,
                Gamma:float, kt:float, 
                omega_rf:float, omega_MW:float,
                toll:float = 1e-4) -> np.ndarray:

    Map = []

    args = [(N, eps, de0, de_rf, Gamma, kt, omega_rf, omega_MW, toll) for de0 in de_MW]

    ncores = mp.cpu_count()

    with mp.Pool(processes=ncores) as pool:
        for res in pool.starmap(get_LZS_Signal, args, chunksize= 1): #len(args)//ncores):

            Map.append(res)
        
        return np.array(Map)
    
def get_LZS_map_rf(N : int, eps : np.ndarray, 
                de_MW:float, de_rf:np.ndarray,
                Gamma:float, kt:float, 
                omega_rf:float, omega_MW:float,
                toll:float = 1e-4) -> np.ndarray:

    Map = []

    args = [(N, eps, de_MW, de0, Gamma, kt, omega_rf, omega_MW, toll) for de0 in de_rf]

    ncores = mp.cpu_count()

    with mp.Pool(processes=ncores) as pool:
        for res in pool.starmap(get_LZS_Signal, args, chunksize= 1): #len(args)//ncores):

            Map.append(res)
        
        return np.array(Map)
