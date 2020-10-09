

import multiprocessing
        


def multiproc_async(func, list_params, name, simul_jobs=4):
    '''
    Execute asynchronous multi-process of given function over given list of parameters

    Parameters
    ------------------
    func : function
        A function to execute multi-processing
    list_params : list

    '''
    if name == '__main__':
        jobs = []
        mp = multiprocessing.Pool(simul_jobs)
        for params in list_params:            
            jobs.append(mp.apply_async(func, params))        
        mp.close()
        mp.join()
        
        return [job.get() for job in jobs]
        