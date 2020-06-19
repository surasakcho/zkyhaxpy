import os
import psutil
import time

def wait_mem(max_mem_pct_used=60, refresh_mins=5, timeout_mins=180):

    mem_pct_used = psutil.virtual_memory()[2]
    waiting_mins = 0
    #print(f'current memory usage : {mem_pct_used}%')
    while mem_pct_used > max_mem_pct_used:     
        print(f'Current memory usage : {mem_pct_used}%... wait for {refresh_mins} mins...')   
        time.sleep(60*refresh_mins)
        waiting_mins = waiting_mins + (refresh_mins)
        mem_pct_used = psutil.virtual_memory()[2]
            
        if mem_pct_used <= max_mem_pct_used:
            print('Continue...')
        assert(waiting_mins < timeout_mins)
