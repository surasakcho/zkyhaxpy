

import multiprocessing


f=open(OUTPUT_LOG, "a+")
f.write('file_name,action,time\n')
f.close()  
        
        
if __name__ == '__main__':
    jobs = []
    mp = multiprocessing.Pool(SIMULTANEOUS_JOBS)
    for i in df_url.index:
        url,file_name = df_url.loc[i,['url','file_name']]           
        jobs.append(mp.apply_async(dl.download_landsat,(url,file_name,DEST_PATH,USER,PASSWORD,CHROME_DRIVER,OUTPUT_LOG,)))
    
    mp.close()
    mp.join()
    
    num=sum([job.get() for job in jobs])
    print('Downloaded {} pages'.format(num)) 