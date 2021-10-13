import datetime

class timer:
  
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.lap_times = []
        self.elapsed_time = None

    def start(self):
        self.start_time = datetime.datetime.now()
        print(f'Start at: {self.start_time}')

    def end(self):
        self.end_time = datetime.datetime.now()        
        print(f'End at: {self.end_time}')
        self.elapsed_time = self.end_time - self.start_time
        print()
        self.show()

    def lap(self):
        curr_lap_time = datetime.datetime.now()
        if len(self.lap_times) > 0:
            prev_lap_time = self.lap_times[-1]
        else:
            prev_lap_time = self.start_time
        latest_lap_elapsed_time = curr_lap_time - prev_lap_time
        self.lap_times.append(curr_lap_time)
        
        print(f'Lap {len(self.lap_times)} time at: {curr_lap_time}')
        print(f'Lap {len(self.lap_times)} time: {latest_lap_elapsed_time}')

    def show(self):
        print('###############################')
        print(f'Timer: {self.name}')          
        if len(self.lap_times) > 0:
            for i, lap_time in enumerate(self.lap_times):
                
                if i == 0:
                    curr_lap_elapsed_time = lap_time - self.start_time
                else:
                    curr_lap_elapsed_time = lap_time - prev_lap_time
                prev_lap_time = lap_time                
                [print(f'Lap {i+1} elapsed time: {curr_lap_elapsed_time}')]
            print(f'Lap {i+2} elapsed time: {self.end_time - prev_lap_time}')
            
        print(f'Total elapsed time: {self.elapsed_time}')
        print('###############################')

timer_1 = timer('Test')
