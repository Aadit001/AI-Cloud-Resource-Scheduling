from .config import *

class Task:
    def __init__(self, task_id, mips, ram, duration):
        self.id = task_id
        self.mips = mips
        self.ram = ram
        self.duration = duration
        self.active = True

    def tick(self):
        self.duration -= 1
        if self.duration <= 0:
            self.active = False
            return False # Task Finished
        return True # Task still running

class Host:
    def __init__(self, host_id):
        self.id = host_id
        self.total_mips = HOST_MIPS
        self.total_ram = HOST_RAM
        self.used_mips = 0
        self.used_ram = 0
        self.active_tasks = []
        self.is_sleep = True # Start in sleep mode

    def get_utilization(self):
        if self.total_mips == 0: return 0
        return self.used_mips / self.total_mips

    def get_power(self):
        # Implementation of Eq 3.2 from your Thesis
        if self.is_sleep and self.used_mips == 0:
            return 0 # Sleep Mode
        
        util = self.get_utilization()
        # Linear Model: k*Pmax + (1-k)*Pmax*u
        return POWER_IDLE + (POWER_MAX - POWER_IDLE) * util

    def allocate(self, task):
        # Can it fit?
        if (self.used_mips + task.mips <= self.total_mips) and \
           (self.used_ram + task.ram <= self.total_ram):
            
            self.used_mips += task.mips
            self.used_ram += task.ram
            self.active_tasks.append(task)
            self.is_sleep = False # Wake up!
            return True
        return False

    def update_state(self):
        # Remove finished tasks
        active_tasks = []
        for t in self.active_tasks:
            if t.tick():
                active_tasks.append(t)
            else:
                # Release resources
                self.used_mips -= t.mips
                self.used_ram -= t.ram
        
        self.active_tasks = active_tasks
        
        # Go to sleep if empty
        if self.used_mips == 0:
            self.is_sleep = True