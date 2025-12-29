import csv
from .entities import Task

class WorkloadLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.all_tasks = []
        self.current_index = 0
        self.load_csv()

    def load_csv(self):
        """Reads the CSV file and converts rows into Task objects."""
        with open(self.filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create Task from CSV Row
                t = Task(
                    task_id=int(row['task_id']),
                    mips=int(row['cpu_rate']),   # Google calls it CPU Rate
                    ram=int(row['mem_usage']),
                    duration=int(row['duration'])
                )
                # We also store the 'arrival time' to simulate real flow
                t.arrival_time = int(row['timestamp'])
                self.all_tasks.append(t)
        
        print(f"Loaded {len(self.all_tasks)} tasks from trace file.")

    def get_tasks_for_time(self, current_time):
        """Returns tasks that 'arrive' at this specific second."""
        arriving_tasks = []
        
        # Check tasks starting from where we left off
        while self.current_index < len(self.all_tasks):
            task = self.all_tasks[self.current_index]
            
            if task.arrival_time <= current_time:
                arriving_tasks.append(task)
                self.current_index += 1
            else:
                # This task is for the future, stop checking
                break
                
        return arriving_tasks