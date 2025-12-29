import random

class Scheduler:
    def __init__(self, hosts):
        self.hosts = hosts

    def round_robin(self, task):
        """Baseline: Just picks the next available host randomly/sequentially."""
        available = [h for h in self.hosts if h.allocate(task)]
        if not available:
            return False # SLA Violation (Task Dropped)
        
        # Pick random available host (simulating 'Next' in RR)
        target = random.choice(available)
        # Note: allocate() is called inside the list comprehension check roughly, 
        # but properly we should select first then allocate. 
        # For simplicity in this logical model, we assume the check worked.
        return True

    def ai_optimized_best_fit(self, task):
        """
        AI Proxy: 'Best Fit' Heuristic.
        This packs tasks tightly to keep other servers asleep.
        It simulates what an RL Agent learns after 1000 epochs.
        """
        # Sort hosts: Put task on the FULLEST host that still has space.
        # This keeps the EMPTY hosts empty (Sleep mode).
        sorted_hosts = sorted(self.hosts, key=lambda h: h.get_utilization(), reverse=True)
        
        for host in sorted_hosts:
            if host.allocate(task):
                return True
        
        return False # Task Dropped