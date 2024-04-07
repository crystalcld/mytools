from workloads.iibench_driver import collectExperimentParams
from executors.vacuum_experiment import VacuumExperiment

class SimulatedVacuum(VacuumExperiment):
    def startExp(self, env_info):
        self.env_info = env_info
        self.initial_size, self.update_speed = collectExperimentParams(self.env_info)

        #print("Environment info (for SimulatedVacuum):")
        #for x in self.env_info:
        #    print ('\t', x, ':', self.env_info[x])

        self.env_info['experiment_id'] += 1

        self.approx_bytes_per_tuple = env_info["approx_bytes_per_tuple"]
        self.used_space = 0
        self.total_space = 0

        self.n_live_tup = self.initial_size
        self.n_dead_tup = 0
        self.seq_tup_read = 0
        self.vacuum_count = 0

        self.step_count = 0
        self.max_steps = env_info['max_seconds']

    def updateUsedSpace(self):
        self.used_space = self.approx_bytes_per_tuple*(self.n_live_tup+self.n_dead_tup)
        if self.used_space > self.total_space:
            self.total_space = self.used_space

    def step(self):
        self.n_dead_tup += self.update_speed
        self.updateUsedSpace()

        if self.n_live_tup > 0:
            # Weigh how many tuples we read per second by how many dead tuples we have.
            self.seq_tup_read += 15*3*self.n_live_tup*((self.n_live_tup/(self.n_live_tup+self.n_dead_tup)) ** 0.5)
            #self.seq_tup_read += 15*3*self.n_live_tup

        self.step_count += 1
        return self.step_count > self.max_steps

    def getTotalAndUsedSpace(self):
        return self.total_space, self.used_space

    def getTupleStats(self):
        return self.n_live_tup, self.n_dead_tup, self.seq_tup_read, self.vacuum_count, 0

    def doVacuum(self):
        self.vacuum_count += 1
        self.n_dead_tup = 0
        self.updateUsedSpace()

        # Reclaim some unused space
        self.total_space = 0.9*self.total_space + 0.1*self.used_space
