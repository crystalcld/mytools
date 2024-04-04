from workloads.iibench_driver import collectExperimentParams

class SimulatedVacuum:
    def startExp(self, env_info):
        self.env_info = env_info
        collectExperimentParams(self.env_info)
        self.initial_size = self.env_info['initial_size']
        self.update_speed = self.env_info['update_speed']
        self.num_cols = self.env_info['num_cols']
        self.num_indexes = self.env_info['num_indexes']
        self.num_partitions = self.env_info['num_partitions']
        self.table_name = self.env_info['table_name']

        print("Environment info (for SimulatedVacuum):")
        for x in self.env_info:
            print ('\t', x, ':', self.env_info[x])

        self.env_info['experiment_id'] += 1

        self.approx_bytes_per_tuple = 100
        self.used_space = 0
        self.total_space = 0

        self.n_live_tup = self.initial_size
        self.n_dead_tup = 0
        self.seq_tup_read = 0
        self.vacuum_count = 0

        self.step_count = 0
        self.max_steps = env_info['max_seconds']

    def step(self):
        self.n_dead_tup += self.update_speed

        sum = self.n_live_tup+self.n_dead_tup
        if sum > 0:
            # Weigh how many tuples we read per second by how many dead tuples we have.
            self.seq_tup_read += 15*3*self.n_live_tup*((self.n_live_tup/sum) ** 0.5)

        self.used_space = self.approx_bytes_per_tuple*sum
        if self.used_space > self.total_space:
            self.total_space = self.used_space

        self.step_count += 1
        return self.step_count >= self.max_steps

    def getTotalAndUsedSpace(self):
        return self.total_space, self.used_space

    def getTupleStats(self):
        return self.n_live_tup, self.n_dead_tup, self.seq_tup_read, self.vacuum_count, 0

    def doVacuum(self):
        self.n_dead_tup = 0
        self.vacuum_count += 1
