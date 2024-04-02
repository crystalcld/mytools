import sys
import time

from learning.autovac_rl import AutoVacEnv
from learning.rl_glue import RLGlue
from learning.rl import Agent, default_network_arch
import psycopg2

from iibench_driver import run_with_default_settings, getParamsFromExperimentId, benchmark

from multiprocessing import Barrier, Process

class PGStatAndVacuum:
    def startExp(self, env_info):
        self.env_info = env_info
        self.db_name = env_info['db_name']
        self.db_host = env_info['db_host']
        self.db_user = env_info['db_user']
        self.db_pwd = env_info['db_pwd']
        self.table_name = env_info['table_name']

        barrier = Barrier(2)
        self.workload_thread = Process(target=run_with_default_settings, args=(barrier, self.env_info))
        self.workload_thread.start()
        # We wait until the workload is initialized and ready to start
        barrier.wait()

        self.conn = psycopg2.connect(dbname=self.db_name, host=self.db_host, user=self.db_user, password=self.db_pwd)
        self.conn.set_session(autocommit=True)
        self.cursor = self.conn.cursor()

        print("Disabling autovacuum...")
        self.cursor.execute("alter table %s set ("
                            "autovacuum_enabled = off,"
                            "autovacuum_vacuum_scale_factor = 0,"
                            "autovacuum_vacuum_insert_scale_factor = 0,"
                            "autovacuum_vacuum_threshold = 0,"
                            "autovacuum_vacuum_cost_delay = 0,"
                            "autovacuum_vacuum_cost_limit = 10000"
                            ")" % self.table_name)

        self.env_info['experiment_id'] += 1

    # Returns True if the run has finished
    def step(self):
        if not self.workload_thread.is_alive():
            return True

        time.sleep(1)
        return False

    def getTotalAndUsedSpace(self):
        try :
            self.cursor.execute("select pg_total_relation_size('public.%s')" % self.table_name)
            total_space = self.cursor.fetchall()[0][0]

            self.cursor.execute("select pg_table_size('public.%s')" % self.table_name)
            used_space = self.cursor.fetchall()[0][0]

            return total_space, used_space
        except psycopg2.errors.UndefinedTable:
            print("Table does not exist.")
            return 0, 0

    def getTupleStats(self):
        self.cursor.execute("select n_live_tup, n_dead_tup, seq_tup_read, vacuum_count, autovacuum_count from pg_stat_user_tables where relname = '%s'" % self.table_name)
        return self.cursor.fetchall()[0]

    def doVacuum(self):
        self.cursor.execute("vacuum %s" % self.table_name)

class SimulatedVacuum:
    def startExp(self, env_info):
        self.env_info = env_info
        self.initial_size, self.update_speed = getParamsFromExperimentId(self.env_info['experiment_id'])
        self.env_info['experiment_id'] += 1
        #print("Params: ", self.initial_size, self.update_speed)

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

def learn(resume_id):
    agent_configs = {
        'network_arch': default_network_arch,

        'batch_size': 8,
        'buffer_size': 50000,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'tau': 0.01 ,
        'seed': 0,
        'num_replay_updates': 5,
        'model_filename' : model1_filename
    }

    if model_type == "simulated":
        instance = SimulatedVacuum()
    elif model_type == "real":
        instance = PGStatAndVacuum()
    else:
        assert("Invalid model type")

    environment_configs = {
        'stat_and_vac': instance,
        'experiment_id': 0,
        'db_name': instance_dbname,
        'db_host': instance_url,
        'db_user': instance_user,
        'db_pwd': instance_password,
        'table_name': 'purchases_index',
        'initial_delay': 5,
        'max_seconds': experiment_duration
    }

    experiment_configs = {
        'num_runs': 1,
        'num_episodes': max_episodes,
        'timeout': 1000
    }

    ### Instantiate the RLGlue class
    rl_glue = RLGlue(AutoVacEnv, Agent)
    rl_glue.do_learn(environment_configs, experiment_configs, agent_configs)

if __name__ == '__main__':
    cmd = sys.argv[1]

    global max_episodes
    max_episodes = int(sys.argv[2])
    global resume_id
    resume_id = int(sys.argv[3])
    global experiment_duration
    experiment_duration = int(sys.argv[4])
    global model_type
    model_type = sys.argv[5]
    global model1_filename
    model1_filename = sys.argv[6]
    global model2_filename
    model2_filename = sys.argv[7]
    global instance_url
    instance_url = sys.argv[8]
    global instance_user
    instance_user = sys.argv[9]
    global instance_password
    instance_password = sys.argv[10]
    global instance_dbname
    instance_dbname = sys.argv[11]

    if cmd == "benchmark":
        benchmark(resume_id)
    elif cmd == "learn":
        learn(resume_id)
    else:
        print("Invalid command")
