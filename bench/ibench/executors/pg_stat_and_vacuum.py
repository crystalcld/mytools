import time
import psycopg2
from workloads.iibench_driver import collectExperimentParams, run_with_default_settings
from multiprocessing import Barrier, Process
from executors.vacuum_experiment import VacuumExperiment

class PGStatAndVacuum(VacuumExperiment):
    def startExp(self, env_info):
        self.env_info = env_info
        self.db_name = env_info['db_name']
        self.db_host = env_info['db_host']
        self.db_user = env_info['db_user']
        self.db_pwd = env_info['db_pwd']

        params = collectExperimentParams(self.env_info)
        self.initial_size = params['initial_size']
        self.update_speed = params['update_speed']
        self.table_name = params['table_name']

        print("Environment info (for PGStatAndVacuum):")
        for x in self.env_info:
            print ('\t', x, ':', self.env_info[x])
        for x in params:
            print ('\t', x, ':', params[x])

        # Connect to Postgres
        conn = psycopg2.connect(dbname=self.db_name, host=self.db_host, user=self.db_user, password=self.db_pwd)
        conn.set_session(autocommit=True)
        self.cursor = conn.cursor()
        print("Resetting stats...")
        self.cursor.execute("SELECT pg_stat_reset()")

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
