from multiprocessing import Queue
import time
import sys
import math

import numpy
from simple_pid import PID
import torch

from learning.rl import RLModel, default_network_arch, softmax_policy

def agent_thread(done_flag, FLAGS, get_conn):
    db_conn = get_conn()
    cursor = db_conn.cursor()
    event_queue = Queue()

    #create logging table
    ddl = "create table if not exists logging (reading_id bigserial primary key, reading_time timestamp without time zone, reading_metadata varchar(4000), reading_name varchar(1000) not null, reading_value_str varchar(4000), reading_value_int bigint, reading_value_float float, reading_value_bool bool, reading_value_time timestamp without time zone)"
    cursor.execute(ddl)

    #pretend we just did an autovacuum
    initial_time = time.time()
    last_autovac_time = initial_time

    current_delay = FLAGS.initial_autovac_delay
    prev_delay = current_delay
    delay_adjustment_count = 0
    vacuum_count = 0

    #cursor.execute("alter system set autovacuum_naptime to %d" % current_delay)
    if FLAGS.control_autovac:
        cursor.execute("alter table %s set ("
                       "autovacuum_enabled = off,"
                       "autovacuum_vacuum_scale_factor = 0,"
                       "autovacuum_vacuum_insert_scale_factor = 0,"
                       "autovacuum_vacuum_threshold = 0,"
                       "autovacuum_vacuum_cost_delay = 0,"
                       "autovacuum_vacuum_cost_limit = 10000"
                       ")" % FLAGS.table_name)
    else:
        cursor.execute("alter table %s reset ("
                       "autovacuum_enabled,"
                       "autovacuum_vacuum_scale_factor,"
                       "autovacuum_vacuum_insert_scale_factor,"
                       "autovacuum_vacuum_threshold,"
                       "autovacuum_vacuum_cost_delay,"
                       "autovacuum_vacuum_cost_limit"
                       ")" % FLAGS.table_name)

    #cursor.execute("select from pg_reload_conf()")

    # Used to log to a file
    class StatStruct():
        def __init__(self, time, vacCount, autovacCount, livePct):
            self.time = time
            self.vacCount = vacCount
            self.autoVacCount = autovacCount
            self.livePct = livePct
    statSeq = []

    range_min = math.log(1/(5*60.0))
    range_max = math.log(1.0)
    #print("Range: ", range_min, range_max)
    pid = PID(Kp=0.5, Ki=0.5, Kd=2.0, setpoint=60.0, output_limits=(range_min, range_max), auto_mode=True)

    count = 0
    live_sum = 0.0
    dead_sum = 0.0
    free_sum = 0.0

    # State for RL model
    if FLAGS.use_learned_model:
        print("Loading model state from file...")
        model_state = torch.load(FLAGS.learned_model_file)
        model = RLModel(default_network_arch)

        model.load_state_dict(model_state['model_state_dict'])
        #model.load_state_dict(model_state.state_dict())

        rng = numpy.random.RandomState(0)
        live_pct_buffer = []
        num_read_deltapct_buffer = []
        num_read_tuples_buffer = []

    while not done_flag.value:
        now = time.time()

        #if FLAGS.enable_logging:
        #pgstattuples = log_tuples(event_queue, cursor,"select * from pgstattuple('purchases_index')",
        #                          ("table_len", "tuple_count", "tuple_len", "tuple_percent", "dead_tuple_count", "dead_tuple_len", "dead_tuple_percent", "free_space", "free_percent"))

        #log_tuples(event_queue, cursor,"select * from pg_stat_user_tables where relname = 'purchases_index'",
        #           ("relid", "schemaname", "relname", "seq_scan", "seq_tup_read", "idx_scan", "idx_tup_fetch",
        #            "n_tup_ins", "n_tup_upd", "n_tup_del", "n_tup_hot_upd", "n_live_tup", "n_dead_tup", "n_mod_since_analyze",
        #            "n_ins_since_vacuum", "last_vacuum", "last_autovacuum", "last_analyze",  "last_autoanalyze", "vacuum_count",
        #            "autovacuum_count", "analyze_count", "autoanalyze_count"))

        #log_tuples(event_queue, cursor, "select pg_visibility_map_summary('purchases_index')", ("pg_visibility_map_summary", ))

        #log_tuples(event_queue, cursor, "select * from pg_sys_cpu_usage_info()",
        #           ("usermode_normal_process_percent", "usermode_niced_process_percent",
        #            "kernelmode_process_percent", "idle_mode_percent", "io_completion_percent",
        #            "servicing_irq_percent", "servicing_softirq_percent", "user_time_percent",
        #            "processor_time_percent", "privileged_time_percent", "interrupt_time_percent"))

        #log_tuples(event_queue, cursor, "select * from pg_sys_memory_info()",
        #           ("total_memory", "used_memory", "free_memory", "swap_total",
        #            "swap_used", "swap_free", "cache_total", "kernel_total", "kernel_paged", "kernel_non_paged",
        #            "total_page_file", "avail_page_file"))

        #log_tuples(event_queue, cursor, "select * from pg_sys_load_avg_info()",
        #           ("load_avg_one_minute", "load_avg_five_minutes", "load_avg_ten_minutes", "load_avg_fifteen_minutes"))

        #log_tuples(event_queue, cursor, "select * from pg_sys_process_info()",
        #           ("total_processes", "running_processes", "sleeping_processes", "stopped_processes", "zombie_processes"))

        #t = pgstattuples[0]
        #live_pct = t[3]
        #dead_pct = t[6]
        #free_pct = t[8]

        cursor.execute("select pg_total_relation_size('public.purchases_index')")
        total_space = cursor.fetchall()[0][0]

        cursor.execute("select pg_table_size('public.purchases_index')")
        used_space = cursor.fetchall()[0][0]

        cursor.execute("select n_live_tup, n_dead_tup, seq_tup_read from pg_stat_user_tables where relname = '%s'" % FLAGS.table_name)
        stats = cursor.fetchall()[0]
        n_live_tup = stats[0]
        n_dead_tup = stats[1]
        seq_tup_read = stats[2]
        #print("Live tup: %d, Dead dup: %d, Seq reads: %d" % (n_live_tup, n_dead_tup, seq_tup_read))

        live_raw_pct = 0.0 if n_live_tup+n_dead_tup == 0 else n_live_tup/(n_live_tup+n_dead_tup)

        print("Total: %d, Used: %d, Live raw pct: %.2f" % (total_space, used_space, live_raw_pct))
        sys.stdout.flush()

        used_pct = used_space/total_space
        live_pct = 100*live_raw_pct*used_pct
        dead_pct = 100*(1.0-live_raw_pct)*used_pct
        free_pct = 100*(1.0-used_pct)

        count += 1
        live_sum += live_pct
        dead_sum += dead_pct
        free_sum += free_pct

        print("Live tuple %% (avg): %.2f, %.2f, Dead tuple %% (avg): %.2f, %.2f, Free space %% (avg): %.2f, %.2f"
              % (live_pct, live_sum / count, dead_pct, dead_sum / count, free_pct, free_sum / count))
        sys.stdout.flush()

        if FLAGS.use_learned_model:
            # generate state
            delta = 0.0 if len(num_read_tuples_buffer) == 0 else seq_tup_read - num_read_tuples_buffer[0]
            if delta < 0:
                delta = 0
            delta_pct = 0.0 if n_live_tup == 0 else delta / n_live_tup

            if len(live_pct_buffer) >= 10:
                live_pct_buffer.pop()
                num_read_deltapct_buffer.pop()
                num_read_tuples_buffer.pop()

            live_pct_buffer.insert(0, live_pct)
            num_read_deltapct_buffer.insert(0, delta_pct)
            num_read_tuples_buffer.insert(0, seq_tup_read)

            l1 = numpy.pad(live_pct_buffer, (0, 10-len(live_pct_buffer)), 'constant', constant_values=(0, 0))
            l2 = numpy.pad(num_read_deltapct_buffer, (0, 10-len(num_read_deltapct_buffer)), 'constant', constant_values=(0, 0))
            # Additional normalization.
            l1 = [(x/100.0)-0.5 for x in l1]
            l2 = [math.log2(x+0.0001) for x in l2]
            state = list(map(float, [*l1, *l2]))
            print("Generated state: ", [round(x, 1) for x in state])
            state = torch.tensor([state])

            # Select action
            action = int(softmax_policy(model, state, rng, default_network_arch['num_actions'], 0.01, False))
            if action == 0:
                # Do not vacuum
                print("Action 0: Idling.")
                current_delay = 5*60
            elif action == 1:
                # Do vacuum
                print("Action 1: Vacuuming...")
                current_delay = 1
            else:
                assert("Invalid action")
        elif FLAGS.enable_pid:
            pid_out = pid(live_pct)
            current_delay = int(math.ceil(1.0/math.exp(pid_out)))
            print("PID output %f, current_delay %d" % (pid_out, current_delay))
            sys.stdout.flush()

        if prev_delay != current_delay:
            prev_delay = current_delay
            delay_adjustment_count += 1
            #print("alter system set autovacuum_naptime to %d" % current_delay)
            #if FLAGS.control_autovac:
                #cursor.execute("alter system set autovacuum_naptime to %d" % current_delay)
                #cursor.execute("select from pg_reload_conf()")

        if FLAGS.control_autovac:
            if int(now-last_autovac_time) >= current_delay:
                last_autovac_time = now
                print("Vacuuming table...")
                sys.stdout.flush()
                cursor.execute("vacuum %s" % FLAGS.table_name)
                vacuum_count += 1

        cursor.execute("select vacuum_count, autovacuum_count from pg_stat_user_tables where relname = '%s'" % FLAGS.table_name)
        internal_vac_count, internal_autovac_count = cursor.fetchall()[0]

        print("%10s ===================> Time %.2f: Vac: %d, Internal vac: %d, Internal autovac: %d" %
              (FLAGS.tag, now-initial_time, vacuum_count, internal_vac_count, internal_autovac_count))
        sys.stdout.flush()
        statSeq.append(StatStruct(now-initial_time, internal_vac_count, internal_autovac_count, live_pct))

        time.sleep(1)

    print("Delay adjustments: ", delay_adjustment_count)
    print("Live tuple: %.2f, Dead tuple: %.2f, Free space: %.2f" % (live_sum / count, dead_sum / count, free_sum / count))
    sys.stdout.flush()

    with open(FLAGS.tag+'_actions.txt', 'w') as f:
        for entry in statSeq:
            f.write("%.2f %d %d %.2f\n" % (entry.time, entry.vacCount, entry.autoVacCount, entry.livePct))
