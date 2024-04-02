import os
import sys
import math

from iibench import apply_options, run_main, run_benchmark

from tqdm.auto import tqdm

def run_with_params(apply_options_only, tag, db_host, db_user, db_pwd, db_name, initial_size, update_speed,
                    initial_delay, max_seconds, control_autovac, enable_pid, enable_learning, rl_model_filename, enable_agent):
    cmd = ["--setup", "--dbms=postgres", "--tag=%s" % tag,
           "--db_user=%s" % db_user, "--db_name=%s" % db_name,
           "--db_host=%s" % db_host, "--db_password=%s" % db_pwd,
           "--max_rows=100000000", "--secs_per_report=120",
           "--query_threads=3", "--delete_per_insert", "--max_seconds=%d" % max_seconds, "--rows_per_commit=10000",
           "--initial_size=%d" % initial_size,
           "--inserts_per_second=%d" % update_speed,
           "--initial_autovac_delay=%d" % initial_delay
           ]
    if control_autovac:
        cmd.append("--control_autovac")
    if enable_pid:
        cmd.append("--enable_pid")
    if enable_learning:
        cmd.append("--enable_learning")

    used_learned_model = len(rl_model_filename) > 0
    if used_learned_model:
        cmd.append("--use_learned_model")
        cmd.append("--learned_model_file=%s" % rl_model_filename)
    if enable_agent:
        cmd.append("--enable_agent")

    print("Running command: ", cmd)
    sys.stdout.flush()

    apply_options(cmd)
    if not apply_options_only:
        if run_main() != 0:
            print("Error. Quitting driver.")
            sys.stdout.flush()
            sys.exit()

    if not enable_learning:
        # Collect and sort query latencies into a single file
        os.system("cat %s_dataQuery_thread_#* | sort -nr > %s_latencies.txt" % (tag, tag))

def benchmark(resume_id):
    id = 0
    for initial_size in tqdm([10000, 100000, 1000000]):
        for update_speed in tqdm([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]):
            id += 1
            if id < resume_id:
                continue
            print("Running experiment %d" % id)
            sys.stdout.flush()

            tag_suffix = "_n%d_size%d_updates%d" % (id, initial_size, update_speed)
            tag1 = "tag_model1%s" % tag_suffix
            tag2 = "tag_model2%s" % tag_suffix
            tag3 = "tag_pid%s" % tag_suffix
            tag4 = "tag_vanilla%s" % tag_suffix

            # Control with RL model #1
            run_with_params(False, tag1, instance_url, instance_user, instance_password, instance_dbname,
                            initial_size, update_speed, 5, experiment_duration, True, False, False,
                            model1_filename, True)

            # Control with RL model #1
            run_with_params(False, tag2, instance_url, instance_user, instance_password, instance_dbname,
                            initial_size, update_speed, 5, experiment_duration, True, False, False,
                            model2_filename, True)

            # Control with PID
            run_with_params(False, tag3, instance_url, instance_user, instance_password, instance_dbname,
                            initial_size, update_speed, 5, experiment_duration, True, True, False,
                            "", True)

            # Control with default autovacuum
            run_with_params(False, tag4, instance_url, instance_user, instance_password, instance_dbname,
                            initial_size, update_speed, 5, experiment_duration, False, False, False,
                            "", True)

            gnuplot_cmd = ("gnuplot -e \"outfile='graph%s.png'; titlestr='Query latency graph (%s)'; filename1='%s_latencies.txt'; filename2='%s_latencies.txt'; filename3='%s_latencies.txt'; filename4='%s_latencies.txt'\" gnuplot_script.txt"
                           % (tag_suffix, tag_suffix, tag1, tag2, tag3, tag4))
            print("Gnuplot command: ", gnuplot_cmd)
            os.system(gnuplot_cmd)

def getParamsFromExperimentId(experiment_id):
    # Vary update speed from 1000 to 128000
    update_speed = math.ceil(1000.0*math.pow(2, experiment_id % 8))
    # Vary initial size from 10^4 to 10^6
    initial_size = math.ceil(math.pow(10, 4 + (experiment_id // 8) % 3))

    return initial_size, update_speed

def run_with_default_settings(barrier, env_info):
    experiment_id = env_info['experiment_id']
    initial_size, update_speed = getParamsFromExperimentId(experiment_id)

    run_with_params(True, "rl_model",
                    env_info['db_host'], env_info['db_user'], env_info['db_pwd'], env_info['db_name'],
                    initial_size, update_speed, env_info['initial_delay'], env_info['max_seconds'],
                    True, False, True, "", False)
    run_benchmark(barrier)
