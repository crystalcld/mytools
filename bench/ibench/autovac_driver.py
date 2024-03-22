import sys
from learning.autovac_rl import AutoVacEnv
from learning.rl_glue import RLGlue
from learning.rl import Agent, default_network_arch

# command line arguments
max_episodes = None
instance_password = None
instance_url = None
instance_user = None
instance_dbname = None
def learn(resume_id):
    agent_configs = {
        'network_arch': default_network_arch,

        'batch_size': 8,
        'buffer_size': 50000,
        'gamma': 0.99,
        'learning_rate': 1e-4,
        'tau':0.01 ,
        'seed':0,
        'num_replay_updates': 5

    }

    environment_configs = {
        'module_name': 'pggrill', # 'pggrill' or 'iibench_driver'
        'function_name': 'run_with_default_settings',
        'initial_size': 1000_000,
        'update_speed': 32_000, # for iibench only
        'update_speed_range': [100, 100_000], # for pggrill only
        'initial_delay': 5,
        'db_name': instance_dbname,
        'db_host': instance_url,
        'db_user': instance_user,
        'db_pwd': instance_password,
        'num_cols_range': [0, 0],
        'num_indexes_range': [0, 0],
        'num_partitions_range': [0, 0],
        'updated_percentage_range': [1, 50],
        'num_workers_range': [1, 50],
        'table_name_fn': 'get_bench_table_name',
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
    max_episodes = int(sys.argv[1])
    instance_url = sys.argv[2]
    instance_user = sys.argv[3]
    instance_password = sys.argv[4]
    instance_dbname = sys.argv[5]

    resume_id = 1
    print("Initial id: ", resume_id)
    sys.stdout.flush()

    #benchmark(resume_id)
    learn(resume_id)
