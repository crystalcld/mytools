import time
import math
from multiprocessing import Barrier, Process

import numpy as np

from learning.environment import BaseEnvironment

class AutoVacEnv(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """

        self.experiment_id = 0
        self.env_info = env_info
        self.workload_fn = env_info['workload_fn']
        self.workload_thread = None

        self.stat_and_vac = env_info['stat_and_vac']

        self.num_live_tuples_buffer = []
        self.num_dead_tuples_buffer = []
        self.num_read_tuples_buffer = []

        self.live_pct_buffer = []
        self.num_read_deltapct_buffer = []
        self.num_read_delta_buffer = []

    def update_stats(self):
        total_space, used_space = self.stat_and_vac.getTotalAndUsedSpace()

        stats = self.stat_and_vac.getTupleStats()
        n_live_tup = stats[0]
        n_dead_tup = stats[1]
        seq_tup_read = stats[2]
        print("Live tup: %d, Dead dup: %d, Seq reads: %d" % (n_live_tup, n_dead_tup, seq_tup_read))

        live_raw_pct = 0.0 if n_live_tup+n_dead_tup == 0 else n_live_tup/(n_live_tup+n_dead_tup)

        used_pct = used_space/total_space
        live_pct = 100*live_raw_pct*used_pct
        dead_pct = 100*(1.0-live_raw_pct)*used_pct
        free_pct = 100*(1.0-used_pct)

        print("Total: %d, Used: %d, Live raw pct: %.2f, Live pct: %.2f"
              % (total_space, used_space, live_raw_pct, live_pct))

        delta = 0.0 if len(self.num_read_tuples_buffer) == 0 else seq_tup_read - self.num_read_tuples_buffer[0]
        if delta < 0:
            delta = 0
        delta_pct = 0.0 if n_live_tup == 0 else delta / n_live_tup

        if len(self.num_live_tuples_buffer) >= 10:
            self.num_live_tuples_buffer.pop()
            self.num_dead_tuples_buffer.pop()
            self.num_read_tuples_buffer.pop()
            self.live_pct_buffer.pop()
            self.num_read_deltapct_buffer.pop()
            self.num_read_delta_buffer.pop()

        self.num_live_tuples_buffer.insert(0, n_live_tup)
        self.num_dead_tuples_buffer.insert(0, n_dead_tup)
        self.num_read_tuples_buffer.insert(0, seq_tup_read)
        self.live_pct_buffer.insert(0, live_pct)
        self.num_read_deltapct_buffer.insert(0, delta_pct)
        self.num_read_delta_buffer.insert(0, delta)

    def generate_state(self):
        l1 = np.pad(self.live_pct_buffer, (0, 10-len(self.live_pct_buffer)), 'constant', constant_values=(0, 0))
        l2 = np.pad(self.num_read_deltapct_buffer, (0, 10-len(self.num_read_deltapct_buffer)), 'constant', constant_values=(0, 0))

        # Additional normalization.
        l1 = [(x/100.0)-0.5 for x in l1]
        l2 = [math.log2(x+0.0001) for x in l2]

        result = list(map(float, [*l1, *l2]))
        print("Generated state: ", [round(x, 1) for x in result])
        return result

    def generate_reward(self, did_vacuum):
        last_live_tup = self.num_live_tuples_buffer[0]
        live_pct = self.live_pct_buffer[0]
        last_read = self.num_read_delta_buffer[0]
        print("Last live tup: %d, Last live %%: %.2f, Last_read: %d"
              % (last_live_tup, live_pct, last_read))

        # -1 unit of reward equivalent to scanning the entire table (live + dead tuples).
        # The reward is intended to be scale free.
        reward = 0
        if last_live_tup > 0:
            reward = (last_read/last_live_tup)*live_pct/100.0
            if live_pct < 50.0:
                # Large penalty for dirty table
                reward -= 2*(50.0-live_pct)

        if did_vacuum:
            # Assume vacuuming is proportionally more expensive than scanning the table once.
            reward -= 100

        print("Returning reward:", reward)
        return reward

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        self.env_info['experiment_id'] = self.experiment_id
        self.experiment_id += 1

        barrier = Barrier(2)
        self.workload_thread = Process(target=self.workload_fn, args=(barrier, self.env_info))
        self.workload_thread.start()
        # We wait until the workload is initialized and ready to start
        barrier.wait()
        print("Starting agent...")

        self.stat_and_vac.startExp()

        self.time_started = time.time()

        self.last_autovac_time = time.time()
        self.delay_adjustment_count = 0

        self.update_stats()
        initial_state = self.generate_state()

        is_terminal = not self.workload_thread.is_alive()
        reward = self.generate_reward(False)
        self.reward_obs_term = (reward, initial_state, is_terminal)
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        time.sleep(1)

        # effect system based on action
        did_vacuum = False
        if action == 0:
            # Not vacuuming
            print("Action 0: Idling.")
            pass
        elif action == 1:
            # Vacuuming
            did_vacuum = True
            self.delay_adjustment_count += 1
            print("Action 1: Vacuuming...")
        else:
            assert("Invalid action")

        self.update_stats()
        current_state = self.generate_state()

        is_terminal = not self.workload_thread.is_alive()
        if is_terminal:
            print("Terminating.")
            stats = self.stat_and_vac.getTupleStats()
            print("Delay adjustments: %d, Internal vac: %d, Internal autovac: %d"
                  % (self.delay_adjustment_count, stats[3], stats[4]))

        # compute reward before doing the vacuum (will clear dead tuples)
        reward = self.generate_reward(did_vacuum)
        if did_vacuum:
            self.stat_and_vac.doVacuum()

        self.reward_obs_term = (reward, current_state, is_terminal)
        return self.reward_obs_term
