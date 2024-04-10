import math
import time

from learning.environment import BaseEnvironment

class AutoVacEnv(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """

        self.experiment_id = 0
        self.env_info = env_info
        self.stat_and_vac = env_info['stat_and_vac']

        self.state_history_length = env_info['state_history_length']

    def update_stats(self):
        total_space, used_space = self.stat_and_vac.getTotalAndUsedSpace()

        stats = self.stat_and_vac.getTupleStats()
        n_live_tup = stats[0]
        n_dead_tup = stats[1]
        seq_tup_read = stats[2]
        print("Live tup: %d, Dead dup: %d, Seq reads: %d" % (n_live_tup, n_dead_tup, seq_tup_read))

        live_raw_pct = 1.0 if n_live_tup+n_dead_tup == 0 else n_live_tup/(n_live_tup+n_dead_tup)

        used_pct = 1.0 if total_space == 0 else used_space/total_space
        live_pct = live_raw_pct*used_pct

        print("Total: %d, Used: %d, Live raw pct: %.2f, Live pct: %.2f"
              % (total_space, used_space, 100.0*live_raw_pct, 100.0*live_pct))

        delta = max(0, seq_tup_read - self.num_read_tuples_buffer[0])
        delta_pct = 0.0 if n_live_tup == 0 else delta / n_live_tup

        self.num_live_tuples_buffer.pop()
        self.num_live_tuples_buffer.insert(0, n_live_tup)
        self.num_dead_tuples_buffer.pop()
        self.num_dead_tuples_buffer.insert(0, n_dead_tup)
        self.num_read_tuples_buffer.pop()
        self.num_read_tuples_buffer.insert(0, seq_tup_read)
        self.live_pct_buffer.pop()
        self.live_pct_buffer.insert(0, live_pct)
        self.num_read_deltapct_buffer.pop()
        self.num_read_deltapct_buffer.insert(0, delta_pct)
        self.num_read_delta_buffer.pop()
        self.num_read_delta_buffer.insert(0, delta)

    def generate_state(self):
        # Normalize raw readings before constructing the environment state.
        l1 = [x-0.5 for x in self.live_pct_buffer]
        l2 = [5.0 if x == 0.0 else math.log2(x) for x in self.num_read_deltapct_buffer]

        result = list(map(float, [*l1, *l2]))
        print("Generated state: ", [round(x, 1) for x in result])
        return result

    def generate_reward(self, did_vacuum):
        last_live_tup = self.num_live_tuples_buffer[0]
        live_pct = self.live_pct_buffer[0]
        last_read = self.num_read_delta_buffer[0]
        print("Last live tup: %d, Last live %%: %.2f, Last_read: %d"
              % (last_live_tup, 100.0*live_pct, last_read))

        # -1 unit of reward equivalent to scanning the entire table (live + dead tuples).
        # The reward is intended to be scale free.
        reward = 0
        if last_read == 0 or last_live_tup == 0:
            # Initial reward
            reward = 50
        else:
            reward = (last_read/last_live_tup)*live_pct
            if live_pct < 0.5:
                # Large penalty for dirty table
                reward -= 200*(0.5-live_pct)

        if did_vacuum:
            # Assume vacuuming is proportionally more expensive than scanning the table once.
            reward -= 75

        print("Returning reward: %.2f" % reward)
        return reward

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        print("Starting agent...")

        # Readings we have obtained for the past several seconds.
        # To start the experiment, pad with some initial values.
        self.num_live_tuples_buffer = [0.0 for _ in range(self.state_history_length)]
        self.num_dead_tuples_buffer = [0.0 for _ in range(self.state_history_length)]
        self.num_read_tuples_buffer = [0.0 for _ in range(self.state_history_length)]
        self.num_read_delta_buffer = [0.0 for _ in range(self.state_history_length)]

        # Those two buffers are used to generate the environment state.
        self.live_pct_buffer = [1.0 for _ in range(self.state_history_length)]
        self.num_read_deltapct_buffer = [0.0 for _ in range(self.state_history_length)]

        self.stat_and_vac.startExp(self.env_info)

        self.delay_adjustment_count = 0
        self.initial_time = time.time()

        self.update_stats()
        initial_state = self.generate_state()

        is_terminal = self.stat_and_vac.step()
        # TODO handle case where we terminate right on the setup.
        assert(not is_terminal)

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

        # effect system based on action
        did_vacuum = False
        if action == 0:
            # Not vacuuming
            print("Action 0: Idling.")
        elif action == 1:
            # Vacuuming
            print("Action 1: Vacuuming...")
            did_vacuum = True
            self.delay_adjustment_count += 1
            # Apply vacuum before collecting new state
            self.stat_and_vac.doVacuum()
        else:
            assert("Invalid action")

        self.update_stats()
        current_state = self.generate_state()

        is_terminal = self.stat_and_vac.step()
        if is_terminal:
            print("Terminating.")
            stats = self.stat_and_vac.getTupleStats()
            print("Time elapsed: %.2f, delay adjustments: %d, internal vac: %d, internal autovac: %d"
                  % ((time.time()-self.initial_time), self.delay_adjustment_count, stats[3], stats[4]))

        # Compute reward after applying the action.
        reward = self.generate_reward(did_vacuum)

        self.reward_obs_term = (reward, current_state, is_terminal)
        return self.reward_obs_term
