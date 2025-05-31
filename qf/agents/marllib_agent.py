from qf.agents.agent import Agent


class MARLLibAgent(Agent):
    def __init__(self, env, model):
        """
        Initializes the MARLLibAgent with the given environment and configuration.
        Parameters:
            env: The environment in which the agents will operate.
            config (dict): Configuration dictionary for the MARLLib agent.
        """
        super().__init__(env)
        self.model = model


    def train(self, total_iterations=100):
        """
        Trains the MARLLib agent for a specified number of iterations.
        Parameters:
            total_iterations (int): Number of training iterations.
        """
        self.model.fit(env, model, stop={'timesteps_total': total_iterations}, share_policy='group')

    def evaluate(self, eval_env, episodes=10):
        """
        Evaluates the MARLLib agent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
        Returns:
            float: The average reward over the evaluation episodes.
        """
        eval_env.set_environment_mode(self.set_env_mode())

        total_rewards = []
        for episode in range(episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = {}
                for agent_id in obs.keys():
                    actions[agent_id] = self.trainer.compute_action(obs[agent_id], policy_id=f"agent_{agent_id}")
                obs, rewards, dones, infos = eval_env.step(actions)
                episode_reward += sum(rewards.values())
                done = all(dones.values())

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward

    def save(self, path):
        """
        Saves the MARLLib agent's model to a file.
        Parameters:
            path (str): Path to save the model.
        """
        self.trainer.save(path)

    def load(self, path):
        """
        Loads the MARLLib agent's model from a file.
        Parameters:
            path (str): Path to load the model from.
        """
        self.trainer.restore(path)