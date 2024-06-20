import typing
import numpy as np
import gymnasium
import sklearn.cluster as skc


def eigenvector_centrality(adj_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(adj_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    largest_eigenvector = eigenvectors[:, max_eigenvalue_index]
    norm_factor = np.linalg.norm(largest_eigenvector)
    normalized_eigenvector = largest_eigenvector / norm_factor
    return normalized_eigenvector


def max_centrality_difference(adj_matrix, centrality_scores):
    max_differences = []

    for node in range(len(adj_matrix)):
        # Find neighbors of the current node
        neighbors = np.nonzero(adj_matrix[node])[0]

        # Calculate the difference between the current node's centrality score and its neighbors' centrality scores
        differences = np.abs(centrality_scores[node] - centrality_scores[neighbors])

        # Update the maximum difference for the current node
        if len(differences) > 0:
            max_difference = np.max(differences)
            max_differences.append(max_difference)
        else:
            max_differences.append(0)

    return max_differences


def top_n_indices(input_list, n):
    # Create a copy of the input list to avoid modifying the original list
    input_copy = input_list.copy()

    # Initialize a boolean list with False values
    bool_list = [False] * len(input_list)

    # Sort the copy of the input list in descending order
    sorted_indices = sorted(range(len(input_copy)), key=lambda i: input_copy[i], reverse=True)

    # Set the first n elements in the boolean list to True
    for i in sorted_indices[:n]:
        bool_list[i] = True

    return bool_list


class TlppoLearner(gymnasium.Env):
    def __init__(self,
                 environment_name: str,
                 on_episode_end: typing.Callable[["TlppoLearner"], None] = lambda _: None,
                 tlppo_dim: typing.Optional[np.ndarray] = None,
                 **kwargs):
        self.environment = gymnasium.make(environment_name, **kwargs)
        self.observation_space = self.environment.observation_space
        if tlppo_dim is None:
            self.tlppo_dim = np.ones(self.observation_space.shape)
        else:
            if self.observation_space.shape != tlppo_dim.shape:
                raise ValueError(f"tlppo_dim must be of same shape than {environment_name} observation space. Received: {tlppo_dim.shape} Expected: {self.observation_space.shape}")
            self.tlppo_dim = tlppo_dim
        self.action_space = self.environment.action_space
        self.action_sequence: typing.List[np.ndarray] = []
        self.observation_sequence: typing.List[typing.Tuple[float, ...]] = []
        self.strategies: typing.List[typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]], float] = []
        self.lppo_states: typing.List[np.ndarray] = []
        self.on_episode_end = on_episode_end
        self.start_states = []
        self.end_states = []

    def reset(self,
              options={},
              seed=None):
        obs, info = self.environment.reset(options=options, seed=seed)
        self.action_sequence = []
        self.observation_sequence = [tuple(obs[self.tlppo_dim])]
        self.start_states.append(tuple(obs[self.tlppo_dim]))
        return obs, info

    def save_strategy(self):
        strategy = (self.action_sequence, self.observation_sequence)
        self.strategies.append(strategy)

    def step(self, action):
        obs, reward, done, truncated, info = self.environment.step(action)
        self.action_sequence.append(action)
        self.observation_sequence.append(tuple(obs[self.tlppo_dim]))
        if done:
            self.save_strategy()
            self.on_episode_end(self)
            self.end_states.append(tuple(obs[self.tlppo_dim]))
        return obs, reward, done, truncated, info

    def update_actions(self,
                       lppo_count: int):
        from itertools import chain
        states = np.array(list(chain.from_iterable([observations for actions, observations in self.strategies])))
        n_clusters = min(lppo_count * 5, len(states))
        kmeans = skc.KMeans(n_clusters=n_clusters, n_init=1)
        kmeans.fit(states)
        centroids = kmeans.cluster_centers_
        adj_matrix = np.zeros((len(centroids), len(centroids)))
        for actions, observations in self.strategies:
            obs_s = np.array(observations)
            obs_cluster = kmeans.predict(obs_s)
            src = obs_cluster[0]
            for index, action in enumerate(actions):
                dst = obs_cluster[index + 1]
                adj_matrix[src][dst] = 1
                adj_matrix[dst][src] = 1
                src = dst
        centrality = eigenvector_centrality(adj_matrix)
        derivative = max_centrality_difference(adj_matrix, centrality)
        lppos = centroids[top_n_indices(derivative, lppo_count)]
        return centroids, adj_matrix, centrality, derivative, lppos
