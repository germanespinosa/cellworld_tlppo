import pulsekit
r = pulsekit.start_profile()
import cellworld_tlppo as ct
import cellworld_gym as cwg
import numpy as np
import matplotlib.pyplot as plt

episode = 0


def on_episode_end(env: ct.TlppoLearner):
    return
    # global episode
    # episode += 1
    # if episode % 10:
    #     return
    # print('Episode ended.')
    # import matplotlib.pyplot as plt
    # lppo, adj_matrix = env.get_lppo()
    # plt.figure(figsize=(8, 8))
    # plt.scatter(lppo[:, 0], lppo[:, 1], color='blue', zorder=2)  # Plot nodes
    # for i in range(adj_matrix.shape[0]):
    #     for j in range(adj_matrix.shape[1]):
    #         if adj_matrix[i][j] == 1:
    #             plt.plot([lppo[i][0], lppo[j][0]], [lppo[i][1], lppo[j][1]], color='black', zorder=1)
    #
    # plt.title('Graph Visualization')
    # plt.grid(True)
    # plt.show()


env_learner = ct.TlppoLearner(environment_name="CellworldBotEvade-v0",
                              on_episode_end=on_episode_end,
                              tlppo_dim=np.array([True, True, False, False, False, False, False, False, False, False, False]),
                              world_name="21_05",
                              use_lppos=False,
                              use_predator=True,
                              max_step=200,
                              time_step=.25,
                              reward_function=cwg.Reward({}),
                              real_time=False,
                              render=False)

step_count = 0
done = True
while step_count < 2000:
    if done:
        env_learner.reset()
    _, _, done, _, _ = env_learner.step(env_learner.action_space.sample())
    step_count += 1
    if step_count % 100 == 0:
        print(f"Step: {step_count}")


while not done:
    _, _, done, _, _ = env_learner.step(len(env_learner.environment.action_list)-1)


nodes, adj_matrix, centrality_scores, derivative, lppo = env_learner.update_actions(lppo_count=25)
end_states = np.array(env_learner.end_states)
print(end_states)
plt.figure(figsize=(8, 8))
plt.scatter(nodes[:, 0], nodes[:, 1],  c=derivative, cmap='viridis', zorder=2, marker='o')  # Plot nodes
plt.scatter(lppo[:, 0], lppo[:, 1],  c="r", zorder=2, marker='o', s=150)  # Plot nodes
plt.scatter(end_states[:, 0], end_states[:, 1],  c="r", zorder=2, marker='o', s=150)  # Plot nodes
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i][j] == 1:
            plt.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color='black', zorder=1, alpha=0.3)

# plt.colorbar(label='Centrality Derivative')
plt.title('Graph Visualization')
plt.grid(True)
plt.show()

env_tester = ct.TlppoTester(environment_name="CellworldBotEvade-v0",
                            lppos=lppo,
                            end_states=end_states,
                            world_name="21_05",
                            use_lppos=False,
                            use_predator=True,
                            max_step=200,
                            time_step=.25,
                            reward_function=cwg.Reward({}),
                            real_time=True,
                            render=True)

print(env_tester.environment.action_list)
for i in range(100):
    print(f"Episode {i+1}")
    env_tester.reset()
    for j in range(10):
        action = env_tester.action_space.sample()
        print(action, env_tester.environment.action_list[action])
        _, _, done, _, _ = env_tester.step(action)
        if done:
            break
    while not done:
        action = len(env_tester.environment.action_list)-1
        print(action, env_tester.environment.action_list[action])
        _, _, done, _, _ = env_tester.step(action)
