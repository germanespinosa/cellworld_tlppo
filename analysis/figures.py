import math
import os
import jsonbind
import matplotlib.pyplot as plt
from cellworld import QuickBundles as QB
import matplotlib.colors as mcolors

title = "Baseline"

conditions = [1, 2, 3, 4, 5]
depths = [1, 2, 3, 4, 5]
budgets = [20, 50, 100, 200, 500]

figures_folder = "figures/results_with_clusters_nobudget_10"
os.makedirs(figures_folder, exist_ok=True)

sim_result_folder = "../results4"
sim_result_file = f"../results4/sim_results_new_nobudget10.json"

if not os.path.exists(sim_result_file):
    clusters_colors = list(mcolors.TABLEAU_COLORS.keys())

    import cellworld as cw

    results = {}

    world = cw.World.get_from_parameters_names("hexagonal",
                                               "canonical",
                                               "21_05")
    clusters_folder = f"{figures_folder}/clusters"
    os.makedirs(clusters_folder, exist_ok=True)

    for condition in conditions:
        results[f"condition{condition}"] = {}
        for depth in depths:
            results[f"condition{condition}"][f"depth{depth}"] = {}
            for budget in budgets:
                experiment_file = os.path.join(f"{sim_result_folder}",
                                               "logs",
                                               f"condition_{condition}",
                                               f"depth_{depth}",
                                               f"budget_{budget}",
                                               f"condition_{condition}_{depth}_{budget}.json")
                if not os.path.isfile(experiment_file):
                    continue
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"] = {}
                print(experiment_file)
                experiment = cw.Experiment.load_from_file(experiment_file)
                total_time = 0
                total_distance = 0
                total_inter_agent_distance = 0
                total_steps = 0
                total_x = 0
                total_used_cells = 0
                total_survived = 0
                total_captured = 0
                clusters = QB.StreamLineClusters(max_distance=.15, streamline_len=500)
                for episode in experiment.episodes:
                    used_cells = set()
                    agent_trajectories = episode.trajectories.split_by_agent()
                    prey_trajectories = agent_trajectories["prey"]
                    if len(prey_trajectories) == 0:
                        continue
                    predator_trajectories = agent_trajectories["predator"]
                    last_frame = prey_trajectories[-1].frame
                    total_time += last_frame * .025
                    src = prey_trajectories[0].location
                    total_x += src.x
                    total_inter_agent_distance += src.dist(predator_trajectories[0].location)
                    if episode.captures:
                        total_captured += 1
                    else:
                        total_survived += 1
                    for prey_step, predator_step in zip(prey_trajectories[1:], predator_trajectories[1:]):
                        used_cells.add(world.cells.find(prey_step.location))
                        total_steps += 1
                        total_distance += src.dist(prey_step.location)
                        src = prey_step.location
                        total_x += src.x
                        inter_agent_distance = src.dist(predator_step.location)
                        if inter_agent_distance < .1:
                            captured = True
                        total_inter_agent_distance += inter_agent_distance
                    total_steps += 1
                    total_used_cells += len(used_cells)
                    clusters.add_trajectory(prey_trajectories)
                clusters.filter_clusters(int(clusters.streamline_count() * .05))
                clusters.assign_unclustered()
                d = cw.Display(world=world)
                d.plot_clusters(clusters=clusters, colors=clusters_colors)
                plt.savefig(f"{clusters_folder}/clusters_condition{condition}_depth{depth}_budget{budget}.png")
                plt.close()
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["cluster_count"] = len(clusters)
                distances = clusters.get_distances()
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["cluster_distance"] = sum(distances)/len(distances)
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_duration"] = total_time/len(experiment.episodes)
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_length"] = total_distance / len(experiment.episodes)
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_predator_prey_distance"] = total_inter_agent_distance / total_steps
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_x"] = total_x / total_steps
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_used_cells"] = total_used_cells / len(experiment.episodes)
                results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_survived"] = total_survived / len(experiment.episodes)

    with open(sim_result_file, "wb") as f:
        jsonbind.dump(obj=results, fp=f)


sim_results = jsonbind.load(open(sim_result_file))
mice_results = jsonbind.load(open("mice_results.json"))

def get_n_colors(n):
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]
    return colors


condition_colormaps = ["Purples", "Blues", "Greens", "Oranges", "Reds"]
condition_color = ["purple", "blue", "green", "orange", "red", "black"]
metrics = ["cluster_count", "cluster_distance", "avg_x", "avg_duration", "avg_length", "avg_predator_prey_distance", "avg_used_cells", "avg_survived"]

for x_axis in metrics:
    for y_axis in metrics:
        if x_axis == y_axis:
            continue

        figure_file = f"{figures_folder}/figures_{x_axis}_{y_axis}.png"
        if os.path.exists(figure_file):
            continue

        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(title)
        ax = fig.add_subplot(111)

        results = {}
        for condition, cr in sim_results.items():
            condition_results = {field: [] for field in metrics}
            condition_results["color"] = []
            for depth, dr in cr.items():
                depth_val = int(depth.replace("depth", ""))
                for budget, br in dr.items():
                    budget_val = int(budget.replace("budget", ""))
                    for r in condition_results:
                        if r in br:
                            condition_results[r].append(br[r])
                    condition_results["color"].append(depth_val)
            results[condition] = condition_results

        for color, (condition, cr) in enumerate(sim_results.items()):
            condition_results = results[condition]
            scatter = ax.scatter(condition_results[x_axis],
                                 condition_results[y_axis],
                                 c=condition_results["color"],
                                 cmap=condition_colormaps[color],
                                 s=50,
                                 alpha=1,
                                 edgecolors=condition_color[color])

            scatter = ax.scatter(x=[sum(condition_results[x_axis]) / len(condition_results[x_axis])],
                                 y=[sum(condition_results[y_axis]) / len(condition_results[y_axis])],
                                 c=condition_color[color],
                                 s=1200,
                                 alpha=.3,
                                 edgecolors='black')

        scatter = ax.scatter([mouse[x_axis] for mouse in mice_results],
                             [mouse[y_axis] for mouse in mice_results],
                             c="black",
                             s=50,
                             alpha=1)

        scatter = ax.scatter([sum([mouse[x_axis] for mouse in mice_results])/len(mice_results)],
                             [sum([mouse[y_axis] for mouse in mice_results])/len(mice_results)],
                             c="black",
                             s=1200,
                             alpha=.3)

        # Adding labels
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        # # Adding a color bar
        # cbar = plt.colorbar(scatter)
        # cbar.set_label('Color Mapping')

        labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4', 'Condition 5', 'Mice']
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=condition_color[i], markersize=10) for i, label in enumerate(labels)]
        plt.legend(handles=handles, title='Colors')

        # Show plot
        plt.savefig(figure_file)
        # plt.show()
        plt.close()


fig, axs = plt.subplots(2, 2, figsize=(12, 12))

groups = ["low_budget-low_depth", "low_budget-high_depth", "high_budget-low_depth", "high_budget-high_depth"]
groups_budgets = [[10, 20], [10, 20], [200, 500], [200, 500]]
groups_depths = [[1, 2], [4, 5], [1, 2], [4, 5]]
groups_xaxis = [0, 1, 0, 1]
groups_yaxis = [0, 0, 1, 1]
for x_axis in metrics:
    for y_axis in metrics:
        if x_axis == y_axis:
            continue

        figure_file = f"{figures_folder}/comp_figures_{x_axis}_{y_axis}.png"
        if os.path.exists(figure_file):
            continue

        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        for title, valid_budgets, valid_depths, x, y in zip(groups, groups_budgets, groups_depths, groups_xaxis,
                                                            groups_yaxis):
            ax = axs[x, y]
            ax.set_title(title)
            results = {}
            for condition, cr in sim_results.items():
                condition_results = {field: [] for field in metrics}
                condition_results["color"] = []
                for r in list(condition_results.keys()):
                    condition_results[f"max_{r}"] = -math.inf
                for depth, dr in cr.items():
                    depth_val = int(depth.replace("depth", ""))
                    if depth_val not in valid_depths:
                        continue
                    for budget, br in dr.items():
                        budget_val = int(budget.replace("budget", ""))
                        if budget_val not in valid_budgets:
                            continue
                        for r in condition_results:
                            if r in br:
                                condition_results[r].append(br[r])
                                if br[r] > condition_results[f"max_{r}"]:
                                    condition_results[f"max_{r}"] = br[r]
                        condition_results["color"].append(depth_val)
                results[condition] = condition_results

            for color, (condition, cr) in enumerate(sim_results.items()):
                condition_results = results[condition]
                if len(condition_results["avg_x"]) == 0:
                    continue
                scatter = ax.scatter(condition_results[x_axis],
                                     condition_results[y_axis],
                                     c=condition_results["color"],
                                     cmap=condition_colormaps[color],
                                     s=50,
                                     alpha=1,
                                     edgecolors=condition_color[color])

                scatter = ax.scatter(x=[sum(condition_results[x_axis]) / len(condition_results[x_axis])],
                                     y=[sum(condition_results[y_axis]) / len(condition_results[y_axis])],
                                     c=condition_color[color],
                                     s=1200,
                                     alpha=.3,
                                     edgecolors='black')


            scatter = ax.scatter([mouse[x_axis] for mouse in mice_results],
                                 [mouse[y_axis] for mouse in mice_results],
                                 c="black",
                                 s=50,
                                 alpha=1)

            scatter = ax.scatter([sum([mouse[x_axis] for mouse in mice_results])/len(mice_results)],
                                 [sum([mouse[y_axis] for mouse in mice_results])/len(mice_results)],
                                 c="black",
                                 s=1200,
                                 alpha=.3)

            # Adding labels
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)

            # # Adding a color bar
            # cbar = plt.colorbar(scatter)
            # cbar.set_label('Color Mapping')

            labels = ['Condition 1', 'Condition 2', 'Condition 3', 'Condition 4', 'Condition 5', 'Mice']
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=condition_color[i], markersize=10) for i, label in enumerate(labels)]
            plt.legend(handles=handles, title='Colors')

            # Show plot
        plt.savefig(figure_file)
        plt.close()

metrics_avg_values = {}

for metric in metrics:
    sum_metric = 0
    count_metric = 0
    metrics_avg_values[metric] = 0
    for mouse_results in mice_results:
        metrics_avg_values[metric] += mouse_results[metric]
    metrics_avg_values[metric] /= len(mice_results)

normalized_sim_results_file = sim_result_file.replace(".json", "_normalized.json")
if os.path.exists(normalized_sim_results_file):
    with open(normalized_sim_results_file, "r") as f:
        normalized_sim_results = jsonbind.load(f)
else:
    normalized_sim_results = {}
    for condition, condition_results in sim_results.items():
        normalized_sim_results[condition] = {}
        for depth, depth_results in condition_results.items():
            normalized_sim_results[condition][depth] = {}
            for budget, budget_results in depth_results.items():
                normalized_sim_results[condition][depth][budget] = {}
                for metric, value in budget_results.items():
                    normalized_value = abs(1 - (value / metrics_avg_values[metric]))
                    normalized_sim_results[condition][depth][budget][metric] = normalized_value
    with open(normalized_sim_results_file, "wb") as f:
        jsonbind.dump(normalized_sim_results, f)

plt.close()

metrics = ["cluster_count", "cluster_distance", "avg_x", "avg_duration", "avg_length", "avg_predator_prey_distance", "avg_used_cells", "avg_survived"]

# metrics = ["cluster_count", "avg_x", "avg_duration", "avg_length", "avg_predator_prey_distance", "avg_used_cells"]


budgets_colors = list(mcolors.TABLEAU_COLORS.keys())


avg_values = [[0 for _ in conditions] for _ in budgets]

for depth in depths:
    figure_file = f"{figures_folder}/distance_by_depth_{depth}.png"
    # if os.path.exists(figure_file):
    #     continue
    fig, ax = plt.subplots()
    ax.set_xticks(conditions)
    ax.set_xticklabels([f"Condition {c}" for c in conditions])

    av_all_budgets = [0 for _ in conditions]
    for i, budget in enumerate(budgets):
        norm_diff = []
        for j, condition in enumerate(conditions):
            sum_diffs = 0
            count_diffs = 0
            for metric in metrics:
                value = normalized_sim_results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"][metric]
                sum_diffs += value
                count_diffs += 1
            norm_diff.append(sum_diffs/count_diffs)
            avg_values[i][j] += sum_diffs/count_diffs
            av_all_budgets[j] += avg_values[i][j] / len(budgets)
        ax.plot(conditions, norm_diff, color=budgets_colors[i])
    # ax.plot(conditions, av_all_budgets, color='black',  linewidth=5, alpha=.5)

    labels = budgets
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=budgets_colors[i], markersize=10) for i, label in enumerate(labels)]

    plt.legend(handles=handles, title='Budgets')
    plt.suptitle(f"Depth {depth}")
    plt.savefig(figure_file)
    plt.close()


figure_file = f"{figures_folder}/distance_by_depth_avg.png"
fig, ax = plt.subplots()
ax.set_xticks(conditions)
ax.set_xticklabels([f"Condition {c}" for c in conditions])

for i, budget in enumerate(budgets):
    ax.plot(conditions, [v / len(depths) for v in avg_values[i]], color=budgets_colors[i])

plt.legend(handles=handles, title='Budgets')
plt.suptitle(f"All Depths")
plt.savefig(figure_file)
plt.close()
