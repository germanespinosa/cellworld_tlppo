import jsonbind
import os
import cellworld as cw

results = {}

world = cw.World.get_from_parameters_names("hexagonal",
                                           "canonical",
                                           "21_05")

for condition in [1, 2, 3, 4, 5]:
    results[f"condition{condition}"] = {}
    for depth in [1, 2, 3, 4, 5]:
        results[f"condition{condition}"][f"depth{depth}"] = {}
        for budget in [10, 20, 50, 100, 200, 500]:
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"] = {}
            experiment_file = os.path.join(f"results",
                                           "logs",
                                           f"condition_{condition}",
                                           f"depth_{depth}",
                                           f"budget_{budget}",
                                           f"condition_{condition}_{depth}_{budget}.json")
            print(experiment_file)
            experiment = cw.Experiment.load_from_file(experiment_file)
            total_time = 0
            total_distance = 0
            total_inter_agent_distance = 0
            total_steps = 0
            total_x = 0
            total_used_cells = 0
            for episode in experiment.episodes:
                used_cells = set()
                agent_trajectories = episode.trajectories.split_by_agent()
                prey_trajectories = agent_trajectories["prey"]
                predator_trajectories = agent_trajectories["predator"]
                last_frame = prey_trajectories[-1].frame
                total_time += last_frame * .025
                src = prey_trajectories[0].location
                total_x += src.x
                total_inter_agent_distance += src.dist(predator_trajectories[0].location)
                for prey_step, predator_step in zip(prey_trajectories[1:], predator_trajectories[1:]):
                    used_cells.add(world.cells.find(prey_step.location))
                    total_steps += 1
                    total_distance += src.dist(prey_step.location)
                    src = prey_step.location
                    total_x += src.x
                    total_inter_agent_distance += src.dist(predator_step.location)
                total_steps += 1
                total_used_cells += len(used_cells)

            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_duration"] = total_time/len(experiment.episodes)
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_length"] = total_time / len(experiment.episodes)
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_predator_prey_distance"] = total_inter_agent_distance / total_steps
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_x"] = total_x / total_steps
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_used_cells"] = total_used_cells / len(experiment.episodes)
            results[f"condition{condition}"][f"depth{depth}"][f"budget{budget}"]["avg_used_cells"] = total_used_cells / len(experiment.episodes)


with open("results/results/sim_results.json", "wb") as f:
    jsonbind.dump(obj=results, fp=f)
