import cellworld as cw
import os


for condition in [1, 2, 3, 4, 5]:
    for depth in [1, 2, 3, 4, 5]:
        for budget in [10, 20, 50, 100, 200, 500]:
            experiment_file = os.path.join(".",
                                           "logs",
                                           f"condition_{condition}",
                                           f"depth_{depth}",
                                           f"budget_{budget}",
                                           f"condition_{condition}_{depth}_{budget}.json")
            if not os.path.isfile(experiment_file):
                continue
            experiment = cw.Experiment.load_from_file(experiment_file)
            experiment: cw.Experiment
            for episode in experiment.episodes:
                episode.captures = []
                agent_trajectories = episode.trajectories.split_by_agent()
                prey_trajectories = agent_trajectories["prey"]
                predator_trajectories = agent_trajectories["predator"]
                for prey_step, predator_step in zip(prey_trajectories, predator_trajectories):
                    inter_agent_distance = prey_step.location.dist(predator_step.location)
                    if inter_agent_distance <= .1:
                        episode.captures.append(prey_step.frame)
            experiment.save(file_path=experiment_file)
