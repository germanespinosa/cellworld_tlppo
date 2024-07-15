import json
import argparse
import cellworld as cw
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

clusters_colors = list(mcolors.TABLEAU_COLORS.keys())


def parse_arguments():
    parser = argparse.ArgumentParser(description='Summarizes an experiment file')
    parser.add_argument('file', type=str, help='file to summarize')
    parser.add_argument('-o', '--output_file', type=str, help='output file', required=True)
    parser.add_argument('-r', '--root_folder', type=str, help='root folder', required=True)
    parser.add_argument('-cf', '--clusters_folder', type=str, help='clusters folder', required=True)
    args = parser.parse_args()
    return args


def process_data(data: dict, world: cw.World, clusters_file: str, processed_data: dict = None):
    if processed_data is None:
        processed_data = {}
    col = data
    if col["values"]:
        values = col["values"]
        if len(values["progress"]) > 0:
            processed_data["avg_x"] = sum(values["progress"])/len(values["progress"])
        else:
            processed_data["avg_x"] = 0

        if len(values["predator_distances"]) > 0:
            processed_data["avg_predator_prey_distance"] = sum(values["predator_distances"]) / len(values["predator_distances"])
        else:
            processed_data["avg_predator_prey_distance"] = 0

        if len(values["used_cells"]) > 0:
            processed_data["avg_used_cells"] = sum(values["used_cells"]) / len(values["used_cells"])
        else:
            processed_data["avg_used_cells"] = 0

        processed_data["avg_duration"] = values["time"] / values["episode_count"]
        processed_data["avg_length"] = values["distance"] / values["episode_count"]
        processed_data["avg_survived"] = values["survived"] / values["episode_count"]
        values["clusters"].filter_clusters(int(values["clusters"].streamline_count() * .05))
        values["clusters"].assign_unclustered()
        d = cw.Display(world=world)
        d.plot_clusters(clusters=values["clusters"], colors=clusters_colors)
        plt.savefig(f"{clusters_file}.png")
        plt.close()
        processed_data["cluster_count"] = len(values["clusters"])
        distances = values["clusters"].get_distances()
        if len(distances) > 0:
            processed_data["cluster_distance"] = sum(distances)/len(distances)
        else:
            processed_data["cluster_distance"] = 0
        processed_data["avg_cluster_count"] = processed_data["cluster_count"]
        processed_data["avg_cluster_distance"] = processed_data["cluster_distance"]

    if "groups" in col:
        groups = col["groups"]
        processed_data["groups"] = {}
        for group_name, group in groups.items():
            processed_data["groups"][group_name] = {}
            process_data(data=group,
                         world=world,
                         clusters_file=f"{clusters_file}_{group_name}",
                         processed_data=processed_data["groups"][group_name])
        sum_cluster_count = 0
        sum_cluster_distance = 0
        for group_name, group in processed_data["groups"].items():
            sum_cluster_count += group["avg_cluster_count"]
            sum_cluster_distance += group["avg_cluster_distance"]
        processed_data["avg_cluster_count"] = sum_cluster_count / len(processed_data["groups"])
        processed_data["avg_cluster_distance"] = sum_cluster_distance / len(processed_data["groups"])
    return processed_data


def summarize_experiments(file_records: list) -> dict:
    import os
    import cellworld.QuickBundles as QB

    def new_values():
        values = {}
        values["episode_count"] = 0
        values["step_count"] = 0
        values["distance"] = 0.0
        values["time"] = 0.0
        values["predator_distances"] = []
        values["used_cells"] = []
        values["progress"] = []
        values["survived"] = 0.0
        values["clusters"] = QB.StreamLineClusters(max_distance=.15, streamline_len=500)
        return values

    data = {}

    def add_value(file_record, value_name, value):
        col = data
        level = -1
        while True:

            if "values" not in col:
                col["values"] = new_values()
            if isinstance(col["values"][value_name], (int, float)):
                col["values"][value_name] += value
            elif isinstance(col["values"][value_name], list):
                col["values"][value_name].append(value)
            elif isinstance(col["values"][value_name], QB.StreamLineClusters):
                col["values"][value_name].add_trajectory(value)
            level += 1
            level_name = f"group_{level}"
            if level_name in file_record:
                group_name = file_record[level_name]
                if "groups" not in col:
                    col["groups"] = {}
                if not file_record[level_name] in col["groups"]:
                    col["groups"][group_name] = {}
                col = col["groups"][group_name]
            else:
                break
    world = None
    for file_record in file_records:
        print(file_record)
        experiment_file = f"{args.root_folder}/{file_record["file"]}"
        if not os.path.isfile(experiment_file):
            raise RuntimeError(f'File {experiment_file} does not exist')
        experiment = cw.Experiment.load_from_file(file_path=experiment_file)
        world = cw.World.get_from_parameters_names(world_configuration_name=experiment.world_configuration_name,
                                                   world_implementation_name="canonical",
                                                   occlusions_name=experiment.occlusions)
        for episode in experiment.episodes:
            episode: cw.Episode
            agents_trajectories = episode.trajectories.split_by_agent()
            if "predator" not in agents_trajectories:
                continue
            if "prey" not in agents_trajectories:
                continue

            prey_trajectory: cw.Trajectories = agents_trajectories["prey"]

            predator_trajectory: cw.Trajectories = agents_trajectories["predator"]

            if len(episode.captures) == 0:
                add_value(file_record, "survived", 1)
            if len(prey_trajectory) == 0:
                continue
            add_value(file_record, "episode_count", 1)
            add_value(file_record, "step_count", len(prey_trajectory))
            last_step: cw.Step = prey_trajectory[0]
            used_cell_ids = set()
            started = False
            for step in prey_trajectory:
                step: cw.Step
                used_cell_ids.add(world.cells.find(step.location))
                if step.location.x > .1:
                    started = True
                if not started:
                    continue
                add_value(file_record, "progress", step.location.x)
                add_value(file_record, "distance", step.location.dist(last_step.location))
                last_step = step
                predator_step = predator_trajectory.get_step_by_frame(frame=step.frame)
                if predator_step:
                    add_value(file_record, "predator_distances", step.location.dist(predator_step.location))
            add_value(file_record, "time", prey_trajectory[-1].time_stamp - prey_trajectory[0].time_stamp)
            add_value(file_record, "used_cells", len(used_cell_ids))
            add_value(file_record, "clusters", prey_trajectory)
    return process_data(data=data,
                        world=world,
                        clusters_file=f"{args.clusters_folder}/clusters")


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    import os
    if not os.path.isfile(args.file):
        print(f"Error: file {args.file} does not exist")
        exit(1)

    with open(args.file, "r") as f:
        test = json.load(f)

    summarized_data = summarize_experiments(test["files"])
    results = {"groups": test["groups"], "data": summarized_data}
    with open(args.output_file, "w") as f:
        json.dump(results, f)

