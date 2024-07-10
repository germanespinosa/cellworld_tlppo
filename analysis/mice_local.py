from json_cpp import JsonObject, JsonList
from cellworld import QuickBundles as QB
from cellworld import *
import pandas as pd
import os

world_name = "030_01_0092"


def DataFile(file_name: str) -> str:
  full_path = f"mice_data/{file_name}"
  return full_path


experiment_list_file = f"mice_data/experiment_list.json"

if not os.path.exists(experiment_list_file):
    import glob
    phases = glob.glob(f"mice_data/{world_name}/*")
    Experiment_list = JsonList()
    for phase_path in phases:
        phase = os.path.basename(phase_path)
        #
        for subject_path in glob.glob(f"mice_data/{world_name}/{phase}/*"):
            subject = os.path.basename(subject_path)
            for iteration_path in glob.glob(f"mice_data/{world_name}/{phase}/{subject}/*"):
                iteration = int(os.path.basename(iteration_path).replace("iteration_", "").replace(".json", ""))
                iteration_path = iteration_path.replace("\\", "/"). replace("mice_data/", "")
                Experiment_list.append(JsonObject(**{"phase": phase, "subject": subject, "iteration": iteration, "file":iteration_path}))
    Experiment_list.save(experiment_list_file)
else:
    Experiment_list = JsonObject.load_from_file(experiment_list_file)


def GetEpisodes(phase="", subject="", iterations=[]):
  def filter(experiment):
    return (experiment.phase == phase or phase == "") and (experiment.subject == subject or subject=="") and (experiment.iteration in iterations or iterations==[])
  experiment_files = Experiment_list.filter(key=filter)

  episodes = Episode_list()
  for experiment_file in experiment_files:
      e = Experiment.load_from_file("mice_data/" + experiment_file.file)
      for ep in e.episodes:
          episodes.append(ep)
  return episodes


import numpy as np
import pandas as pd
import random
from cellworld import *

occlusions = world_name
w = World.get_from_parameters_names('hexagonal', "canonical", occlusions)
subject_str = 'prey'
scalar = 1.0
mv = 2/2.34

min_x, max_x = 0.05, 0.95

mice =['FMM19', 'FMM20', 'MMM18', 'MMM20']

result_list = []

for subject in mice:
    experiment_files = []
    x_location_data = []

    sum_x = 0
    steps = 0

    sum_distance = 0
    sum_inter_agent_distance = 0
    sum_time = 0
    sum_used_cells = 0
    count_episodes = 0
    sum_captured = 0
    sum_survived = 0
    clusters = QB.StreamLineClusters(max_distance=.15, streamline_len=500)
    episodes = GetEpisodes(subject=subject)
    for episode in episodes:
        if episode.captures:
            sum_captured += 1
        else:
            sum_survived += 1

        try:
            t = episode.trajectories
        except Exception as ex:
            print(ex)
            continue
        atr = t.split_by_agent()
        if "prey" not in atr:
            continue
        ht = atr["prey"]
        pt: Trajectories = atr["predator"]
        clusters.add_trajectory(ht)
        prey_ust = ht.get_unique_steps()
        src = None
        start_time_stamp = 0
        end_time_stamp = 0
        used_cells = set()
        is_episode = 0
        for l in prey_ust:
            if l.location.x > max_x:
                break
            if src is None and l.location.x < min_x:
                continue
            if start_time_stamp is None:
                start_time_stamp = l.time_stamp

            used_cells.add(w.cells.find(l.location))
            steps += 1
            if src is not None:
                sum_distance += src.location.dist(l.location)
            ps = pt.get_step_by_frame(l.frame)
            if ps:
                sum_inter_agent_distance += l.location.dist(ps.location)
            src = l
            sum_x += l.location.x
            end_time_stamp = l.time_stamp
            is_episode = 1
        count_episodes += is_episode
        sum_used_cells += len(used_cells)
        sum_time += end_time_stamp - start_time_stamp
    results = {}
    clusters.filter_clusters(int(clusters.streamline_count() * .05))
    clusters.assign_unclustered()
    results["subject"] = subject
    results["cluster_count"] = len(clusters)
    distances = clusters.get_distances()
    results["cluster_distance"] = sum(distances)/len(distances)
    results["sum_x"] = sum_x
    results["avg_x"] = sum_x/steps
    results["sum_length"] = sum_distance
    results["avg_length"] = sum_distance/count_episodes
    results["sum_predator_prey_distance"] = sum_inter_agent_distance
    results["avg_predator_prey_distance"] = sum_inter_agent_distance/steps
    results["sum_used_cells"] = sum_used_cells
    results["avg_used_cells"] = sum_used_cells/count_episodes
    results["sum_survived"] = sum_survived
    results["avg_survived"] = sum_survived/count_episodes
    results["sum_duration"] = sum_time
    results["avg_duration"] = sum_time/count_episodes
    results["episodes"] = count_episodes
    results["steps"] = steps
    result_list.append(results)

import json

with open(f'mice_{world_name}_results.json', 'w') as f:
    json.dump(result_list, f)
