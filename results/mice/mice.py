from json_cpp import JsonObject
from cellworld import Episode_list, QuickBundles as QB
import pandas as pd


def DownloadDataFile(file_name:str, is_zipped:bool = False) -> bool:
  import requests
  import os
  import zipfile

  if is_zipped:
    full_filename = file_name + ".zip"
  else:
    full_filename = file_name

  data_path = "mice_data"
  full_path = "%s/%s" % (data_path, full_filename)

  if not os.path.exists(full_path):
      print("Downloading data from %s" % full_path)
      data = requests.get("https://raw.githubusercontent.com/cellworld-neuro/public_data/main/%s" % full_filename)
      dir_name = os.path.dirname(full_path)
      os.makedirs(dir_name, exist_ok=True)
      with open(full_path, "wb") as f:
        f.write(data.content)
        f.close()
  else:
      print("Getting from cache %s" % full_path)

  if is_zipped:
    with zipfile.ZipFile(full_path, 'r') as zip_ref:
      zip_ref.extractall(data_path)

def DataFile(file_name:str)-> str:
  full_path = "mice_data/%s" % file_name
  return full_path

DownloadDataFile("21_05/experiment_list.json")
Experiment_list = JsonObject.load_from_file(DataFile("21_05/experiment_list.json"))

def GetEpisodes(phase="", subject="", iterations=[]):
  def filter(experiment):
    return (experiment.phase == phase or phase == "") and (experiment.subject == subject or subject=="") and (experiment.iteration in iterations or iterations==[])
  experiment_files = Experiment_list.filter(key=filter)

  episodes = Episode_list()
  for experiment_file in experiment_files:
      DownloadDataFile(experiment_file.file)
      e = pd.read_hdf(DataFile(experiment_file.file))
      for k, ep in e.iloc[:].iterrows():
          if ep['bad_episode'] == 1:
              continue
          episode = ep['ep_data']
          episodes.append(episode)
  return episodes


experiment_files = Experiment_list

for f in experiment_files:
  DownloadDataFile(f.file)
  df = pd.read_hdf(DataFile(f.file))


import numpy as np
import pandas as pd
import random
from cellworld import *

occlusions = '21_05'
w = World.get_from_parameters_names('hexagonal', "canonical", occlusions)
subject_str = 'prey'
scalar = 1.0
mv = 2/2.34

min_x, max_x = 0.05, 0.95
mice =['FMM9', 'FMM10', 'FMM13', 'FMM14', 'MMM10', 'MMM11', 'MMM13', 'MMM14']

result_list = []

for subject in mice:
    experiment_files = []
    def filter(experiment):
        return (experiment.phase == ['RT']) and (experiment.subject == subject)

    RT_experiment_files = Experiment_list.filter(key=filter)[-3:]
    experiment_files += RT_experiment_files

    def filter(experiment):
        return (experiment.phase == 'R') and (experiment.subject == subject)

    R_experiment_files = Experiment_list.filter(key=filter)
    experiment_files += R_experiment_files

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
    for f in experiment_files:
        DownloadDataFile(f.file)
        print(f.file)
        df = pd.read_hdf(DataFile(f.file))
        for e, ep in df.iloc[:].iterrows():
            if ep['bad_episode'] == 1:
                continue
            episode = ep['ep_data']
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
            ht = atr["prey"]
            pt: Trajectories = atr["predator"]
            clusters.add_trajectory(ht)
            count_episodes += 1
            prey_ust = ht.get_unique_steps()
            src = None
            start_time_stamp = None
            end_time_stamp = None
            used_cells = set()
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

with open('mice_21_05_results.json', 'w') as f:
    json.dump(result_list, f)
