from json_cpp import JsonObject
import pandas as pd
import cellworld as cw

Experiment_list = JsonObject.load_from_file("21_05/experiment_list.json")
experiment_files = Experiment_list
for f in experiment_files:
    print(f.file)
    df = pd.read_hdf(f.file)
    experiment = cw.Experiment(world_configuration_name="hexagonal", world_implementation_name="mice", subject_name=f["subject"],duration=30)

    for e, ep in df.iloc[:].iterrows():
        if ep['bad_episode'] == 1:
            continue
        episode = ep['ep_data']
        new_episode = cw.Episode(start_time=episode.start_time, end_time=episode.end_time, captures=episode.captures)
        for step in episode.trajectories:
            new_step = cw.Step(data=step.data,
                               location=cw.Location(step.location.x, step.location.y),
                               frame=step.frame,
                               time_stamp=step.time_stamp,
                               agent_name=step.agent_name,
                               rotation=step.rotation)
            new_episode.trajectories.append(new_step)
        experiment.episodes.append(new_episode)
    new_file = f.file.replace(".hdf", ".json")
    experiment.save(new_file)
    f.file = new_file
