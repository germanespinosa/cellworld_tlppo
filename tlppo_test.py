import math
import time

import cellworld_gym as cg
from cellworld_game import save_video_output, save_log_output
from test_conditions import get_tlppo, get_belief_state_components
import os

create_video = False
result_folders = "New_Baseline"
os.makedirs(result_folders, exist_ok=True)
episode_count = 100

for condition in [1, 2, 3, 4, 5]:
    for depth in [1, 2, 3, 4, 5]:
        process = False
        for budget in [20, 50, 100, 200, 500]:
            data_point_results_folder = f"results/{result_folders}/logs/condition_{condition}/depth_{depth}/budget_{budget}"
            data_point_videos_folder = f"results/{result_folders}/videos/condition_{condition}/depth_{depth}/budget_{budget}"
            data_point_experiment_name = f"condition_{condition}_{depth}_{budget}"

            data_point_file = f"{data_point_experiment_name}.json"
            data_point_file_path = f"{data_point_results_folder}/{data_point_file}"
            print(f"Condition: {condition}, Depth: {depth}, Budget: {budget}", end="")
            if os.path.exists(data_point_file_path):
                print("... already exists")
                continue
            process = True
            print("")
            bs_components = get_belief_state_components(condition=condition)

            environment = cg.BotEvadeBeliefEnv(world_name="21_05",
                                               real_time=False,
                                               render=create_video,
                                               use_lppos=False,
                                               use_predator=True,
                                               belief_state_components=bs_components,
                                               belief_state_probability=1)

            save_log_output(model=environment.model,
                            experiment_name=data_point_experiment_name,
                            log_folder=data_point_results_folder,
                            save_checkpoint=False)

            if create_video:
                environment.model.render_agent_visibility = "prey"
                save_video_output(model=environment.model,
                                  video_folder=data_point_videos_folder)

            tlppo = get_tlppo(environment=environment,
                              depth=depth,
                              budget=budget)
            tree = None
            for i in range(episode_count):
                print(f"Episode: {i}")
                obs, _ = environment.reset()
                finished, truncated = False, False
                puff_count = 0
                while not finished and not truncated:
                    action = tlppo.get_action(point=environment.model.prey.state.location,
                                              discount=0.0,
                                              exploration=math.inf)
                    obs, reward, finished, truncated, info = environment.step(action=action)
                    show_steps = False
            environment.close()
            del environment
            del tlppo
        if process:
            time.sleep(10)
