import math
import time

import cellworld_gym as cg
from cellworld_game import save_video_output, save_log_output
from test_conditions import get_tlppo, get_belief_state_components
import os

create_video = False
result_folders = "TLPPO_FINAL"
os.makedirs(result_folders, exist_ok=True)
episode_count = 100

worlds1 = ['00_01', '00_02', '00_03', '00_04', '00_05', '00_06', '00_07', '00_08', '00_09', '01_01', '01_02', '01_03', '01_04', '01_05', '01_06', '01_07', '01_08', '01_09', '02_01', '02_02', '02_03', '02_04', '02_05', '02_06', '02_07', '02_08', '02_09', '03_01', '03_02', '03_03', '03_04', '03_05', '03_06', '03_07', '03_08', '03_09', '04_01', '04_02', '04_03', '04_04', '04_05', '04_06', '04_07', '04_08', '04_09']
worlds2 = ['05_01', '05_02', '05_03', '05_04', '05_05', '05_06', '05_07', '05_08', '05_09', '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '07_01', '07_02', '07_03', '07_04', '07_05', '07_06', '07_07', '07_08', '07_09', '08_01', '08_02', '08_03', '08_04', '08_05', '08_06', '08_07', '08_08', '08_09', '09_01', '09_02', '09_03', '09_04', '09_05', '09_06', '09_07', '09_08', '09_09']
worlds3 = ['10_01', '10_02', '10_03', '10_04', '10_05', '10_06', '10_07', '10_08', '10_09', '11_01', '11_02', '11_03', '11_04', '11_05', '11_06', '11_07', '11_08', '11_09', '12_01', '12_02', '12_03', '12_04', '12_05', '12_06', '12_07', '12_08', '12_09', '13_01', '13_02', '13_03', '13_04', '13_05', '13_06', '13_07', '13_08', '13_09', '14_01', '14_02', '14_03', '14_04', '14_05', '14_06', '14_07', '14_08', '14_09']

for world in worlds1:
    for condition in [4]:
        for depth in [1, 5]:
            process = False
            for budget in [20, 500]:
                data_point_results_folder = f"results/{result_folders}/logs/{world}_condition_{condition}/depth_{depth}/budget_{budget}"
                data_point_videos_folder = f"results/{result_folders}/videos/{world}_condition_{condition}/depth_{depth}/budget_{budget}"
                data_point_experiment_name = f"{world}_condition_{condition}_{depth}_{budget}"

                data_point_file = f"{data_point_experiment_name}.json"
                data_point_file_path = f"{data_point_results_folder}/{data_point_file}"
                print(f"Condition: {condition}, World: {world}, Depth: {depth}, Budget: {budget}", end="")
                if os.path.exists(data_point_file_path):
                    print("... already exists")
                    continue
                process = True
                print("")
                bs_components = get_belief_state_components(condition=condition)

                environment = cg.BotEvadeBeliefEnv(world_name=world,
                                                   real_time=False,
                                                   render=create_video,
                                                   use_lppos=False,
                                                   use_predator=True,
                                                   belief_state_components=bs_components,
                                                   belief_state_probability=1,
                                                   prey_max_forward_speed=.5,
                                                   prey_max_turning_speed=5.0,
                                                   predator_prey_turning_speed_ratio=1.0,
                                                   predator_prey_forward_speed_ratio=1.0)

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
                    while not finished and not truncated and environment.model.prey_data.puff_count == 0:
                        action = tlppo.get_action(point=environment.model.prey.state.location,
                                                  discount=0.0,
                                                  exploration=math.inf)
                        obs, reward, finished, truncated, info = environment.step(action=action)
                        show_steps = False
                time.sleep(1)
                environment.close()
                del environment
                del tlppo
            if process:
                time.sleep(10)
