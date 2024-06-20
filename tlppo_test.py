import math
import time

import cellworld_gym as cg
from cellworld_game import save_video_output, save_log_output
from test_conditions import get_tlppo, get_belief_state_components
import os

create_video = False
result_folders = "results6"
os.makedirs(result_folders, exist_ok=True)
episode_count = 1000

for condition in [1, 2, 3, 4, 5]:
    for depth in [3]:
        process = False
        for budget in [20, 50, 100, 200, 500]:
            result_file = f"{result_folders}/logs/condition_{condition}/depth_{depth}/budget_{budget}/condition_{condition}_{depth}_{budget}.json"
            print(f"Condition: {condition}, Depth: {depth}, Budget: {budget}", end="")
            if os.path.exists(result_file):
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

            save_log_output(environment.model,
                            f"condition_{condition}_{depth}_{budget}",
                            f"{result_folders}/logs/condition_{condition}/depth_{depth}/budget_{budget}",
                            save_checkpoint=False)

            if create_video:
                environment.model.render_agent_visibility = "prey"
                save_video_output(environment.model,
                                  f"{result_folders}/videos/condition_{condition}/depth_{depth}/budget_{budget}")


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
