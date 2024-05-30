import cellworld_gym as cg
from cellworld_game import save_video_output, Point
import cellworld_tlppo.belief_state as belief
import cellworld_tlppo as ct
import cellworld as cw


env = cg.BotEvadeEnv(world_name="21_05",
                     real_time=False,
                     render=True,
                     use_lppos=False,
                     use_predator=True)


# save_video_output(env.model, "./videos")

gc = belief.GaussianDiffusionComponent(.25 / 2.34 * .25)
dc = belief.DirectedDiffusionComponent(.20 / 2.34 * .25)

bs = belief.BeliefState(arena=env.model.arena,
                        occlusions=env.model.occlusions,
                        definition=100,
                        components=[gc, dc])


env.model.view.add_render_step(bs.render, z_index=5)


def get_probability(a, b):
    print("probability", algo.__get_probability__(bs.probability_distribution, b))


env.model.view.on_mouse_button_up = get_probability

obs: cg.BotEvadeObservation
# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 50

connection_graph = ct.Graph(nodes={label: ct.State(point)
                                   for label, point
                                   in enumerate(env.model.loader.full_action_list)})

for src_label, cnn in enumerate(env.model.loader.options_graph):
    for dst_label in cnn:
        connection_graph.connect(src_label=src_label, dst_label=dst_label)


def reward_function(point, puff_probability):
    distance_to_goal = Point.distance(src=point, dst=(1.0, 0.5))
    return -distance_to_goal - puff_probability * 100, distance_to_goal > env.model.goal_threshold
    # return -distance_to_goal, distance_to_goal > env.model.goal_threshold


algo = ct.TLPPO(graph=connection_graph,
                puff_radius=env.model.puff_threshold,
                robot_belief_state=bs,
                reward_fn=reward_function,
                visibility=env.model.visibility,
                depth=5,
                budget=100,
                speed=env.model.prey.max_forward_speed * env.time_step,
                navigation=env.loader.navigation)

save_video_output(model=env.model, video_folder="./videos")

bs.tick()
for i in range(100):
    print(i)
    obs, _ = env.reset()
    bs.reset()
    finished, truncated = False, False
    while not finished and not truncated:
        prey_location = (obs.prey_x, obs.prey_y)
        bs.update_self_location(self_location=prey_location)
        if env.model.predator_visible:
            bs.update_other_location((obs.predator_x, obs.predator_y))
        else:
            visibility_polygon = env.model.visibility.get_visibility_polygon(src=prey_location,
                                                                             direction=0,
                                                                             view_field=360)
            bs.update_visibility(visibility_polygon=visibility_polygon)
        bs.tick()
        tree = algo.get_action(point=prey_location, discount=.4)
        action = tree.root.select(0).label
        obs, reward, finished, truncated, info = env.step(action=action)
