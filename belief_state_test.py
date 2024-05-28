import random
from cellworld_game import BotEvade, save_video_output, Point
import cellworld_tlppo.belief_state as belief
import cellworld_tlppo as ct
import cellworld as cw


bot_evade = BotEvade(world_name="21_05",
                     puff_cool_down_time=.5,
                     puff_threshold=.1,
                     goal_threshold=.05,
                     time_step=.025,
                     real_time=False,
                     render=True,
                     use_predator=True)

# save_video_output(bot_evade, ".")

gc = belief.GaussianDiffusionComponent(.25 / 2.34 * .25)
dc = belief.DirectedDiffusionComponent(.20 / 2.34 * .25)

bs = belief.BeliefState(arena=bot_evade.arena,
                        occlusions=bot_evade.occlusions,
                        definition=100,
                        components=[gc, dc])


bot_evade.view.add_render_step(bs.render, z_index=200)
bot_evade.reset()
# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 50

action_count = len(bot_evade.loader.full_action_list)
visibility_polygon = bot_evade.visibility.get_visibility_polygon(src=bot_evade.prey.state.location,
                                                                 direction=bot_evade.prey.state.direction,
                                                                 view_field=360)
bs.update_visibility(visibility_polygon=visibility_polygon)


connection_graph = ct.Graph()
for cell in bot_evade.loader.world.cells:
    connection_graph.add_node(state=ct.State(cell.location.get_values()), label=cell.id)
cell_graph = cw.Graph.create_connection_graph(world=bot_evade.loader.world)
for cell in bot_evade.loader.world.cells:
    for conn in cell_graph[cell]:
        connection_graph.connect(cell.id, conn)
p = connection_graph.get_nearest(point=(.5, .5))
lppo_graph = connection_graph.get_subgraph(bot_evade.loader.lppo)


def reward_function(point, puff_probability):
    reward = 0
    _continue = True
    if Point.distance(src=point, dst=(1.0, 0.5)) < bot_evade.goal_threshold:
        reward = 1
        _continue = False
    return reward - puff_probability * 1000, _continue


algo = ct.TLPPO(graph=lppo_graph,
                puff_radius=bot_evade.puff_threshold,
                robot_belief_state=bs,
                reward_fn=reward_function,
                visibility=bot_evade.visibility,
                depth=20,
                budget=1000,
                speed=bot_evade.prey.max_forward_speed,
                navigation=bot_evade.loader.navigation)

print(p)
bs.tick()
steps = 0
tim_between_actions = 5
while bot_evade.running:
    algo.get_action(discount=.9,
                    location=bot_evade.prey.state.location,
                    exploration=.1)
    destination = random.choice(bot_evade.loader.open_locations)
    bot_evade.prey.set_destination(destination)
    last_destination_time += tim_between_actions

    for i in range(10):
        steps += 1
        bot_evade.step()
        if bot_evade.predator_visible:
            bs.update_other_location(bot_evade.predator.state.location)
            bs.update_self_location(bot_evade.prey.state.location)

    visibility_polygon = bot_evade.visibility.get_visibility_polygon(src=bot_evade.prey.state.location,
                                                                     direction=bot_evade.prey.state.direction,
                                                                     view_field=360)

    bs.update_visibility(visibility_polygon=visibility_polygon)
    # print(bs.get_probability((.5, .5), .1))
    bs.tick()

print(steps)