import cellworld_gym as cg
from cellworld_game import save_video_output, Point, CoordinateConverter
import cellworld_tlppo.belief_state as belief
import cellworld_tlppo as ct
import cellworld as cw


env = cg.BotEvadeEnv(world_name="21_05",
                     real_time=False,
                     render=True,
                     use_lppos=False,
                     use_predator=True)


save_video_output(env.model, "./videos")

gc = belief.GaussianDiffusionComponent(.30 / 2.34 * .25)
dc = belief.DirectedDiffusionComponent(.20 / 2.34 * .25)

bs = belief.BeliefState(arena=env.model.arena,
                        occlusions=env.model.occlusions,
                        definition=100,
                        components=[gc, dc],
                        other_size=6)

obs: cg.BotEvadeObservation
# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 50

connection_graph = ct.Graph(nodes={label: ct.State(point)
                                   for label, point
                                   in enumerate(env.model.loader.full_action_list)})

for src_label, cnn in enumerate(env.model.loader.options_graph):
    connection_graph.connect(src_label=src_label, dst_label=src_label)
    for dst_label in cnn:
        connection_graph.connect(src_label=src_label, dst_label=dst_label, bi=False)


def reward_function(point, puff_probability):
    distance_to_goal = Point.distance(src=point, dst=(1.0, 0.5))
    reward = - distance_to_goal * 10 - puff_probability * 1000
    return reward, distance_to_goal > env.model.goal_threshold


algo = ct.TLPPO(graph=connection_graph,
                puff_radius=env.model.puff_threshold,
                robot_belief_state=bs,
                reward_fn=reward_function,
                visibility=env.model.visibility,
                depth=2,
                budget=50,
                speed=env.model.prey.max_forward_speed * env.time_step,
                navigation=env.loader.navigation)

steps = []
rewards = []
tree = None

show_steps = False

if env.model.render:

    def get_probability(a, b):
        print(f" {b} - probability: ", algo.robot_belief_state.get_probability_in_radius(point=b,
                                                                                         radius=env.model.puff_threshold))


    env.model.view.add_render_step(bs.render, z_index=5)
    env.model.view.on_mouse_button_up = get_probability

    def render_graph(surface, coordinate_converter: CoordinateConverter, node: ct.TreeNode = None):
        if tree is None:
            return

        import pygame
        if node is None:
            node = tree.root
            pygame.draw.circle(surface=surface,
                               color=(100, 0, 100),
                               center=coordinate_converter.from_canonical(tree.root.state.point),
                               radius=20,
                               width=2)

        radius = 10 #coordinate_converter.from_canonical(env.model.puff_threshold)
        font = pygame.font.Font(None, 20)
        location = coordinate_converter.from_canonical(node.state.point)
        if node.visits:
            text_surface = font.render(f'{int(node.value)}', True, (0, 0, 0))
            surface.blit(text_surface, location)
            pygame.draw.circle(surface=surface,
                               color=(80, 80, 0),
                               center=location,
                               radius=radius,
                               width=2)
            for child in node.children:
                dst = coordinate_converter.from_canonical(child.state.point)
                pygame.draw.line(surface,
                                 (120, 120, 0),
                                 location,
                                 dst,
                                 2)

                pygame.draw.circle(surface=surface,
                                   color=(0, 100, 0),
                                   center=dst,
                                   radius=20,
                                   width=2)
            action_point = coordinate_converter.from_canonical(tree.root.get_best().state.point)
            pygame.draw.circle(surface=surface,
                               color=(200, 120, 40),
                               center=action_point,
                               radius=radius,
                               width=5)
            for child in node.children:
                dst = coordinate_converter.from_canonical(child.state.point)
                if child.visits:
                    text_surface = font.render(f'{int(child.value)}', True, (0, 100, 0))
                    surface.blit(text_surface, dst)
                else:
                    text_surface = font.render('NA', True, (0, 100, 0))
                    surface.blit(text_surface, dst)

                # if child.data:
                #     steps, rewards = child.data
                #     step = steps[0]
                #     reward = rewards[0]
                #     location = coordinate_converter.from_canonical(step)
                #     text_surface = font.render(f'{int(reward)}', True, (0, 0, 0))
                #     surface.blit(text_surface, location)
                #
                #     pygame.draw.circle(surface=surface,
                #                        color=(120, 120, 0),
                #                        center=location,
                #                        radius=20,
                #                        width=5)
                # if child.data:
                #     steps, rewards = child.data
                #     for step, reward in zip(steps, rewards):
                #         text_surface = font.render(f'{int(reward)}', True, (0, 100, 0))
                #         surface.blit(text_surface, location)
                #         location = coordinate_converter.from_canonical(step)
                #         pygame.draw.circle(surface=surface,
                #                            color=(0, 100, 0),
                #                            center=location,
                #                            radius=20,
                #                            width=2)
                # render_graph(surface=surface, coordinate_converter=coordinate_converter, node=child)
        else:
            pygame.draw.circle(surface=surface,
                               color=(80, 80, 80),
                               center=location,
                               radius=radius//3,
                               width=2)

    def render_steps(surface, coordinate_converter: CoordinateConverter):

        import pygame
        if tree is None:
            return
        steps, rewards = [], []
        node = tree.root
        while node.visits:
            node = node.select(0)
            if node.data:
                steps, rewards = node.data

        font = pygame.font.Font(None, 20)

        pygame.draw.circle(surface=surface,
                           color=(120, 0, 120),
                           center=coordinate_converter.from_canonical(tree.root.state.point),
                           radius=20,
                           width=5)

        for step, reward in zip(steps, rewards):
            location = coordinate_converter.from_canonical(step)
            text_surface = font.render(f'{int(reward)}', True, (0, 0, 0))
            surface.blit(text_surface, location)

            pygame.draw.circle(surface=surface,
                               color=(120, 120, 0),
                               center=location,
                               radius=20,
                               width=5)

    # env.model.view.add_render_step(render_graph, z_index=205)
    # env.model.view.add_render_step(render_steps, z_index=90)

bs.tick()
for i in range(100):
    print(i)
    obs, _ = env.reset()
    bs.reset()
    finished, truncated = False, False
    puff_count = 0
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

        if not env.model.paused:
            bs.tick()
            tree = algo.get_action(point=prey_location, discount=0.0, previous_tree=tree)
            action = tree.root.get_best().label
            obs, reward, finished, truncated, info = env.step(action=action)
            show_steps = False
        else:
            env.model.step()
