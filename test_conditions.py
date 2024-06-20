import cellworld_belief as belief
import cellworld_game as game
import cellworld_gym as cg
import cellworld_tlppo as ct


def get_belief_state_components(condition: int):
    NB = belief.NoBeliefComponent()
    LOS = belief.LineOfSightComponent(other_scale=.5)
    V = belief.VisibilityComponent()
    D = belief.DiffusionComponent()
    GD = belief.GaussianDiffusionComponent()
    DD = belief.DirectedDiffusionComponent()
    O = belief.OcclusionsComponent()
    A = belief.ArenaComponent()
    M = belief.MapComponent()
    NL = belief.ProximityComponent()

    components = []
    if condition == 0:
        components = [NL, M]
    if condition == 1:
        components = [NB, LOS, M]
    elif condition == 2:
        components = [V, LOS, M]
    elif condition == 3:
        components = [GD, V, LOS, M]
    elif condition == 4:
        components = [DD, V, LOS, M]
    elif condition == 5:
        components = [DD, GD, V, LOS, M]
    return components


def get_tlppo(depth: int,
              budget: int,
              environment: cg.BotEvadeBeliefEnv) -> ct.TLPPO:
    def reward_function(point, puff_probability):
        distance_to_goal = game.Point.distance(src=point, dst=environment.model.goal_location)
        reward = - distance_to_goal * 10 - puff_probability * 1000
        return reward, distance_to_goal > environment.model.goal_threshold

    connection_graph = ct.Graph(nodes={label: ct.State(point)
                                for label, point
                                in enumerate(environment.model.loader.full_action_list)})

    for src_label, cnn in enumerate(environment.model.loader.options_graph):
        connection_graph.connect(src_label=src_label, dst_label=src_label)
        for dst_label in cnn:
            connection_graph.connect(src_label=src_label, dst_label=dst_label, bi=False)

    algo = ct.TLPPO(graph=connection_graph,
                    puff_radius=environment.model.puff_threshold,
                    robot_belief_state=environment.belief_state,
                    reward_fn=reward_function,
                    depth=depth,
                    budget=budget,
                    speed=environment.model.prey.max_forward_speed * environment.time_step,
                    navigation=environment.loader.navigation)

    return algo