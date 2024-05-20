import pygame
import random
from cellworld_game import BotEvade, CoordinateConverter
import cellworld_tlppo.belief_state as belief

bot_evade = BotEvade(world_name="21_05",
                     puff_cool_down_time=.5,
                     puff_threshold=.1,
                     goal_threshold=.05,
                     time_step=.025,
                     real_time=True,
                     render=True,
                     use_predator=True)

gc = belief.GaussianDiffusionComponent(.20 / 2.34 * .25)
dc = belief.DirectedDiffusionComponent(.35 / 2.34 * .25)

bs = belief.BeliefState(arena=bot_evade.arena,
                        occlusions=bot_evade.occlusions,
                        definition=100,
                        components=[gc])


def render_bs(screen, coordinate_converter: CoordinateConverter):
    # Convert tensor to NumPy array and get max probability
    
    prob_matrix = bs.probability_distribution.cpu().numpy()
    max_prob = prob_matrix.max()
    if not max_prob:
        return
    cell_size = 10

    # Get dimensions
    height, width = prob_matrix.shape
    screen_width, screen_height = width * cell_size, height * cell_size

    # Precompute color surface for efficiency
    color_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
    color_surface.fill((255, 0, 0))

    for i in range(height):
        for j in range(width):
            prob = prob_matrix[i, j]
            alpha = int(255 * (prob / max_prob))
            color_surface.set_alpha(alpha)
            x, y = coordinate_converter.from_canonical((bs.points[i][j].x, bs.points[i][j].y))

            # Blit (copy) the colored surface onto the main screen
            screen.blit(color_surface, (x-cell_size/2, y-cell_size/2))


bot_evade.view.add_render_step(render_bs)
bot_evade.reset()
# prey
puff_cool_down = 0
last_destination_time = -3
random_actions = 5

action_count = len(bot_evade.loader.full_action_list)

while bot_evade.running:
    if bot_evade.time > last_destination_time + 2:
        if bot_evade.goal_achieved or random_actions == 0:
            destination = bot_evade.goal_location
            random_actions = 5
        else:
            random_actions -= 1
            destination = random.choice(bot_evade.loader.open_locations)

        bot_evade.prey.set_destination(destination)
        last_destination_time += 2
    bs.tick()
    bs.update_self_location(bot_evade.prey.state.location)
    if bot_evade.predator_visible:
        bs.update_other_location(bot_evade.predator.state.location)
    else:
        visibility_polygon, a = bot_evade.visibility.get_visibility_polygon(location=bot_evade.prey.state.location,
                                                                            direction=bot_evade.prey.state.direction,
                                                                            view_field=360)
        bs.update_visibility(visibility_polygon=visibility_polygon)
    bot_evade.step()

