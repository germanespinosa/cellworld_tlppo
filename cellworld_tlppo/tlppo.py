import math
import typing
import cellworld_game as cg
import torch

from .graph import Graph
from .mcts import Tree, TreeNode
from cellworld_belief import BeliefState
from .steps import Steps


class TLPPO:
    def __init__(self,
                 graph: Graph,
                 navigation: cg.Navigation,
                 robot_belief_state: BeliefState,
                 reward_fn: typing.Callable[[typing.Tuple[float, float], float], typing.Tuple[float, bool]],
                 budget: int = 100,
                 depth: int = 20,
                 puff_radius: float = 0.1,
                 speed: float = .25):
        self.graph = graph
        self.navigation = navigation
        self.robot_belief_state = robot_belief_state
        self.budget = budget
        self.depth = depth
        self.speed = speed
        self.reward_fn = reward_fn
        self.puff_radius = puff_radius
        self.render_active = False
        self.render_best_plan = False
        if self.robot_belief_state.model.render:
            import pygame
            def on_key_down(key: pygame.key):
                if key == pygame.K_t:
                    self.render_active = not self.render_active
                if key == pygame.K_x:
                    self.render_best_plan = not self.render_best_plan

            self.robot_belief_state.model.view.add_event_handler("key_down", on_key_down)
            self.robot_belief_state.model.view.add_render_step(self.render, z_index=26)
            self.robot_belief_state.model.view.add_render_step(self.render_plan, z_index=206)
        self.last_tree = None
        self.last_point = None

    def evaluate_trajectory(self,
                            src: cg.Point.type,
                            trajectory: typing.List[cg.Point.type],
                            evaluation_cache: typing.Dict[cg.Line.type, typing.Tuple[float, bool]]) -> float:
        iteration_reward: float = 0

        for step in trajectory:
            line = (src, step)
            if line in evaluation_cache:
                step_reward, _continue = evaluation_cache[line]
            else:
                robot_probability = self.robot_belief_state.get_probability_in_distance_to_segment(src=src,
                                                                                                   dst=step,
                                                                                                   distance=self.puff_radius)
                step_reward, _continue = self.reward_fn(step,
                                                        robot_probability)
                evaluation_cache[line] = (step_reward, _continue)

            iteration_reward += step_reward
            if not _continue:
                break
            src = step
        return iteration_reward

    def get_trajectory(self,
                       node: TreeNode,
                       exploration: float) -> typing.Tuple[typing.List[cg.Point.type], TreeNode]:
        src = node.state.point
        step_count = 0
        step_remainder = 0
        trajectory: typing.List[cg.Point.type] = []
        _continue = True
        while _continue and step_count < self.depth:
            node = node.select(c=exploration)
            dst = node.state.point
            path = self.navigation.get_path(src=src,
                                            dst=dst)
            steps = Steps(start=src,
                          stops=path,
                          step_size=self.speed,
                          pending=step_remainder)
            for step in steps:
                trajectory.append(step)
                step_count += 1
                a, _continue = self.reward_fn(step, 0)
                if not _continue or step_count >= self.depth:
                    break
            step_remainder = steps.pending
            src = dst
        return trajectory, node

    def get_action(self,
                   point: typing.Tuple[float, float],
                   exploration: float = math.sqrt(2),
                   discount: float = .1):

        evaluation_cache: typing.Dict[cg.Line.type, typing.Tuple[float, bool]] = {}
        self.last_point = point

        if self.last_tree is None:
            tree = Tree(graph=self.graph,
                        point=point)
        else:
            # evaluates previous plan
            previous_reward_value = self.last_tree.root.value

            self.last_tree.move(point=point)

            trajectory, destination_node = self.get_trajectory(node=self.last_tree.root,
                                                               exploration=0)
            iteration_reward = self.evaluate_trajectory(src=point,
                                                        trajectory=trajectory,
                                                        evaluation_cache=evaluation_cache)

            destination_node.propagate_reward(reward=iteration_reward,
                                              discount=discount,
                                              initial_value=True)

            if self.last_tree.root.value >= previous_reward_value:
                tree = self.last_tree
            else:
                tree = Tree(graph=self.graph,
                            point=point)

        for i in range(self.budget):
            trajectory, destination_node = self.get_trajectory(node=tree.root,
                                                               exploration=exploration)
            iteration_reward = self.evaluate_trajectory(src=point, trajectory=trajectory,
                                                        evaluation_cache=evaluation_cache)
            destination_node.propagate_reward(reward=iteration_reward,
                                              discount=discount)
        self.last_tree = tree

        node = self.last_tree.root.get_best()
        if cg.Point.distance(node.state.point, point) < self.speed:
            if node.children:
                node = node.get_best()

        return node.label

    def render(self, screen, coordinate_converter: cg.CoordinateConverter):
        if not self.render_active:
            return
        import pygame
        # Create a surface capable of handling alpha

        trajectory, destination_node = self.get_trajectory(node=self.last_tree.root,
                                                           exploration=0)
        src = self.last_tree.root.state.point
        robot_probability = torch.zeros(self.robot_belief_state.shape, dtype=torch.bool, device=self.robot_belief_state.device)
        for step in trajectory:
            robot_probability |= torch.logical_or(robot_probability,
                                                  self.robot_belief_state.get_mask_in_distance_to_segment(src=src,
                                                                                                          dst=step,
                                                                                                          distance=self.puff_radius))
            src = step

        heatmap_surface = pygame.Surface(robot_probability.shape[::-1], pygame.SRCALPHA)
        pix_array = pygame.PixelArray(heatmap_surface)
        for y in range(robot_probability.shape[1]):
            for x in range(robot_probability.shape[0]):
                value = robot_probability[robot_probability.shape[0] - x - 1, y]
                pix_array[y, x] = (0, 0, 255, 30 if value else 0)

        # Delete the pixel array to unlock the surface
        del pix_array

        # Scale the surface to the window size
        scaled_heatmap = pygame.transform.scale(heatmap_surface,
                                                size=(coordinate_converter.screen_width,
                                                      coordinate_converter.screen_height))
        screen.blit(scaled_heatmap, (0, 0))

    def render_plan(self,
                    surface,
                    coordinate_converter: cg.CoordinateConverter):
        import pygame
        surface: pygame.Surface
        size = 30
        font = pygame.font.Font(None, size)
        if self.render_best_plan and self.last_tree:
            trajectory, destination_node = self.get_trajectory(node=self.last_tree.root,
                                                               exploration=0)
            current_step_point = coordinate_converter.from_canonical(self.last_point)
            step_number = 0
            for step in trajectory:
                step_point = coordinate_converter.from_canonical(step)
                pygame.draw.line(surface,
                                 (0, 180, 0),
                                 current_step_point,
                                 step_point,
                                 2)
                pygame.draw.circle(surface=surface,
                                   color=(0, 0, 255),
                                   center=step_point,
                                   radius=5,
                                   width=2)
                text = font.render(f'{step_number}', True, (0, 0, 0))
                surface.blit(text, cg.Point.add(step_point, (size//2, -size//2)))
                current_step_point = step_point
                step_number += 1
