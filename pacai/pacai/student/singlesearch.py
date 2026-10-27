"""
In this file, you will implement code relating to simple single-agent searches.
"""

import collections
import itertools
import random
import typing

import pacai.agents.searchproblem
import pacai.core.action
import pacai.core.agent
import pacai.core.agentaction
import pacai.core.board
import pacai.core.gamestate
import pacai.core.search
import pacai.pacman.board
import pacai.search.common
import pacai.search.distance
import pacai.search.food
import pacai.search.position

import pacai.util.containers

def graph_search(problem, fringe, heuristic = None):
    best_cost = {}
    expanded_cost = {}
    visited = set()
    counter = itertools.count()
    start_node = problem.get_starting_node()

    use_priority = isinstance(fringe, pacai.util.containers.PriorityQueue)

    def push(node, actions, cost):
        item = (next(counter), node, actions, cost)

        if use_priority:
            priority = cost
            # If a heuristic is provided (e.g. aster), then add it to the priority.
            if heuristic is not None:
                priority += heuristic(node, problem)
            fringe.push(item, priority)
        else:
            fringe.push(item)

    # Push the starting node onto the fringe.
    best_cost[start_node] = 0
    push(start_node, [], 0)

    # Main search loop.
    while not fringe.is_empty():
        (_, node, actions, cost) = fringe.pop()

        if (use_priority):
            if cost != best_cost.get(node):
                continue

            if node in expanded_cost and expanded_cost[node] <= cost:
                continue
            expanded_cost[node] = cost
        else:
            if node in visited:
                continue
            visited.add(node)

        if problem.is_goal_node(node):
            problem.complete(node)
            return pacai.core.search.SearchSolution(actions, cost, goal_node = node)

        for successor in problem.get_successor_nodes(node):
            new_cost = cost + successor.cost
            if (use_priority and new_cost < best_cost.get(successor.node, float('inf'))):
                best_cost[successor.node] = new_cost
                push(successor.node, actions + [successor.action], new_cost)
            elif (not use_priority and successor.node not in visited):
                push(successor.node, actions + [successor.action], new_cost)

    # Return an explicit unsuccessful solution when no goal lis reachable.
    return pacai.core.search.SearchSolution([], float('inf'), goal_node = None)

def depth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements depth first search (DFS).
    This means that it will search the deepest nodes in the search tree first.
    See: https://en.wikipedia.org/wiki/Depth-first_search .
    """

    # *** Your Code Here ***
    return graph_search(problem, pacai.util.containers.Stack())

def breadth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements breadth first search (BFS).
    This means that it will search nodes based on what level in search tree they appear.
    See: https://en.wikipedia.org/wiki/Breadth-first_search .
    """

    # *** Your Code Here ***
    return graph_search(problem, pacai.util.containers.Queue())

def uniform_cost_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements uniform cost search (UCS).
    This means that it will search nodes with a lower total cost first.
    See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs .
    """

    # *** Your Code Here ***
    return graph_search(problem, pacai.util.containers.PriorityQueue())

def astar_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements A* search (pronounced "A Star search").
    This means that it will search nodes with a lower combined cost and heuristic first.
    See: https://en.wikipedia.org/wiki/A*_search_algorithm .
    """

    # *** Your Code Here ***
    return graph_search(problem, pacai.util.containers.PriorityQueue(), heuristic)

class CornersSearchNode(pacai.core.search.SearchNode):
    """
    A search node the can be used to represent the corners search problem.

    You get to implement this search node however you want.
    """

    def __init__(self, state, parent=None, action=None, cost=0, depth=0) -> None:
        """ Construct a search node to help search for corners. """

        # *** Your Code Here ***
        # Remember that you can also add argument to your constructor.
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth

    def getState(self):
        return self.state

    def getPosition(self):
        return self.state[0]

    def getDepth(self):
        return self.depth

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, CornersSearchNode) and self.state == other.state

class CornersSearchProblem(pacai.core.search.SearchProblem[CornersSearchNode]):
    """
    A search problem for touching the four different corners in a board.

    You may assume that very board is surrounded by walls (e.g., (0, 0) is a wall),
    and that the position diagonally inside from the walled corner is the location we are looking for.
    For example, if we had a square board that was 10x10, then we would be looking for the following corners:
     - (1, 1) -- North-West / Upper Left
     - (1, 8) -- North-East / Upper Right
     - (8, 1) -- South-West / Lower Left
     - (8, 8) -- South-East / Lower Right
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            cost_function: pacai.core.search.CostFunction = pacai.search.common.unit_cost_function,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        # *** Your Code Here ***
        self.game_state = game_state
        self.start_position = game_state.get_agent_position(0)
        self.cost_function = cost_function

        self.corners = game_state.board.get_corners(offset = 1)

    def get_starting_node(self) -> CornersSearchNode:
        # *** Your Code Here ***
        visited = tuple(self.start_position == corner for corner in self.corners)
        start_state = (self.start_position, visited)

        return CornersSearchNode(
            state = start_state,
            parent = None,
            action = None,
            cost = 0,
            depth = 0
        )

    def is_goal_node(self, node: CornersSearchNode) -> bool:
        # *** Your Code Here ***
        (_, visited) = node.getState()
        return all(visited)

    def complete(self, goal_node: CornersSearchNode) -> None:
        self.position_history.append(goal_node.getPosition())

    def get_successor_nodes(self, node: CornersSearchNode) -> list[pacai.core.search.SuccessorInfo]:
        # *** Your Code Here ***
        successors = []
        (position, visited) = node.getState()
        board = self.game_state.board

        for action, next_pos in board.get_neighbors(position):
            new_visited = list(visited)
            if next_pos in self.corners:
                index = self.corners.index(next_pos)
                new_visited[index] = True
            new_visited = tuple(new_visited)

            new_state = (next_pos, new_visited)

            successor_node = CornersSearchNode(
                state = new_state,
                parent = node,
                action = action,
                cost = 0,
                depth = node.getDepth() + 1
            )
            successor_cost = self.cost_function(successor_node)
            successor_node.cost = successor_cost

            successors.append(
                pacai.core.search.SuccessorInfo(
                    successor_node,
                    action,
                    successor_cost
                )
            )

        self.expanded_node_count += 1
        if node not in self.visited_nodes:
            self.position_history.append(position)
            self.visited_nodes.add(node)

        return successors

def corners_heuristic(node: CornersSearchNode, problem: CornersSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for CornersSearchProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # *** Your Code Here ***
    (_, visited) = node.getState()
    remaining = [corner for (corner, was_visited) in zip(problem.corners, visited) if not was_visited]
    if (len(remaining) == 0):
        return 0

    # Find the cheapest ordering of the remaining corners in Manhattan distance.
    # Ignoring walls can only make this estimate smaller than the true path.
    return min(
        sum(
            pacai.search.distance.manhattan_distance(a, b)
            for (a, b) in zip(path, path[1:])
        )
        for path in ((node.getPosition(),) + order for order in itertools.permutations(remaining))
    )

def food_heuristic(node: pacai.search.food.FoodSearchNode, problem: pacai.search.food.FoodSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for the FoodSearchProblem.
    """

    # *** Your Code Here ***
    remaining = node.remaining_food
    if (len(remaining) == 0):
        return 0

    board = problem.state.board
    distance_cache = getattr(problem, '_food_heuristic_distance_cache', None)
    if distance_cache is None:
        distance_cache = {}
        problem._food_heuristic_distance_cache = distance_cache

    def distances_from(source):
        if source not in distance_cache:
            distance = {source: 0}
            fringe = collections.deque([source])
            while fringe:
                current = fringe.popleft()
                for (_, neighbor) in board.get_neighbors(current):
                    if neighbor not in distance:
                        distance[neighbor] = distance[current] + 1
                        fringe.append(neighbor)
            distance_cache[source] = distance
        return distance_cache[source]

    mst_cache = getattr(problem, '_food_heuristic_mst_cache', None)
    if mst_cache is None:
        mst_cache = {}
        problem._food_heuristic_mst_cache = mst_cache

    if remaining not in mst_cache:
        connected = {remaining[0]}
        unconnected = set(remaining[1:])
        mst_cost = 0
        while unconnected:
            best_edge = float('inf')
            best_food = None
            for connected_food in connected:
                food_distances = distances_from(connected_food)
                for food in unconnected:
                    distance = food_distances.get(food, float('inf'))
                    if distance < best_edge:
                        best_edge = distance
                        best_food = food
            if best_food is None:
                # Disconnected foods are not expected, but Manhattan remains admissible.
                best_food = min(unconnected)
                best_edge = min(
                    pacai.search.distance.manhattan_distance(connected_food, best_food)
                    for connected_food in connected
                )
            mst_cost += best_edge
            connected.add(best_food)
            unconnected.remove(best_food)
        mst_cache[remaining] = mst_cost

    current_distance = distances_from(node.position)
    nearest_food = min(current_distance.get(food, float('inf')) for food in remaining)
    if nearest_food == float('inf'):
        nearest_food = min(
            pacai.search.distance.manhattan_distance(node.position, food)
            for food in remaining
        )

    return nearest_food + mst_cache[remaining]


class ClosestDotSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    Search for a path to all the food by greedily searching for the next closest food again and again
    (util we have reached all the food).

    This agent is left to you to fill out.
    But make sure to take your time and think.
    The final solution is quite simple if you take your time to understand everything up until this point and leverage
    pacai.agents.searchproblem.GreedySubproblemSearchAgent and pacai.student.problem.AnyMarkerSearchProblem.
    pacai.agents.searchproblem.GreedySubproblemSearchAgent is already implemented,
    but you should take some time to understand it.
    pacai.student.problem.AnyMarkerSearchProblem (below in this file) has not yet been implemented,
    but is the quickest and easiest way to implement this class.

    Hint:
    Remember that you can call a parent class' `__init__()` method from a child class' `__init__()` method.
    (See pacai.student.problem.AnyMarkerSearchProblem for an example.)
    Child classes will generally always call their parent's `__init__()` method
    (if a child class does not implement `__init__()`, then the parent's `__init__()` is automatically called).
    This call does not need to be the first line in the method,
    and you can pass whatever you want to the parent's `__init__()`.
    """

    # *** Your Code Here ***
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(
            problem = AnyMarkerSearchProblem,
            solver = breadth_first_search,
            **kwargs
        )

class AnyMarkerSearchProblem(pacai.search.position.PositionSearchProblem):
    """
    A search problem for finding a path to any instance of the specified board marker (e.g., food, wall, power capsule).

    This search problem is just like the pacai.search.position.PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    You may modify the `__init__()` if you want, the other methods should be fine as-is.
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            target_marker: pacai.core.board.Marker = pacai.pacman.board.MARKER_PELLET,
            **kwargs: typing.Any) -> None:
        super().__init__(game_state, **kwargs)

        # *** Your Code Here ***
        self.target_marker = target_marker

    def is_goal_node(self, node: pacai.search.position.PositionSearchNode) -> bool:
        # *** Your Code Here ***
        return self.target_marker in self.board.get(node.position)

class ApproximateSearchAgent(pacai.core.agent.Agent):
    """
    A search agent that tries to perform an approximate search instead of an exact one.
    In other words, this agent is okay with a solution that is "good enough" and not necessarily optimal.
    """
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._closest_dot_agent = ClosestDotSearchAgent(**kwargs)

    def game_start_full(self,
                       agent_index: int,
                       suggested_seed: int,
                       initial_state: pacai.core.gamestate.GameState,
                       ) -> pacai.core.agentaction.AgentAction:
        super().game_start_full(agent_index, suggested_seed, initial_state)
        return self._closest_dot_agent.game_start_full(agent_index, suggested_seed, initial_state)

    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        return self._closest_dot_agent.get_action(state)