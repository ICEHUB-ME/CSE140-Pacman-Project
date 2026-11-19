import typing
from collections import deque

import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.board
import pacai.core.gamestate
import pacai.pacman.board


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """Create one food attacker and one home-territory defender."""

    return [
        pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.OffensiveCaptureAgent"),
        pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.OffensiveCaptureAgent"),
    ]


class _CaptureAgent(pacai.core.agent.Agent):
    """Shared maze-distance and movement helpers for the student agents."""

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._distances: dict[
            pacai.core.board.Position,
            dict[pacai.core.board.Position, int]
        ] = {}
        self._current_board_width = 0
        self._current_board_height = 0
        self._beliefs: dict[int, dict[pacai.core.board.Position, float]] = {}
        self._belief_confidence: dict[int, float] = {}
        self._last_home_food: set[pacai.core.board.Position] = set()

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._current_board_width = initial_state.board.width
        self._current_board_height = initial_state.board.height
        positions = [
            pacai.core.board.Position(row, col)
            for row in range(initial_state.board.height)
            for col in range(initial_state.board.width)
            if not initial_state.board.is_wall(pacai.core.board.Position(row, col))
        ]

        # Run one breadth-first search from each open cell. This gives us maze
        # distances with predictable startup cost on the larger contest boards.
        for start in positions:
            distances = {start: 0}
            frontier = deque([start])
            while (frontier):
                current = frontier.popleft()
                next_distance = distances[current] + 1
                for (_, neighbor) in initial_state.board.get_neighbors(current):
                    if (neighbor in distances):
                        continue
                    distances[neighbor] = next_distance
                    frontier.append(neighbor)
            self._distances[start] = distances

        self._last_home_food = {
            food for food in initial_state.board.get_marker_positions(pacai.pacman.board.MARKER_PELLET)
            if self._is_home_side(food)
        }
        self._update_opponent_beliefs(
            typing.cast(pacai.capture.gamestate.GameState, initial_state),
            use_food_evidence = False)

    def _distance(self,
            first: pacai.core.board.Position,
            second: pacai.core.board.Position) -> float:
        return float(self._distances.get(first, {}).get(second, 1000))

    def _is_home_side(self, position: pacai.core.board.Position) -> bool:
        is_left_side = position.col < (self._current_board_width / 2)
        return is_left_side == (self.agent_index % 2 == 0)

    @staticmethod
    def _normalize_belief(
            weights: dict[pacai.core.board.Position, float]) -> dict[pacai.core.board.Position, float]:
        if (not weights):
            return {}

        # Bound work and memory if an opponent stays hidden for a long time.
        if (len(weights) > 250):
            weights = dict(sorted(weights.items(), key = lambda item: item[1], reverse = True)[:250])

        total = sum(weights.values())
        if (total <= 0.0):
            return {}

        return {position: weight / total for (position, weight) in weights.items()}

    def _diffuse_belief(self,
            belief: dict[pacai.core.board.Position, float],
            board: pacai.core.board.Board) -> dict[pacai.core.board.Position, float]:
        propagated: dict[pacai.core.board.Position, float] = {}
        for position, probability in belief.items():
            neighbors = [next_position for (_, next_position) in board.get_neighbors(position)]
            destinations = [position, *neighbors]
            share = probability / len(destinations)
            for destination in destinations:
                propagated[destination] = propagated.get(destination, 0.0) + share

        return self._normalize_belief(propagated)

    def _update_opponent_beliefs(self,
            state: pacai.capture.gamestate.GameState,
            use_food_evidence: bool = True) -> None:
        visible_positions = state.get_opponent_positions(agent_index = self.agent_index)
        opponent_indexes = [
            index for index in state.get_agent_indexes()
            if (index % 2) != (self.agent_index % 2)
        ]

        for opponent_index in opponent_indexes:
            if (opponent_index in visible_positions):
                position = visible_positions[opponent_index]
                self._beliefs[opponent_index] = {position: 1.0}
                self._belief_confidence[opponent_index] = 1.0
            else:
                previous_belief = self._beliefs.get(opponent_index, {})
                if (previous_belief):
                    self._beliefs[opponent_index] = self._diffuse_belief(previous_belief, state.board)
                else:
                    start_position = state.board.get_agent_initial_position(
                        pacai.core.board.Marker(str(opponent_index)))
                    if (start_position is not None):
                        self._beliefs[opponent_index] = {start_position: 1.0}
                        self._belief_confidence[opponent_index] = 0.75
                self._belief_confidence[opponent_index] = (
                    0.97 * self._belief_confidence.get(opponent_index, 0.0))

        if (not use_food_evidence):
            return

        current_home_food = {
            food for food in state.board.get_marker_positions(pacai.pacman.board.MARKER_PELLET)
            if self._is_home_side(food)
        }
        newly_missing = self._last_home_food - current_home_food
        self._last_home_food = current_home_food
        if (not newly_missing):
            return

        evidence: dict[pacai.core.board.Position, float] = {}
        for food_position in newly_missing:
            evidence[food_position] = evidence.get(food_position, 0.0) + 3.0
            for (_, neighbor) in state.board.get_neighbors(food_position):
                evidence[neighbor] = evidence.get(neighbor, 0.0) + 1.0
        evidence = self._normalize_belief(evidence)

        for opponent_index in opponent_indexes:
            if (opponent_index in visible_positions):
                continue

            old_belief = self._beliefs.get(opponent_index, {})
            combined = {position: 0.35 * probability for (position, probability) in old_belief.items()}
            for (position, probability) in evidence.items():
                combined[position] = combined.get(position, 0.0) + 0.65 * probability
            self._beliefs[opponent_index] = self._normalize_belief(combined)
            self._belief_confidence[opponent_index] = max(
                self._belief_confidence.get(opponent_index, 0.0), 0.8)

    def _likely_invaders(self,
            state: pacai.capture.gamestate.GameState) -> tuple[dict[pacai.core.board.Position, float], float]:
        visible_invaders = state.get_invader_positions(agent_index = self.agent_index)
        if (visible_invaders):
            visible_belief = {position: 1.0 for position in visible_invaders.values()}
            return self._normalize_belief(visible_belief), 1.0

        belief: dict[pacai.core.board.Position, float] = {}
        confidence = 0.0
        visible_positions = state.get_opponent_positions(agent_index = self.agent_index)
        for opponent_index, positions in self._beliefs.items():
            if (opponent_index in visible_positions):
                continue

            confidence_weight = self._belief_confidence.get(opponent_index, 0.0)
            for (position, probability) in positions.items():
                if (self._is_home_side(position)):
                    weighted_probability = probability * confidence_weight
                    belief[position] = belief.get(position, 0.0) + weighted_probability
                    confidence += weighted_probability

        if (confidence < 0.12):
            return {}, 0.0

        return self._normalize_belief(belief), min(confidence, 1.0)

    def _hidden_ghost_risk(self,
            state: pacai.capture.gamestate.GameState,
            candidate: pacai.core.board.Position) -> float:
        visible_positions = state.get_opponent_positions(agent_index = self.agent_index)
        risk = 0.0
        for opponent_index, positions in self._beliefs.items():
            if (opponent_index in visible_positions):
                continue

            for (ghost_position, probability) in positions.items():
                if (self._is_home_side(ghost_position)):
                    continue

                distance = self._distance(candidate, ghost_position)
                if (distance <= 2):
                    risk += (probability * self._belief_confidence.get(opponent_index, 0.0)
                            * (3 - distance) / 3.0)

        return min(risk, 1.0)

    def _select_action(self,
            state: pacai.capture.gamestate.GameState,
            action_scores: dict[pacai.core.action.Action, float]) -> pacai.core.action.Action:
        legal_actions = state.get_legal_actions()
        if (not legal_actions):
            return pacai.core.action.STOP

        previous_actions = state.get_agent_actions(self.agent_index)
        reverse_action = None
        if (previous_actions):
            reverse_action = state.get_reverse_action(previous_actions[-1])

        best_action = legal_actions[0]
        best_score = float('-inf')

        for action in legal_actions:
            score = action_scores.get(action, 0.0)
            if (action == reverse_action):
                score -= 0.15

            if (score > best_score):
                best_score = score
                best_action = action

        return best_action


class OffensiveCaptureAgent(_CaptureAgent):
    """Attack food and defend home territory when an invader may be present."""

    def __init__(self, row_region: int = -1, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._food_target: pacai.core.board.Position | None = None
        self._row_region = row_region

    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        state = typing.cast(pacai.capture.gamestate.GameState, state)
        self._update_opponent_beliefs(state)
        position = state.get_agent_position(self.agent_index)
        if (position is None):
            # A respawn should start a fresh route from the spawn point instead
            # of sending the attacker back toward the same dangerous target.
            self._food_target = None
            return pacai.core.action.STOP

        legal_actions = state.get_legal_actions()
        if (not legal_actions):
            return pacai.core.action.STOP

        if (self._is_home_side(position)):
            invader_belief, confidence = self._likely_invaders(state)
            # Diffused spawn beliefs can leak across the midline and should not
            # make the attacker abandon its route. Visible invaders still have
            # confidence 1.0; strong food-loss evidence also clears this bar.
            if (confidence >= 0.65):
                scores: dict[pacai.core.action.Action, float] = {}
                scared = state.is_scared(self.agent_index)
                for action in legal_actions:
                    next_position = position.apply_action(action)
                    expected_distance = sum(
                        probability * self._distance(next_position, invader)
                        for (invader, probability) in invader_belief.items())
                    score = (3.0 if scared else -3.0) * confidence * expected_distance
                    if (not scared):
                        score += 100.0 * confidence * invader_belief.get(next_position, 0.0)
                    if (not self._is_home_side(next_position)):
                        score -= 30.0
                    scores[action] = score

                return self._select_action(state, scores)

        food_positions = state.get_food(agent_index = self.agent_index)
        if (self._row_region in {0, 1}):
            region_food = {
                food for food in food_positions
                if int(food.row >= (state.board.height / 2)) == self._row_region
            }
            if (region_food):
                food_positions = region_food
        if (not food_positions):
            return legal_actions[0]

        dangerous_ghosts = {
            opponent_index: opponent_position
            for (opponent_index, opponent_position) in state.get_nonscared_opponent_positions(
                    agent_index = self.agent_index).items()
            if state.is_ghost(opponent_index)
        }

        def target_cost(food: pacai.core.board.Position) -> float:
            cost = self._distance(position, food)
            if (not self._is_home_side(food)):
                for ghost_position in dangerous_ghosts.values():
                    ghost_distance = self._distance(food, ghost_position)
                    if (ghost_distance <= 4):
                        cost += 2.0 * (5 - ghost_distance)
                cost += 4.0 * self._hidden_ghost_risk(state, food)
            return cost

        nearest_food = min(food_positions, key = target_cost)
        if ((self._food_target not in food_positions)
                or (target_cost(nearest_food) + 3 < target_cost(
                    typing.cast(pacai.core.board.Position, self._food_target)))):
            self._food_target = nearest_food
        target = typing.cast(pacai.core.board.Position, self._food_target)

        capsule_positions = {
            capsule for capsule in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
            if not self._is_home_side(capsule)
        }
        closest_ghost_distance = min(
            (self._distance(position, ghost) for ghost in dangerous_ghosts.values()),
            default = 1000.0)

        # Treat capsules as tactical objectives when a defender is approaching.
        # Prefer capsules that are reachable soon and not already guarded by a ghost.
        if ((closest_ghost_distance <= 8) and capsule_positions):
            def capsule_cost(capsule: pacai.core.board.Position) -> float:
                ghost_distance = min(
                    (self._distance(capsule, ghost) for ghost in dangerous_ghosts.values()),
                    default = 1000.0)
                guarded_penalty = max(0.0, 5.0 - ghost_distance) * 1.5
                return self._distance(position, capsule) + guarded_penalty

            capsule = min(capsule_positions, key = capsule_cost)
            if (capsule_cost(capsule) <= self._distance(position, target) + 5):
                target = capsule

        scores: dict[pacai.core.action.Action, float] = {}
        for action in legal_actions:
            next_position = position.apply_action(action)
            distance_to_target = self._distance(next_position, target)
            score = -2.0 * distance_to_target

            if (next_position in food_positions):
                score += 16.0
            if (next_position in capsule_positions):
                score += 20.0

            for ghost_position in dangerous_ghosts.values():
                if (self._is_home_side(next_position)):
                    continue

                ghost_distance = self._distance(next_position, ghost_position)
                if (ghost_distance <= 1):
                    score -= 100.0
                elif (ghost_distance == 2):
                    score -= 35.0
                elif (ghost_distance == 3):
                    score -= 12.0
                elif (ghost_distance == 4):
                    score -= 4.0

            if (not self._is_home_side(next_position)):
                score -= 100.0 * self._hidden_ghost_risk(state, next_position)

            scores[action] = score

        return self._select_action(state, scores)

"""
class DefensiveCaptureAgent(_CaptureAgent):
    Guard home territory, chase likely invaders, and patrol when it is clear.

    def _patrol_target(self) -> pacai.core.board.Position | None:
        Choose an open home-side tile near the center line, favoring mid-board rows.
        half = self._current_board_width / 2
        patrol_col = int(half) - 1 if self.agent_index % 2 == 0 else int(half)
        center_row = self._current_board_height / 2
        candidates = [
            position
            for position in self._distances
            if position.col == patrol_col and self._is_home_side(position)
        ]
        if not candidates:
            return None
        return min(candidates, key = lambda item: abs(item.row - center_row))

    def get_action(self, state: pacai.core.gamestate.GameState) -> pacai.core.action.Action:
        state = typing.cast(pacai.capture.gamestate.GameState, state)
        self._update_opponent_beliefs(state)
        position = state.get_agent_position(self.agent_index)
        if position is None:
            return pacai.core.action.STOP

        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP

        invader_belief, confidence = self._likely_invaders(state)
        scared = state.is_scared(self.agent_index)
        patrol_target = self._patrol_target()
        scores: dict[pacai.core.action.Action, float] = {}

        for action in legal_actions:
            next_position = position.apply_action(action)
            if confidence > 0.0:
                expected_distance = sum(
                    probability * self._distance(next_position, invader)
                    for invader, probability in invader_belief.items())
                # Chase when safe; keep distance from invaders while scared.
                score = (2.5 if scared else -2.5) * confidence * expected_distance
                if not scared:
                    score += 100.0 * confidence * invader_belief.get(next_position, 0.0)
            elif patrol_target is not None:
                score = -2.0 * self._distance(next_position, patrol_target)
            else:
                score = 0.0

            if not self._is_home_side(next_position):
                score -= 50.0
            scores[action] = score

        return self._select_action(state, scores)
"""