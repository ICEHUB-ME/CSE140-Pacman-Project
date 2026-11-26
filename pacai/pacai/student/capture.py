import typing
from collections import deque

import pacai.capture.gamestate
import pacai.core.action
import pacai.core.agent
import pacai.core.agentinfo
import pacai.core.board
import pacai.core.gamestate
import pacai.core.features
import pacai.pacman.board
import pacai.student.learning


# Belief tracking: cap state growth while retaining useful uncertainty, decay
# stale sightings slowly, and let direct food-loss evidence outweigh diffusion.
DISTANCE_FALLBACK = 1000.0  # Large sentinel for unreachable/missing maze distances.
MAX_BELIEF_POSITIONS = 250  # Bounds belief work without pruning ordinary game states.
INITIAL_HIDDEN_CONFIDENCE = 0.75  # A spawn is plausible, but not a confirmed location.
BELIEF_CONFIDENCE_DECAY = 0.97  # Gradual decay keeps recently hidden opponents relevant.
FOOD_LOSS_EVIDENCE_WEIGHT = 3.0  # Missing food strongly points to its own tile.
FOOD_NEIGHBOR_EVIDENCE_WEIGHT = 1.0  # Adjacent tiles remain possible with weaker evidence.
OLD_BELIEF_WEIGHT = 0.35  # Preserve some prior uncertainty when food disappears.
NEW_EVIDENCE_WEIGHT = 0.65  # Favor fresh food-loss evidence over the older estimate.
FOOD_EVIDENCE_CONFIDENCE_FLOOR = 0.8  # Strong evidence should trigger meaningful response.
MIN_INVADER_CONFIDENCE = 0.12  # Ignore tiny probabilities created by belief diffusion.
HIDDEN_GHOST_RISK_RADIUS = 2  # Risk is local; farther hidden ghosts have little immediate threat.
HIDDEN_GHOST_RISK_SCALE = 3.0  # Converts the two-tile proximity gradient to a 0–1 range.

# Action scoring: these weights express the relative value of objectives and risk.
REVERSE_ACTION_PENALTY = 0.15  # Slightly discourage oscillation without forbidding retreat.
DEFENSE_CONFIDENCE_THRESHOLD = 0.65  # Switch roles only when evidence is reasonably strong.
DEFENSE_DISTANCE_WEIGHT = 3.0  # Scale pursuit distance enough to prioritize nearby invaders.
DEFENSE_CAPTURE_BONUS = 100.0  # Strongly favor a move that lands on a likely invader.
DEFENSE_CROSSING_PENALTY = 30.0  # Keep a defender on its home side absent a compelling chase.
FOOD_GHOST_ALERT_DISTANCE = 4  # Treat ghosts within four maze steps as threatening a food target.
FOOD_GHOST_DISTANCE_PENALTY = 2.0  # Raise the cost for each step a ghost approaches the food.
FOOD_GHOST_RISK_WEIGHT = 4.0  # Make uncertain hidden-ghost danger affect target selection.
FOOD_TARGET_SWITCH_MARGIN = 3.0  # Avoid changing targets for a negligible path-cost improvement.
TEAM_FOOD_REGION_PENALTY = 16.0  # Keep attackers in separate board halves when their assigned half is empty.
GHOST_DISTANCE_FALLBACK = DISTANCE_FALLBACK  # Same large sentinel when no dangerous ghost is visible.
CAPSULE_GHOST_ALERT_DISTANCE = 8  # Consider a capsule when a defender is within practical reach.
CAPSULE_GUARD_RADIUS = 5.0  # Ghosts within this range make a capsule increasingly risky.
CAPSULE_GUARD_PENALTY_WEIGHT = 1.5  # Moderate the capsule penalty so guarded capsules can still be chosen.
CAPSULE_EXTRA_TRAVEL_ALLOWANCE = 5.0  # Allow a small detour for a useful capsule.
TARGET_DISTANCE_WEIGHT = 2.0  # Reward progress toward the selected food or capsule.
FOOD_REWARD = 16.0  # Reward collecting food while keeping route progress influential.
CAPSULE_REWARD = 20.0  # Capsules get a slightly larger immediate reward than food.
GHOST_DANGER_PENALTIES = {1: 100.0, 2: 35.0, 3: 12.0, 4: 4.0}  # Sharp nearby penalty tapers with distance.
HIDDEN_GHOST_SCORE_PENALTY = 100.0  # Strongly discourage moves into likely hidden-ghost danger.
TRAP_ALERT_DISTANCE = 3.0  # Start choosing escape routes before a nearby opponent can close in.
TRAP_ESCAPE_WEIGHT = 20.0  # Make gaining distance from a nearby threat a primary objective.
TRAP_ROUTE_COST = 0.35  # Discount distant safe spots so escape plans remain practical.
TRAP_BAIT_WEIGHT = 12.0  # If escape is blocked, draw the threat toward the isolated agent.
TRAP_ALLY_SEPARATION_WEIGHT = 2.0  # Encourage a baiting agent to pull danger away from its teammate.


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """Create one food attacker and one home-territory defender."""

    return [
        pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.OffensiveCaptureAgent",
            role = "offense",
            row_region = 0,
            training = True,
            learning_rate = 0.2,
            discount_rate = 0.9,
            exploration_rate = 0.2),
        pacai.core.agentinfo.AgentInfo(
            name = f"{__name__}.OffensiveCaptureAgent",
            role = "offense",
            row_region = 1,
            training = True,
            learning_rate = 0.2,
            discount_rate = 0.9,
            exploration_rate = 0.2),
    ]


class _CaptureAgent(pacai.student.learning.ApproximateQLearningAgent):
    """Shared maze-distance helpers and approximate Q-learning support."""

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(feature_extractor_func = _capture_rl_features, **kwargs)
        if (not hasattr(self, "role")):
            self.role = "offense"
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
        super().game_start(initial_state)

    def _distance(self,
            first: pacai.core.board.Position,
            second: pacai.core.board.Position) -> float:
        return float(self._distances.get(first, {}).get(second, DISTANCE_FALLBACK))

    def _is_home_side(self, position: pacai.core.board.Position) -> bool:
        is_left_side = position.col < (self._current_board_width / 2)
        return is_left_side == (self.agent_index % 2 == 0)

    @staticmethod
    def _normalize_belief(
            weights: dict[pacai.core.board.Position, float]) -> dict[pacai.core.board.Position, float]:
        if (not weights):
            return {}

        # Bound work and memory if an opponent stays hidden for a long time.
        if (len(weights) > MAX_BELIEF_POSITIONS):
            weights = dict(sorted(weights.items(), key = lambda item: item[1], reverse = True)[:MAX_BELIEF_POSITIONS])

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
                        self._belief_confidence[opponent_index] = INITIAL_HIDDEN_CONFIDENCE
                self._belief_confidence[opponent_index] = (
                    BELIEF_CONFIDENCE_DECAY * self._belief_confidence.get(opponent_index, 0.0))

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
            evidence[food_position] = evidence.get(food_position, 0.0) + FOOD_LOSS_EVIDENCE_WEIGHT
            for (_, neighbor) in state.board.get_neighbors(food_position):
                evidence[neighbor] = evidence.get(neighbor, 0.0) + FOOD_NEIGHBOR_EVIDENCE_WEIGHT
        evidence = self._normalize_belief(evidence)

        for opponent_index in opponent_indexes:
            if (opponent_index in visible_positions):
                continue

            old_belief = self._beliefs.get(opponent_index, {})
            combined = {position: OLD_BELIEF_WEIGHT * probability for (position, probability) in old_belief.items()}
            for (position, probability) in evidence.items():
                combined[position] = combined.get(position, 0.0) + NEW_EVIDENCE_WEIGHT * probability
            self._beliefs[opponent_index] = self._normalize_belief(combined)
            self._belief_confidence[opponent_index] = max(
                self._belief_confidence.get(opponent_index, 0.0), FOOD_EVIDENCE_CONFIDENCE_FLOOR)

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

        if (confidence < MIN_INVADER_CONFIDENCE):
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
                if (distance <= HIDDEN_GHOST_RISK_RADIUS):
                    risk += (probability * self._belief_confidence.get(opponent_index, 0.0)
                            * (HIDDEN_GHOST_RISK_RADIUS + 1 - distance) / HIDDEN_GHOST_RISK_SCALE)

        return min(risk, 1.0)

    def _avoid_or_draw_threat(self,
            state: pacai.capture.gamestate.GameState,
            position: pacai.core.board.Position,
            legal_actions: list[pacai.core.action.Action],
            action_scores: dict[pacai.core.action.Action, float]) -> None:
        """Escape nearby lethal opponents, or draw them away if every route is blocked."""
        if state.is_pacman(self.agent_index):
            threats = {
                index: opponent_position
                for index, opponent_position in state.get_nonscared_opponent_positions(
                    agent_index = self.agent_index).items()
                if state.is_ghost(index)
            }
        elif state.is_scared(self.agent_index):
            threats = {
                index: opponent_position
                for index, opponent_position in state.get_invader_positions(
                    agent_index = self.agent_index).items()
            }
        else:
            threats = {}

        if not threats:
            return

        current_distance = min(self._distance(position, enemy) for enemy in threats.values())
        if current_distance > TRAP_ALERT_DISTANCE:
            return

        next_positions = {action: position.apply_action(action) for action in legal_actions}
        next_distances = {
            action: min(self._distance(next_position, enemy) for enemy in threats.values())
            for action, next_position in next_positions.items()
        }

        # A dead end often requires moving toward a threat briefly before the
        # corridor opens into a safer area. Score each first step by the best
        # threat distance reachable from it, discounted by the route length.
        route_potential: dict[pacai.core.action.Action, float] = {}
        for action, next_position in next_positions.items():
            reachable = self._distances.get(next_position, {})
            route_potential[action] = max(
                (
                    min(self._distance(destination, enemy) for enemy in threats.values())
                    - TRAP_ROUTE_COST * route_distance
                    for destination, route_distance in reachable.items()
                ),
                default = next_distances[action])

        has_escape = any(potential > current_distance for potential in route_potential.values())

        if has_escape:
            # Prefer a first step that leads to a safer area, even when it
            # initially moves closer to the opponent to get out of a dead end.
            for action, potential in route_potential.items():
                action_scores[action] = action_scores.get(action, 0.0) + TRAP_ESCAPE_WEIGHT * (
                    potential - current_distance)
            return

        # There is no move that increases the gap. Make the endangered agent
        # the opponent's closest target while maximizing its separation from
        # its teammate, so the chase is pulled away from the rest of the team.
        allies = state.get_ally_positions(agent_index = self.agent_index)
        for action, next_position in next_positions.items():
            distance_to_threat = next_distances[action]
            ally_distance = min(
                (self._distance(next_position, ally) for ally in allies.values()),
                default = 0.0)
            action_scores[action] = action_scores.get(action, 0.0) + (
                TRAP_BAIT_WEIGHT * (current_distance - distance_to_threat)
                + TRAP_ALLY_SEPARATION_WEIGHT * ally_distance)

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
                score -= REVERSE_ACTION_PENALTY

            if (score > best_score):
                best_score = score
                best_action = action

        return best_action


class CaptureQLearningAgent(_CaptureAgent):
    """Approximate Q-learner with role-specific capture-game features."""

    def __init__(self, role: str = "offense", **kwargs: typing.Any) -> None:
        if (role not in {"offense", "defense"}):
            raise ValueError("CaptureQLearningAgent role must be 'offense' or 'defense'.")

        self.role = role
        super().__init__(**kwargs)


def _capture_rl_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    """Describe the results of an action for approximate Q-learning."""
    capture_state = typing.cast(pacai.capture.gamestate.GameState, state)
    capture_agent = typing.cast(CaptureQLearningAgent, agent)
    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features["bias"] = 1.0

    position = capture_state.get_agent_position(capture_agent.agent_index)
    if (position is None):
        return features

    successor = typing.cast(pacai.capture.gamestate.GameState,
        capture_state.generate_successor(action, rng = capture_agent.rng))
    next_position = successor.get_agent_position(capture_agent.agent_index)
    if (next_position is None):
        features["respawned"] = 1.0
        return features

    scale = max(capture_state.board.width, capture_state.board.height, 1)
    features["stopped"] = float(action == pacai.core.action.STOP)

    previous_actions = capture_state.get_agent_actions(capture_agent.agent_index)
    if (previous_actions):
        reverse_action = capture_state.get_reverse_action(previous_actions[-1])
        features["reversed"] = float(action == reverse_action)

    score_change = (successor.get_normalized_score(capture_agent.agent_index)
        - capture_state.get_normalized_score(capture_agent.agent_index))
    features["score_change"] = score_change

    if (capture_agent.role == "offense"):
        current_food = capture_state.get_food(agent_index = capture_agent.agent_index)
        next_food = successor.get_food(agent_index = capture_agent.agent_index)
        features["food_eaten"] = float(len(next_food) < len(current_food))

        if (current_food):
            if (len(next_food) < len(current_food)):
                # Count collecting a pellet as progress even when the next one is farther away.
                features["food_progress"] = 1.0
            else:
                current_distance = min(capture_agent._distance(position, food) for food in current_food)
                next_distance = min(capture_agent._distance(next_position, food) for food in next_food)
                features["food_progress"] = (current_distance - next_distance) / scale

        current_capsules = capture_state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
        next_capsules = successor.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
        features["capsule_eaten"] = float(len(next_capsules) < len(current_capsules))

        if (not successor.is_ghost(capture_agent.agent_index)):
            ghosts = successor.get_nonscared_opponent_positions(agent_index = capture_agent.agent_index)
            if (ghosts):
                nearest_ghost = min(capture_agent._distance(next_position, ghost) for ghost in ghosts.values())
                features["ghost_proximity"] = 1.0 / (nearest_ghost + 1.0)
    else:
        current_invaders = capture_state.get_invader_positions(agent_index = capture_agent.agent_index)
        next_invaders = successor.get_invader_positions(agent_index = capture_agent.agent_index)
        features["home_side"] = float(successor.is_ghost(capture_agent.agent_index))
        features["invader_captured"] = float(len(next_invaders) < len(current_invaders))

        if (current_invaders):
            old_distance = min(capture_agent._distance(position, invader)
                for invader in current_invaders.values())
            if (next_invaders):
                new_distance = min(capture_agent._distance(next_position, invader)
                    for invader in next_invaders.values())
                features["invader_progress"] = (old_distance - new_distance) / scale
            else:
                features["invader_progress"] = 1.0

        if (successor.is_scared(capture_agent.agent_index) and next_invaders):
            nearest_invader = min(capture_agent._distance(next_position, invader)
                for invader in next_invaders.values())
            features["scared_invader_proximity"] = 1.0 / (nearest_invader + 1.0)

    return features


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
            if (confidence >= DEFENSE_CONFIDENCE_THRESHOLD):
                scores: dict[pacai.core.action.Action, float] = {}
                scared = state.is_scared(self.agent_index)
                for action in legal_actions:
                    next_position = position.apply_action(action)
                    expected_distance = sum(
                        probability * self._distance(next_position, invader)
                        for (invader, probability) in invader_belief.items())
                    score = (DEFENSE_DISTANCE_WEIGHT if scared else -DEFENSE_DISTANCE_WEIGHT) * confidence * expected_distance
                    if (not scared):
                        score += DEFENSE_CAPTURE_BONUS * confidence * invader_belief.get(next_position, 0.0)
                    if (not self._is_home_side(next_position)):
                        score -= DEFENSE_CROSSING_PENALTY
                    scores[action] = score

                return self._select_action(state, scores)

        food_positions = state.get_food(agent_index = self.agent_index)
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
            if (self._row_region in {0, 1}
                    and int(food.row >= (state.board.height / 2)) != self._row_region):
                # Give each teammate a preferred upper or lower entry. Keep a
                # fallback path to other food if that region has been cleared.
                cost += TEAM_FOOD_REGION_PENALTY
            if (not self._is_home_side(food)):
                for ghost_position in dangerous_ghosts.values():
                    ghost_distance = self._distance(food, ghost_position)
                    if (ghost_distance <= FOOD_GHOST_ALERT_DISTANCE):
                        cost += FOOD_GHOST_DISTANCE_PENALTY * (FOOD_GHOST_ALERT_DISTANCE + 1 - ghost_distance)
                cost += FOOD_GHOST_RISK_WEIGHT * self._hidden_ghost_risk(state, food)
            return cost

        nearest_food = min(food_positions, key = target_cost)
        if ((self._food_target not in food_positions)
                or (target_cost(nearest_food) + FOOD_TARGET_SWITCH_MARGIN < target_cost(
                    typing.cast(pacai.core.board.Position, self._food_target)))):
            self._food_target = nearest_food
        target = typing.cast(pacai.core.board.Position, self._food_target)

        capsule_positions = {
            capsule for capsule in state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
            if not self._is_home_side(capsule)
        }
        closest_ghost_distance = min(
            (self._distance(position, ghost) for ghost in dangerous_ghosts.values()),
            default = GHOST_DISTANCE_FALLBACK)

        # Treat capsules as tactical objectives when a defender is approaching.
        # Prefer capsules that are reachable soon and not already guarded by a ghost.
        if ((closest_ghost_distance <= CAPSULE_GHOST_ALERT_DISTANCE) and capsule_positions):
            def capsule_cost(capsule: pacai.core.board.Position) -> float:
                ghost_distance = min(
                    (self._distance(capsule, ghost) for ghost in dangerous_ghosts.values()),
                    default = GHOST_DISTANCE_FALLBACK)
                guarded_penalty = max(0.0, CAPSULE_GUARD_RADIUS - ghost_distance) * CAPSULE_GUARD_PENALTY_WEIGHT
                return self._distance(position, capsule) + guarded_penalty

            capsule = min(capsule_positions, key = capsule_cost)
            if (capsule_cost(capsule) <= self._distance(position, target) + CAPSULE_EXTRA_TRAVEL_ALLOWANCE):
                target = capsule

        scores: dict[pacai.core.action.Action, float] = {}
        for action in legal_actions:
            next_position = position.apply_action(action)
            distance_to_target = self._distance(next_position, target)
            score = -TARGET_DISTANCE_WEIGHT * distance_to_target

            if (next_position in food_positions):
                score += FOOD_REWARD
            if (next_position in capsule_positions):
                score += CAPSULE_REWARD

            for ghost_position in dangerous_ghosts.values():
                if (self._is_home_side(next_position)):
                    continue

                ghost_distance = self._distance(next_position, ghost_position)
                for danger_distance, penalty in GHOST_DANGER_PENALTIES.items():
                    if (ghost_distance <= danger_distance):
                        score -= penalty
                        break

            if (not self._is_home_side(next_position)):
                score -= HIDDEN_GHOST_SCORE_PENALTY * self._hidden_ghost_risk(state, next_position)

            scores[action] = score

        self._avoid_or_draw_threat(state, position, legal_actions, scores)
        return self._select_action(state, scores)


class DefensiveCaptureAgent(_CaptureAgent):
    """Guard home territory, chase likely invaders, and patrol when it is clear."""

    def _patrol_target(self) -> pacai.core.board.Position | None:
        """Choose an open home-side tile near the center line, favoring mid-board rows."""
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

        self._avoid_or_draw_threat(state, position, legal_actions, scores)
        return self._select_action(state, scores)
