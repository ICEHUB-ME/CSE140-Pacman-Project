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
REVERSE_ACTION_PENALTY = 1.0  # Discourage reversals unless they make objective progress.
DEFENSE_ENTER_URGENCY = 0.78  # Require a credible, strategically urgent invasion to enter defense.
DEFENSE_EXIT_URGENCY = 0.62  # Keep defending until the invasion is clearly less urgent.
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
DEAD_END_ALERT_DISTANCE = 7.0  # Start checking escape races before a pursuer reaches the corridor.
DEAD_END_ESCAPE_BONUS = 1000.0  # Let a viable escape override local danger penalties.
DEAD_END_DELAY_BONUS = 300.0  # Prefer the move that buys time when the exit race is lost.
DEAD_END_EXIT_PROGRESS_BONUS = 1000.0  # Keep moving toward an exit when waiting would trap the agent.
INTERCEPT_LATE_PENALTY = 3.0  # Penalize reaching a likely path after the invader.
INTERCEPT_MARGIN_REWARD = 0.25  # Prefer a small arrival lead without chasing distant points.
SCARED_SAFE_DISTANCE = 2.0  # Keep just outside an invader's immediate capture range.
SCARED_DISTANCE_WEIGHT = 18.0  # Favor the safe-distance band without fleeing indefinitely.
SCARED_DEATH_PENALTY = 500.0  # Avoid a scared collision except in the explicit respawn strategy.
SCARED_FOOD_BONUS = 35.0  # A food pickup can outweigh a small temporary safety cost.
SCARED_Q_VALUE_WEIGHT = 1.0  # Let learned scared-state preferences break tactical ties.
SCARED_AMBUSH_WEIGHT = 6.0  # Approach an invader's route while maintaining a safe gap.
SCARED_ATTACK_TIMER_MIN = 6  # Attack instead of waiting when plenty of scared time remains.
SCARED_ATTACK_SCORE_THRESHOLD = -20.0  # A meaningful deficit favors an offensive counter-race.
SCARED_ATTACK_FOOD_THRESHOLD = 3  # Few remaining enemy pellets make an attack especially valuable.
SCARED_SACRIFICE_TIMER_MIN = 12  # Respawn is worth considering only with a long timer remaining.
SCARED_SACRIFICE_FOOD_DISTANCE = 4.0  # Do not abandon an invader already near our food.
ENDGAME_FOOD_THRESHOLD = 5  # Evaluate short food runs during the closing stages.
ENDGAME_FOOD_RUN_WEIGHT = 24.0  # Reward each pellet the route can plausibly secure.
ENDGAME_TARGET_THRESHOLD = 3  # Switch to direct collection when only a few pellets remain.
ENDGAME_TARGET_SWITCH_MARGIN = 5.0  # Keep the current closing target unless another is much better.
ENDGAME_PROGRESS_BONUS = 45.0  # Keep making progress despite moderate nearby-ghost penalties.
FINAL_FOOD_BONUS = 5000.0  # Securing the final pellet takes priority over ordinary danger penalties.
ENTRY_Q_VALUE_WEIGHT = 0.1  # Translate learned entry value into a route-selection bonus.
ENTRY_EXPLORATION_WINDOW = 6.0  # Explore entries whose route cost is close to the best route.
ENTRY_MISMATCH_PENALTY = 100.0  # Keep movement aligned with the chosen crossing.


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """Create one of each behavior class; both can switch roles dynamically."""

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
        self._boundary_entries: dict[
            str,
            tuple[pacai.core.board.Position, pacai.core.board.Position]
        ] = {}
        self._entry_by_away_position: dict[pacai.core.board.Position, str] = {}
        self._active_entry_id: str | None = None
        self._planned_entry_id: str | None = None
        self._planned_entry_target: pacai.core.board.Position | None = None
        self._dead_end_depth: dict[pacai.core.board.Position, int] = {}
        self._defense_mode: bool | None = None

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

        self._dead_end_depth = self._compute_dead_end_depth(initial_state.board, positions)

        # Each open adjacency between home and away territory is one entry
        # choice. Row and boundary column make ids consistent for either team.
        for home_position in positions:
            if (not self._is_home_side(home_position)):
                continue

            for (_, away_position) in initial_state.board.get_neighbors(home_position):
                if (self._is_home_side(away_position)):
                    continue

                boundary_col = min(home_position.col, away_position.col)
                entry_id = f"{home_position.row}_{boundary_col}"
                self._boundary_entries[entry_id] = (home_position, away_position)
                self._entry_by_away_position[away_position] = entry_id

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
    def _compute_dead_end_depth(
            board: pacai.core.board.Board,
            positions: list[pacai.core.board.Position]) -> dict[pacai.core.board.Position, int]:
        """Peel leaf corridors from the maze graph; remaining cells form its 2-core."""
        remaining = set(positions)
        degrees = {
            position: len(board.get_neighbors(position))
            for position in positions
        }
        layer = [position for (position, degree) in degrees.items() if (degree <= 1)]
        depths: dict[pacai.core.board.Position, int] = {}
        depth = 1

        while (layer):
            next_layer = []
            for position in layer:
                if (position not in remaining):
                    continue

                remaining.remove(position)
                depths[position] = depth
                for (_, neighbor) in board.get_neighbors(position):
                    if (neighbor not in remaining):
                        continue

                    degrees[neighbor] -= 1
                    if (degrees[neighbor] <= 1):
                        next_layer.append(neighbor)

            layer = list(set(next_layer))
            depth += 1

        return depths

    def _dead_end_exit(self,
            position: pacai.core.board.Position) -> pacai.core.board.Position | None:
        """Return the nearest tile outside a pruned dead-end corridor."""
        if (position not in self._dead_end_depth):
            return None

        core_positions = [point for point in self._distances if (point not in self._dead_end_depth)]
        if (not core_positions):
            return None

        return min(core_positions, key = lambda point: self._distance(position, point))

    def _interception_target(self,
            state: pacai.capture.gamestate.GameState,
            position: pacai.core.board.Position,
            invaders: dict[int, pacai.core.board.Position]) -> pacai.core.board.Position | None:
        """Pick a reachable tile ahead of an invader on its likely route to food."""
        candidates: list[tuple[float, pacai.core.board.Position]] = []
        home_positions = [point for point in self._distances if (self._is_home_side(point))]

        for invader_index, invader_position in invaders.items():
            target_food = state.get_food(agent_index = invader_index)
            if (not target_food):
                continue

            nearest_food = sorted(
                target_food,
                key = lambda food: self._distance(invader_position, food))[:3]
            for food in nearest_food:
                invader_food_distance = self._distance(invader_position, food)
                for intercept in home_positions:
                    invader_eta = self._distance(invader_position, intercept)
                    distance_after_intercept = self._distance(intercept, food)

                    # Only consider tiles ahead of the invader on a plausible
                    # shortest route to that food pellet.
                    if (distance_after_intercept >= invader_food_distance):
                        continue
                    if (invader_eta + distance_after_intercept > invader_food_distance + 1.0):
                        continue

                    our_eta = self._distance(position, intercept)
                    arrival_margin = invader_eta - our_eta
                    late_cost = max(0.0, -arrival_margin) * INTERCEPT_LATE_PENALTY
                    lead_reward = min(max(0.0, arrival_margin), 4.0) * INTERCEPT_MARGIN_REWARD
                    cost = our_eta + 0.2 * distance_after_intercept + late_cost - lead_reward
                    candidates.append((cost, intercept))

        if (not candidates):
            return None

        return min(candidates, key = lambda item: item[0])[1]

    def _plan_boundary_entry(self,
            state: pacai.capture.gamestate.GameState,
            position: pacai.core.board.Position,
            target: pacai.core.board.Position,
            row_region: int,
            dangerous_ghosts: dict[int, pacai.core.board.Position]) -> str | None:
        """Select a boundary route using approximate Q weights and epsilon exploration."""
        if (not self._boundary_entries):
            return None

        if ((self._planned_entry_id in self._boundary_entries)
                and (self._planned_entry_target == target)):
            return self._planned_entry_id

        route_costs: dict[str, float] = {}
        route_values: dict[str, float] = {}
        for entry_id, (home_entry, away_entry) in self._boundary_entries.items():
            cost = self._distance(position, home_entry) + 1.0 + self._distance(away_entry, target)

            if (row_region in {0, 1}
                    and int(home_entry.row >= (state.board.height / 2)) != row_region):
                cost += 6.0

            for ghost_position in dangerous_ghosts.values():
                ghost_distance = self._distance(away_entry, ghost_position)
                if (ghost_distance <= FOOD_GHOST_ALERT_DISTANCE):
                    cost += FOOD_GHOST_DISTANCE_PENALTY * (FOOD_GHOST_ALERT_DISTANCE + 1 - ghost_distance)

            cost += FOOD_GHOST_RISK_WEIGHT * self._hidden_ghost_risk(state, away_entry)
            route_costs[entry_id] = cost

            # These are learned Approximate Q weights. The active route feature
            # is updated from each transition until the agent returns home.
            route_values[entry_id] = (
                self.weights.get(f"entry_route:{entry_id}", 0.0)
                + self.weights.get(f"entry_crossing:{entry_id}", 0.0))

        best_cost = min(route_costs.values())
        candidates = [
            entry_id for (entry_id, cost) in route_costs.items()
            if (cost <= best_cost + ENTRY_EXPLORATION_WINDOW)
        ]
        exploration_rate = self.exploration_rate / (1.0 + (0.01 * self.training_epoch))
        if (self.training and (self.rng.random() < exploration_rate)):
            selected_entry = self.rng.choice(candidates)
        else:
            selected_entry = max(
                route_costs,
                key = lambda entry_id: (-route_costs[entry_id]
                    + ENTRY_Q_VALUE_WEIGHT * route_values[entry_id]))

        self._planned_entry_id = selected_entry
        self._planned_entry_target = target
        return selected_entry

    def _remember_boundary_action(self,
            position: pacai.core.board.Position,
            action: pacai.core.action.Action) -> None:
        """Track the entry selected when this action crosses into enemy territory."""
        next_position = position.apply_action(action)
        if (self._is_home_side(position) and not self._is_home_side(next_position)):
            self._active_entry_id = self._entry_by_away_position.get(next_position)
            self._planned_entry_id = None
            self._planned_entry_target = None

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

    def _defense_urgency(self,
            state: pacai.capture.gamestate.GameState,
            invader_belief: dict[pacai.core.board.Position, float],
            confidence: float) -> float:
        """Blend invader confidence, distance to our base, and score context."""
        if ((not invader_belief) or (confidence <= 0.0)):
            return 0.0

        base_positions = []
        for index in state.get_agent_indexes():
            if ((index % 2) == (self.agent_index % 2)):
                base = state.board.get_agent_initial_position(pacai.core.board.Marker(str(index)))
                if (base is not None):
                    base_positions.append(base)

        board_scale = max(self._current_board_width, self._current_board_height, 1)
        if (base_positions):
            base_proximity = sum(
                probability * max(0.0, 1.0 - min(
                    self._distance(position, base) for base in base_positions) / board_scale)
                for (position, probability) in invader_belief.items())
        else:
            base_proximity = 0.0

        normalized_score = state.get_normalized_score(agent_index = self.agent_index)
        score_context = max(-1.0, min(1.0, normalized_score / 100.0))
        urgency = confidence * (0.55 + 0.35 * base_proximity + 0.10 * score_context)
        return max(0.0, min(urgency, 1.0))

    def _should_defend(self, urgency: float) -> bool:
        """Apply hysteresis so small urgency changes do not flip roles each turn."""
        if (self._defense_mode is None):
            self._defense_mode = (urgency >= DEFENSE_ENTER_URGENCY)
        elif (self._defense_mode):
            if (urgency <= DEFENSE_EXIT_URGENCY):
                self._defense_mode = False
        elif (urgency >= DEFENSE_ENTER_URGENCY):
            self._defense_mode = True

        return self._defense_mode

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

    def _food_run_capacity(self,
            start: pacai.core.board.Position,
            food_positions: set[pacai.core.board.Position],
            defenders: dict[int, pacai.core.board.Position],
            limit: int = ENDGAME_FOOD_THRESHOLD) -> int:
        """Estimate how many nearby pellets are reachable before a defender."""
        remaining = set(food_positions)
        current = start
        elapsed = 0.0
        collected = 0

        while (remaining and collected < limit):
            ordered_food = sorted(remaining, key = lambda food: self._distance(current, food))
            selected_food = None
            selected_distance = 0.0
            for food in ordered_food:
                travel = self._distance(current, food)
                arrival_time = elapsed + travel
                defender_time = min(
                    (self._distance(defender, food) for defender in defenders.values()),
                    default = DISTANCE_FALLBACK)
                # A defender that can reach the pellet first can contest that
                # leg; stop the route estimate rather than counting it as safe.
                if (defender_time <= arrival_time):
                    continue
                selected_food = food
                selected_distance = travel
                break

            if (selected_food is None):
                break

            elapsed += selected_distance
            collected += 1
            current = selected_food
            remaining.remove(selected_food)

        return collected

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

        next_positions = {action: position.apply_action(action) for action in legal_actions}
        next_distances = {
            action: min(self._distance(next_position, enemy) for enemy in threats.values())
            for action, next_position in next_positions.items()
        }

        current_distance = min(self._distance(position, enemy) for enemy in threats.values())
        if (state.is_scared(self.agent_index)):
            # A scared defender should hold a survivable gap, not maximize
            # distance forever. Food is rewarded when it is actually edible
            # from this state (for example, on the crossing step).
            food_positions = state.get_food(agent_index = self.agent_index)
            timer = state.scared_timers.get(self.agent_index, 0)
            safe_distance = min(SCARED_SAFE_DISTANCE, max(1.0, float(timer)))
            mdp_state = self.mdp_state_class(position = position, game_state = state)
            for action, next_position in next_positions.items():
                distance = next_distances[action]
                action_scores[action] = action_scores.get(action, 0.0) - (
                    SCARED_DISTANCE_WEIGHT * abs(distance - safe_distance))
                if (next_position in food_positions):
                    action_scores[action] += SCARED_FOOD_BONUS
                # These learned Q-values use scared_food and scared_safety
                # features below; hand-authored shaping remains the cold-start.
                action_scores[action] += SCARED_Q_VALUE_WEIGHT * self.get_qvalue(
                    mdp_state, state, action)
            return

        dead_end_exit = self._dead_end_exit(position)
        if ((dead_end_exit is not None) and (current_distance <= DEAD_END_ALERT_DISTANCE)):
            escape_time = self._distance(position, dead_end_exit)
            opponent_time = min(
                self._distance(enemy, dead_end_exit)
                for enemy in threats.values())

            if (escape_time < opponent_time):
                # Commit to the exit race. This bonus intentionally outweighs
                # local danger and food scores, including a first step that
                # briefly moves closer to the pursuing opponent.
                for action, next_position in next_positions.items():
                    next_escape_time = self._distance(next_position, dead_end_exit)
                    progress = escape_time - next_escape_time
                    action_scores[action] = action_scores.get(action, 0.0) + (
                        DEAD_END_ESCAPE_BONUS * progress)
                return

            # The pursuer can reach the exit first. Buy time by moving to the
            # tile with the greatest immediate separation and never choosing
            # STOP when another legal move improves that separation.
            best_distance = max(next_distances.values())
            for action, distance in next_distances.items():
                next_position = next_positions[action]
                exit_progress = escape_time - self._distance(next_position, dead_end_exit)
                action_scores[action] = action_scores.get(action, 0.0) + (
                    DEAD_END_DELAY_BONUS * (distance - best_distance)
                    + DEAD_END_EXIT_PROGRESS_BONUS * exit_progress)
            return

        if current_distance > TRAP_ALERT_DISTANCE:
            return

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
        if (not allies):
            # Baiting only helps when another ally can benefit. If this is the
            # team's only active agent, retreat toward home while avoiding a
            # move directly onto a lethal opponent whenever possible.
            home_targets = [point for point in self._distances if self._is_home_side(point)]
            home_target = min(
                home_targets,
                key = lambda point: self._distance(position, point),
                default = position)
            for action, next_position in next_positions.items():
                home_progress = (self._distance(position, home_target)
                    - self._distance(next_position, home_target))
                safety_progress = next_distances[action] - current_distance
                action_scores[action] = action_scores.get(action, 0.0) + (
                    DEAD_END_EXIT_PROGRESS_BONUS * home_progress
                    + TRAP_ESCAPE_WEIGHT * safety_progress)
                if (next_distances[action] == 0.0):
                    action_scores[action] -= DEAD_END_ESCAPE_BONUS
            return

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
            action_scores: dict[pacai.core.action.Action, float],
            action_progress: dict[pacai.core.action.Action, float] | None = None) -> pacai.core.action.Action:
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
            if ((action == reverse_action)
                    and ((action_progress is None) or (action_progress.get(action, 0.0) <= 0.0))):
                score -= REVERSE_ACTION_PENALTY

            if (score > best_score):
                best_score = score
                best_action = action
            elif ((score == best_score)
                    and (best_action == pacai.core.action.STOP)
                    and (action != pacai.core.action.STOP)):
                # STOP appears first in the framework's legal-action list.
                # Prefer movement on ties so a flat score cannot cause a stall.
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
    capture_agent = typing.cast(_CaptureAgent, agent)
    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features["bias"] = 1.0

    position = capture_state.get_agent_position(capture_agent.agent_index)
    if (position is None):
        return features

    candidate_position = position.apply_action(action)
    successor = typing.cast(pacai.capture.gamestate.GameState,
        capture_state.generate_successor(action, rng = capture_agent.rng))
    next_position = successor.get_agent_position(capture_agent.agent_index)

    crossing_entry_id = None
    if (capture_agent._is_home_side(position)
            and not capture_agent._is_home_side(candidate_position)):
        crossing_entry_id = capture_agent._entry_by_away_position.get(candidate_position)

    route_entry_id = crossing_entry_id
    if ((route_entry_id is None) and not capture_agent._is_home_side(position)):
        route_entry_id = capture_agent._active_entry_id

    if (route_entry_id is not None):
        features[f"entry_route:{route_entry_id}"] = 1.0
        if (crossing_entry_id is not None):
            features[f"entry_crossing:{crossing_entry_id}"] = 1.0

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

    if (capture_state.is_scared(capture_agent.agent_index)):
        invaders = capture_state.get_invader_positions(agent_index = capture_agent.agent_index)
        next_distance = min(
            (capture_agent._distance(next_position, invader) for invader in invaders.values()),
            default = SCARED_SAFE_DISTANCE + scale)
        safe_margin = next_distance - SCARED_SAFE_DISTANCE
        features["scared_safety"] = -abs(safe_margin) / scale
        features["scared_safe_side"] = float(safe_margin >= 0.0)
        timer = capture_state.scared_timers.get(capture_agent.agent_index, 0)
        features["scared_timer"] = min(timer, 40) / 40.0
        current_food = capture_state.get_food(agent_index = capture_agent.agent_index)
        next_food = successor.get_food(agent_index = capture_agent.agent_index)
        features["scared_food"] = float(len(next_food) < len(current_food))

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

    def get_action(self,
            state: pacai.core.gamestate.GameState,
            _already_updated: bool = False,
            _force_offense: bool = False) -> pacai.core.action.Action:
        state = typing.cast(pacai.capture.gamestate.GameState, state)
        if (not _already_updated):
            self.update(state)
            self._update_opponent_beliefs(state)
        position = state.get_agent_position(self.agent_index)
        if (position is None):
            # A respawn should start a fresh route from the spawn point instead
            # of sending the attacker back toward the same dangerous target.
            self._food_target = None
            self._active_entry_id = None
            self._planned_entry_id = None
            self._planned_entry_target = None
            return pacai.core.action.STOP

        if (self._is_home_side(position)):
            # The previous return move has now been learned from.
            self._active_entry_id = None
        else:
            self._planned_entry_id = None
            self._planned_entry_target = None

        legal_actions = state.get_legal_actions()
        if (not legal_actions):
            return pacai.core.action.STOP

        invader_belief, confidence = self._likely_invaders(state)
        # Both classes share this urgency check and can delegate to the other
        # behavior without creating a second learning state. This also lets an
        # attacker return home when a serious invasion develops mid-raid.
        defense_urgency = self._defense_urgency(state, invader_belief, confidence)
        if ((not _force_offense) and self._should_defend(defense_urgency)):
            return DefensiveCaptureAgent._get_defensive_action(self, state)

        food_positions = state.get_food(agent_index = self.agent_index)
        if (not food_positions):
            return self._select_action(state, {})

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
                        cost += FOOD_GHOST_DISTANCE_PENALTY * (
                            FOOD_GHOST_ALERT_DISTANCE + 1 - ghost_distance)
                cost += FOOD_GHOST_RISK_WEIGHT * self._hidden_ghost_risk(state, food)
            return cost

        endgame_food_run = len(food_positions) <= ENDGAME_TARGET_THRESHOLD
        if (endgame_food_run):
            # In the closing run, choose the nearest remaining objective
            # directly instead of letting stale risk estimates keep the agent
            # parked or route it away from the last few pellets.
            nearest_food = min(food_positions, key = lambda food: self._distance(position, food))
        else:
            nearest_food = min(food_positions, key = target_cost)
        final_food_run = len(food_positions) == 1
        if (self._food_target not in food_positions):
            self._food_target = nearest_food
        elif (endgame_food_run):
            current_target = typing.cast(pacai.core.board.Position, self._food_target)
            if ((nearest_food != current_target)
                    and (target_cost(nearest_food) + ENDGAME_TARGET_SWITCH_MARGIN
                        < target_cost(current_target))):
                self._food_target = nearest_food
        elif (target_cost(nearest_food) + FOOD_TARGET_SWITCH_MARGIN < target_cost(
                typing.cast(pacai.core.board.Position, self._food_target))):
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
        if ((not endgame_food_run)
                and (closest_ghost_distance <= CAPSULE_GHOST_ALERT_DISTANCE) and capsule_positions):
            def capsule_cost(capsule: pacai.core.board.Position) -> float:
                ghost_distance = min(
                    (self._distance(capsule, ghost) for ghost in dangerous_ghosts.values()),
                    default = GHOST_DISTANCE_FALLBACK)
                guarded_penalty = max(0.0, CAPSULE_GUARD_RADIUS - ghost_distance) * CAPSULE_GUARD_PENALTY_WEIGHT
                return self._distance(position, capsule) + guarded_penalty

            capsule = min(capsule_positions, key = capsule_cost)
            if (capsule_cost(capsule) <= self._distance(position, target) + CAPSULE_EXTRA_TRAVEL_ALLOWANCE):
                target = capsule

        planned_entry_id = None
        if (self._is_home_side(position) and not self._is_home_side(target)):
            planned_entry_id = self._plan_boundary_entry(
                state, position, target, self._row_region, dangerous_ghosts)

        current_route_distance = self._distance(position, target)
        if ((planned_entry_id is not None) and self._is_home_side(position)):
            home_entry, away_entry = self._boundary_entries[planned_entry_id]
            current_route_distance = (self._distance(position, home_entry)
                + 1.0 + self._distance(away_entry, target))

        scores: dict[pacai.core.action.Action, float] = {}
        route_distances: dict[pacai.core.action.Action, float] = {}
        for action in legal_actions:
            next_position = position.apply_action(action)
            if ((planned_entry_id is not None) and self._is_home_side(position)):
                home_entry, away_entry = self._boundary_entries[planned_entry_id]
                if (self._is_home_side(next_position)):
                    distance_to_target = (self._distance(next_position, home_entry)
                        + 1.0 + self._distance(away_entry, target))
                elif (next_position == away_entry):
                    distance_to_target = self._distance(next_position, target)
                else:
                    distance_to_target = (self._distance(next_position, target)
                        + ENTRY_MISMATCH_PENALTY)
            else:
                distance_to_target = self._distance(next_position, target)
            route_distances[action] = distance_to_target
            score = -TARGET_DISTANCE_WEIGHT * distance_to_target

            if (next_position in food_positions):
                score += FINAL_FOOD_BONUS if final_food_run else FOOD_REWARD
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

            if (len(food_positions) <= ENDGAME_FOOD_THRESHOLD):
                remaining_after_move = food_positions - {next_position}
                capacity = self._food_run_capacity(
                    next_position, remaining_after_move, dangerous_ghosts)
                if (final_food_run):
                    score += FINAL_FOOD_BONUS * capacity
                else:
                    score += ENDGAME_FOOD_RUN_WEIGHT * capacity

            if (final_food_run and next_position in food_positions):
                # A final pellet ends the game; safe reachability was checked
                # by the route-capacity estimate above, so don't let generic
                # fear penalties drown out the winning move.
                score += FINAL_FOOD_BONUS

            scores[action] = score

        self._avoid_or_draw_threat(state, position, legal_actions, scores)
        if (endgame_food_run and (target in food_positions)):
            # Apply the closing-run objective after generic threat shaping so
            # it cannot erase all forward movement when the route is still
            # open. Preserve the threat model for immediately dangerous tiles.
            for action in legal_actions:
                next_position = position.apply_action(action)
                if (next_position in food_positions):
                    scores[action] += ENDGAME_PROGRESS_BONUS
                    continue

                next_threat_distance = min(
                    (self._distance(next_position, ghost) for ghost in dangerous_ghosts.values()),
                    default = DISTANCE_FALLBACK)
                if (next_threat_distance <= 1.0):
                    continue

                if ((planned_entry_id is not None) and self._is_home_side(position)):
                    home_entry, away_entry = self._boundary_entries[planned_entry_id]
                    if (self._is_home_side(next_position)):
                        next_route_distance = (self._distance(next_position, home_entry)
                            + 1.0 + self._distance(away_entry, target))
                    elif (next_position == away_entry):
                        next_route_distance = self._distance(next_position, target)
                    else:
                        next_route_distance = (self._distance(next_position, target)
                            + ENTRY_MISMATCH_PENALTY)
                else:
                    next_route_distance = self._distance(next_position, target)

                route_progress = current_route_distance - next_route_distance
                scores[action] += ENDGAME_PROGRESS_BONUS * max(0.0, route_progress)

        action_progress = {
            candidate: current_route_distance - route_distances[candidate]
            for candidate in legal_actions
        }
        action = self._select_action(state, scores, action_progress = action_progress)
        if ((action == pacai.core.action.STOP)
                and (closest_ghost_distance > TRAP_ALERT_DISTANCE)):
            progressing_actions = [
                candidate for candidate in legal_actions
                if ((candidate != pacai.core.action.STOP)
                    and (route_distances[candidate] < current_route_distance)
                    and (min(
                        (self._distance(position.apply_action(candidate), ghost)
                            for ghost in dangerous_ghosts.values()),
                        default = DISTANCE_FALLBACK) > 1.0))
            ]
            if (progressing_actions):
                # If the threat is not nearby, a stale risk score must not
                # leave the attacker idle while a safe objective step exists.
                action = max(
                    progressing_actions,
                    key = lambda candidate: (scores[candidate],
                        current_route_distance - route_distances[candidate]))

        self._remember_boundary_action(position, action)
        return action


class DefensiveCaptureAgent(_CaptureAgent):
    """Guard home territory, chase likely invaders, and patrol when it is clear."""

    def __init__(self, row_region: int = -1, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._food_target: pacai.core.board.Position | None = None
        self._row_region = row_region

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
        self.update(state)
        self._update_opponent_beliefs(state)
        position = state.get_agent_position(self.agent_index)
        if position is None:
            return pacai.core.action.STOP

        invader_belief, confidence = self._likely_invaders(state)
        defense_urgency = self._defense_urgency(state, invader_belief, confidence)
        if (not self._should_defend(defense_urgency)):
            return OffensiveCaptureAgent.get_action(self, state, _already_updated = True)

        return self._get_defensive_action(state)

    def _get_defensive_action(self,
            state: pacai.capture.gamestate.GameState) -> pacai.core.action.Action:
        """Run the defensive behavior while preserving this agent's Q state."""
        position = state.get_agent_position(self.agent_index)
        if (position is None):
            return pacai.core.action.STOP

        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return pacai.core.action.STOP

        invader_belief, confidence = self._likely_invaders(state)
        scared = state.is_scared(self.agent_index)
        visible_invaders = state.get_invader_positions(agent_index = self.agent_index)

        if (scared):
            timer = state.scared_timers.get(self.agent_index, 0)
            our_modifier = ((self.agent_index % 2) * 2) - 1
            food_on_our_side = state.get_food(team_modifier = -our_modifier)
            if (invader_belief and food_on_our_side):
                threat_to_food = sum(
                    probability * min(self._distance(invader, food) for food in food_on_our_side)
                    for (invader, probability) in invader_belief.items())
            else:
                threat_to_food = DISTANCE_FALLBACK

            our_food_remaining = state.food_count(agent_index = self.agent_index)
            score = state.get_normalized_score(agent_index = self.agent_index)
            should_counterattack = (
                timer >= SCARED_ATTACK_TIMER_MIN
                and threat_to_food >= SCARED_SACRIFICE_FOOD_DISTANCE + 1.0
                and ((score <= SCARED_ATTACK_SCORE_THRESHOLD)
                    or (our_food_remaining <= SCARED_ATTACK_FOOD_THRESHOLD)))
            if (should_counterattack):
                # If invaders are not close to our food, use the scared window
                # to race for their remaining food instead of waiting at home.
                return OffensiveCaptureAgent.get_action(
                    self, state, _already_updated = True, _force_offense = True)

            if ((timer >= SCARED_SACRIFICE_TIMER_MIN) and visible_invaders
                    and (min(self._distance(position, invader)
                        for invader in visible_invaders.values()) <= 1.0)
                    and (threat_to_food >= SCARED_SACRIFICE_FOOD_DISTANCE)):
                # A deliberate collision clears our timer; respawn puts us back
                # near base to defend, and the invader is still far from food.
                sacrifice_actions = [
                    action for action in state.get_legal_actions()
                    if (position.apply_action(action) in visible_invaders.values())
                ]
                if (sacrifice_actions):
                    return sacrifice_actions[0]

        interception_target = None
        if (visible_invaders):
            interception_target = self._interception_target(state, position, visible_invaders)
        patrol_target = DefensiveCaptureAgent._patrol_target(self)
        scores: dict[pacai.core.action.Action, float] = {}

        for action in legal_actions:
            next_position = position.apply_action(action)
            if interception_target is not None:
                score = -(SCARED_AMBUSH_WEIGHT if scared else DEFENSE_DISTANCE_WEIGHT) * self._distance(
                    next_position, interception_target)
                if (scared):
                    score -= SCARED_DEATH_PENALTY * sum(
                        1.0 for invader in visible_invaders.values()
                        if next_position == invader)
                else:
                    score += 100.0 * sum(
                        1.0 for invader in visible_invaders.values()
                        if next_position == invader)
            elif confidence > 0.0:
                if (scared):
                    expected_gap_error = sum(
                        probability * abs(self._distance(next_position, invader)
                            - SCARED_SAFE_DISTANCE)
                        for (invader, probability) in invader_belief.items())
                    score = -SCARED_DISTANCE_WEIGHT * confidence * expected_gap_error
                else:
                    expected_distance = sum(
                        probability * self._distance(next_position, invader)
                        for (invader, probability) in invader_belief.items())
                    score = -DEFENSE_DISTANCE_WEIGHT * confidence * expected_distance
                    score += 100.0 * confidence * invader_belief.get(next_position, 0.0)
            elif patrol_target is not None:
                score = -2.0 * self._distance(next_position, patrol_target)
            else:
                score = 0.0

            if not self._is_home_side(next_position):
                score -= 50.0
            scores[action] = score

        self._avoid_or_draw_threat(state, position, legal_actions, scores)
        action = self._select_action(state, scores)
        self._remember_boundary_action(position, action)
        return action
