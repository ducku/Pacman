"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.directions import Directions
from pacai.student import search
from pacai.core.distance import maze

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        self.remainingGoals = tuple(c for c in self.corners if c != self.startingPosition)
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        return None

    def startingState(self):
        return (self.startingPosition, self.remainingGoals)

    def isGoal(self, state):
        (coordinates, remaining) = state
        if not remaining:
            return True
        return False

    def successorStates(self, state):
        successors = []

        for action in Directions.CARDINAL:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.
                remaining = state[1]
                newRemaining = tuple(c for c in remaining if c != (nextx, nexty))
                successors.append((((nextx, nexty), newRemaining), action, 1))

        self._numExpanded += 1
        if (state not in self._visitedLocations):
            self._visitedLocations.add(state)

        return successors
    
    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    coordinates, remaining = state
    x, y = coordinates

    result = 0
    while remaining:
        closestHeuristic = float("inf")
        closestCorner = None
        for goalX, goalY in remaining:
            manDist = abs(goalX - x) + abs(goalY - y)
            if manDist < closestHeuristic:
                closestHeuristic = manDist
                closestCorner = (goalX, goalY)
        remaining = set(c for c in remaining if c != closestCorner)
        x, y = closestCorner
        result += closestHeuristic

    return result

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state
    x, y = position
    gameState = problem.startingGameState

    if "furthestFood" not in problem.heuristicInfo:
        problem.heuristicInfo["furthestFood"] = (-1, -1)
        problem.heuristicInfo["foodDP"] = {}

    foodSet = set()
    for i in range(foodGrid.getWidth()):
        for j in range(foodGrid.getHeight()):
            if foodGrid[i][j]:
                foodSet.add((i, j))
                
    furthestFood = (-1, -1)
    furthestDistance = float("-inf")

    for foodPosition in foodSet:
        for otherFoodPosition in foodSet:
            if foodPosition == otherFoodPosition:
                continue
            dist = None
            distStr = str(foodPosition) + str(otherFoodPosition)
            if distStr in problem.heuristicInfo["foodDP"]:
                dist = problem.heuristicInfo["foodDP"][distStr]
            else:
                dist = maze(foodPosition, otherFoodPosition, gameState)
                problem.heuristicInfo["foodDP"][distStr] = dist
                problem.heuristicInfo["foodDP"][str(otherFoodPosition) + str(foodPosition)] = dist
            if dist > furthestDistance:
                furthestFood = (foodPosition, otherFoodPosition)
                furthestDistance = dist
    problem.heuristicInfo["furthestFood"] = furthestFood
        
    positionToFood = 0
    if furthestFood and furthestFood != (-1, -1):
        positionToFood = min([maze(position, fp, gameState) for fp in furthestFood])
    elif foodSet:
        positionToFood = min([maze(position, fp, gameState) for fp in foodSet])
    if furthestDistance == float("-inf"):
        furthestDistance = 0

    return furthestDistance + positionToFood
    

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        return search.uniformCostSearch(AnyFoodSearchProblem(gameState))

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.startingPosition = gameState.getPacmanPosition()
        self.food = gameState.getFood()

    def startingState(self):
        return (self.startingPosition)

    def isGoal(self, state):
        return state in self.food.asList()

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
