import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core import directions


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()

        # *** Your Code Here ***
        gPos = successorGameState.getGhostPositions()
        bonus = 0
        ghostDist = [distance.manhattan(newPosition, ghostPos) for ghostPos in gPos]
        ghostDist.append(float("inf"))
        closestGhost = min(ghostDist)

        if closestGhost == 1:
            bonus -= 40
        elif closestGhost >= 2 and closestGhost <= 5:
            bonus -= 5 - closestGhost

        newFood = successorGameState.getFood().asList()
        if len(newFood) < len(oldFood):
            bonus += 30
        
        food_dist = [distance.manhattan(newPosition, foodPos) for foodPos in newFood]
        if not food_dist:
            food_dist = [distance.manhattan(newPosition, oldFood[0])]
        closestFood = min(food_dist)

        if action == "Stop":
            bonus -= 50

        return successorGameState.getScore() - closestFood + bonus

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def get_max(gameState, depth, agent):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("-inf")
            resultAction = directions.Directions.STOP
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                value, a = mini_max(successor, depth + 1, agent + 1)
                if value > resultValue:
                    resultValue = value
                    resultAction = action
            return resultValue, resultAction

        def get_min(gameState, depth, agent):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("inf")
            resultAction = directions.Directions.STOP
            
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                value, a = mini_max(successor, depth, nextAgent)
                if value < resultValue:
                    resultValue = value
                    resultAction = action
            return resultValue, resultAction

        def mini_max(gameState, depth, agent):
            if gameState.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(gameState), directions.Directions.STOP
            
            if agent == 0:
                return get_max(gameState, depth, agent)
            else:
                return get_min(gameState, depth, agent)

        value, action = mini_max(gameState, 0, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def get_max(gameState, depth, agent, alpha, beta):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("-inf")
            resultAction = directions.Directions.STOP
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                value, a = mini_max(successor, depth + 1, agent + 1, alpha, beta)
                alpha = max(alpha, value)
                if value > resultValue:
                    resultValue = value
                    resultAction = action
                if alpha >= beta:
                    break
                
            return resultValue, resultAction

        def get_min(gameState, depth, agent, alpha, beta):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("inf")
            resultAction = directions.Directions.STOP
            
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                value, a = mini_max(successor, depth, nextAgent, alpha, beta)
                beta = min(beta, value)
                if value < resultValue:
                    resultValue = value
                    resultAction = action
                if alpha >= beta:
                    break
            return resultValue, resultAction

        def mini_max(gameState, depth, agent, alpha, beta):
            if gameState.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(gameState), directions.Directions.STOP
            
            if agent == 0:
                return get_max(gameState, depth, agent, alpha, beta)
            else:
                return get_min(gameState, depth, agent, alpha, beta)

        alpha = float("-inf")
        beta = float("inf")
        value, action = mini_max(gameState, 0, 0, alpha, beta)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def get_max(gameState, depth, agent):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("-inf")
            resultAction = directions.Directions.STOP
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                value, a = expecti_max(successor, depth + 1, agent + 1)
                if value > resultValue:
                    resultValue = value
                    resultAction = action
            return resultValue, resultAction

        def get_random(gameState, depth, agent):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            valuesSum = 0
            resultActions = []
            
            for action in legalMoves:
                if action == directions.Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                nextAgent = agent + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                value, a = expecti_max(successor, depth, nextAgent)
                valuesSum += value
                resultActions.append(a)

            average = valuesSum / len(resultActions)
            return average, random.choice(resultActions)

        def expecti_max(gameState, depth, agent):
            if gameState.isOver() or depth >= self.getTreeDepth():
                return self.getEvaluationFunction()(gameState), directions.Directions.STOP
            
            if agent == 0:
                return get_max(gameState, depth, agent)
            else:
                return get_random(gameState, depth, agent)

        value, action = expecti_max(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()

    ghostDist = [distance.manhattan(pacmanPosition, g.getPosition()) for g in ghostStates]
    ghostDist.append(float("inf"))
    closestGhost = min(ghostDist)

    bonus = 0
    if closestGhost == 1:
        bonus -= 10000
    elif closestGhost >= 2 and closestGhost <= 5:
        bonus -= 5 - closestGhost
    
    food_dist = [distance.manhattan(pacmanPosition, foodPos) for foodPos in food]
    if not food_dist:
        food_dist = [0]
    closestFood = min(food_dist)
    furthestFood = max(food_dist)

    features = [currentGameState.getScore(), closestFood, len(food), bonus, furthestFood]
    multiplier = [10, -10, -10, 30, -10]

    return sum([f * m for f, m in zip(features, multiplier)])

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
