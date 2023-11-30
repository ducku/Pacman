from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions

class MiniMaxAgent(ReflexCaptureAgent):
    def __init__(self, index, depth, **kwargs):
        super().__init__(index)
        self.maxDepth = depth

    # Generates the next opponent, if all opponent's index has been visited,
    # returns self's index
    def getNextAgent(self, gameState, agent):
        opponentIndices = self.getOpponents(gameState)
        biggerIndices = [index for index in opponentIndices if index > agent]
        # if no opponent agent's index is bigger, go back to self's index, call get_max next
        if not biggerIndices:
            return self.index

        # return the next biggest agent index
        return min(biggerIndices)

    def chooseAction(self, gameState):
        def get_max(gameState, depth, agent, alpha, beta):
            legalMoves = gameState.getLegalActions(agentIndex=agent)
            resultValue = float("-inf")
            resultAction = Directions.STOP
            for action in legalMoves:
                if action == Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                # Pass -1 to make sure all opponent indices are available
                nextAgent = self.getNextAgent(gameState, -1)
                value, a = mini_max(successor, depth + 1, nextAgent, alpha, beta)
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
            resultAction = Directions.STOP
            
            for action in legalMoves:
                if action == Directions.STOP:
                    continue
                successor = gameState.generateSuccessor(agent, action)
                nextAgent = self.getNextAgent(gameState, agent)
                value, a = mini_max(successor, depth, nextAgent, alpha, beta)
                beta = min(beta, value)
                if value < resultValue:
                    resultValue = value
                    resultAction = action
                if alpha >= beta:
                    break
            return resultValue, resultAction

        def mini_max(gameState, depth, agent, alpha, beta):
            if gameState.isOver() or depth >= self.maxDepth:
                return self.evaluate(gameState, Directions.STOP), Directions.STOP
            
            if agent == self.index:
                return get_max(gameState, depth, agent, alpha, beta)
            else:
                return get_min(gameState, depth, agent, alpha, beta)

        alpha = float("-inf")
        beta = float("inf")
        value, action = mini_max(gameState, 0, self.index, alpha, beta)
        return action


class OffenseAgent(MiniMaxAgent):
    def __init__(self, index, depth, **kwargs):
        super().__init__(index, depth)

    def getFeatures(self, gameState, action):
        features = {}
        if gameState.isOver():
            return features
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToGhost': -1
        }


class DefenseAgent(MiniMaxAgent):
    def __init__(self, index, depth, **kwargs):
        super().__init__(index, depth)

    def getFeatures(self, gameState, action):
        features = {}

        if gameState.isOver():
            return features

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }


def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffenseAgent(firstIndex, 3),
        DefenseAgent(secondIndex, 3),
    ]
