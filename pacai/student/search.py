from pacai.util.priorityQueue import PriorityQueue
"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
actionToVector = {
    "South": (0, -1),
    "North": (0, 1),
    "West": (-1, 0),
    "East": (1, 0)
}


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    
    # *** Your Code Here ***
    stack = []
    visited = set()

    stack.append(problem.startingState())
    visited.add(problem.startingState())

    prevNode = {}
    goal = None
    while stack:
        node = stack.pop()
        if problem.isGoal(node):
            goal = node
            break
        for state, direction, cost in problem.successorStates(node):
            if state not in visited:
                stack.append(state)
                visited.add(state)
                prevNode[state] = (node, direction)

    curr = goal
    actions = []
    while curr in prevNode:
        prevState, action = prevNode[curr]
        actions.insert(0, action)
        curr = prevState
        
    return actions


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    queue = []
    visited = set()

    queue.append(problem.startingState())
    visited.add(problem.startingState())

    prevNode = {}
    goal = None
    while queue:
        node = queue.pop(0)
        if problem.isGoal(node):
            goal = node
            break
        for state, direction, cost in problem.successorStates(node):
            if state not in visited:
                queue.append(state)
                visited.add(state)
                prevNode[state] = (node, direction)

    curr = goal
    actions = []
    while curr in prevNode:
        prevState, action = prevNode[curr]
        actions.insert(0, action)
        curr = prevState
        
    return actions

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    pq = PriorityQueue()
    visited = set()
    visited.add(problem.startingState())
    pq.push((problem.startingState(), 0), 0)

    prevNode = {}
    goal = None
    while not pq.isEmpty():
        node, cost_so_far = pq.pop()
        if problem.isGoal(node):
            goal = node
            break
        for state, direction, cost in problem.successorStates(node):
            if state not in visited:
                new_cost = cost + cost_so_far
                pq.push((state, new_cost), new_cost)
                visited.add(state)
                prevNode[state] = (node, direction)
        
    curr = goal
    actions = []
    while curr in prevNode:
        prevState, action = prevNode[curr]
        actions.insert(0, action)
        curr = prevState
        
    return actions

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    pq = PriorityQueue()
    visited = set()
    visited.add(problem.startingState())
    pq.push((problem.startingState(), 0), 0)

    prevNode = {}
    goal = None
    while not pq.isEmpty():
        node, cost_so_far = pq.pop()
        if problem.isGoal(node):
            goal = node
            break
        for state, direction, cost in problem.successorStates(node):
            if state not in visited:
                new_cost = cost + cost_so_far + heuristic(state, problem)
                pq.push((state, new_cost), new_cost)
                visited.add(state)
                prevNode[state] = (node, direction)
        
    curr = goal
    actions = []
    while curr in prevNode:
        prevState, action = prevNode[curr]
        actions.insert(0, action)
        curr = prevState
        
    return actions
