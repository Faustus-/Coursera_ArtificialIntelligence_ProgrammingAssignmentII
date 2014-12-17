# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"
    "*** YOUR CODE HERE ***"
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    "initialize score"
    score = successorGameState.getScore()
    "pacman"
    pacPosition = successorGameState.getPacmanPosition()
    "food"
    foodPositions = successorGameState.getFood().asList()
    foodDistances = [util.manhattanDistance(pacPosition, food) for food in foodPositions]
    "ghosts"
    ghostPositions = successorGameState.getGhostPositions()
    ghostStates = successorGameState.getGhostStates()
    ghostDistances = [util.manhattanDistance(pacPosition, ghost) for ghost in ghostPositions]
    
    closestGhostDistance = min(ghostDistances)
    if len(foodDistances) > 0:
        closestFoodDistance = min(foodDistances)
    else:
        closestFoodDistance = -10
        
    if closestGhostDistance <= 1:
        score = score - 100 * closestGhostDistance
    else:
        score = score + 2 * closestGhostDistance - 5 * closestFoodDistance
    return score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    """check the max score in the next step for pacman agent.
    parameters: gameState, deepCounter, lastMove
    return: (bestScore, bestMove)
    """
    def pacmanAction(gameState, deepCounter, lastMove):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isWin() or gameState.isLose() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = -inf"
            bestScore = float('inf') * (-1)
            "get all legal moves"
            moves = gameState.getLegalActions(self.index)
            "find bestScore and corresponding bestMove"
            for move in moves:
                moveScore = (ghostAction(gameState.generateSuccessor(self.index, move), deepCounter, move, gameState.getNumAgents()-1))[0] 
                if moveScore> bestScore:
                    bestMove = move
                    bestScore = moveScore
            return (bestScore, bestMove) 

    """check the min score in the next step around all ghost agents.
    parameters: gameState, deepCounter, lastMove, numGhosts
    return: (bestScore, bestMove)
    """
    def ghostAction(gameState, deepCounter, lastMove, numGhosts):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isLose() or gameState.isWin() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = inf"
            bestScore = float('inf')
            numAgents = gameState.getNumAgents()
            "get legal actions for current ghost agent"
            moves = gameState.getLegalActions(numAgents - numGhosts)
            "find bestScore and corresponding bestMove for the current ghost agent"
            for move in moves:
                if numGhosts == 1:
                    "compete with pacman agent"
                    moveScore = (pacmanAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter-1, move))[0] 
                else:
                    "cooperate with other ghost agents"
                    moveScore = (ghostAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter, move, numGhosts-1))[0] 
                if moveScore < bestScore:
                    bestMove = move
                    bestScore = moveScore
            return (bestScore, bestMove)
        
    if self.index == 0:
        bestAction = (pacmanAction(gameState, self.depth, ""))[1]
    else:
        bestAction = (ghostAction(gameState, self.depth, ""))[1]
    return bestAction

    # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    
    """check the max score in the next step for pacman agent.
    parameters: gameState, deepCounter, lastMove
    return: (bestScore, bestMove)
    """
    def pacmanAction(gameState, deepCounter, lastMove, alpha, beta):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isWin() or gameState.isLose() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = -inf"
            bestScore = float('inf') * (-1)
            "get all legal moves"
            moves = gameState.getLegalActions(self.index)
            "find bestScore and corresponding bestMove"
            for move in moves:
                moveScore = (ghostAction(gameState.generateSuccessor(self.index, move), deepCounter, move, gameState.getNumAgents()-1, alpha, beta))[0] 
                if moveScore> bestScore:
                    bestMove = move
                    bestScore = moveScore
                if bestScore > beta:
                    break
                alpha = max(alpha, bestScore)
            return (bestScore, bestMove) 

    """check the min score in the next step around all ghost agents.
    parameters: gameState, deepCounter, lastMove, numGhosts
    return: (bestScore, bestMove)
    """
    def ghostAction(gameState, deepCounter, lastMove, numGhosts, alpha, beta):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isLose() or gameState.isWin() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = inf"
            bestScore = float('inf')
            numAgents = gameState.getNumAgents()
            "get legal actions for current ghost agent"
            moves = gameState.getLegalActions(numAgents - numGhosts)
            "find bestScore and corresponding bestMove for the current ghost agent"
            for move in moves:
                if numGhosts == 1:
                    "compete with pacman agent"
                    moveScore = (pacmanAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter-1, move, alpha, beta))[0] 
                else:
                    "cooperate with other ghost agents"
                    moveScore = (ghostAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter, move, numGhosts-1, alpha, beta))[0] 
                if moveScore < bestScore:
                    bestMove = move
                    bestScore = moveScore
                if bestScore < alpha:
                    break
                beta = min(beta, bestScore)
            return (bestScore, bestMove)

    alpha = float('inf') * (-1)
    beta = -1 * alpha        
    if self.index == 0:
        bestAction = (pacmanAction(gameState, self.depth, "", alpha, beta))[1]
    else:
        bestAction = (ghostAction(gameState, self.depth, "", alpha, beta))[1]
    return bestAction
    

    
    "util.raiseNotDefined()"

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    
    """check the max score in the next step for pacman agent.
    parameters: gameState, deepCounter, lastMove
    return: (bestScore, bestMove)
    """
    def pacmanAction(gameState, deepCounter, lastMove):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isWin() or gameState.isLose() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = -inf"
            bestScore = float('inf') * (-1)
            "get all legal moves"
            moves = gameState.getLegalActions(self.index)
            "find bestScore and corresponding bestMove"
            for move in moves:
                moveScore = (ghostAction(gameState.generateSuccessor(self.index, move), deepCounter, move, gameState.getNumAgents()-1))[0] 
                if moveScore> bestScore:
                    bestMove = move
                    bestScore = moveScore
            return (bestScore, bestMove) 

    """check the min score in the next step around all ghost agents.
    parameters: gameState, deepCounter, lastMove, numGhosts
    return: (bestScore, bestMove)
    """
    def ghostAction(gameState, deepCounter, lastMove, numGhosts):
        "if current state is win or lose or reach the deepest layer, return"
        if gameState.isLose() or gameState.isWin() or deepCounter == 0:
            return (self.evaluationFunction(gameState), lastMove)
        else:
            "initialize best score = inf"
            bestScore = float('inf')
            numAgents = gameState.getNumAgents()
            "get legal actions for current ghost agent"
            moves = gameState.getLegalActions(numAgents - numGhosts)
            "find bestScore and corresponding bestMove for the current ghost agent"
            totalScore = 0
            bestMove = random.sample(moves, 1)[0]
            for move in moves:
                p = 1.0 / len(moves)
                if numGhosts == 1:
                    "compete with pacman agent"
                    moveScore = (pacmanAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter-1, move))[0] 
                else:
                    "cooperate with other ghost agents"
                    moveScore = (ghostAction(gameState.generateSuccessor(numAgents - numGhosts, move), deepCounter, move, numGhosts-1))[0] 
                totalScore = totalScore + p * moveScore
            return (totalScore, bestMove)
        
    if self.index == 0:
        bestAction = (pacmanAction(gameState, self.depth, ""))[1]
    else:
        bestAction = (ghostAction(gameState, self.depth, ""))[1]
    return bestAction
    
    "util.raiseNotDefined()"

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  
  "initialize score"
  score = currentGameState.getScore()
  "pacman"
  pacPosition = currentGameState.getPacmanPosition()
  
  "food"
  foodPositions = currentGameState.getFood().asList()
  foodDistances = [util.manhattanDistance(pacPosition, food) for food in foodPositions]
  
  "ghosts"
  ghostPositions = currentGameState.getGhostPositions()
  ghostStates = currentGameState.getGhostStates()
  ghostDistances = [util.manhattanDistance(pacPosition, ghost) for ghost in ghostPositions]
  
  
  closestGhostDistance = min(ghostDistances)
  if len(foodDistances) > 0:
      closestFoodDistance = min(foodDistances)
  else:
      closestFoodDistance = -10
  
  if closestGhostDistance <= 1:
      score = score - 100 * closestGhostDistance
  else:
      score = score + 2 * closestGhostDistance - closestFoodDistance
  
  return score
  "util.raiseNotDefined()"

# Abbreviation
better = betterEvaluationFunction


