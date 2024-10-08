from game import Directions, Agent, Actions

import numpy as np
from util import raiseNotDefined



class ValueEstimationAgent(Agent):
  """
    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a 
    state and a policy given respectively by,
    
    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)
    
    Both ValueIterationAgent and QLearningAgent inherit 
    from this agent. While a ValueIterationAgent has
    a model of the environment via a MarkovDecisionProcess
    (see mdp.py) that is used to estimate Q-Values before
    ever actually acting, the QLearningAgent estimates 
    Q-Values while acting in the environment. 
  """
  
  def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.gamma = float(gamma)
    self.numTraining = int(numTraining)
    
  ####################################
  #    Override These Functions      #  
  ####################################
  def getQValue(self, state, action):
    """
    Should return Q(state,action)
    """
    raiseNotDefined()
    
  def getValue(self, state):
    """
    What is the value of this state under the best action? 
    Concretely, this is given by
    
    V(s) = max_{a in actions} Q(s,a)
    """
    raiseNotDefined()  
    
  def getPolicy(self, state):
    """
    What is the best action to take in the state. Note that because
    we might want to explore, this might not coincide with getAction
    Concretely, this is given by
    
    policy(s) = arg_max_{a in actions} Q(s,a)
    
    If many actions achieve the maximal Q-value,
    it doesn't matter which is selected.
    """
    raiseNotDefined()  
    
  def getAction(self, state):
    """
    state: can call state.getLegalActions()
    Choose an action and return it.   
    """
    raiseNotDefined() 


class BaseAgent:
    def __init__(self):
        ReinforcementAgent.__init__(self, **args)
        "You can initialize Q-values here..."
        
        "*** YOUR CODE HERE ***"
  
    def getQValue(self, state, action):
        """
          Returns Q(state,action)    
          Should return 0.0 if we never seen
          a state or (state,action) tuple 
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()

    
    def getValue(self, state):
        """
          Returns max_action Q(state,action)        
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()
    
    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()
    
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """  
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        raiseNotDefined()
        
        return action
  
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a 
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()