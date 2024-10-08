# textGridworldDisplay.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
from tabulate import tabulate

class TextGridworldDisplay:
  
  def __init__(self, gridworld):
    self.gridworld = gridworld

  def start(self):
    pass
  
  def pause(self):
    pass
  
  def displayValues(self, agent, currentState = None, message = None):
    if message != None:
      print(message)
    values = util.Counter()
    policy = {}
    states = self.gridworld.getStates()
    for state in states:
      values[state] = agent.getValue(state)
      policy[state] = agent.getPolicy(state)
    printValues(self.gridworld, values, policy, currentState)
  
  def displayNullValues(self, currentState = None, message = None):
    if message != None: print(message)
    printNullValues(self.gridworld, currentState)

  def displayQValues(self, agent, currentState = None, message = None):
    if message != None: print(message)
    qValues = util.Counter()
    states = self.gridworld.getStates()
    for state in states:
      for action in self.gridworld.getPossibleActions(state):
        qValues[(state, action)] = agent.getQValue(state, action)
    printQValues(self.gridworld, qValues, currentState)


def insert(source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

def printValues(gridWorld, values, policy=None, currentState = None):
  grid = gridWorld.grid
  maxLen = 11
  newRows = []
  for y in range(grid.height):
    newRow = []
    for x in range(grid.width):
      state = (x, y)
      value = values[state]
      action = None
      if policy != None and state in policy:
        action = policy[state]          
      actions = gridWorld.getPossibleActions(state)   
      
      if action not in actions and 'exit' in actions:
        action = 'exit'
      valString = None

      if action == 'exit':
        valString = 'e: %.2f' % value
      else:
        valString = '%.2f,' % value

      if grid[x][y] == 'S':
        valString = 'S: %.2f'  % value
      if grid[x][y] == '#':
        valString = '#'
 
      if currentState == state:
         valString = insert(valString, '*', 1)

      if action == 'east':
        valString += ' >'
      elif action == 'west':
        valString +=  ' <'
      elif action == 'north':
        valString += ' ^'
      elif action == 'south':
        valString +=  ' v'


      text = valString 
    
      newRow.append(text)
    newRows.append(newRow)

  # reverse to print
  newRows.reverse()
  print(tabulate(newRows))


def printNullValues(gridWorld, currentState = None):
  grid = gridWorld.grid
  maxLen = 11
  newRows = []
  for y in range(grid.height):
    newRow = []
    for x in range(grid.width):
      state = (x, y)

      action = None
        
      if grid[x][y] == 'S':
          valString = 'S'
      elif grid[x][y] == '#':
          valString = '#'
      elif type(grid[x][y]) == float or type(grid[x][y]) == int:
          valString = ('%.2f' % float(grid[x][y]))
      else:
          valString = ' '
 
      if currentState == state:
         valString += '*'

      text = valString 
    
      newRow.append(text)
    newRows.append(newRow)

  # reverse to print
  newRows.reverse()
  print(tabulate(newRows))



def printQValues(gridWorld, qValues, currentState=None): # gridWorld, values, policy=None, currentState = None
  grid = gridWorld.grid
  maxLen = 11
  newRows = []
  for y in range(grid.height):
    newRow = []

    top_row = []
    mid_row = []
    btm_row = []
    bbtm_row = []

    for x in range(grid.width):

      top_print = [' ',' ',' ']
      mid_print = [' ',' ',' ']
      btm_print = [' ',' ',' ']
      bbtm_print = ['-----', '-----', '-----']

      state = (x, y)
      actions = gridWorld.getPossibleActions(state)
      if actions == None or len(actions) == 0:
        actions = [None]
      bestQ = max([qValues[(state, action)] for action in actions])
      bestActions = [action for action in actions if qValues[(state, action)] == bestQ]

      qStrings = dict([(action, "%.2f" % qValues[(state, action)]) for action in actions])

      northString = ('north' in qStrings and qStrings['north']) or ' '
      southString = ('south' in qStrings and qStrings['south']) or ' '
      eastString = ('east' in qStrings and qStrings['east']) or ' '
      westString = ('west' in qStrings and qStrings['west']) or ' '
      exitString = ('exit' in qStrings and qStrings['exit']) or ' '

      top_print[1] = northString
      btm_print[1] = southString
      mid_print[2] = eastString + mid_print[2]
      mid_print[0] = mid_print[0] + westString
      mid_print[1] = mid_print[1] + exitString

      stringToPrint = 'w/n/s/e: %s, %s, %s, %s, ex:%s, A: '%(westString, northString, southString, eastString, exitString)

      if 'north' in bestActions:
        top_print[1] =  '/' + top_print[1] + '\\'
      if 'south' in bestActions:
        btm_print[1] = '\\' + btm_print[1] + '/'
      if 'east' in bestActions:
        mid_print[2] =  mid_print[2] + ' >'
      if 'west' in bestActions:
        mid_print[0] = '<' + mid_print[0]
      if 'exit' in bestActions:
        mid_print[1] = '[' + mid_print[1] +  ']'


      if grid[x][y] == '#':
        mid_print[1] = '#' + mid_print[1] +  '#'
      if state == gridWorld.getStartState():
        mid_print[1] = 'Start ' + mid_print[1] 
      if state == currentState:
        mid_print[1] = '*' + mid_print[1] 

      top_print[0] = '| ' + top_print[0]
      mid_print[0] = '| ' + mid_print[0]
      btm_print[0] = '| ' + btm_print[0]


      text = stringToPrint 
    
      newRow.append(text)
      top_row.append(top_print)
      mid_row.append(mid_print)
      btm_row.append(btm_print)
      bbtm_row.append(bbtm_print)


    top_row = [item for sublist in top_row for item in sublist]
    mid_row = [item for sublist in mid_row for item in sublist]
    btm_row = [item for sublist in btm_row for item in sublist]
    bbtm_row = [item for sublist in bbtm_row for item in sublist]

    # reverse order 
    newRows.append(bbtm_row)
    newRows.append(btm_row)
    newRows.append(mid_row)
    newRows.append(top_row)

  # reverse to print
  newRows.reverse()
  print(tabulate(newRows))

