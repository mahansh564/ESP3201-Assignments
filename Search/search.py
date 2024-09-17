# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "ucs": ucs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
    }.get(searchMethod)(maze)


class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position



def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start, objectives = maze.getStart(), maze.getObjectives()
    # BFS has a queue data structure i.e. FIFO order
    queue, explored = [[start]], set()
    
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in explored:
            continue
        if node in objectives:
            return path
        explored.add(node)
        for neighbor in maze.getNeighbors(*node):
            queue.append(path + [neighbor])
    
    return []


def dfs(maze):
    """
    Runs DFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start, objectives = maze.getStart(), maze.getObjectives()
    # DFS has a stack data structure i.e. LIFO order
    stack, explored = [[start]], set()
    
    while stack:
        path = stack.pop()
        node = path[-1]
        if node in explored:
            continue
        if node in objectives:
            return path
        explored.add(node)
        for neighbor in maze.getNeighbors(*node):
            stack.append(path + [neighbor])
    
    return []


def ucs(maze):
    """
    Runs ucs for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start, objectives = maze.getStart(), maze.getObjectives()
    # UCS has a priority-queue data structure i.e. FIFO order ordered by path-cost
    frontier, explored = [(0, [start])], set()
    
    while frontier:
        cost, path = frontier.pop(0)
        node = path[-1]
        if node in explored:
            continue
        if node in objectives:
            return path
        explored.add(node)
        for neighbor in maze.getNeighbors(*node):
            frontier.append((cost + 1, path + [neighbor]))
    
    return []

def astar_search(maze, start, objective):
    frontier = [Node(None, start)]
    explored = set()

    while frontier:
        current_node = min(frontier, key=lambda node: node.f)
        frontier.remove(current_node)

        if current_node.position == objective:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        explored.add(current_node.position)

        for neighbor in maze.getNeighbors(*current_node.position):
            if neighbor in explored:
                continue

            neighbor_node = Node(current_node, neighbor)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = abs(neighbor[0] - objective[0]) + abs(neighbor[1] - objective[1])
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            frontier.append(neighbor_node)

    return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_search(maze, maze.getStart(), maze.getObjectives()[0])


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start, objectives = maze.getStart(), maze.getObjectives()
    full_path = []

    # Find the closest objective using manhattan distance. Find path to that objective, then find the next closest objective and repeat.
    while objectives:
        closest_objective = min(objectives, key=lambda obj: abs(start[0] - obj[0]) + abs(start[1] - obj[1]))
        path = astar_search(maze, start, closest_objective)
        
        if not path:
            return []  # No valid path found
        
        if full_path:
            path = path[1:]
        full_path.extend(path)
        
        start = closest_objective
        objectives.remove(closest_objective)
    
    return full_path

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start, objectives = maze.getStart(), maze.getObjectives()
    full_path = []
    
    # Find the closest objective using manhattan distance. Find path to that objective, then find the next closest objective and repeat.
    while objectives:
        closest_objective = min(objectives, key=lambda obj: abs(start[0] - obj[0]) + abs(start[1] - obj[1]))
        path = astar_search(maze, start, closest_objective)
        
        if not path:
            return []  # No valid path found
        
        if full_path:
            path = path[1:]
        full_path.extend(path)
        
        start = closest_objective
        objectives.remove(closest_objective)
    
    return full_path