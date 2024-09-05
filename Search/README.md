Modified from https://courses.grainger.illinois.edu/ece448/sp2020/MPs/mp1/assignment1.html

## Implement:
Write your search algorithms in *search.py* and do not edit any other files, except for testing.

## Requirements:
```
python3
pygame
```
## Running:
The main file to run the mp is mp1.py:

```
usage: mp1.py [-h] [--method {bfs,dfs,dijkstra,astar, astar_corner}] [--scale SCALE]
              [--fps FPS] [--save SAVE]
              filename
```

Examples of how to run MP1:
```
python mp1.py map/bigMaze.txt --method dfs
```
```
python mp1.py map/tinyMaze.txt --scale 30 --fps 10 --method bfs
```

For help run:
```
python mp1.py -h
```
Help Output:
```
positional arguments:
  filename              path to maze file [REQUIRED]

optional arguments:
  -h, --help            show this help message and exit
  --method {dfs, bfs, dijkstra, astar}
                        search method - default bfs
  --scale SCALE         scale - default: 20
  --fps FPS             fps for the display - default 30
  --save SAVE           save output to image file - default not saved
```
