from collections import deque
import heapq

dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def valid(g, x, y):
    return 0 <= x < len(g) and 0 <= y < len(g[0]) and g[x][y] != 1

def find_t(g):
    return next(((i, j) for i, row in enumerate(g) for j, cell in enumerate(row) if cell == 'T'), None)

def path(came_from, cur):
    p = []
    while cur in came_from:
        cur, d = came_from[cur]
        p.append(d)
    return p[::-1]

def find_path_bfs(g, start):
    t = find_t(g)
    if not t: return []
    q, seen, came_from = deque([start]), {start}, {}
    while q:
        x, y = q.popleft()
        if (x, y) == t: return path(came_from, (x, y))
        for d, (dx, dy) in dirs.items():
            nx, ny = x + dx, y + dy
            if valid(g, nx, ny) and (nx, ny) not in seen:
                q.append((nx, ny))
                seen.add((nx, ny))
                came_from[(nx, ny)] = ((x, y), d)
    return []

def find_path_dfs(g, start):
    t = find_t(g)
    if not t: return []
    stack, seen, came_from = [start], {start}, {}
    while stack:
        x, y = stack.pop()
        if (x, y) == t: return path(came_from, (x, y))
        for d, (dx, dy) in dirs.items():
            nx, ny = x + dx, y + dy
            if valid(g, nx, ny) and (nx, ny) not in seen:
                stack.append((nx, ny))
                seen.add((nx, ny))
                came_from[(nx, ny)] = ((x, y), d)
    return []

def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_path_a_star(g, start):
    t = find_t(g)
    if not t: return []
    open_set = [(0, start)]
    came_from, gscore, fscore = {}, {start: 0}, {start: h(start, t)}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == t: return path(came_from, cur)
        for d, (dx, dy) in dirs.items():
            n = (cur[0] + dx, cur[1] + dy)
            if valid(g, *n):
                tg = gscore[cur] + 1
                if tg < gscore.get(n, float('inf')):
                    came_from[n] = (cur, d)
                    gscore[n] = tg
                    fscore[n] = tg + h(n, t)
                    heapq.heappush(open_set, (fscore[n], n))
    return []

grid = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 'T']
]

start = (0, 0)

path_bfs = find_path_bfs(grid, start)
path_dfs = find_path_dfs(grid, start)
path_a_star = find_path_a_star(grid, start)

print("BFS Path:", path_bfs)
print("DFS Path:", path_dfs)
print("A* Path :", path_a_star)