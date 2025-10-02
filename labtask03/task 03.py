import math
cap1, cap2, target = 4, 3, 2
start = (0, 0)
visited = set()
path = []
def is_goal(state):
    return state[0] == target or state[1] == target
def dfs(state):
    x, y = state
    if state in visited:
        return False
    visited.add(state)
    if is_goal(state):
        return True
    successors = []
    if x < cap1: successors.append(("Fill x", (cap1, y)))
    if y < cap2: successors.append(("Fill y", (x, cap2)))
    if x > 0: successors.append(("Empty x", (0, y)))
    if y > 0: successors.append(("Empty y", (x, 0)))
    if x > 0 and y < cap2:
        t = min(x, cap2 - y)
        successors.append((f"Pour x -> y ({t}L)", (x - t, y + t)))
    if y > 0 and x < cap1:
        t = min(y, cap1 - x)
        successors.append((f"Pour y -> x ({t}L)", (x + t, y - t)))
    for action, nxt in successors:
        path.append((action, nxt))
        if dfs(nxt):
            return True
        path.pop()
    return False
if dfs(start):
    print("Start => (x: 0L, y: 0L)")
    for i, (action, (x, y)) in enumerate(path, start=1):
        print(f"{i}. {action} => (x: {x}L, y: {y}L)")
else:
    print("No solution found.")
