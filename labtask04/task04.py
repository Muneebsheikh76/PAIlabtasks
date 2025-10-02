n = 4
ls = [[" " for _ in range(n)] for _ in range(n)]  
def is_safe(row, col):
    for i in range(row):
        if ls[i][col] == "Q":
            return False
    i, j = row, col
    while i >= 0 and j >= 0:
        if ls[i][j] == "Q":
            return False
        i -= 1
        j -= 1
    i, j = row, col
    while i >= 0 and j < n:
        if ls[i][j] == "Q":
            return False
        i -= 1
        j += 1
    return True
def solve(row=0):
    if row == n:
        print_solution()
        print("\n" + "="*20 + "\n")
        return
    for col in range(n):
        if is_safe(row, col):
            ls[row][col] = "Q"
            solve(row + 1)
            ls[row][col] = " "
def print_solution():
    for i in range(n):
        for j in range(n):
            print(f" {ls[i][j]} ", end="")
            if j < n - 1:
                print("|", end="")
        print()
        if i < n - 1:
            for j in range(n):
                print("- -", end="")
                if j < n - 1:
                    print(" ", end="")
            print()