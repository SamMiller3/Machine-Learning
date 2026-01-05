# 22/12/25 Sudoku solver using Constraint Propagation and the Minimum Remaining Value heuristic

def sudoku_solver(sudoku):
    # Initialise constraint sets
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    blocks = [set() for _ in range(9)]
    
    def block_id(i, j):
        return (i // 3) * 3 + (j // 3)
    
    # Find empty cells and constraints
    empty_cells = []
    for i in range(9):
        for j in range(9):
            val = sudoku[i, j]
            if val == 0:
                empty_cells.append((i, j))
            else:
                b = block_id(i, j)
                if val in rows[i] or val in cols[j] or val in blocks[b]:
                    return np.full((9, 9), -1)
                rows[i].add(val)
                cols[j].add(val)
                blocks[b].add(val)
    
    if not empty_cells: # Sudoku is already solved
        return sudoku
    
    # Find possible values for each cell
    def get_possible(i, j):
        b = block_id(i, j)
        used = rows[i] | cols[j] | blocks[b]
        return [num for num in range(1, 10) if num not in used]
    
    def backtrack(cell_idx):
        if cell_idx == len(empty_cells): # Solution is complete
            return True
        
        # Find cell with minimum remaining values (MRV heuristic)
        min_options = 10
        best_idx = cell_idx
        for idx in range(cell_idx, len(empty_cells)):
            i, j = empty_cells[idx]
            options = get_possible(i, j)
            if len(options) < min_options:
                min_options = len(options)
                best_idx = idx
                if min_options == 0:
                    return False
                if min_options == 1:
                    break
        
        # Swap to process this cell next
        empty_cells[cell_idx], empty_cells[best_idx] = empty_cells[best_idx], empty_cells[cell_idx]
        
        i, j = empty_cells[cell_idx]
        b = block_id(i, j)
        possible = get_possible(i, j)
        
        if not possible:
            # Swap back before returning
            empty_cells[cell_idx], empty_cells[best_idx] = empty_cells[best_idx], empty_cells[cell_idx]
            return False
        
        for val in possible:
            # Place value
            sudoku[i, j] = val
            rows[i].add(val)
            cols[j].add(val)
            blocks[b].add(val)
            
            if backtrack(cell_idx + 1):
                return True
            
            # Backtrack
            sudoku[i, j] = 0
            rows[i].remove(val)
            cols[j].remove(val)
            blocks[b].remove(val)
        
        # Swap back before returning
        empty_cells[cell_idx], empty_cells[best_idx] = empty_cells[best_idx], empty_cells[cell_idx]
        return False
    
    if backtrack(0):
        return sudoku
    else:
        return np.full((9, 9), -1)
