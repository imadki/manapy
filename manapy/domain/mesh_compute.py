from manapy.backends.compile_fun import compile

def _append(cells: 'int[:, :]', cells_item: 'int[:, :]', counter: 'int'):
  for i in range(len(cells_item)):
    cells[counter, 0:len(cells_item[i])] = cells_item[i]
    cells[counter, -1] = len(cells_item[i])
    counter += 1

def _append_1d(arr_dest: 'int[:]', arr_src: 'int[:]', counter: 'int'):
  for i in range(len(arr_src)):
    arr_dest[counter] = arr_src[i]
    counter += 1

# public
append = compile(_append)
append_1d = compile(_append_1d)