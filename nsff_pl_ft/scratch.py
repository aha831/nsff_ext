from kornia import create_meshgrid

grid = create_meshgrid(2, 3, normalized_coordinates=False)[0] # (H, W, 2)
x, y = grid.unbind(-1)

print(type(grid), grid)
print(type(x), x)
print(type(y), y)