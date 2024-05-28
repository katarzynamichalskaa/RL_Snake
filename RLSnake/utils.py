import random

# global variables
DISPLAY_BOOL = True
GENERATE_OBSTACLES = True
MODEL_LOADING_BOOL = True


# global functions
def random_coords(dims, unit):
    return tuple(round(random.randrange(0, dim - unit) / 10.0) * 10.0 for dim in dims)
