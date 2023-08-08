
# this is obsolete
# use draggan_utils.py instead


def list2dict(points,targets):
    assert len(points) == len(targets)
    dict = {}
    for i in range(len(points)):
        dict[i] = {'start': points[i], 'target': targets[i]}
    return dict
