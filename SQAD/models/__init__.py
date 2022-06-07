from .sqad import SQAD

def sqad():
    net = SQAD(1, 16, 3, [1,2])
    net.bandwise = False
    return net