"""Helper module to easily create uniform distributions of particles"""

from __future__ import print_function
import numpy

def uniform_distribution_hcp2D(dx, xmin, xmax, ymin, ymax, adjust=False):
    """Hexagonal closed packing arrangement in 2D"""
    dy = 0.5 * numpy.sqrt(3.0) * dx
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy
    
    # since we will be shifting each alternate row by dxb2, we use
    # xstart as dx/4
    xstart = xmin + 0.25 * dx

    ystart = ymin+dyb2

    # adjust ymax so that particles can fill a periodic region
    if adjust:
        _y = numpy.arange(ystart, ymax, dy)
        ymax = _y[-1] + 1.5*dy
    
    # create the points
    x, y = numpy.mgrid[xstart:xmax:dx,
                       ystart:ymax:dy]

    # each alternate row is shifted by dxb2
    x[:,::2] += dxb2
    x = x.ravel(); y = y.ravel()

    print('HCP packing domain: xmin, xmax, ymin, ymax =  ', xmin, xmax, ymin,
            ymax)
    print('HCP packing particles: xmin, xmax, ymin, ymax = ', x.min(), x.max(),
            y.min(), y.max())
    print('Particle spacings: dx, dy = ', dx, dy)
    print('Offset: xmin, xmax = ', x.min()-xmin, xmax-x.max())
    print('Offset: ymin, ymax = ', y.min()-ymin, ymax-y.max())

    return x, y, dx, dy, xmin, xmax, ymin, ymax

def uniform_distribution_cubic2D(dx, xmin, xmax, ymin, ymax, nrows=None):
    """Cubic lattice arrangement in 2D"""
    dy = dx
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy
    
    if nrows is not None:
        ymax = nrows * dy

    xstart = xmin + dxb2
    ystart = ymin + dyb2
    x, y = numpy.mgrid[xstart:xmax:dx,
                       ystart:ymax:dy]

    x = x.ravel(); y = y.ravel()

    print('Cubic packing domain: xmin, xmax, ymin, ymax =  ', xmin, xmax, ymin,
            ymax)
    print('Cubic packing particles: xmin, xmax, ymin, ymax = ', x.min(),
            x.max(), y.min(), y.max())
    print('Particle spacings: dx, dy = ', dx, dy)
    print('Offset: xmin, xmax = ', x.min()-xmin, xmax-x.max())
    print('Offset: ymin, ymax = ', y.min()-ymin, ymax-y.max())

    return x, y, dx, dy, xmin, xmax, ymin, ymax

def get_number_density_hcp(dx, dy, kernel, h0):

    # create a dummy particle distribution with the reference spacings
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy

    xstart = 0.25 * dx
    ystart = dyb2

    # create the points
    x, y = numpy.mgrid[xstart:1.0:dx,
                       ystart:1.0:dy]

    # each alternate row is shifted by dxb2
    x[:,::2] += dxb2

    # the target point
    nrows, ncols = x.shape
    x0, y0 = x[nrows/2, ncols/2], y[nrows/2, ncols/2]

    x = x.ravel(); y = y.ravel()

    # now do a kernel sum
    wij_sum = 0.0
    for i in range(x.size):
        xij = x0 - x[i]
        yij = y0 - y[i]
        zij = 0.0
        
        rij = numpy.sqrt( xij**2 + yij**2 + zij**2 )
        wij_sum += kernel.kernel( [xij, yij, zij], rij, h0 )
                
    return wij_sum

def generate_bcc3D(a, xmin, xmax, ymin, ymax, zmin, zmax):
    """Body centered packing arrangement in 3D"""
    # a = side of unit cell
    #
    # | z
    # |
    # |_________x
    # \
    #  \
    #   \y
    #
    dx = dy = a
    dz = 0.5 * a
    dxb2 = 0.5 * dx
    dyb2 = 0.5 * dy
    dzb2 = 0.5 * dz

    # since we will be shifting each alternate layer by dxb2 and dyb2, we use
    # xstart as dx/4 and dy/4
    xstart = xmin + 0.5 * dxb2
    ystart = xmin + 0.5 * dyb2
    zstart = zmin + dzb2

    # create the points
    x, y, z = numpy.mgrid[xstart:xmax:dx, ystart:ymax:dy, zstart:zmax:dz]

    # each alternate layer is shifted
    x[:, :, ::2] += dxb2
    y[:, :, ::2] += dyb2

    print(f'BCC packing domain: {xmin = }, {xmax = }, {ymin = }, {ymax = },'
          f' {zmin = }, {zmax = }')

    print(f'BCC packing particles: {x.min() = }, {x.max() = }, {y.min() = }, '
          f'{y.max() = }, {z.min() = }, {z.max() = }')

    print(f'Particle spacings: \n'
          f'{numpy.min(x[:, :, 1] - x[:, :, 0]) = }, \n'
          f'{numpy.min(y[:, :, 1] - y[:, :, 0]) = }, \n'
          f'{numpy.min(z[:, :, 1] - z[:, :, 0]) = }')

    print(f'Offsets: \n'
          f'{x.min() - xmin = } \n'
          f'{y.min() - ymin = } \n'
          f'{z.min() - zmin = } \n'
          f'{x.max() - xmax = } \n'
          f'{y.max() - ymax = } \n'
          f'{z.max() - zmax = }')

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    return x, y, z