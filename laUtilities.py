import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotSetup(xmin = -6.0, xmax = 6.0, ymin = -2.0, ymax = 4.0, size=(6,4)):
    """
    basics of 2D plot setup
    defaults: xmin = -6.0, xmax = 6.0, ymin = -2.0, ymax = 4.0, size=(6,4)
    size is by default 6 inches by 4 inches
    """
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    ax.axes.set_xlim([xmin, xmax])
    return ax

def formatEqn(coefs, b):
    """
    format a set of coefficients as a linear equation in text
    """
    leadingLabel = {-1: '-{} x_{}', 0: '', 1: '{} x_{}'}
    followingLabel = {-1: ' - {} x_{}', 0: '', 1: ' + {} x_{}'}
    nterms = len(coefs)
    i = 0
    # skip initial terms with coefficient zero
    while ((i < nterms) and (np.sign(coefs[i]) == 0)):
        i += 1
    # degenerate equation 
    if (i == nterms):
        return '0 = {}'.format(b)
    # first term is formatted slightly differently
    if (np.abs(coefs[i]) == 1):
        label = leadingLabel[np.sign(coefs[i])].format('',i+1)
    else:
        label = leadingLabel[np.sign(coefs[i])].format(np.abs(coefs[i]),i+1)
    # and the rest of the terms if any exist
    for j in range(i+1,len(coefs)):
        if (np.abs(coefs[j]) == 1):
            label = label + followingLabel[np.sign(coefs[j])].format('',j+1)
        else:
            label = label + followingLabel[np.sign(coefs[j])].format(np.abs(coefs[j]),j+1)
    label = label + ' = {}'.format(b)
    return label

def plotPoint (ax, x1, x2, color='r'):
    ax.plot(x1, x2, '{}o'.format(color))

def plotVec (ax, x1, color='r'):
    ax.plot(x1[0], x1[1], '{}o'.format(color))

def plotArrow (ax, x1, x2):
    ax.arrow(0.0, 0.0, x1, x2)

def plotArrowVec(ax, v, start = [0,0], head_width=0.2, head_length=0.2, length_includes_head = True, color='Red'):
    try:
        ax.arrow(start[0],start[1],v[0]-start[0],v[1]-start[1],head_width=head_width, head_length=head_length, length_includes_head = length_includes_head, color=color)
    # if the arrow length is zero, raises an IndexError
    except IndexError:
        pass

def plotLinEqn (a1, a2, b, format='-', color='r'):
    """
    plot line line corresponding to the linear equation
    a1 x + a2 y = b
    """
    [xmin, xmax] = plt.xlim()
    x1 = xmin
    y1 = (b - (x1 * a1))/float(a2)
    x2 = xmax
    y2 = (b - (x2 * a1))/float(a2)
    plt.plot([x1, x2],[y1, y2], format, label='${}$'.format(formatEqn([a1, a2],b)),color=color)

def centerAxes (ax):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    bounds = np.array([ax.axes.get_xlim(), ax.axes.get_ylim()])
    ax.plot(bounds[0][0],bounds[1][0],'')
    ax.plot(bounds[0][1],bounds[1][1],'')
        
def plotSetup3d(xmin = -3.0, xmax = 3.0, ymin = -3.0, ymax = 3.0, zmin = -3.0, zmax = 3.0, figsize=(6,4)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_xlim([xmin, xmax])
    ax.axes.set_ylim([ymin, ymax])
    ax.axes.set_zlim([zmin, zmax])
    ax.axes.set_xlabel('$x_1$',size=15)
    ax.axes.set_ylabel('$x_2$',size=15)
    ax.axes.set_zlabel('$x_3$',size=15)
    return ax

def plotPoint3d (ax, x1, x2, x3, color='r'):
    ax.plot([x1], [x2], '{}o'.format(color), zs=[x3])
    
def plotLinEqn3d(ax, l1, color='Green'):
    """
    plot the plane corresponding to the linear equation
    a1 x + a2 y + a3 z = b
    where l1 = [a1, a3, a3, b]
    """
    pts = intersectionPlaneCube(ax, l1)
    ptlist = np.array([np.array(i) for i in pts])
    x = ptlist[:,0]
    y = ptlist[:,1]
    z = ptlist[:,2]
    if (len(x) > 2):
        try:
            triang = mp.tri.Triangulation(x, y)
        except:
            # this happens where there are triangles parallel to the z axis
            # so some points in the x,y plane are repeated (which is illegal for a triangulation)
            # this is a hack but it works!
            try:
                triang = mp.tri.Triangulation(x, z)
                triang.y = y
            except:
                triang = mp.tri.Triangulation(z, y)
                triang.x = x
        ax.plot_trisurf(triang, z, color=color, alpha=0.3, linewidth=0, shade=False)

def intersectionPlaneCube(ax, l1):
    # returns the vertices of the polygon defined by the intersection of a plane
    # and the rectangular prism defined by the limits of the axes
    bounds = np.array([ax.axes.get_xlim(), ax.axes.get_ylim(), ax.axes.get_zlim()])
    coefs = l1[0:3]
    b = l1[3]
    points = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                corner = [x, y, z]
                # 24 corner-pairs 
                for i in range(3):
                    # but only consider each edge once (12 unique edges)
                    if corner[i] == 0:
                        # we are looking for the intesection of the line defined by
                        # the two constant values with the plane
                        isect = (b - np.sum([coefs[k] * bounds[k][corner[k]] for k in range(3) if k != i]))/float(coefs[i])
                        if ((isect >= bounds[i][0]) & (isect <= bounds[i][1])):
                            pt = [bounds[k][corner[k]] for k in range(3)]
                            pt[i] = isect
                            points.append(tuple(pt))
    return set(points)

def plotIntersection3d(ax, eq1, eq2, type='-',color='Blue'):
    """
    plot the intersection of two linear equations in 3d
    """
    bounds = np.array([ax.axes.get_xlim(), ax.axes.get_ylim(), ax.axes.get_zlim()])
    tmp = np.array([np.array(eq1), np.array(eq2)])
    A = tmp[:,:-1]
    b = tmp[:,-1]
    ptlist = []
    for i in range(3):
        vars = [k for k in range(3) if k != i]
        A2 = A[:][:,vars]
        for j in range(2):
            b2 = b - bounds[i,j] * A[:,i]
            try:
                pt = np.linalg.inv(A2).dot(b2)
            except:
                continue
            if (pt[0] >= bounds[vars[0]][0]) & (pt[0] <= bounds[vars[0]][1]) & (pt[1] >= bounds[vars[1]][0]) & (pt[1] <= bounds[vars[1]][1]):
                point = [0,0,0]
                point[vars[0]] = pt[0]
                point[vars[1]] = pt[1]
                point[i] = bounds[i,j]
                ptlist.append(point)
    ptlist = np.array(ptlist).T
    ax.plot(ptlist[0,:], ptlist[1,:], type, zs = ptlist[2,:], color=color)

def plotCube(ax, pt, color='Blue'):
    """
    plot a 3d wireframe parallelipiped with one corner on the origin
    """
    endpoints = np.concatenate((np.array([[0,0,0]]),np.array([pt])))
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                # we are plotting each line twice; not bothering to fix this
                corner = [endpoints[x,0],endpoints[y,1], endpoints[z,2]]
                # from each corner, plot the edges adjacent to that corner
                ax.plot([endpoints[x,0],endpoints[1-x,0]],[endpoints[y,1],endpoints[y,1]],zs=[endpoints[z,2],endpoints[z,2]],color=color)
                ax.plot([endpoints[x,0],endpoints[x,0]],[endpoints[y,1],endpoints[1-y,1]],zs=[endpoints[z,2],endpoints[z,2]],color=color)
                ax.plot([endpoints[x,0],endpoints[x,0]],[endpoints[y,1],endpoints[y,1]],zs=[endpoints[z,2],endpoints[1-z,2]],color=color)
                
def plotSpan3d(ax, u, v, color='Blue'):
    """
    Plot the plane that is the span of u and v
    """
    # we are looking for a single equation ax1 + bx2 + cx3 = 0
    # it is homogeneous because it is a subspace (span)
    # we have two solutions [a b c]'u = 0 and [a b c]'v = 0
    # this corresponds to a linear system in [a b c]
    # with coefficient matrix [u; v; 0]
    A = np.array([u, v])
    # put A in reduced row echelon form
    # assumes the line connecting the two points is
    # not parallel to any axes!
    A[0] = A[0]/A[0][0]
    A[1] = A[1] - A[1][0] * A[0]
    A[1] = A[1] / A[1][1]
    A[0] = A[0] - A[0][1] * A[1]
    # now use c=1 to fix a single solution
    a = -A[0][2]
    b = -A[1][2]
    c = 1.0
    plotLinEqn3d(ax, [a, b, c, 0.0], color)
    
