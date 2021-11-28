import numpy as np
import matplotlib as mp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import json
import qrcode
import hashlib

url_base = 'https://www.cs.bu.edu/faculty/crovella/cs132-figures'

class two_d_figure:

    def __init__(self,
                     fig_name,
                     xmin = -6.0,
                     xmax = 6.0,
                     ymin = -2.0,
                     ymax = 4.0,
                     size=(6,4)):
        """
        basics of 2D plot setup
        defaults: xmin = -6.0, xmax = 6.0, ymin = -2.0, ymax = 4.0, size=(6,4)
        size is by default 6 inches by 4 inches
        """
        self.fig = plt.figure(figsize=size)
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        self.ax.axes.set_xlim([xmin, xmax])

    def plotPoint (self, x1, x2, color='r'):
        self.ax.plot(x1, x2, '{}o'.format(color))

    def plotVec (self, x1, color='r'):
        self.ax.plot(x1[0], x1[1], '{}o'.format(color))

    def plotArrow (self, x1, x2):
        self.ax.arrow(0.0, 0.0, x1, x2)

    def plotArrowVec(self,
                         v,
                         start = [0,0],
                         head_width=0.2,
                         head_length=0.2,
                         length_includes_head = True,
                         color='Red'):
        try:
            self.ax.arrow(start[0],
                        start[1],
                        v[0]-start[0],
                        v[1]-start[1],
                        head_width=head_width,
                        head_length=head_length,
                        length_includes_head = length_includes_head,
                        color=color)
        # if the arrow length is zero, raises an IndexError
        except IndexError:
            pass

    def plotLinEqn (self, a1, a2, b, format='-', color='r', alpha=1.0):
        """
        plot line line corresponding to the linear equation
        a1 x + a2 y = b
        """
        if (a2 != 0):
            # line is not parallel to y axis
            [xmin, xmax] = plt.xlim()
            x1 = xmin
            y1 = (b - (x1 * a1))/float(a2)
            x2 = xmax
            y2 = (b - (x2 * a1))/float(a2)
            plt.plot([x1, x2],
                     [y1, y2],
                     format,
                     label='${}$'.format(formatEqn([a1, a2],b)),
                     color=color,
                     alpha=alpha)
        else:
            # line is parallel to y axis
            [ymin, ymax] = plt.ylim()
            y1 = ymin
            x1 = b / float(a1)
            y2 = ymax
            x2 = b /float(a1)
            plt.plot([x1, x2],
                     [y1, y2],
                     format,
                     label='${}$'.format(formatEqn([a1, a2],b)),
                     color=color,
                     alpha=alpha)
            

    def centerAxes (self):
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['top'].set_color('none')
        # self.ax.spines['left'].set_smart_bounds(True)
        # self.ax.spines['bottom'].set_smart_bounds(True)
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        bounds = np.array([self.ax.axes.get_xlim(),
                               self.ax.axes.get_ylim()])
        self.ax.plot(bounds[0][0],bounds[1][0],'')
        self.ax.plot(bounds[0][1],bounds[1][1],'')

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

      from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def perp_sym(vertex, pt1, pt2, size):
    ''' Construct the two lines needed to create a perpendicular-symbol
    at vertex vertex and heading toward points pt1 and pt2, given size
    Usage: 
    perpline1, perpline2 = perp_sym(...)
    plt.plot(perpline1[0], perpline1[1], 'k', lw = 1)
    plt.plot(perpline2[0], perpline2[1], 'k', lw = 1)
    '''
    arm1 = pt1 - vertex
    arm2 = pt2 - vertex
    arm1unit = arm1 / np.linalg.norm(arm1)
    arm2unit = arm2 / np.linalg.norm(arm2)
    leg1 = np.array([vertex, vertex + (size * arm1unit)]) + (size * arm2unit)
    leg2 = np.array([vertex, vertex + (size * arm2unit)]) + (size * arm1unit)
    return((leg1.T, leg2.T))
    
class three_d_figure:
    
    def __init__ (self,
                      fig_num,
                      fig_desc = '',
                      xmin = -3.0,
                      xmax = 3.0,
                      ymin = -3.0,
                      ymax = 3.0,
                      zmin = -3.0,
                      zmax = 3.0,
                      figsize=(6,4),
                      qr = None,
                      displayAxes = True,
                      equalAxes = True):

        if len(fig_num) != 2:
            raise ValueError('fig_num should be (lec, fig)')

        fig_name = f'Figure {fig_num[0]}.{fig_num[1]}'
        self.fig_num = fig_num
        
        # possible values: None (no QR code displayed),
        # url (url based QR code displayed), direct
        valid_qr = [None, 'url', 'direct']
        self.qr = qr  
        if self.qr not in valid_qr:
            raise ValueError('Invalid qr argument')
        self.fig = plt.figure(figsize=figsize)
        if self.qr == None:
            # only plot the figure, no QR code
            self.ax = self.fig.add_subplot(111, projection='3d')
                                               #, proj_type='ortho')
            # this is not implemented in mp3d as of Apr 2020
            # self.ax.set_aspect('equal')
            if equalAxes:
                set_axes_equal(self.ax)
        else:
            # plot the figure and the QR code next to it
            self.ax = self.fig.add_subplot(121, projection='3d', position=[0,0,1,1])
            self.ax2 = self.fig.add_subplot(122,position=[1.2, 0.125, 0.75, 0.75])
        # self.ax.axes.set_title(fig_desc)
        self.equalAxes = equalAxes
        self.ax.axes.set_xlim([xmin, xmax])
        self.ax.axes.set_ylim([ymin, ymax])
        self.ax.axes.set_zlim([zmin, zmax])
        self.ax.axes.set_xlabel('$x_1$',size=15)
        self.ax.axes.set_ylabel('$x_2$',size=15)
        self.ax.axes.set_zlabel('$x_3$',size=15)
        self.desc = {}
        self.desc['FigureName'] = fig_name
        self.desc['FigureType'] = 'three_d_with_axes'
        self.desc['FigureDescription'] = fig_desc
        self.desc['xmin'] = xmin
        self.desc['xmax'] = xmax
        self.desc['ymin'] = ymin
        self.desc['ymax'] = ymax
        self.desc['zmin'] = zmin
        self.desc['zmax'] = zmax
        self.desc['xlabel'] = 'x_1'
        self.desc['ylabel'] = 'x_2'
        self.desc['zlabel'] = 'x_3'
        self.desc['objects'] = []
        self.desc['displayAxes'] = displayAxes

    # at present, this only hides axes in the json (app)
    # axes are draw in matplotlib in all cases
    def hideAxes(self):
        self.desc['displayAxes'] = False
        # can we use plt.axes('off') here? 

    def showAxes(self):
        self.desc['displayAxes'] = True
        
    def plotPoint (self, x1, x2, x3, color='r', alpha=1.0):
        # do the plotting
        self.ax.plot([x1], [x2], '{}o'.format(color), zs=[x3])
        # save the graphics element
        hex_color = colors.to_hex(color)
        self.desc['objects'].append(
            {'type': 'point',
             'transparency': alpha,
             'color': hex_color,
             'points': [{'x': float(x1), 'y': float(x2), 'z': float(x3)}]})

    def plotLinEqn(self, l1, color='Green', alpha=0.3):
        """
        plot the plane corresponding to the linear equation
        a1 x + a2 y + a3 z = b
        where l1 = [a1, a2, a3, b]
        """
        pts = self.intersectionPlaneCube(l1)
        ptlist = np.array([np.array(i) for i in pts])
        x = ptlist[:,0]
        y = ptlist[:,1]
        z = ptlist[:,2]
        if (len(x) > 2):
            try:
                triang = mp.tri.Triangulation(x, y)
            except:
                # this happens where there are triangles parallel to
                # the z axis so some points in the x,y plane are
                # repeated (which is illegal for a triangulation)
                # this is a hack but it works!
                try:
                    triang = mp.tri.Triangulation(x, z)
                    triang.y = y
                except:
                    triang = mp.tri.Triangulation(z, y)
                    triang.x = x
            # save the graphics element
            hex_color = colors.to_hex(color)
            self.desc['objects'].append(
                {'type': 'polygonsurface',
                 'color': hex_color,
                 'transparency': alpha,
                 'points': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in pts],
                 'triangleIndices': [int(y) for x in triang.triangles
                                         for y in x]})
            # do the plotting
            self.ax.plot_trisurf(triang,
                                     z,
                                     color=color,
                                     alpha=alpha,
                                     linewidth=0,
                                     shade=False)

    def intersectionPlaneCube(self, l1):
        ''' 
        returns the vertices of the polygon defined
        by the intersection of a plane
        and the rectangular prism defined by the limits of the axes
        '''
        bounds = np.array([self.ax.axes.get_xlim(),
                               self.ax.axes.get_ylim(),
                               self.ax.axes.get_zlim()])
        coefs = l1[:3]
        b = l1[3]
        points = []
        for x, y, z in itertools.product([0,1],repeat=3):
            corner = [x, y, z]
            # 24 corner-pairs 
            for i in range(3):
                # but only consider each edge once (12 unique edges)
                if corner[i] == 1:
                    continue
                # we are looking for the intesection of the line defined by
                # the two constant values with the plane
                if coefs[i] == 0.0:
                    continue
                isect = (b - np.sum([coefs[k] * bounds[k][corner[k]]
                            for k in range(3) if k != i]))/float(coefs[i])
                if ((isect >= bounds[i][0]) & (isect <= bounds[i][1])):
                    pt = [bounds[k][corner[k]] for k in range(3)]
                    pt[i] = isect
                    points.append(tuple(pt))
        return set(points)

    def text(self, x, y, z, mpl_label, json_label, size, color='k'):
        hex_color = colors.to_hex(color)
        self.desc['objects'].append({
            'type': 'text', 
            'content': json_label,
            'size': size,
            'color': hex_color,
            'points': [{'x': float(x), 'y': float(y), 'z': float(z)}]})
        self.ax.text(x, y, z, mpl_label, size=size)

    def set_title(self, mpl_title, json_title = None,
                      number_fig = True, size = 12):
        if number_fig:
            self.fig.suptitle(f'Figure {self.fig_num[0]:d}.{self.fig_num[1]:d}')
        self.ax.set_title(mpl_title, size=size)
        if json_title == None:
            json_title = mpl_title
        self.desc['objects'].append({'type': 'title', 'label': json_title})

    def plotLine(self, in_ptlist, color, line_type='-', alpha=1.0):
        ptlist = [[float(i) for i in j] for j in in_ptlist]
        hex_color = colors.to_hex(color)
        self.desc['objects'].append({'type': 'line',
                                     'color': hex_color,
                                     'transparency': alpha,
                                     'linetype': line_type,
             'points': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in ptlist]})
        ptlist = np.array(ptlist).T
        self.ax.plot(ptlist[0,:],
                         ptlist[1,:],
                         line_type,
                         zs = ptlist[2,:],
                         color=color)

    def plotPerpSym(self, vertex, pt1, pt2, size):
        ''' Plot in 3D the two lines needed to create a perpendicular-symbol
        at vertex vertex and heading toward points pt1 and pt2, given size
        '''
        perpline1, perpline2 = perp_sym(vertex, pt1, pt2, size)
        self.plotLine([perpline1[:,0], perpline1[:,1]], 'k', '-')
        self.plotLine([perpline2[:,0], perpline2[:,1]], 'k', '-')
        
    def plotIntersection(self, eq1, eq2, line_type='-',color='Blue'):
        """
        plot the intersection of two linear equations in 3d
        """
        hex_color = colors.to_hex(color)
        bounds = np.array([self.ax.axes.get_xlim(),
                               self.ax.axes.get_ylim(),
                               self.ax.axes.get_zlim()])
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
                if ((pt[0] >= bounds[vars[0]][0])
                    & (pt[0] <= bounds[vars[0]][1])
                    & (pt[1] >= bounds[vars[1]][0])
                    & (pt[1] <= bounds[vars[1]][1])):
                    point = [0,0,0]
                    point[vars[0]] = pt[0]
                    point[vars[1]] = pt[1]
                    point[i] = bounds[i,j]
                    ptlist.append(point)
        self.plotLine(ptlist, color, line_type)

    def plotCube(self, pt, color='Blue'):
        """
        plot a 3d wireframe parallelipiped with one corner on the origin
        """
        endpoints = np.concatenate((np.array([[0,0,0]]),np.array([pt])))
        for x, y, z in itertools.product([0, 1], repeat=3):
            # we are plotting each line twice; not bothering to fix this
            corner = [endpoints[x,0],endpoints[y,1], endpoints[z,2]]
            # from each corner, plot the edges adjacent to that corner
            if (x == 0):
                ptlist = [[endpoints[x,0], endpoints[y,1], endpoints[z,2]],
                        [endpoints[1-x,0], endpoints[y,1], endpoints[z,2]]]
                self.plotLine(ptlist, color)
            if (y == 0):
                ptlist = [[endpoints[x,0], endpoints[y,1], endpoints[z,2]],
                        [endpoints[x,0], endpoints[1-y,1], endpoints[z,2]]]
                self.plotLine(ptlist, color)
            if (z == 0):
                ptlist = [[endpoints[x,0], endpoints[y,1], endpoints[z,2]],
                        [endpoints[x,0], endpoints[y,1], endpoints[1-z,2]]]
                self.plotLine(ptlist, color)

    def plotSpan(self, u, v, color='Blue'):
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
        self.plotLinEqn([a, b, c, 0.0], color)

    def plotQF(self, qf_mat, color='Red', alpha=1.0):
        """
        Plot the quadratic form that is given by 2x2 symmetric matrix qf_mat
        """
        # helper functions
        
        # evaluate the qf at a particular x, y
        def eval_qf(qf, x, y):
            xvec = np.array([x, y])
            return xvec.T @ qf @ xvec

        # find the portion that is contained within two ranges r1 and r2
        def range_intersect(r1, r2):
            if np.all(np.isnan([r1[0], r2[0]])):
                lo = np.nan
            else:
                lo = np.nanmax([r1[0], r2[0]])
            if np.all(np.isnan([r1[1], r2[1]])):
                hi = np.nan
            else:
                hi = np.nanmin([r1[1], r2[1]])
            return [lo, hi]

        # find the union of two ranges r1 and r2
        def range_union(r1, r2):
            if np.all(np.isnan([r1[0], r2[0]])):
                lo = np.nan
            else:
                lo = np.nanmin([r1[0], r2[0]])
            if np.all(np.isnan([r1[1], r2[1]])):
                hi = np.nan
            else:
                hi = np.nanmax([r1[1], r2[1]])
            return [lo, hi]

        xmin, xmax = self.ax.axes.get_xlim()
        ymin, ymax = self.ax.axes.get_ylim()
        zmin, zmax = self.ax.axes.get_zlim()

        # first find the limits of x and y for grid creation
        # the limits will occur along the eigenvectors of the QF
        e, v = np.linalg.eig(qf_mat)

        # we will build a grid on which we'll evaluate the QF.
        # to draw the boundary of the surface precisely,
        # the boundaries of the grid need to exactly fall where
        # the QF crosses either the upper or lower Z bounding planes.
        # furthermore, it is important for the boundary to exactly
        # hit the points where the most extreme points of the ellipse
        # fall.

        # first we find the points where the extreme points of the ellipse
        # fall.  If this is an indefinite QF, this will find the point
        # where the saddle intersects the z planes

        # this helper function computes those extreme points, ie,
        # x_limit is the +/- value of x that lies on a given eigenvector
        # and for which the qf = some_z.
        # note that when we are on a line, the QF becomes a stanard
        # quadratic, and we solve for where it equals z by using the
        # quadratic formula (-b +/ sqrt(b2 - 4ac))/2a.
        # this fn may return nans, which we handle in range computation later
        def axes_limit(qf, z, evec):
            if evec[0] != 0:
                # for any eigenvector evec, y = alpha x,
                # so alpha = y/x = evec[1]/evec[0]
                alf = evec[1]/evec[0]
                # substituting y = alf x into the quadratric form yields
                denom = (qf[0, 0] + alf * (qf[0, 1] + qf[1, 0])
                             + alf**2 * qf[1, 1])
                if (denom == 0) or ((z / denom) < 0):
                    return [[np.nan, np.nan],[np.nan, np.nan]]
                else:
                    x = np.sqrt(z / denom)
                    return [[-x, -alf*x], [x, alf*x]]
            else:
                # if evec[0] = 0, the evec is parallel to the y axis, so
                # switch places of x and y since y can't be given in terms of x
                alf = evec[0]/evec[1]
                denom = (qf[1, 1] + alf * (qf[0, 1] + qf[1, 0])
                            + alf**2 * qf[0, 0])
                if (denom == 0) or ((z / denom) < 0) :
                    return [[np.nan, np.nan],[np.nan, np.nan]]
                else:
                    y = np.sqrt(z / denom)
                    return [[-alf*y, -y], [alf*y, y]]

        # considering both eigenvectors, find the range of x for y = zmax
        r1 = axes_limit(qf_mat, zmax, v[:,0])
        r2 = axes_limit(qf_mat, zmax, v[:,1])
        # take the union of the x ranges given by the two eignvectors
        xrange_max = range_union([r1[0][0], r1[1][0]], [r2[0][0], r2[1][0]])
        # same thing for zmin
        r1 = axes_limit(qf_mat, zmin, v[:,0])
        r2 = axes_limit(qf_mat, zmin, v[:,1])
        xrange_min = range_union([r1[0][0], r1[1][0]], [r2[0][0], r2[1][0]])
        # final xrange is union of ranges for zmin and zmax
        final_xrange = range_union(xrange_max, xrange_min)
        # but not extending beyond the plotting box
        final_xrange = range_intersect([xmin, xmax], final_xrange)

        gridsize = 50

        # the x values of the grid
        x_vals = np.linspace(final_xrange[0], final_xrange[1], gridsize)

        # now for each x value, we need to compute the y bounds of the grid
        # making sure the fall (if needed) on the exact point where the
        # sruface passes through the zmin or zmax planes

        # helper function to solve a single quadratic equation
        # a x**2 + b x + c = 0
        def quad_zeros(a, b, c):
            disc = b**2 - 4*a*c
            if disc < 0:
                return [np.nan, np.nan]
            elif a == 0:
                return [np.nan, np.nan]
            else:
                return sorted([(-b - np.sqrt(disc))/(2*a),
                                   (-b + np.sqrt(disc))/(2*a)])

        # helper function to find the y values for which the QF
        # evaluated along a given constant x line crosses a given z value
        def solve_y(qf_in, x, z):
            A = qf_in[1, 1]
            B = (qf_in[1, 0] + qf_in[0, 1]) * x
            C = (qf_in[0, 0] * x**2) - z
            return quad_zeros(A, B, C)

        # the grid points
        X = []
        Y = []
        Z = []


        for x in x_vals:
            y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmax)
            if np.isnan(y_min_intcpt):
                # surface does not cross the zmax plane
                y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmin)
                if np.isnan(y_min_intcpt):
                    # surface does not cross the zmin plane
                    ztest = eval_qf(qf_mat, x, ymin)
                    if ((ztest <= zmax) & (ztest >= zmin)):
                        # surface lies entirely within z boundaries
                        # (this should not happen)
                        X += gridsize*[x]
                        valids = list(np.linspace(ymin, ymax, gridsize))
                        Y += valids
                        Z += [eval_qf(qf_mat, x, y)
                                for x, y in zip(gridsize*[x], valids)]
                    else:
                        # surface lies entirely outside z boundaries
                        pass
                else:
                    # surface crosses zmin but not zmax plane
                    X += gridsize*[x]
                    y_start, y_end = range_intersect(
                        [y_min_intcpt, y_max_intcpt], [ymin, ymax])
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, x, y)
                              for x, y in zip(gridsize*[x], valids)] 
            else:
                # surface does cross zmax plane
                # need to decide if range in between crossings
                # is in or out of visualization cube
                # WORK ON THIS -- GENERALIZE to zmin plane --
                # THEN generalize notion of axes_limits computed above
                # note that triangulation may have trouble
                # dealing with region between, where surface is out of box
                # perhaps put one point in between, with a zvalue of nan?
                y_start, y_end = range_intersect(
                    [y_min_intcpt, y_max_intcpt], [ymin, ymax])
                y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmin)
                if np.isnan(y_min_intcpt):
                    # surface does not cross the zmin plane
                    X += gridsize*[x]
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, x, y)
                              for x, y in zip(gridsize*[x], valids)] 
                else:
                    # surface crosses zmin and zmax planes
                    # have already taken zmax crossings into account
                    # in y_start, y_end
                    y_start, y_end = range_intersect(
                        [y_min_intcpt, y_max_intcpt], [y_start, y_end])
                    X += gridsize*[x]
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, x, y)
                              for x, y in zip(gridsize*[x], valids)]

        # now that we have the grid defined, define the triangles
        # to form the triangulation for the surface.   We do our
        # own triangulation because the standard (delaunay) triangulation
        # introduces artifacts at the edge of a concave surface.
        # trigulation scheme used ensures that triangles are only formed
        # among adjacent points on the grid (never further away)
        def coord_to_ndx(i, j):
            return i*gridsize + j

        def triang1(i, j):
            return [coord_to_ndx(i, j), coord_to_ndx(i+1, j),
                        coord_to_ndx(i, j+1)]

        def triang2(i, j):
            return [coord_to_ndx(i+1, j), coord_to_ndx(i+1, j+1),
                        coord_to_ndx(i, j+1)]

        triangles = []
        for i in range(gridsize-1):
            for j in range(gridsize-1):
                triangles.append(triang1(i,j))
                triangles.append(triang2(i,j))  

        triang = mp.tri.Triangulation(X, Y, triangles=triangles)

        # and plot it!
        self.ax.plot_trisurf(triang,
                            Z,
                            color=color,
                            alpha=alpha,
                            linewidth=0)

        hex_color = colors.to_hex(color)
        self.desc['objects'].append(
                {'type': 'quadraticform',
                     'color': hex_color,
                 'transparency': alpha,
                 'a11': qf_mat[0][0],
                 'a12': qf_mat[0][1],
                 'a21': qf_mat[1][0],
                 'a22': qf_mat[1][1]
                })

    def plotGeneralQF(self, qf_mat, vec, cnst, color='Red', alpha=1.0):
        """
        Plot the general quadratic form 
        let Q = qf_mat, V = vec, C = cnst
        then plot x'Qx + v'x + C
        """
        # helper functions
        
        # evaluate the qf at a particular x, y
        def eval_qf(qf, vec, cnst, x, y):
            xvec = np.array([x, y])
            return xvec.T @ qf @ xvec + vec.T @ xvec + cnst

        # find the portion that is contained within two ranges r1 and r2
        def range_intersect(r1, r2):
            if np.all(np.isnan([r1[0], r2[0]])):
                lo = np.nan
            else:
                lo = np.nanmax([r1[0], r2[0]])
            if np.all(np.isnan([r1[1], r2[1]])):
                hi = np.nan
            else:
                hi = np.nanmin([r1[1], r2[1]])
            return [lo, hi]

        # find the union of two ranges r1 and r2
        def range_union(r1, r2):
            if np.all(np.isnan([r1[0], r2[0]])):
                lo = np.nan
            else:
                lo = np.nanmin([r1[0], r2[0]])
            if np.all(np.isnan([r1[1], r2[1]])):
                hi = np.nan
            else:
                hi = np.nanmax([r1[1], r2[1]])
            return [lo, hi]

        xmin, xmax = self.ax.axes.get_xlim()
        ymin, ymax = self.ax.axes.get_ylim()
        zmin, zmax = self.ax.axes.get_zlim()

        # first find the limits of x and y for grid creation
        # the limits will occur along the eigenvectors of the QF
        e, v = np.linalg.eig(qf_mat)

        # we will build a grid on which we'll evaluate the QF.
        # to draw the boundary of the surface precisely,
        # the boundaries of the grid need to exactly fall where
        # the QF crosses either the upper or lower Z bounding planes.
        # furthermore, it is important for the boundary to exactly
        # hit the points where the most extreme points of the ellipse
        # fall.

        # first we find the points where the extreme points of the ellipse
        # fall.  If this is an indefinite QF, this will find the point
        # where the saddle intersects the z planes

        # this helper function computes those extreme points, ie,
        # x_limit is the +/- value of x that lies on a given eigenvector
        # and for which the qf = some_z.
        # note that when we are on a line, the QF becomes a stanard
        # quadratic, and we solve for where it equals z by using the
        # quadratic formula (-b +/ sqrt(b2 - 4ac))/2a.
        # this fn may return nans, which we handle in range computation later
        def axes_limit(qf, z, evec):
            if evec[0] != 0:
                # for any eigenvector evec, y = alpha x,
                # so alpha = y/x = evec[1]/evec[0]
                alf = evec[1]/evec[0]
                # substituting y = alf x into the quadratric form yields
                denom = (qf[0, 0] + alf * (qf[0, 1] + qf[1, 0])
                             + alf**2 * qf[1, 1])
                if (denom == 0) or ((z / denom) < 0):
                    return [[np.nan, np.nan],[np.nan, np.nan]]
                else:
                    x = np.sqrt(z / denom)
                    return [[-x, -alf*x], [x, alf*x]]
            else:
                # if evec[0] = 0, the evec is parallel to the y axis, so
                # switch places of x and y since y can't be given in terms of x
                alf = evec[0]/evec[1]
                denom = (qf[1, 1] + alf * (qf[0, 1] + qf[1, 0])
                            + alf**2 * qf[0, 0])
                if (denom == 0) or ((z / denom) < 0) :
                    return [[np.nan, np.nan],[np.nan, np.nan]]
                else:
                    y = np.sqrt(z / denom)
                    return [[-alf*y, -y], [alf*y, y]]

        # considering both eigenvectors, find the range of x for y = zmax
        r1 = axes_limit(qf_mat, zmax, v[:,0])
        r2 = axes_limit(qf_mat, zmax, v[:,1])
        # take the union of the x ranges given by the two eignvectors
        xrange_max = range_union([r1[0][0], r1[1][0]], [r2[0][0], r2[1][0]])
        # same thing for zmin
        r1 = axes_limit(qf_mat, zmin, v[:,0])
        r2 = axes_limit(qf_mat, zmin, v[:,1])
        xrange_min = range_union([r1[0][0], r1[1][0]], [r2[0][0], r2[1][0]])
        # final xrange is union of ranges for zmin and zmax
        final_xrange = range_union(xrange_max, xrange_min)
        # but not extending beyond the plotting box
        final_xrange = range_intersect([xmin, xmax], final_xrange)

        gridsize = 50

        # the x values of the grid
        x_vals = np.linspace(final_xrange[0], final_xrange[1], gridsize)

        # now for each x value, we need to compute the y bounds of the grid
        # making sure the fall (if needed) on the exact point where the
        # sruface passes through the zmin or zmax planes

        # helper function to solve a single quadratic equation
        # a x**2 + b x + c = 0
        def quad_zeros(a, b, c):
            disc = b**2 - 4*a*c
            if disc < 0:
                return [np.nan, np.nan]
            elif a == 0:
                return [np.nan, np.nan]
            else:
                return sorted([(-b - np.sqrt(disc))/(2*a),
                                   (-b + np.sqrt(disc))/(2*a)])

        # helper function to find the y values for which the QF
        # evaluated along a given constant x line crosses a given z value
        def solve_y(qf_in, x, z):
            A = qf_in[1, 1]
            B = (qf_in[1, 0] + qf_in[0, 1]) * x
            C = (qf_in[0, 0] * x**2) - z
            return quad_zeros(A, B, C)

        # the grid points
        X = []
        Y = []
        Z = []


        for x in x_vals:
            y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmax)
            if np.isnan(y_min_intcpt):
                # surface does not cross the zmax plane
                y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmin)
                if np.isnan(y_min_intcpt):
                    # surface does not cross the zmin plane
                    ztest = eval_qf(qf_mat, vec, cnst, x, ymin)
                    if ((ztest <= zmax) & (ztest >= zmin)):
                        # surface lies entirely within z boundaries
                        # (this should not happen)
                        X += gridsize*[x]
                        valids = list(np.linspace(ymin, ymax, gridsize))
                        Y += valids
                        Z += [eval_qf(qf_mat, vec, cnst, x, y)
                                for x, y in zip(gridsize*[x], valids)]
                    else:
                        # surface lies entirely outside z boundaries
                        pass
                else:
                    # surface crosses zmin but not zmax plane
                    X += gridsize*[x]
                    y_start, y_end = range_intersect(
                        [y_min_intcpt, y_max_intcpt], [ymin, ymax])
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, vec, cnst, x, y)
                              for x, y in zip(gridsize*[x], valids)] 
            else:
                # surface does cross zmax plane
                # need to decide if range in between crossings
                # is in or out of visualization cube
                # WORK ON THIS -- GENERALIZE to zmin plane --
                # THEN generalize notion of axes_limits computed above
                # note that triangulation may have trouble
                # dealing with region between, where surface is out of box
                # perhaps put one point in between, with a zvalue of nan?
                y_start, y_end = range_intersect(
                    [y_min_intcpt, y_max_intcpt], [ymin, ymax])
                y_min_intcpt, y_max_intcpt = solve_y(qf_mat, x, zmin)
                if np.isnan(y_min_intcpt):
                    # surface does not cross the zmin plane
                    X += gridsize*[x]
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, vec, cnst, x, y)
                              for x, y in zip(gridsize*[x], valids)] 
                else:
                    # surface crosses zmin and zmax planes
                    # have already taken zmax crossings into account
                    # in y_start, y_end
                    y_start, y_end = range_intersect(
                        [y_min_intcpt, y_max_intcpt], [y_start, y_end])
                    X += gridsize*[x]
                    valids = list(np.linspace(y_start, y_end, gridsize))
                    Y += valids 
                    Z += [eval_qf(qf_mat, vec, cnst, x, y)
                              for x, y in zip(gridsize*[x], valids)]

        # now that we have the grid defined, define the triangles
        # to form the triangulation for the surface.   We do our
        # own triangulation because the standard (delaunay) triangulation
        # introduces artifacts at the edge of a concave surface.
        # trigulation scheme used ensures that triangles are only formed
        # among adjacent points on the grid (never further away)
        def coord_to_ndx(i, j):
            return i*gridsize + j

        def triang1(i, j):
            return [coord_to_ndx(i, j), coord_to_ndx(i+1, j),
                        coord_to_ndx(i, j+1)]

        def triang2(i, j):
            return [coord_to_ndx(i+1, j), coord_to_ndx(i+1, j+1),
                        coord_to_ndx(i, j+1)]

        triangles = []
        for i in range(gridsize-1):
            for j in range(gridsize-1):
                triangles.append(triang1(i,j))
                triangles.append(triang2(i,j))  

        triang = mp.tri.Triangulation(X, Y, triangles=triangles)

        # and plot it!
        self.ax.plot_trisurf(triang,
                            Z,
                            color=color,
                            alpha=alpha,
                            linewidth=0)

        hex_color = colors.to_hex(color)
        self.desc['objects'].append(
                {'type': 'quadraticform',
                     'color': hex_color,
                 'transparency': alpha,
                 'a11': qf_mat[0][0],
                 'a12': qf_mat[0][1],
                 'a21': qf_mat[1][0],
                 'a22': qf_mat[1][1]
                })

    def rotate(self, start=0, end=360, increment=5):
        from matplotlib import animation
        # return an animation that rotates the figure using
        # this nifty js viewer
        mp.rcParams['animation.html'] = 'jshtml'

        # putting plt.show() works for %matplotlib notebook
        def display(angle, *fargs):
            fargs[0].view_init(azim=angle)
            # plt.show()

        return mp.animation.FuncAnimation(self.fig, 
                                    display, 
                                    frames=np.arange(start, end, increment), 
                                    fargs=[self.ax],
                                    interval=100, 
                                    repeat=False)
        
    def save(self, qrviz = None):
        file_name = f'Fig{self.fig_num[0]:02d}.{self.fig_num[1]:d}'
        
        if self.equalAxes:
            set_axes_equal(self.ax)
        fname = 'json/{}.json'.format(file_name)
        with open(fname, 'w') as fp:
            json.dump(self.desc, fp, indent=2)
        if self.qr != None:
            qr_code = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=3,
                border=4
                )
            m = hashlib.sha256()
            if self.qr == 'direct':
                m.update(self.json().encode('utf-8'))
                d = m.digest().hex()
                qr_code.add_data("b"+self.json()+d)
            elif self.qr == 'url':
                url_string = url_base + '/' + file_name + '.json'
                m.update(url_string.encode('utf-8'))
                d = m.digest().hex()
                qr_code.add_data("a"+url_string+d)
            qr_code.make(fit=True)
            img = qr_code.make_image()
            if qrviz == 'show':
                self.ax2.imshow(img, cmap="gray")
                # self.ax2.imshow(img)
                self.ax2.set_axis_off()
                return None
            elif qrviz == 'save':
                return img
            # plt.subplots_adjust(wspace=1.)
            # plt.tight_layout()

    def dont_save(self):
        return

    def json(self):
        return(json.dumps(self.desc))

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
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    bounds = np.array([ax.axes.get_xlim(), ax.axes.get_ylim()])
    ax.plot(bounds[0][0],bounds[1][0],'')
    ax.plot(bounds[0][1],bounds[1][1],'')

def noAxes(ax):
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
        
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
    where l1 = [a1, a2, a3, b]
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
    bounds = np.array([ax.axes.get_xlim(),
                           ax.axes.get_ylim(),
                           ax.axes.get_zlim()])
    coefs = l1[:3]
    b = l1[3]
    points = []
    for x, y, z in itertools.product([0,1],repeat=3):
        corner = [x, y, z]
        # 24 corner-pairs 
        for i in range(3):
            # but only consider each edge once (12 unique edges)
            if corner[i] == 1:
                continue
            # we are looking for the intersection of the line defined by
            # the two constant values with the plane
            if coefs[i] == 0.0:
                continue
            isect = (b - np.sum([coefs[k] * bounds[k][corner[k]]
                                     for k in range(3) if k != i]))/float(coefs[i])
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
    
