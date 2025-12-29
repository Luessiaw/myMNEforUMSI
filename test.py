"""
===========================
More triangular 3D surfaces
===========================

Two additional examples of plotting surfaces with triangular mesh.

The first demonstrates use of plot_trisurf's triangles argument, and the
second sets a `.Triangulation` object's mask and passes the object directly
to plot_trisurf.
"""
from scipy.spatial import Delaunay
import matplotlib.tri as mtri
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from myMNE.mathTools import *

from myMNE.visualise import *

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.set_xlim([-0.11,0.11])
ax.set_ylim([-0.11,0.11])
ax.set_zlim([-0.11,0.11])
ax.set_aspect("equal")

f = lambda p:p[2]
plotFunctionOnSphere(ax,f=f,radius=0.11,vmin=0,vmax=0.11, num=400)

fb_points = np.array(fibonacci_sphere(800)[:400])
# for p in fb_points:
#     p*= 1.02
#     ax.scatter(p[0],p[1],p[2])

xs = fb_points[:,0]
ys = fb_points[:,1]
zs = fb_points[:,2]

triangulation = mtri.Triangulation(xs,ys)
for k,triangle in enumerate(triangulation.triangles):
    verts = []
    for i in range(3):
        x = xs[triangle[i]]
        y = ys[triangle[i]]
        z = zs[triangle[i]]
        verts.append(np.array([x,y,z]))
    collection = Poly3DCollection([verts],)
    ax.add_collection3d(collection)    

boundPoints = np.array(convex_hull(np.array(list(zip(xs,ys)))))
xu,yu = boundPoints.transpose()
zu = np.sqrt(1-xu**2-yu**2)
pu = np.stack([xu,yu,zu]).transpose()
pu = np.vstack([pu,pu[0]])

theta = np.arctan2(pu[:,1],pu[:,0])
xd = np.cos(theta)
yd = np.sin(theta)
zd = np.zeros_like(xd)
pd = np.stack([xd,yd,zd]).transpose()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.scatter(fb_points[:,0],fb_points[:,1])
ax2.plot(pu[:,0],pu[:,1])

rgbs = ["red","green","blue"]
for i in range(xu.size):
    p1u = pu[i]
    p2u = pu[i+1]
    p1d = pd[i]
    p2d = pd[i+1]
    verts = []
    verts.append([p1u,p1d,p2d])
    verts.append([p1u,p2u,p2d])
    collection = Poly3DCollection(verts,facecolor=rgbs[i%3])
    ax.add_collection3d(collection)    

plt.show()


# rgbas = []
# for x in np.linspace(0,1,20):
#     rgba = cmap(norm(x))
#     rgbas.append([x,]+list(rgba))
# rgbas = np.array(rgbas)
# print(rgbas)