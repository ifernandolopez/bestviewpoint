import numpy as np 
import math
import sys
import itertools
import model3d
from OpenGL.GL import*
from OpenGL.GLU import*
from OpenGL.GLUT import*
from typing import List, Tuple, Dict

# Helper functions
IEdge = Tuple[int, int]

class AdjMatrix:
    def __init__(self, n_vertices):
        self.adj_matrix: np.ndarray = np.zeros((n_vertices, n_vertices))
        self.iedges: List[IEdge] = []
    def addEdge(self, iv0: int, iv1: int):
        if iv0 > iv1:
            iv0, iv1 = iv1, iv0
        if self.adj_matrix[iv0-1,iv1-1] > 0:
            return
        self.adj_matrix[iv0-1,iv1-1] += 1
        self.iedges.append((iv0, iv1))
    def getIndexedEdges(self) -> List[IEdge]:
        return self.iedges
    
def distance2D(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

class Intersection:
    INTERSECTING = 0
    NON_INTERSECTING = 1
    PARALLEL = 2

def intersection(p1, p2, p3, p4):
    """ Analyze the intersection between the segment p1-p2 and the segment p3-p4
        Returns 0 do intersect 
                1 do not intersect
                2 are parallel or overlapped """
    a, b, c = p2 - p1, p3 - p4,  p1 - p3
    den = round(a[1]*b[0] - a[0]*b[1], 5)
    # We first analyze the cases where the denominator is zero: quasi-paralled or overlapped
    if den < 0.05 and den > -0.05:
        return Intersection.PARALLEL
    # Then we detect the NO intersection case
    alpha_num = round(b[1]*c[0] - b[0]*c[1], 5)
    beta_num = round(a[0]*c[1] - a[1]*c[0], 5)
    if den > 0 and (alpha_num < 0 or alpha_num > den or beta_num < 0 or beta_num > den):
        return Intersection.NON_INTERSECTING
    if den < 0 and (alpha_num > 0 or alpha_num < den or beta_num > 0 or beta_num < den):
        return Intersection.NON_INTERSECTING
    # Otherwise, there is an intersection: it can be at a vertex, side or cross intersection
    alpha_side = alpha_num == 0.0 or alpha_num == den # Detect if the alpha segment ends in the other
    beta_side = beta_num == 0.0 or beta_num == den # Detect if the beta segment ends in the other
    if alpha_side or beta_side: # Vertex or side intersection
        return Intersection.INTERSECTING
    # Cross intersecton
    return Intersection.INTERSECTING

def are_adjacent_iedges(ie1: IEdge, ie2: IEdge):
    return ie1[0] == ie2[0] or ie1[0] == ie2[1] or ie1[1] == ie2[0] or ie1[1] == ie2[1]

def tight_angle(model_2d, iedges_pair: Tuple[IEdge, IEdge]):
    origin_vertex: List[int] = []
    arrow_vertices: list[int] = []
    for v in np.concatenate(iedges_pair):
        if (v in arrow_vertices):
            arrow_vertices.remove(v)
            origin_vertex.append(v)
        else:
            arrow_vertices.append(v)
    origin_vertex_2d:np.ndarray = model_2d.vertices[np.array(origin_vertex)-1][:,0:2]
    arrow_vertices_2d:np.ndarray = model_2d.vertices[np.array(arrow_vertices)-1][:,0:2]
    vectors_2d:np.ndarray = arrow_vertices_2d - origin_vertex_2d
    acos = np.dot(*vectors_2d) / (np.linalg.norm(vectors_2d[0]) * np.linalg.norm(vectors_2d[1]))
    return acos > 0.985 # vectors_2d form an angle less than 10º

def point_segment_dist(p, p0, p1):
    d = p1 - p0
    v = p - p0
    num = np.dot(d, v)
    if num <= 0:
        return distance2D(p, p0)
    den = np.dot(d,d)
    if num > den:
        return distance2D(p, p1)
    t0 = num / den
    q = p0 + d*t0 
    return distance2D(p, q)

def segments_dist(p1, p2, p3, p4):
    """ Return the min distance between the segment p1-p2 and the segment p3-p4 """
    return min(point_segment_dist(p1, p3, p4), point_segment_dist(p2, p3, p4), point_segment_dist(p3, p1, p2), point_segment_dist(p4, p1, p2))

def point_in_polygon(q, vertices):
    odd = False
    n = len(vertices)
    for i in range(n):
        a, b = vertices[i], vertices[(i+1)%n] # Get the edge vertices
        # If the ray intersects the edge
        if ((a[1] <= q[1] and b[1] > q[1]) # test upward crossing
           or (a[1] > q[1] and b[1] <= q[1])): # test downward crossing
           # Compute the actual edge-ray x-coordinate intersection
           t2 = (q[1]-a[1]) / (b[1]-a[1])
           if q[0] < a[0] + (b[0]-a[0])*t2:
               odd = not odd
    # If the number of crossings was odd, the point is in the polygon
    return odd

def polygon_inside(outer: np.ndarray,inner:np.ndarray) -> bool:
    """ Test if the inner polygon is inside the outer polygon """
    outer_maxs = np.amax(outer, axis = 0)
    outer_mins = np.amin(outer, axis = 0)
    inner_maxs = np.amax(inner, axis = 0)
    inner_mins = np.amin(inner, axis = 0)
    # Compare bounding boxes 
    # If the bounding box of the inner goes outside the bounding box of the outer, the inner is not inside
    if (inner_maxs>outer_maxs).any() or (inner_mins<outer_mins).any():
       return False
    # If any of the inner vertex is outside the outer polygon, the inner polygon is not inside
    for p in inner:
        if not point_in_polygon(p, outer):
            return False
    return True
# 3D model functions

class Model2D:
    def __init__(self, model_3d):
        self.model3D = model3d
        self.vertices: np.ndarray = None     # Projected xy vertices
        self.vertices_z_distances: np.ndarray = None # For occlusion detection
        self.ifaces: List[List] = []         # Projected Indexed Face Set (IFS) with indexed faces
        self.iedges: List[IEdge] = []        # Projected indexed edges, without duplicates
        self.areas: Dict[int, float] = {}    # Projected faces area. Or 0.0 if any point has been proyected outside the clipping window
        self.cachedProfits = None
        self.cachedPenalties = None
        
    def getFaceVectices(self, i):
        """ i is zero indexed """
        """ Returns an np.parray wth as many rows as vertices """
        return self.vertices[np.array(self.ifaces[i])-1]

def auxComputeProjectedFacesArea(model_2d):
    """ Compute the projected faces area """
    """ Constraint: This area is 0.0 if any point has been proyected outside the clipping window (-1,-1) to (1, 1) """
    for i in range(len(model_2d.ifaces)):
        vertices = model_2d.getFaceVectices(i)
        # Trigger the contraint
        if np.max(vertices[:,[0,1]])>1.0 or np.min(vertices[:,[0,1]])<-1.0:
            model_2d.areas = 0.0
            return 
        vertices = np.vstack([vertices,vertices[0]]) # We duplicate the first vertex at the end 
        downwards, upwards = 0.0, 0.0
        for j in range(0,len(vertices)-1):
            downwards += vertices[j,0]*vertices[j+1,1]
            upwards += vertices[j,1]*vertices[j+1,0] 
        face_area = 0.5*(upwards-downwards)
        model_2d.areas[i+1] = face_area  

def auxDetectOccludedFaces(model_2d):
    # Get the CCW faces
    front_ifaces = [key for key, value in model_2d.areas.items() if value<0]
    # Sort the front faces my minimin distance
    def depth(i):
        iface = model_2d.ifaces[i-1]
        face_vertices_z_distances = model_2d.vertices_z_distances[np.array(iface)-1]
        return np.min(face_vertices_z_distances)
    front_ifaces.sort(key = depth)
    # Search occluded faces and remove them from the front_ifaces
    occluded_faces = []
    pos_i, n_front_faces = 0, len(front_ifaces)
    while (pos_i<n_front_faces-1):
        i = front_ifaces[pos_i]
        pos_j = pos_i+1
        while (pos_j < n_front_faces):
            j = front_ifaces[pos_j]
            if polygon_inside(outer = model_2d.getFaceVectices(i-1)[:,:-1], inner = model_2d.getFaceVectices(j-1)[:,:-1]):
                occluded_faces.append(j)
                del front_ifaces[pos_j]
                n_front_faces -= 1
            else:
                pos_j += 1
        pos_i +=1
    # Invert the area of occluded faces
    for i in occluded_faces:
        model_2d.areas[i] *= -1

def compute2dModel(model_3d: model3d.Model3D, Mpers, Mmv) -> Model2D:
    """ Return a Model2D with the projection of the the model_3d according to Mpers a Mmv
        The Mpers and Mmv were constructed using the projection, rho, theta, phi an fovy of the model_3d """
    if model_3d.cached2dModel != None:
        return model_3d.cached2dModel
    Mcomposed = np.matmul(Mmv, Mpers)
    model_2d = Model2D(model_3d)
    # First, we project the vertices
    n_vertices = len(model_3d.vertices)
    # We multiply in homogeneous coordinates to obtain the clipping coordinates
    vertices_4d_homogeneous = np.c_[model_3d.vertices, np.ones(n_vertices)]
    model_2d.vertices = np.matmul(vertices_4d_homogeneous,Mcomposed)
    # Keep the z distances of the vertices
    model_2d.vertices_z_distances = model_2d.vertices[:,2]
    # Normalize the homogeneous clipping coordinates
    if model_3d.projection == model3d.ProjectionType.PERSPECTIVE:
        model_2d.vertices = (model_2d.vertices[:,:].T / model_2d.vertices[:,-1]).T
    # Reconvert to geometric coordinates
    model_2d.vertices = model_2d.vertices[:,:3]
    # Second, we obtain the projected faces
    model_2d.ifaces = model_3d.ifaces.copy()
    # Third, we measure the area of the projected faces
    auxComputeProjectedFacesArea(model_2d)
    # Fourth, we detect occluded faces, The sign of occluded faces in changed
    if model_2d.areas != 0.0:
        auxDetectOccludedFaces(model_2d)
    # Fifth, we obtain the edges, removing duplicates
    adj_matrix = AdjMatrix(n_vertices)
    for iface in model_2d.ifaces:
        for i in range(len(iface)-1):
            iv0, iv1 = iface[i],iface[i+1]
            adj_matrix.addEdge(iv0,iv1)
        iv0, iv1 = iface[-1],iface[0] # Close the polygon edge
        adj_matrix.addEdge(iv0,iv1)
    model_2d.iedges = adj_matrix.getIndexedEdges()
    model_3d.cached2dModel = model_2d
    return model_2d

def draw2dModel(model_2d) -> None:
    """ Draw the 2D model in normalized coordinates """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glColor(0.0, 0.0, 0.0, 1.0)
    glLineWidth(1)
    glBegin(GL_LINES)
    for ie in model_2d.iedges:
        v0 = model_2d.vertices[ie[0]-1]
        v1 = model_2d.vertices[ie[1]-1]
        glVertex3fv(v0)
        glVertex3fv(v1)
    glEnd()
    
# Profit functions
def balance(f,b):
    return (f*b) / (f+b) * (2*f) / (f+b)

def profitsProjectedFacesArea(model_2d, is_top_view):
    """ Compute a profit for the projected faces area """
    """ Constraint: This area is 0.0 if any point has been proyected outside the clipping window (-1,-1) to (1, 1) """
    """ Return (faces_area, visibility_ratio)"""
    if model_2d.cachedProfits != None:
        return model_2d.cachedProfits
    total_area = 0.0
    n_front_faces, n_back_faces = 0, 0
    # Trigger the contraint
    if model_2d.areas == 0.0:
        return (-100.0, 0.0, n_front_faces, n_back_faces)
    for face_area in model_2d.areas.values():  
        if face_area < 0.0: # CCW faces are front faces
            n_front_faces += 1
        else: # CW faces are back faces
            n_back_faces += 1     
        total_area += np.abs(face_area)
    if is_top_view:
        total_area += total_area*model3d.Model3D.TOP_VIEW_AREA_INFLATION_PERCENTAGE
    model_2d.cachedProfits = (total_area, balance(n_front_faces, n_back_faces), n_front_faces, n_back_faces )
    return model_2d.cachedProfits

def repulsion_force(d, t):
        inv_t = 1/t
        exponent = inv_t*(inv_t*d-0.5)
        if exponent > 10: # Speed up avoiding calculate r when r->0
            exponent = 10
            r = 0
        else:
            r = 1 / (1+np.exp(exponent))
        return r

def penaltiesCloseVerticesCrossedAndCloseEdges(model_2d):
    """ Return three values with the close vertices' repulsion, crossed and close edges penalties """
    if model_2d.cachedPenalties != None:
        return model_2d.cachedPenalties
    # Compute close vertices repulsion
    unique_pairs = itertools.combinations(model_2d.vertices, 2)
    vertices_repulsion = 0.0
    for pair in unique_pairs:
        vertices_repulsion += repulsion_force(distance2D(*pair), 0.1)
    # Compute crossed and close edges penalties
    unique_edge_pairs = list(itertools.combinations(model_2d.iedges, 2))
    n_crosses = 0
    close_edges_penalty = 0.0
    for iedges_pair in unique_edge_pairs:
        edges_pair_2d = model_2d.vertices[np.array(iedges_pair)-1][:,:,:-1]
        i = intersection(*edges_pair_2d[0], *edges_pair_2d[1])
        if (i == Intersection.INTERSECTING and not are_adjacent_iedges(*iedges_pair)):
            n_crosses += 1
        elif (i == Intersection.PARALLEL): # Quasi-parallel edges
            if not are_adjacent_iedges(*iedges_pair) or tight_angle(model_2d,iedges_pair):
                d = segments_dist(*edges_pair_2d[0], *edges_pair_2d[1])
                r = repulsion_force(d, 0.05)
                close_edges_penalty += r
    n_edges = len(model_2d.iedges)
    vertices_repulsion /= n_edges
    crosses_penalty = n_crosses/n_edges
    close_edges_penalty /= n_edges
    model_2d.cachedPenalties = [vertices_repulsion, crosses_penalty, close_edges_penalty]
    return model_2d.cachedPenalties

if __name__ == '__main__':
     # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    WndId = glutCreateWindow('')
    # Perform the test
    model_3d = model3d.loadModel(model3d.Model3D.OBJS_DIR + '/cube.obj')
    model_3d.projection = model3d.ProjectionType.PERSPECTIVE
    Mpers = model3d.computeProjectionMatrix(model_3d, aspect_ratio = 1.0)
    Mmv = model3d.computeModelviewMatrix(model_3d)
    model_2d = compute2dModel(model_3d, Mpers, Mmv)
    print('Vertices 2D:\n' + str(np.round(model_2d.vertices,3)))
    print('IFS 2D:\n' + str(model_2d.ifaces))
    print('IEDGES 2D:' + str(model_2d.iedges))
    print('AREAS 2D:' + str({key: np.round(value,3) for key, value in model_2d.areas.items()}))
    profit_area, balance_ratio, f, b = profitsProjectedFacesArea(model_2d, model_3d.top_view())
    print('Profit area:(%.3f*%.2f)=%.3f b(f=%d, b=%d)=%.2f' % (profit_area, balance_ratio, profit_area*balance_ratio, f, b, balance_ratio))
    vertices_repulsion, n_crosses, edges_repulsion = penaltiesCloseVerticesCrossedAndCloseEdges(model_2d)
    print('Close vertices total repulsion force:', vertices_repulsion)
    print('Cross edges:', n_crosses)
    print('Close edges total repulsion foce:', edges_repulsion)
