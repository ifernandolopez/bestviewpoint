import numpy as np 
import math
import itertools
import model3d
from OpenGL.GL import*
from OpenGL.GLU import*
from OpenGL.GLUT import*

# Helper functions

class AdjMatrix:
    def __init__(self, n_vertices):
        self.adj_matrix = np.zeros((n_vertices, n_vertices))
        self.iedges = []
    def addEdge(self, iv0, iv1):
        if iv0 > iv1:
            iv0, iv1 = iv1, iv0
        if self.adj_matrix[iv0-1,iv1-1] > 0:
            return
        self.adj_matrix[iv0-1,iv1-1] += 1
        self.iedges.append([iv0, iv1])
    def getIndexedEdges(self):
        return self.iedges
    
def distance2D(a, b):
    return np.sqrt(np.sum((a-b)**2))

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
    den = a[1]*b[0] - a[0]*b[1]
    # We first analyze the cases where the denominator is zero: quasi-paralled or overlapped
    if np.isclose(den, 0, atol = 5e-2):
        return Intersection.PARALLEL
    # Then we detect the NO intersection case
    alpha_num = b[1]*c[0] - b[0]*c[1]
    beta_num = a[0]*c[1] - a[1]*c[0]
    if den > 0 and (alpha_num <= 0 or alpha_num > den or beta_num <= 0 or beta_num >= den):
        return Intersection.NON_INTERSECTING
    if den < 0 and (alpha_num >= 0 or alpha_num < den or beta_num >= 0 or beta_num <= den):
        return Intersection.NON_INTERSECTING
    # Otherwise, there is an intersection: it can be at a vertex, side or cross intersection
    return Intersection.INTERSECTING

def are_adjacent_iedges(ie1, ie2):
    return ie1[0] == ie2[0] or ie1[0] == ie2[1] or ie1[1] == ie2[0] or ie1[1] == ie2[1]

def tight_angle(model_2d, iedges_pair):
    origin_vertex = []
    arrow_vertices = []
    for v in np.concatenate(iedges_pair):
        if (v in arrow_vertices):
            arrow_vertices.remove(v)
            origin_vertex.append(v)
        else:
            arrow_vertices.append(v)
    origin_vertex_2d = model_2d.vertices[np.array(origin_vertex)-1][:,0:2]
    arrow_vertices_2d = model_2d.vertices[np.array(arrow_vertices)-1][:,0:2]
    vectors_2d = arrow_vertices_2d - origin_vertex_2d
    try:
        acos = np.dot(*vectors_2d) / (np.linalg.norm(vectors_2d[0]) * np.linalg.norm(vectors_2d[1]))
    except(ex):
        print(ex)
        print(vectors_2d[0])
        print(vectors_2d[1])
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
    
# 3D model functions

class Model2D:
    def __init__(self, model_3d):
        self.model3D = model3d
        self.vertices = None   # Projected xy vertices
        self.ifaces = []          # Projected Indexed Face Set (IFS) with indexed faces
        self.iedges = []       # Projected indexed edges, without duplicates

def compute2dModel(model_3d, Mpers, Mmv):
    """ Return a Model2D with the projection of the the model_3d according to Mpers a Mmv
        In turn, the Mpers and Mmv were consctructed using the projection, rho, theta, phi an fovy of the model_3d """
    Mcomposed = np.matmul(Mmv, Mpers)
    model_2d = Model2D(model_3d)
    # First, we project the vertices
    n_vertices = len(model_3d.vertices)
    # We multiply in homogeneous coordinates
    vertices_4d_homogeneous = np.c_[model_3d.vertices, np.ones(n_vertices)]
    model_2d.vertices = np.matmul(vertices_4d_homogeneous,Mcomposed)
    # Normalize the homogeneous coordinates and reconvert it to geometric coordinates
    model_2d.vertices = (model_2d.vertices[:,:].T / model_2d.vertices[:,-1]).T
    model_2d.vertices = model_2d.vertices[:,:3]
    # Second, we obtain the projected the faces
    model_2d.ifaces = model_3d.ifaces.copy()
    # Third, we obtain the edges, removing duplicates
    adj_matrix = AdjMatrix(n_vertices)
    for iface in model_2d.ifaces:
        for i in range(len(iface)-1):
            iv0, iv1 = iface[i],iface[i+1]
            adj_matrix.addEdge(iv0,iv1)
        iv0, iv1 = iface[-1],iface[0] # Close the polygon edge
        adj_matrix.addEdge(iv0,iv1)
    model_2d.iedges = adj_matrix.getIndexedEdges()
    return model_2d

def draw2dModel(model_2d):
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
    return (f*b-max(f,b)) / max(f,b)

def profitProjectedFacesArea(model_2d):
    """ Compute a profit for the projected faces area """
    """ Constraint: This area is 0.0 if any point has been proyected outside the clipping window (-1,-1) to (1, 1) """
    """ Return (faces_area, visibility_ratio)"""
    faces_area = 0.0
    n_front_faces, n_back_faces = 0, 0
    for iface in model_2d.ifaces:
        vertices = model_2d.vertices[np.array(iface)-1]
        # Trigger the contraint
        if np.max(vertices[:,[0,1]])>1.0 or np.min(vertices[:,[0,1]])<-1.0:
            return (-100.0, 1.0)
        vertices = np.vstack([vertices,vertices[0]]) # We duplicate the first vertex at the end 
        downwards, upwards = 0.0, 0.0
        for i in range(0,len(vertices)-1):
            downwards += vertices[i,0]*vertices[i+1,1]
            upwards += vertices[i,1]*vertices[i+1,0]   
        face_area = upwards-downwards   
        if face_area < 0.0: # CCW faces are front faces
            n_front_faces += 1
        else: # CW faces are back faces
            n_back_faces += 1     
        faces_area += np.abs(face_area)
    return (0.5 * faces_area, balance(n_front_faces, n_back_faces), n_front_faces, n_back_faces )

def repulsion_force(d, t):
        inv_t = 1/t
        exponent = inv_t*(inv_t*d-0.5)
        if exponent > 10: # Speed up avoiding calculate r when r->0
            exponent = 10
            r = 0
        else:
            r = 1 / (1+np.exp(exponent))
        return r
    
def penaltyCloseVertices(model_2d):
    unique_pairs = itertools.combinations(model_2d.vertices, 2)
    sum = 0.0
    for pair in unique_pairs:
        sum += repulsion_force(distance2D(*pair), 0.1)
    return sum

def penaltyCrossedAndCloseEdges(model_2d):
    """ Return two values with the crossed and close edges penalty """
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
    crosses_penalty = n_crosses/math.log(len(model_2d.iedges),6)
    return [crosses_penalty, close_edges_penalty]

if __name__ == '__main__':
     # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    WndId = glutCreateWindow('')
    # Perform the test
    model_3d = model3d.loadModel(model3d.Model3D.OBJS_DIR + '/cross.obj')
    Mpers = model3d.computeProjectionMatrix(model_3d, 1.0)
    Mmv = model3d.computeModelviewMatrix(model_3d)
    model_2d = compute2dModel(model_3d, Mpers, Mmv)
    print('Vertices 2D:\n' + str(np.round(model_2d.vertices,3)))
    print('IFS 2D:\n' + str(model_2d.ifaces))
    print('IEDGES 2D:' + str(model_2d.iedges))
    profit_area, balance_ratio, f, b = profitProjectedFacesArea(model_2d)
    print('Profit area:(%.3f*%.2f)=%.3f b(f=%d, b=%d)=%.2f' % (profit_area, balance_ratio, profit_area*balance_ratio, f, b, balance_ratio))
    vertices_repulsion = penaltyCloseVertices(model_2d)
    print('Close vertices total repulsion force:', vertices_repulsion)
    [n_crosses, edges_repulsion] = penaltyCrossedAndCloseEdges(model_2d)
    print('Cross edges:', n_crosses)
    print('Close edges total repulsion foce:', edges_repulsion)
