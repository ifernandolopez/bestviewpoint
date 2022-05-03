from OpenGL.GL import*
from OpenGL.GLU import*
from OpenGL.GLUT import*
import numpy as np 
sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))
tand = lambda degrees: np.tan(np.deg2rad(degrees))

class ProjectionType:
    ORTOGONAL = 0
    CABINET = 1
    PERSPECTIVE = 2
    _cabinetMatrix = None
    @staticmethod
    def cabinetMatrix():
        if (ProjectionType._cabinetMatrix != None):
            return ProjectionType._cabinetMatrix
        alfa = 63.4 * np.pi / 180.
        phi = 45 * np.pi / 180.
        vpx = np.cos (phi) / np.tan (alfa)
        vpy = np.sin (phi) / np.tan (alfa)
        ProjectionType._cabinetMatrix = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            vpx, vpy, 1, 0,
            0.0, 0.0, 0.0, 1.0)
        return ProjectionType._cabinetMatrix

class Model3D:
    DEFAULT_RHO = None
    DEFAULT_THETA = 0
    DEFAULT_PHI = 90
    DEFAULT_FOVY = 60
    OBJS_DIR = 'bestviewpoint/objs' 
    EDGES_COLOR = (0.1, 0.5, 0.9, 1.0)
    FACES_COLOR = (0.7, 0.8, 0.9, 0.6)
    TOP_VIEW_AREA_INFLATION_PERCENTAGE = 0.1
    def __init__(self):
        self.filename = None
        self.projection = ProjectionType.PERSPECTIVE
        self.rho, self.theta, self.phi = Model3D.DEFAULT_RHO, Model3D.DEFAULT_THETA, Model3D.DEFAULT_PHI
        self.fovy = Model3D.DEFAULT_FOVY
        self.vertices = None    # xyz per vertex
        self.ifaces = []           # Indexed Face Set (IFS) with indexed faces
        self.center = None
        self.minRadius = None
    def top_view(self):
        return self.phi < 90

def loadModel(path):
    """ Return a Model with the vertices and an Indexed Face Set (IFS) with indexed faces
        Skip textels and normals, materials, etc """
    model_3d = Model3D()
    model_3d.filename = path
    file = open(path, 'r')
    model_3d.vertices = []
    for line in file:
        type = line[0:2]
        if (type == "v "):
            vertex = [float(cood) for cood in line[2:].split()]
            model_3d.vertices.append(np.array(vertex))
        elif (type == "f "):
            face_indexes = [int(token.split('/')[0]) for token in line[2:].split()]
            model_3d.ifaces.append(face_indexes)    
    file.close() 
    model_3d.vertices = np.array(model_3d.vertices)
    # Compute the center
    model_3d.center = ( np.amax(model_3d.vertices, axis=0) + np.amin(model_3d.vertices,axis=0) ) / 2
    # Compute the radius covering all the model
    model_3d.minRadius = 0.0
    for vertex in model_3d.vertices:
        distance=np.sqrt((vertex[0]-model_3d.center[0])**2 + (vertex[1]-model_3d.center[1])**2 + (vertex[2]-model_3d.center[2])**2)
        if (distance > model_3d.minRadius):
            model_3d.minRadius = distance
    if model_3d.DEFAULT_RHO == None:
        model_3d.DEFAULT_RHO = model_3d.rho = 2 * model_3d.minRadius
    return model_3d

def computeProjectionMatrix(model_3d, aspect_ratio):
    """ Compute the projection matrix Mpers for the selected proyection """
    """ Uses the projection indicated in the model_3d.projection attribute """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    near, far = 0.01, 2*model_3d.rho
    fustrum_size = model_3d.rho
    if model_3d.projection == ProjectionType.ORTOGONAL:
        glOrtho(-fustrum_size/2, fustrum_size/2, -fustrum_size/2, fustrum_size/2, near, far)
    elif model_3d.projection == ProjectionType.CABINET:
        glLoadMatrixd(ProjectionType.cabinetMatrix())
        glOrtho(-fustrum_size/2, fustrum_size/2, -fustrum_size/2, fustrum_size/2, near, far)
    else:
        gluPerspective(model_3d.fovy, aspect_ratio, near, far)
    Mpers = glGetFloatv(GL_PROJECTION_MATRIX)
    return Mpers
        
def computeModelviewMatrix(model_3d):
    """ Compute the projection matrix Mmv that changes the point of view in model_3d to position rho, theta, phi """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    rho = model_3d.rho
    theta = model_3d.theta
    phi = model_3d.phi
    phi = 1.1e-15 if (phi==0) else phi # Hack to prevent failure in glLookAt when x=0, z=0 and v_up = (0,1,0)
    gluLookAt(rho*sind(theta)*sind(phi),rho*cosd(phi),rho*cosd(theta)*sind(phi), 0,0,0, 0,1,0)
    # Move the center of the object to the cordinates origin
    glTranslated(-model_3d.center[0],-model_3d.center[1],model_3d.center[2])
    Mmv = glGetFloatv(GL_MODELVIEW_MATRIX)
    return Mmv

def draw3dFaces(model_3d, mode, show_face):
    glPolygonMode(GL_FRONT_AND_BACK , mode)
    if show_face==-1:
        ifaces = model_3d.ifaces
    else:
        ifaces = [model_3d.ifaces[show_face]]
    for iface in ifaces:
        glBegin(GL_POLYGON)
        for i in iface:
            glVertex3fv(model_3d.vertices[i-1])
        glEnd()
    glPolygonMode(GL_FRONT_AND_BACK , GL_FILL)
    
def draw3dModel(model_3d, wireframe, show_face):
    """ Draw the 3D model in object coordinates """
    glColor4d(*Model3D.EDGES_COLOR)
    glLineWidth(3.0)
    # 1. Paint dashed edges with polygon and culling off, and blending on
    glEnable(GL_POLYGON_OFFSET_LINE)
    glPushAttrib(GL_ENABLE_BIT)
    glLineStipple(1, 0x000F)
    glEnable(GL_LINE_STIPPLE)
    draw3dFaces(model_3d, GL_LINE, show_face)
    glPopAttrib()
    glDisable(GL_POLYGON_OFFSET_LINE)
    # 2. Paint faces with polygon offset, culling and blending on 
    glEnable(GL_CULL_FACE)
    if not wireframe:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4d(*Model3D.FACES_COLOR)
        glEnable(GL_POLYGON_OFFSET_FILL)
        draw3dFaces(model_3d, GL_FILL, show_face)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)
    # 3. Paint edges with, poligon offset, and blending off, culling on
    glColor4d(*Model3D.EDGES_COLOR)
    draw3dFaces(model_3d, GL_LINE, show_face)
    glDisable(GL_CULL_FACE)

if __name__ == '__main__':
    # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(500, 500)
    WndId = glutCreateWindow('')
    # Perform the test
    model_3d = loadModel(Model3D.OBJS_DIR + '/cube.obj')
    print('Vertices 3D:\n', np.round(model_3d.vertices,3))
    print('IFS:\n', model_3d.ifaces)
    aspect_ratio = 1.0
    Mpers = computeProjectionMatrix(model_3d, aspect_ratio)
    print('GL_PROJECTION_MATRIX:\n' + str(np.round(Mpers,3)))
    Mmv = computeModelviewMatrix(model_3d)
    print('GL_MODELVIEW_MATRIX:\n' + str(np.round(Mmv,3)))
