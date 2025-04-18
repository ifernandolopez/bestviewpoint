import os
from OpenGL.GL import*
from OpenGL.GLU import*
from OpenGL.GLUT import*
import model3d
import model2d
import optimization
import numpy as np
import copy
import sys
import glob
import pyinstrument

# Global variables
WndTitle = 'Starting'
WndTopDecorativeGap = 10
WndBottomLegendGap = 170
WndWidth, WndHeight = 1000, 500 + WndTopDecorativeGap + WndBottomLegendGap
WndId:int  = None
Current3DModel: model3d.Model3D = None
ShowFace: int = -1 # Face index, or -1 for all faces
Wireframe: bool = False

# Tentative3DModel and Tentative2DModel states used during optimization:
#   None - Optimization has no been executed, or has finished
#   Objects - Optimization is executing
Tentative3DModel: model3d.Model3D = None
Tentative2DModel: model2d.Model2D = None

# Optimizer states:
#   False - Optimization has not been executed, or has been reset
#   Object - Executing
#   True - Optimization finished
Optimizer = False

# Variables to enable profiling
WithProfiler, Profiler = False, None

def isOptimizing():
    global Optimizer
    return Optimizer != False and Optimizer != True

def acceptingInteractiveChanges():
    global Optimizer
    return not isOptimizing()

def resetOptimizerIfFinished():
    global Optimizer
    if Optimizer==True:
        Optimizer = False
        
def resetGLMatrices():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

def drawStringBitmaps (x, y, color, str):
    """ Output string in x,y normalized coordinates using GLUT bitmaps """
    position = (x, y, 0.99)
    glColor4fv(color)
    glRasterPos3fv(position)
    for c in str:
         glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

def profitInfo(model_3d, model_2d):
    """ Return profit info for the parameters
        (area, balance_ratio, crosses_repulsion, vertices_repulsion, edges_repulsion, legend) """
    area, balance_ratio,f, b = model2d.profitsProjectedFacesArea(model_2d, model_3d.top_view())
    vertices_repulsion, crosses_repulsion, edges_repulsion = model2d.penaltiesCloseVerticesCrossedAndCloseEdges(model_2d)
    profit = area*balance_ratio 
    penalty = crosses_repulsion + vertices_repulsion + edges_repulsion
    total = np.round(profit/(1+penalty),2)
    if total == -0.0:
        total = 0.0
    profit_legend = 'Profit: %.2f/(1+%.2f)=%.2f' % (profit, penalty, total)
    return (area, balance_ratio, f, b, crosses_repulsion, vertices_repulsion, edges_repulsion, profit_legend)

def povLegend(model_3d):
    rho_info = 'Rho=' + "%.2f" % model_3d.rho
    theta_info = ' Theta=' + "%.0f" % model_3d.theta + 'º'
    phi_info = ' Phi=' + "%.0f" % model_3d.phi + 'º'
    return rho_info + theta_info + phi_info

def draw3dInfo(model_2d):
    """ Draw the legend in the left 3D scene panel """
    pov_legend = povLegend(Current3DModel)
    optimization_legend = ''
    if Optimizer == True:
        optimization_legend = 'Optimization done'
    elif Optimizer != False:
        if isinstance(Optimizer, optimization.SAOptimizer):
            optimization_legend = 'SA Optimizing T: %.2f' % Optimizer.T
        else:
            optimization_legend = 'TS Optimizing Pending iters: %d' % Optimizer.pending_it
        if Optimizer.has_cooled:
                optimization_legend += ' (cooling)'
    area, balance_ratio, f, b, crosses_repulsion, vertices_repulsion, edges_repulsion, profit_legend = profitInfo(Current3DModel, model_2d)    
    if Wireframe:
        material_legend = 'Wireframe'
    else:
        material_legend = 'Solid'
    if (ShowFace != -1):
        material_legend += ' (face ' + str(ShowFace+1) + ')'
    if Current3DModel.projection == model3d.ProjectionType.ORTOGONAL:
        projection_legend = 'Ortogonal projection, '
    elif Current3DModel.projection == model3d.ProjectionType.CABINET:
        projection_legend = 'Oblique cabinet projection, '
    else:
        projection_legend = 'Perspective projection, '
    top_view_legend = ''
    if Current3DModel.top_view():
        top_view_legend = '(top_view)'
    drawStringBitmaps(-0.95, -0.51, Current3DModel.EDGES_COLOR, optimization_legend)
    drawStringBitmaps(-0.95, -0.60, Current3DModel.EDGES_COLOR, projection_legend + material_legend)
    drawStringBitmaps(-0.95, -0.69, Current3DModel.EDGES_COLOR, pov_legend)
    drawStringBitmaps(-0.95, -0.78, Current3DModel.EDGES_COLOR, profit_legend)
    area_legend = 'Area: %.2f*%.1f=%.2f balance(f=%d,b=%d)=%.1f %s' % (area, balance_ratio, area*balance_ratio, f, b, balance_ratio, top_view_legend)
    drawStringBitmaps(-0.95, -0.87, Current3DModel.EDGES_COLOR, area_legend)
    repulsion_legend = 'Repulsion (V: %.2f, C:%.2f, E: %.2f)'  % (vertices_repulsion, crosses_repulsion, edges_repulsion)
    drawStringBitmaps(-0.95, -0.96, Current3DModel.EDGES_COLOR, repulsion_legend)

def draw2dInfo(model_2d):
    """ Draw the legend in the right optimization panel """
    global Tentative3DModel, Tentative2DModel
    assert(Tentative3DModel != None and Tentative2DModel != None)
    black = (0.0, 0.0, 0.0, 1.0)
    pov_legend = povLegend(Tentative3DModel)
    drawStringBitmaps(-1.0, -0.69, black, pov_legend)
    area, balance_ratio, f, b, crosses_repulsion, vertices_repulsion, edges_repulsion, profit_legend = profitInfo(Tentative3DModel, Tentative2DModel)
    drawStringBitmaps(-1.0, -0.78, black, profit_legend)
    top_view_legend = ''
    if Tentative3DModel.top_view():
        top_view_legend = '(top_view)'
    area_legend = 'Area: %.2f*%.1f=%.2f balance(f=%d,b=%d)=%.1f %s' % (area, balance_ratio, area*balance_ratio, f, b, balance_ratio, top_view_legend)
    drawStringBitmaps(-1.0, -0.87, black, area_legend)
    repulsion_legend = 'Repulsion (V: %.2f, C:%.2f, E: %.2f)'  % (-vertices_repulsion, -crosses_repulsion, -edges_repulsion)
    drawStringBitmaps(-1.0, -0.96, black, repulsion_legend)
    
def displayCB():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, WndBottomLegendGap, WndWidth//2, WndHeight-WndBottomLegendGap-WndTopDecorativeGap)
    Mpers = model3d.computeProjectionMatrix(Current3DModel, get_aspect_ratio())
    Mmv = model3d.computeModelviewMatrix(Current3DModel)
    model3d.draw3dModel(Current3DModel, Wireframe, ShowFace)
    glViewport(0, 0, WndWidth//2, WndHeight) # The 3D viewport lower-left corner and the width-height in pixels
    model_2d = model2d.compute2dModel(Current3DModel, Mpers, Mmv)
    resetGLMatrices()
    draw3dInfo(model_2d)
    if isOptimizing(): # Draw the optimization right panel
        glViewport(WndWidth//2, WndBottomLegendGap, WndWidth//2, WndHeight-WndBottomLegendGap-WndTopDecorativeGap) # The projection viewport lower-left corner and the width-height in pixels
        model2d.draw2dModel(Tentative2DModel)
        glViewport(WndWidth//2, 0, WndWidth//2, WndHeight)
        draw2dInfo(Tentative2DModel)
    glutSwapBuffers()

def get_aspect_ratio():
    return WndWidth/2.0/(WndHeight-WndBottomLegendGap-WndTopDecorativeGap)

def resizeCB(w, h):
    """  Prevert changes in the aspect ratio because otherwise the area changes in perspective projection """
    global WndWidth, WndHeight, Reshaping
    new_aspect_ratio = w/2.0/(h-WndBottomLegendGap-WndTopDecorativeGap)
    if new_aspect_ratio > 1.0:
        w = int(2.0 * (h-WndBottomLegendGap-WndTopDecorativeGap))
    elif new_aspect_ratio < 1.0:
        h = int(w/2.0 + WndBottomLegendGap + WndTopDecorativeGap)
    if new_aspect_ratio > 1.0 or new_aspect_ratio < 1.0:
        WndWidth, WndHeight = int(w), int(h)
        glutReshapeWindow( WndWidth, WndHeight)
    if w!=WndWidth or h!= WndHeight:
        glutPostRedisplay()
    
def specialKeyCB(key, x, y):
    global Optimizer
    if not acceptingInteractiveChanges():
        return
    resetOptimizerIfFinished()
    if ( key == GLUT_KEY_LEFT ):
        Current3DModel.theta = (Current3DModel.theta + 5) % 360
    elif ( key == GLUT_KEY_RIGHT ):
        Current3DModel.theta = (Current3DModel.theta - 5) % 360
    elif ( key == GLUT_KEY_UP):
        if (Current3DModel.phi<180):
            Current3DModel.phi += 5
    elif ( key == GLUT_KEY_DOWN ):
        if (Current3DModel.phi>0):
         Current3DModel.phi -= 5
    elif ( key == GLUT_KEY_HOME ):
        if (Current3DModel.rho>Current3DModel.minRadius):
            Current3DModel.rho -= 0.5
    elif ( key == GLUT_KEY_END ):
        Current3DModel.rho += 0.5
    Current3DModel.flushCache()
    glutPostRedisplay()

def keyboardCB(key, x, y):
    global ShowFace, Wireframe, Optimizer, Profiler
    upper_face_keys = {b'!':1, b'"':2, b'\xc2':3, b'$':4, b'%':5, b'&':6, b'/':7, b'(':8, b')':9, b'=':0}
    if key in upper_face_keys.keys():
        key = str(10 + upper_face_keys[key])
    if key.isdigit():
        face_index = int(key)
        if glutGetModifiers() == GLUT_ACTIVE_CTRL:
            face_index += 10
        if (face_index==ShowFace):
            ShowFace = -1 # Disable show face
        else:
            if face_index >= len(Current3DModel.ifaces):
                return
            ShowFace = face_index # Enable show face
        glutPostRedisplay()
    elif key == b'w':
        Wireframe = not Wireframe
        glutPostRedisplay()
    elif key == b' ':
        if not acceptingInteractiveChanges():
            return
        resetOptimizerIfFinished()
        Current3DModel.rho, Current3DModel.theta, Current3DModel.phi = Current3DModel.DEFAULT_RHO, Current3DModel.DEFAULT_THETA, Current3DModel.DEFAULT_PHI
        Current3DModel.flushCache()
        glutPostRedisplay()
    elif key == b'p' or key == b'P':
        if not acceptingInteractiveChanges():
            return
        resetOptimizerIfFinished()
        Current3DModel.projection = (Current3DModel.projection+1) % 3
        Current3DModel.flushCache()
        glutPostRedisplay()
    elif key == b's' or key == b'S' or key == b't' or key == b'T':
        if Optimizer==False or Optimizer==True:
            grid_percentage = 0.05
            grid_steps = grid_percentage * np.array((2 * Current3DModel.minRadius, 360, 180))
            rho, theta, phi = [2*Current3DModel.minRadius,2.5*Current3DModel.minRadius], [0, 360-grid_steps[1]], [0,180-grid_steps[2]]
            domains = [rho, theta, phi]
            if key == b's' or key == b'S':
                OptimizerClass = optimization.SAOptimizer
            else:
                OptimizerClass = optimization.TSOptimizer
            Optimizer = OptimizerClass(domains, tentative_3d_model_cost_fn, grid_steps)
            start_sol = [Current3DModel.rho, Current3DModel.theta, Current3DModel.phi]
            Optimizer.restart(start_sol)
            glutIdleFunc(idleCB)
            if WithProfiler:
                Profiler = pyinstrument.Profiler()
                Profiler.start()
        else:
            glutIdleFunc(None)
            Optimizer = False
            glutPostRedisplay()
            if WithProfiler:
                Profiler.stop()
                Profiler.print()
                Profiler = None
        return
    elif key == b'\x1b' or key == b'q'  or key == b'Q':
        glutDestroyWindow (WndId)
             
def tentative_3d_model_cost_fn(tentative_sol):
    global Tentative3DModel, Tentative2DModel
    Tentative3DModel = copy.copy(Current3DModel) # Shallow copy of the outermost container (without cache)
    Tentative3DModel.flushCache()
    Tentative3DModel.rho, Tentative3DModel.theta, Tentative3DModel.phi = tentative_sol
    Mpers = model3d.computeProjectionMatrix(Tentative3DModel, get_aspect_ratio())
    Mmv = model3d.computeModelviewMatrix(Tentative3DModel)
    Tentative2DModel = model2d.compute2dModel(Tentative3DModel, Mpers, Mmv)
    resetGLMatrices()
    area, balance_ratio, f, b = model2d.profitsProjectedFacesArea(Tentative2DModel, Tentative3DModel.top_view())
    vertices_repulsion, crosses_repulsion, edges_repulsion = model2d.penaltiesCloseVerticesCrossedAndCloseEdges(Tentative2DModel)
    total = -(area*balance_ratio) / (1 + vertices_repulsion + crosses_repulsion + edges_repulsion)
    return total

def idleCB():
    global Optimizer, RedrawOptimizationCount, Profiler
    assert Optimizer != False and Optimizer!=True
    if Optimizer.hasFinished():
        glutIdleFunc(None)
        Optimizer = True
        Tentative3DModel = None
        Tentative2DModel = None
        if WithProfiler:
            Profiler.stop()
            Profiler.print()
            Profiler = None
        glutPostRedisplay()
        return
    better_solution_found = Optimizer.step()
    if better_solution_found:
        Current3DModel.rho, Current3DModel.theta, Current3DModel.phi = Optimizer.best_sol
        Current3DModel.flushCache()
    glutPostRedisplay()

def popupMenuCB(value):
    if isOptimizing():
        return value
    loadModel(value)
    glutPostRedisplay()
    return value

def createPopupMenu():
    global ObjFiles
    glutCreateMenu(popupMenuCB)
    ObjFiles = sorted([os.path.split(f)[-1] for f in glob.glob(model3d.Model3D.OBJS_DIR + '/*.obj')])
    for i,f in enumerate(ObjFiles):
        glutAddMenuEntry(f, i)
    glutAttachMenu(GLUT_RIGHT_BUTTON)

def loadModel(file_index):
    global Current3DModel, ObjFiles
    Current3DModel = model3d.loadModel(model3d.Model3D.OBJS_DIR + '/' + ObjFiles[file_index])
    title = 'Arrows (move) Space (Restore) S (SA optimizer) T (TS optimizer) W (Wireframe/Solid) Number (ShowFace) ' \
          + 'Right Click (Load): %s' % ObjFiles[file_index]
    glutSetWindowTitle(title)

if __name__ == '__main__':
    # Initialize GLUT
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(WndWidth, WndHeight)
    WndId = glutCreateWindow(WndTitle)
    # Set the global OpenGL state
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glPolygonOffset(1.0,1.0)
    glLineStipple(1, 0x000F)
    glEnable(GL_DEPTH_TEST)
    # Set OpenGL callbacks
    glutDisplayFunc(displayCB)
    glutReshapeFunc(resizeCB)
    glutKeyboardFunc(keyboardCB)
    glutSpecialFunc(specialKeyCB)
    # Load the model and show it
    createPopupMenu()
    loadModel(26)
    glutMainLoop()
