from copy import copy
import numpy as np
import vtk
import matplotlib.pyplot as plt
from vedo import printc
from skimage import exposure as ex
from sklearn.decomposition import PCA, SparsePCA
import networkx as nx

from skimage import io
from skimage.color import rgb2gray
from skimage import filters
from skimage.filters.rank import mean_bilateral
from skimage.filters import rank
from skimage.morphology import ball, disk


def graphToCoords(graph, graphnodes=None):
  coords = []
  if type(graphnodes) == type(None):
    graphnodes = graph.nodes
  for node in graphnodes:
    coords.append(graph.nodes[node]["pos"])
  return coords


def simplifyGraph(G, debug=False):
  ''' Reduce graph containing all skeletonised voxels to only branch points.
      Loop over the graph until all nodes of degree 2 have been 
      removed and their incident edges fused
      params:
      G: networkx graph
      returns:
      networkx graph 
  '''
  g = G.copy()
  # use while loop as we break the for loop once we remove a mid-point node
  # so loop until there are no nodes with degree = 2 (line to line).
  # We are left with only degree >= 3 (branch) or degree = 1 (end points)
  stopLoop = False # for forcing loop to stop if error found in getting edges
  while any(degree==2 for _, degree in g.degree):
    if stopLoop and sum(degree==2 for _, degree in g.degree)<=1:
      break 
    # prevent error `dictionary changed size during iteration` 
    g0 = g.copy() 
    for node, degree in g.degree():
      # print(g.nodes[node]["pos"])
      if degree==2:
        # for directed graphs we need to maintain direction 
        # (which point is in vs out)
        if g.is_directed(): 
          if len(list(g.in_edges(node))) == 0 or len(list(g.out_edges(node))) == 0:
            # prevent strange issue where no in_edge for some nodes
            stopLoop = True # force while loop to stop
            continue
          a0,b0 = list(g.in_edges(node))[0]
          a1,b1 = list(g.out_edges(node))[0]

        else:
          edges = g.edges(node)
          edges = list(edges)#.__iter__())
          a0,b0 = edges[0]
          a1,b1 = edges[1]

        # decide which nodes to save and which to delete
        if a0 != node:
          e0 = copy(a0)
        else:
          e0 = copy(b0)

        if a1 != node:
          e1 = copy(a1)
        else:
          e1 = copy(b1)

        # remove midpoint and merge two adjacent edges to become one
        g0.remove_node(node)
        g0.add_edge(e0, e1)
        break

    g = g0.copy()

  if debug:
    print("\nsimplified graph")
    print(nx.info(g))

  return g

def advance_tCounter(t_c, s, neighbors):
    '''
        Add t_c for this iteration of the loop (part of step 2b(8))
        t_c refers to t_counter. The t that is used in equation (A.3)
    '''

    #-get index of t list which holds s
    sIndex = np.where((t_c[:,0] == s[0])*\
                        (t_c[:,1] == s[1])*\
                        (t_c[:,2] == s[2]))[0]

    #-add one to the index
    t_c[sIndex,3] = t_c[sIndex,3] + 1
    
    for n in neighbors:
        #-same, for each neighboring node
        nIndex = np.where((t_c[:,0] == n[0])*\
                                            (t_c[:,1] == n[1])*\
                                            (t_c[:,2] == n[2]))[0]
        t_c[nIndex,3] = t_c[nIndex,3] + 1
    
    return t_c

def alignToMean(pointCloud):
    '''
        Align a surface point-cloud to its mean
    '''
    return (pointCloud - np.mean(pointCloud, axis=0) )

def amendMatches(checkArr, oVal, nVal, filterCols=[0,None], filterRows=[0, None]):
    '''
        Usage:
        checkArr = 2D array to check for matches in.
        oVal = old value to find matching indexes
        nVal = new value to replace old value with

        Description:
        Finds matches of an old value (oVal) in a 2D array (checkArr).
        The array is then returned with first match replaced by the new value (nVal)

    '''
    checkArr[np.where((
                        checkArr[filterRows[0]:filterRows[1], filterCols[0]:filterCols[1]]
                        ==
                        oVal).all(axis=1))[0], filterCols[0]:filterCols[1]] \
            = nVal
    return checkArr

def arrayRowIntersection(nodes,edgeNodes1, edgeNodes2):
    '''
        USAGE:
        nodes = 2D array of coordinates (y,x = *, 3)
        edgeNodes1 = 2D array of coordinates from first half of edge array
        edgeNodes1 = 2D array of coordinates from second half of edge array


        DESCRIPTION:
        List comprehension to return node list exempt of nodes without edges,
        by checking for presence of each node as node 1 or node 2 in edge list.
    '''
    return np.delete(nodes,
                    np.where(
                            (np.isin(nodes, edgeNodes1, invert=True).all(axis=1) & 
                            (np.isin(nodes, edgeNodes2, invert=True).all(axis=1) ) )
                            ),
            axis=0)


def bilateralfilter(img, rad=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        rad: int, radius of filtering kernel
  '''
  return mean_bilateral(img, disk(rad))

def cleanSTLtags(workdir):
    #-Check correct formatting
    for file in workdir:
        print("checking", file)
        rewrite = False
        f = open(file, "r")
        fileLines = f.readlines()
        f.close()
        print("top lines are")
        print(fileLines[0:2], "\n\n\n")
        if "3D Slicer" in fileLines[0]:
            fileLines[0] = "solid\n"
            rewrite = True
            print(fileLines[0], "\n")
        if "endsolid" not in fileLines[-1]:
            fileLines.append("endsolid\n")
            rewrite = True
            print(fileLines[-1], "\n")
        
        if rewrite == True:
            fOut = open(file, "w")
            print("rewriting file", file)
            [fOut.write(outputLine) for outputLine in fileLines]
            fOut.close()
        elif rewrite == False:
            print("no rewriting needed")

        del fileLines

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def doPCA(c, expl_var=0.95, plotPCA=False):
    '''
      Args:
        c = 2D array of components

      Returns:
        pca, variation of each mode
        k, amount of modes to reach desired variance ratio
    ''' 
    # sc = StandardScaler().fit_transform(c)
    pca = PCA(svd_solver="full")#, n_components = 0.99)
    # pca = SparsePCA(alpha=0.5)
    pca.fit(c)
    varRat = pca.explained_variance_ratio_
    k = np.where(np.cumsum(varRat)>expl_var)[0][0]
    print("Reduced to {} components from {} for {}% variation".format(
                                                  k,len(c),expl_var*100)
          )

    return pca, k

def doSparsePCA(c, expl_var=0.95, plotPCA=False):
    '''
      Args:
        c = 2D array of components

      Returns:
        pca, variation of each mode
        k, amount of modes to reach desired variance ratio
    ''' 
    # sc = StandardScaler().fit_transform(c)
    pca = SparsePCA()#, n_components = 0.99)
    pca.fit(c)
    varRat = pca.explained_variance_ratio_
    k = np.where(np.cumsum(varRat)>expl_var)[0][0]
    print("Reduced to {} components from {} for {}% variation".format(
                                                  k,len(c),expl_var*100)
          )

    return pca, k

def euclideanDist(x, y):
    '''
        Finds the euclidean distance between two arrays x, y.
        Calculated using pythagoras theorem
    '''
    if x.size <= 3:
        return np.sqrt(np.sum((x-y) ** 2))
    else:
        return np.sqrt(np.sum((x-y) ** 2, axis = 1))

def findOccurencesOfPoint(pointList, point, point2="0"):
    '''
        Return a 1D array of all indexes of point matches in a list of points
    '''
    if point2=="0":
        if pointList.shape[1] == 7:
            indexArr = np.where((pointList[:,0]==point[0]) & (pointList[:,1]==point[1]) & (pointList[:,2]==point[2]) | \
                                (pointList[:,3]==point[0]) & (pointList[:,4]==point[1]) & (pointList[:,5]==point[2]) )[0]
        elif pointList.shape[1] == 4 or  pointList.shape[1] == 3:
            indexArr = np.where((pointList[:,0]==point[0]) & (pointList[:,1]==point[1]) & (pointList[:,2]==point[2]))[0]
        else:
            print("ERROR, non-determined shape found", pointList.shape)
            exit()
    else:
        if pointList.shape[1] == 7:
            indexArr = np.where((
                                (pointList[:,0]==point[0]) & (pointList[:,1]==point[1]) & (pointList[:,2]==point[2]) & \
                                (pointList[:,3]==point2[0]) & (pointList[:,4]==point2[1]) & (pointList[:,5]==point2[2])
                                ) |
                                (
                                (pointList[:,0]==point2[0]) & (pointList[:,1]==point2[1]) & (pointList[:,2]==point2[2]) & \
                                (pointList[:,3]==point[0]) & (pointList[:,4]==point[1]) & (pointList[:,5]==point[2])
                                )
                                )[0]
    return indexArr

def he(img):
    '''
        Histogram equalisation for contrast enhancement
        https://github.com/AndyHuang1995/Image-Contrast-Enhancement
    '''
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def loadXR(file):
  '''
    take input X-ray as png (or similar) and convert to grayscale np array

    can add any necessary pre-processing steps in here, 
    such as utils.he (histogram equalisation contrast enhancement) 
  '''
  g_im = rgb2gray(io.imread(file))
  g_im = he(g_im) / 255
  return g_im

def localNormalisation(img, rad=2):
  '''edge preserving filter
    args: img: np.ndarry, image to filtr
        rad: int, radius of filtering kernel
  '''
  g_i = filters.gaussian(img, rad)
  sigma = np.sqrt( filters.gaussian( (img - g_i)**2 ) )
  return (img - g_i) / sigma

def mahalanobisDist(x, y):
    '''
        Finds the mahalanobis distance between two arrays x, y
        Calculated based on the inverse covariance matrix of the two arrays
        and the difference of each array (delta)
    '''
    delta = x-y
    if len(np.where(delta == delta[0])[0]) == delta.size:
        return 0
    X = np.vstack([x,y])

    V = np.cov(X.T)

    if 0 in np.diagonal(V):
        print("SINGULAR MATRIX, EXITING")
        exit()
    VI = np.linalg.inv(V)

    if np.sum(np.dot(delta, VI) * delta) < 0:
        return 10000000

    return np.sqrt(np.sum(np.dot(delta, VI) * delta, axis = -1))    


def getNeighboringNodes(edgeList, node):
    '''
        Returns a list of nodes that share an edge with the given node
    '''
    neighbors = [] #initialise empty list
    nIndexArr = findOccurencesOfPoint(edgeList, node)
    potentialNeighbors = np.array(edgeList[nIndexArr][:,:-1])
    neighbors = potentialNeighbors[:,[0,1,2]] #-set temp array of correct length to overwrite in for loop
    edge1 = potentialNeighbors[:,[0,1,2]]
    edge2 = potentialNeighbors[:,[3,4,5]]
    deleteList1, deleteList2 = findOccurencesOfPoint(edge1, node), findOccurencesOfPoint(edge2, node)

    for i, neighbor in enumerate(potentialNeighbors):
        if i in deleteList1: #-if node is first coordinate in edge, set neighbor to second coord
            neighbors[i] = edge2[i]
        elif i in deleteList2: #-if node is second coordinate in edge, set neighbor to first coord
            neighbors[i] = edge1[i]

    return neighbors #-return list of all neighboring nodes

def readSurfaceFile(fileName, align=True, scaler=1000):
    from vtk.util.numpy_support import vtk_to_numpy
    #-Read stl files
    reader = vtk.vtkSTLReader() #initialise reader
    reader.SetFileName(fileName) 
    reader.Update() #initialise changes 
    #-get surface triangle cell data
    polydata = reader.GetOutput() 
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()
    pointCloud = vtk_to_numpy(points.GetData())*scaler
    if align:
        pointCloud = alignToMean(pointCloud)

    return pointCloud

def removeDuplicateNumPyRows(arr2D):
    #removes duplicate rows from input 2D array
    return np.unique([tuple(row) for row in arr2D])

def remove_values_from_list(the_list, val, returnList="def"):
    #-if no list fed in to return, initialise new list
    if returnList == "def":
        returnList = []
    #-for each node in a single edge
    for i in the_list[:2]:
        #-if i (edge node) and val (s) not equal, return i (edge node) 
        if not np.array_equal(i, val):
            returnList.append(i)
    return returnList

def return_t_counter(t, point):
    '''
        Get t (number of times value has been s or neighbor of s) for a given node.
    '''
    if point.size == 3:
        point = point.reshape(3,)
        n = t[np.where((t[:,0] == point[0])*\
                                        (t[:,1] == point[1])*\
                                        (t[:,2] == point[2])) ][0,3]
    else:
        n = np.array([t[np.where((t[:,0] == p[0])*\
                                    (t[:,1] == p[1])*\
                                    (t[:,2] == p[2])) ][0,3] for p in point])
    return n

def plotLoss(lossList, scale="linear", dir="./", stage=""):
    plt.close()
    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(0,len(lossList)), lossList, lw=1)
    ax.set_title(stage+" estimation loss function")
    ax.set_yscale("linear")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    plt.savefig("loss"+stage+".pdf")
    return None

def saveHistogram(distances, fileLabel=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1)
    axes[0].hist(distances, bins=100)
    axes[1].hist(distances, bins=100, range=[8, 15])
    # axes[0].set_ylim(0, len(distances))
    # plt.show()
    plt.xlabel('Distance (voxels)')
    axes[0].set_ylabel('Number of coords')
    axes[1].set_ylabel('Number of coords')

    axes[0].set_title('Number of coords (ymax = largest coord)')
    axes[1].set_title('Large distanced histogram')
    plt.subplots_adjust(hspace=0.6)
    plt.savefig("histogram"+str(fileLabel)+".png")
    return None

def trainTestSplit(inputData, train_size=0.9):
    '''
      Args:
         inputData (2D array): (3*num landmarks, num samples)
         train_size (float): 0 < train_size < 1

      Splits data into training and testing randomly based on a set 
      test size
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(inputData, 
                                    train_size=train_size)
    print("set sizes (train | test)")
    print("\t",train.shape, "|",test.shape)
    return train, test


def writeAdjustedSTL(stlIn, newPoints, outname):
    #-Read the mean shape
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stlIn)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    n_points = points.GetNumberOfPoints()

    surfP = []
    #-Adjust points based on mode
    for j in range(0, n_points):
        p = np.array(points.GetPoint(j))
        #-find matching node index (stl and node have diff index)
        # loc = np.argmin( euclideanDist(stlIn, p) )
        points.SetPoint(j, newPoints[j])

    polydata.GetPointData()

    #-Write the shape
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(outname)
    writer.SetInputData(polydata)
    writer.Write()
    print("Finished writing")
    return None


def vtkBatchShowFiles(inputFiles, textPrint=""):
    from vtkplotter import Plotter, Points, Text2D, load
    vp = Plotter(N=len(inputFiles))
    v = [None] * len(inputFiles)
    for i, readFile in enumerate(inputFiles):
        if textPrint == "":
            showTxt=readFile
        else:
            showTxt=textPrint
        v[i] = load(readFile) #Points(readSurfaceFile(readFile)[::10], c='b', r=3)
        if i == len(inputFiles)-1:
            print("TRUE")
            vp.show(v[i], Text2D(showTxt), at=i, interactive = 1)
        else:
            vp.show(v[i], Text2D(showTxt), at=i)
    return None

def vtkplotNumPyPoints(inP, secondP="def"):
    '''
        Similar to a paraview rendering. 
        For GAMEs landmarking, args are:
            inP (2D np arr): landmarks
            secondP (2D np arr): base coordinate set for comparison
    ''' 
    from vtkplotter import Plotter, Points
    vp = Plotter(axes=3)
    # vp = Plotter(N=1, axes=0)
    vp += Points(inP, c='b', r=10) 

    # show(plotPoints, at=0, zoom=1.2, interactive=1)
    if secondP == "def":
        vp.show(interactive=1)
    else:
        vp += Points(secondP, c='r', r=2)
        vp.show(interactive=1)

    # plotPoints = vp.Points(inputPoints)#, r=3).legend("plotting points")
    # text1 = vp.Text2D('Landmarks of right upper lobe', c='blue')
    # # show(plotPoints, at=0, zoom=1.2, interactive=1)
    # if initialPoints == "def":
    #   vp.show([(plotPoints, text1)], N=1)
    # else:
    #   plotStl = vp.Points(initialPoints)
    #   text2 = vp.Text2D('Original right upper lobe', c='blue')
    #   vp.show([(plotPoints, text1), (plotStl,text2)], N=2)

    return None

def heightToWidthCalc():
    import matplotlib.pyplot as plt 

    lNums = {"RUL": "4",
             "RML": "5",
             "RLL": "6",
             "LUL": "7",
             "LLL": "8"
             }

    basedir = "/home/josh/3DSlicer/luna16Upload/"
    lobes = ["RUL", "RML", "RLL"]
    lobeDirs = [glob(basedir+"*/*_"+lNums[i]+"*.stl") for i in lobes]

    width = np.zeros(len(lobeDirs[0]))
    height = np.zeros(len(lobeDirs[0]))
    depth = np.zeros(len(lobeDirs[0]))

    for i, pDir in enumerate(lobeDirs[0]):
        patient = pDir.split("/")[-2]
        print(patient)
        rul = userUtils.readSurfaceFile(pDir, False)
        rml = userUtils.readSurfaceFile(lobeDirs[1][i], False)
        rll = userUtils.readSurfaceFile(lobeDirs[2][i], False)
        rlung = np.vstack((rul, rml, rll))
        width[i] = rlung[:,0].max() - rlung[:,0].min()
        depth[i] = rlung[:,1].max() - rlung[:,1].min()
        height[i] = rlung[:,2].max() - rlung[:,2].min()
        print("H/W =",height[i]/width[i])
        # break

    plt.close()
    plt.boxplot(height/width)
    plt.show()
    return height/width