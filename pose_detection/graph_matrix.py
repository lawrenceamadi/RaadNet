'''
    Find maximum weighted cycle in graph
    modified algorithm from:
    1. https://www.geeksforgeeks.org/find-minimum-weight-cycle-undirected-graph/
    2. http://www.gitta.info/Accessibiliti/en/html/Dijkstra_learningObject1.html

    Algorithm
    1) Modify edge weights by subtracting from max, ie graphEdges = np.max(graphEdges) - graphEdges
        And reset all 0s in previous graphEdges to 0s in new graphEdges
        This converts maximum weighted cycle graph problem to minimum weighted cycle graph problem,
        while preserving the non-negative property of the graph to necessary to run Dijkstra SP Algorithm
    2) Traverse every node in the first frame-group one-by-one
        I. choose node (A) and loop through all edges from some node (Z) in last frame-group to chosen node (A)
            a. For each discovered 'cycle' connecting edge (Z->A),
                temporarily disable edge in the graph (set to 0 or [MAX + 1])
            b. Find the shortest path from A->Z using Dijkstra’s shortest path algorithm
            c. Add edge weight Z->A to found shortest path and re-enable the edge (set back to original weight
            d. Update min_weight_cycle if needed (record path configuration and weight if minimum)
    3. Reset graphEdge weight to original. ie. graphEdges = MAX - graphEdges
    4. Return maximum weighted cycle path and corresponding weight
'''

import numpy as np
import cv2 as cv
import sys

def max_weight_cycle(graphEdges, activeEdges, candsPerGrp):
    '''

    :param graphEdges: edges in graph matrix: matrix of shape=(nodesPerKpt, nodesPerKpt), dtype=np.float32
    :return:
    '''
    # 1) Modify edge weights
    assert (graphEdges.shape == (96, 96))
    assert (np.min(graphEdges) >= 0 and np.max(graphEdges) < np.inf)
    maxEdgeWeight = np.max(graphEdges)
    newGraphEdges = maxEdgeWeight - graphEdges
    newGraphEdges = np.where(activeEdges, newGraphEdges, np.inf)
    assert (np.sum(np.abs(newGraphEdges - graphEdges)) != 0)
    pathConfig, cycleWeight = find_min_weighted_cycle(newGraphEdges, candsPerGrp)
    assert (len(pathConfig) > 0)
    # 3) and 4) implicit implementation
    return pathConfig, path_weighted_distance(graphEdges, pathConfig)


def min_weight_cycle(graphEdges, activeEdges, candsPerGrp):
    '''

    :param graphEdges: edges in graph matrix: matrix of shape=(nodesPerKpt, nodesPerKpt), dtype=np.float32
    :return:
    '''
    assert (graphEdges.shape == (96, 96))
    assert (np.min(graphEdges) >= 0 and np.max(graphEdges) < np.inf)
    newGraphEdges = np.where(activeEdges, graphEdges, np.inf)
    pathConfig, cycleWeight = find_min_weighted_cycle(newGraphEdges, candsPerGrp)
    assert (len(pathConfig) > 0)
    # 3) and 4) implicit implementation
    return pathConfig, path_weighted_distance(newGraphEdges, pathConfig)


def find_min_weighted_cycle(directedGraphEdges, candsPerGrp):
    '''

    :param directedGraphEdges: edges in graph matrix: matrix of shape=(nodesPerKpt, nodesPerKpt), dtype=np.float32
    :return:
    '''
    assert (np.all(directedGraphEdges >= 0))
    assert (directedGraphEdges.ndim == 2 and directedGraphEdges.shape[0] == directedGraphEdges.shape[1])

    numOfNodes = directedGraphEdges.shape[0]
    numOfGroups = int(numOfNodes / candsPerGrp)
    assert (numOfGroups == 16)
    sinOfLastGroup = candsPerGrp * (numOfGroups - 1)
    einOfLastGroup = candsPerGrp * numOfGroups
    assert (einOfLastGroup == numOfNodes and (einOfLastGroup - sinOfLastGroup) == candsPerGrp)
    minCycleWeight = np.inf
    #print('initial graph min cycle weight: ', minCycleWeight)
    minCycleConfig = []
    # 2) Traverse every node in the first frame-group
    # We assume there is a connected path from each node in the first group to some node in the last group
    # Note there may not be a connected path from each node in first group to all nodes in the last group
    for nidA in range(candsPerGrp):
        pathFromBtoA = False
        # 2) I. choose node (A) and loop through all edges from some node (Z)
        for nidZ in range(sinOfLastGroup, einOfLastGroup):
            #print(nidZ)
            cycleEdgeWeight = directedGraphEdges[nidZ, nidA]
            if cycleEdgeWeight != 0:
                pathFromBtoA = True
                # 2) I. a. Temporarily deactivate discovered 'cycle' connecting edge (Z->A)
                #directedGraphEdges[nidZ, nidA] = 0
                # 2) I. b. Find the shortest path from A->Z using Dijkstra’s shortest path algorithm
                pathConfig, pathWeight = dijkstraSP(nidA, nidZ, directedGraphEdges, numOfNodes)
                #print('\tfound opt path candidate (A, B, length, path)', nidA, nidZ, len(pathConfig), pathConfig)
                # 2) I. c. Add edge (Z->A) weight to SP distance and Reactivate edge, set back to original weight
                cycleWeight = cycleEdgeWeight + pathWeight
                #print('dj cycle weight: ', cycleWeight)
                #directedGraphEdges[nidZ, nidA] = cycleEdgeWeight
                if cycleWeight < minCycleWeight:
                    # 2) I. d. Update min_weight_cycle and min_cycle_path
                    assert (len(pathConfig) == numOfGroups)  # because path should consist of one node in each group
                    minCycleWeight = cycleEdgeWeight
                    minCycleConfig = pathConfig
                    #print('new min cycle (weight and path): ', minCycleWeight, minCycleConfig)

        assert(pathFromBtoA)

    return minCycleConfig, minCycleWeight


def dijkstraSP(soureNodeID, sinkNodeID, directedGraphEdges, numOfNodes):
    assert (soureNodeID >= 0 and sinkNodeID >= 0)
    distanceToNode = np.full(shape=(numOfNodes), fill_value=np.inf, dtype=np.float32)
    prevLinkNode = np.full(shape=(numOfNodes), fill_value=-1, dtype=np.int32)
    pendingNodes = list(range(numOfNodes))
    assert (pendingNodes[0] == 0 and pendingNodes[len(pendingNodes) - 1] == numOfNodes - 1)
    #print('pending nodes: ', pendingNodes)
    #print('src -> sink: ', soureNodeID, sinkNodeID)
    distanceToNode[soureNodeID] = 0
    prevLinkNode[soureNodeID] = soureNodeID

    while len(pendingNodes) > 0:
        nidU = smallest_distance_node(pendingNodes, distanceToNode)

        if nidU == -1:
            return [], np.inf

        pendingNodes.remove(nidU)
        if nidU == sinkNodeID:
            assert(not(sinkNodeID in pendingNodes))
            return trace_path(soureNodeID, sinkNodeID, prevLinkNode), distanceToNode[soureNodeID]

        # Update nearest distance from source node to neighborhood nodes V of node U
        for nidV in range(numOfNodes):
            edgeWeightFromUtoV = directedGraphEdges[nidU, nidV] # directed edge: nidU -> nidV
            if edgeWeightFromUtoV != np.inf:# and nidV in pendingNodes:
                altDistance = distanceToNode[nidU] + edgeWeightFromUtoV
                if altDistance < distanceToNode[nidV]:
                    distanceToNode[nidV] = altDistance
                    prevLinkNode[nidV] = nidU

    return trace_path(soureNodeID, sinkNodeID, prevLinkNode), distanceToNode[soureNodeID]


def smallest_distance_node(pendingNodeList, distanceToNode):
    '''
    Find the next connected, pending node with the smallest distance from source node
        The node must be connected to the source node (passed to dijkstra's algorithm
        Hence, there must be a path from the chosen node to the source node
        Therefore, the distance from the source node to the chosen node CANNOT be infinity
    :param pendingNodeList:
    :param distanceToNode:
    :return: index of the nearest (distance from source node), connected, pending node
                OR -1 if at the end of connected path
    '''
    minNode = -1
    minDist = np.inf
    for node in pendingNodeList:
        if distanceToNode[node] < minDist:
            minNode = node
            minDist = distanceToNode[node]
    #print('smallest (node, dist, grp): ', minNode, minDist, int(minNode / 6))
    return minNode


def trace_path(srcNode, sinkNode, previousNodeLink):
    # path in reverse order
    path = []
    nodeV = sinkNode
    while nodeV != srcNode:
        path.append(nodeV)
        #print(path)
        nodeV = previousNodeLink[nodeV]
        assert (nodeV != -1)
        if len(path) > len(previousNodeLink)**2:
            #print('**Warning, may result to infinite loop, something went wrong')
            sys.exit()
    path.append(nodeV)
    return path


def path_weighted_distance(directedGraphEdges, pathConfig):
    # traverse and compute
    distance = 0
    nodeV = pathConfig[0]
    for nodeU in range(1, len(pathConfig)):
        distance += directedGraphEdges[nodeU, nodeV]
        nodeV = nodeU
    return distance


def visualize_graph(graphNodeTags, graphEdgeWgts, activeEdges, candsPerGrp=6, kptID=None, kptLB=None):
    # 0:white, 1:green, 2:blue, 3:yellow, 4:purple, 5:red
    nodeColor = {0: (255, 255, 255), 1: (  0, 100,   0), 2: (255,   0,   0),
                 3: (  0, 255, 255), 4: (255,   0, 255), 5: (  0,   0, 200)}
    graphGUI = np.full(shape=(350, 1000, 3), fill_value=50, dtype=np.uint8)
    xpad, ypad = 120, 80

    # Label image
    cv.putText(graphGUI, 'Frames: ', (20, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    if isinstance(kptID, int):
        kptLabel = 'Kpt: {}'.format(kptID + 1)
        cv.putText(graphGUI, kptLabel, (20, 170), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    if kptLB:
        cv.putText(graphGUI, kptLB, (20, 190), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # Highlight invalid keypoints
    for sin in range(0, graphNodeTags.shape[0], candsPerGrp):
        # if the node-of-origin used for reconstruction invalid, its tag will be -3
        if np.any(graphNodeTags[sin + 1 : sin + candsPerGrp] == -3):
            # group belongs to an invalid keypoint
            gid = int(sin / candsPerGrp)
            x = xpad + (gid * 50)
            cv.rectangle(graphGUI, (x - 15, 65), (x + 15, 295), (0, 0, 255), 1)
            if gid == 0:
                x = 925
                cv.rectangle(graphGUI, (x - 15, 65), (x + 15, 295), (0, 0, 255), 1)

    for nodeU in range(graphEdgeWgts.shape[0]):
        # Draw Node
        nodeUtag = graphNodeTags[nodeU]
        groupX, groupY, u_x, u_y = get_node_position(nodeU, xpad, ypad)
        cv.circle(graphGUI, (u_x, u_y), 7, nodeColor[abs(nodeUtag)], -1)
        cv.putText(graphGUI, str(groupX), (u_x - 4, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        if groupX == 0:
            e_x = 925
            cv.circle(graphGUI, (u_x, u_y), 9, (0, 255, 0), 1)
            cv.circle(graphGUI, (e_x, u_y), 9, (0, 0, 255), 1)
            cv.circle(graphGUI, (e_x, u_y), 7, nodeColor[abs(nodeUtag)], -1)
            cv.putText(graphGUI, str(groupX), (e_x - 4, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        for nodeV in range(graphEdgeWgts.shape[1]):
            # Draw Edge
            if activeEdges[nodeU][nodeV]:
                nodeVtag = graphNodeTags[nodeV]
                groupX, groupY, v_x, v_y = get_node_position(nodeV, xpad, ypad, checkEndOfCycle=True)
                b, g, r = get_edge_color(nodeUtag, nodeVtag, nodeColor)
                cv.line(graphGUI, (u_x, u_y), (v_x, v_y), (b, g, r), 1)

    return graphGUI


def visualize_opt_path(graphGUIimg, revOptPath, graphNodeTags):
    # 0:white, 1:green, 2:blue, 3:yellow, 4:purple, 5:red
    nodeColor = {0: (255, 255, 255), 1: (0, 100, 0), 2: (255, 0, 0),
                 3: (0, 255, 255), 4: (255, 0, 255), 5: (0, 0, 200)}
    xpad, ypad = 120, 80
    for i in range(len(revOptPath) - 1):
        nodeV = revOptPath[i]
        nodeU = revOptPath[i + 1]
        b, g, r = get_edge_color(graphNodeTags[nodeU], graphNodeTags[nodeV], nodeColor)
        v_gX, v_gY, v_x, v_y = get_node_position(nodeV, xpad, ypad)
        u_gX, u_gY, u_x, u_y = get_node_position(nodeU, xpad, ypad)
        cv.line(graphGUIimg, (u_x, u_y), (v_x, v_y), (0, 0, 0), 9)
        cv.line(graphGUIimg, (u_x, u_y), (v_x, v_y), (b, g, r), 2)

    # add cycle edge
    nodeU = revOptPath[0]
    nodeV = revOptPath[len(revOptPath) - 1]
    u_gX, u_gY, u_x, u_y = get_node_position(nodeU, xpad, ypad)
    v_gX, v_gY, v_x, v_y = get_node_position(nodeV, xpad, ypad, checkEndOfCycle=True)
    b, g, r = get_edge_color(graphNodeTags[nodeU], graphNodeTags[nodeV], nodeColor)
    cv.line(graphGUIimg, (u_x, u_y), (v_x, v_y), (0, 0, 0), 9)
    cv.line(graphGUIimg, (u_x, u_y), (v_x, v_y), (b, g, r), 2)

    return graphGUIimg


def get_edge_color(nodeUtag, nodeVtag, nodeColor):
    # red: edge between two adjacent nodes in chain reconstructed an from invalid keypoint
    # yellow: edge between two adjacent nodes in chain reconstructed from a valid keypoint
    # white: edge to original hpe node (node may or may not be an invalid keypoint)
    # blue: edge to 2nd node in chain (node may or may not be an invalid keypoint)
    # green: edge to 1st node in chain (node may or may not be an invalid keypoint)
    if nodeUtag < 0 and nodeVtag < 0 and abs(nodeUtag - nodeVtag) == 1:
        # edge in a chain reconstructed from an invalid keypoint
        return 0, 0, 255 # red

    tagDiff = nodeUtag - nodeVtag
    if tagDiff == -1 and nodeUtag > 0 and nodeVtag > 0:
        return 0, 255, 255 # yellow

    return nodeColor[abs(nodeVtag)]


def get_node_position(node, xpad, ypad, checkEndOfCycle=False):
    groupX = int(node / 6)
    groupY = node % 6
    x = xpad + (groupX * 50) if not(groupX == 0 and checkEndOfCycle) else 925
    y = ypad + (groupY * 40)
    return groupX, groupY, x, y



