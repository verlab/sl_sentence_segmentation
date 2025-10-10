import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout=='openpose (137)':
            # open pose with 137 pose keypoints (body + face + hands)
            self.num_node = 137
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,15], [15,17], [0,16], [16,18],
                [1,8], [9,8], [9,10], [10,11], [11,24],
                [11,22], [22,23], [8,12], [12,13], [13,14],
                [14,21], [14,19], [19,20],
                # face
                # contour
                [25,26], [26,27], [27,28], [28,29], [29,30],
                [30,31], [31,32], [32,33], [33,34], [34,35], [35,36],
                [36,37], [37,38], [38,39], [39,40], [40,41],
                # eyebrows
                [42,43], [43,44], [44,45], [45,46],
                [47,48], [48,49], [49,50], [50,51],
                # nose
                [52,53], [53,54], [54,55],
                [56,57], [57,58], [58,59], [59,60],
                # eyes
                [61,62], [62,63], [63,64], [64,65], [65,66], [66,61],
                [67,68], [68,69], [69,70], [70,71], [71,72], [72,67],
                # mouth
                [73,74], [74,75], [75,76], [76,77], [77,78], [78,79],
                [79,80], [80,81], [81,82], [82,83], [83,84], [84,73],
                [85,86], [86,87], [87,88], [88,89], [89,90],
                [90,91], [91,92], [92,85],
                # left hand
                [95,96], [96,97], [97,98], [98,99],
                [95,100], [100,101], [101,102], [102,103],
                [95,104], [104,105], [105,106], [106,107],
                [95,108], [108,109], [109,110], [110,111],
                [95, 112], [112,113], [113,114], [114,115],
                # right hand
                [116,117], [117,118], [118,119], [119,120],
                [116,121], [121,122], [122,123], [123,124],
                [116,125], [125,126], [126,127], [127,128],
                [116,129], [129,130], [130,131], [131,132],
                [116,133], [133,134], [134,135], [135,136]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        
        elif layout=='openpose (25, body only)':
            # open pose body with 25 keypoints (body)
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,15], [15,17], [0,16], [16,18],
                [1,8], [9,8], [9,10], [10,11], [11,24],
                [11,22], [22,23], [8,12], [12,13], [13,14],
                [14,21], [14,19], [19,20],
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=='openpose (127, no legs and feet)':
            # open pose with 127 keypoints (body without legs and feet + face + hands)
            self.num_node = 127
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body (indexes altered to match the removal of legs and feet keypoints)
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,11], [11,13], [0,12], [12,14],
                [1,8], [9,8], [8,10],
                # face
                #contour
                [15,16], [16,17], [17,18], [18,19], [19,20],
                [20,21], [21,22], [22,23], [23,24], [24,25], [25,26],
                [26,27], [27,28], [28,29], [29,30], [30,31],
                # eyebrows
                [32,33], [33,34], [34,35], [35,36],
                [37,38], [38,39], [39,40], [40,41],
                # nose
                [42,43], [43,44], [44,45],
                [46,47], [47,48], [48,49], [49,50],
                # eyes
                [51,52], [52,53], [53,54], [54,55], [55,56], [56,51],
                [57,58], [58,59], [59,60], [60,61], [61,62], [62,57],
                # mouth
                [63,64], [64,65], [65,66], [66,67], [67,68], [68,69],
                [69,70], [70,71], [71,72], [72,73], [73,74], [74,63],
                [75,76], [76,77], [77,78], [78,79], [79,80],
                [80,81], [81,82], [82,75],
                # left hand
                [85,86], [86,87], [87,88], [88,89],
                [85,90], [90,91], [91,92], [92,93],
                [85,94], [94,95], [95,96], [96,97],
                [85,98], [98,99], [99,100], [100,101],
                [85, 102], [102,103], [103,104], [104,105],
                # right hand
                [106,107], [107,108], [108,109], [109,110],
                [106,111], [111,112], [112,113], [113,114],
                [106,115], [115,116], [116,117], [117,118],
                [106,119], [119,120], [120,121], [121,122],
                [106,123], [123,124], [124,125], [125,126]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=='openpose (15, body only, no legs and feet)':
            # open pose body with 15 keypoints (body without legs and feet)
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body (indexes altered to match the removal of legs and feet keypoints)
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,11], [11,13], [0,12], [12,14],
                [1,8], [9,8], [8,10]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=='openpose (85, body and face, no legs and feet)':
            # open pose with 85 keypoints (body without legs and feet + face)
            self.num_node = 85
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body (indexes altered to match the removal of legs and feet keypoints)
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,11], [11,13], [0,12], [12,14],
                [1,8], [9,8], [8,10],
                # face
                [15,16], [16,17], [17,18], [18,19], [19,20],
                [20,21], [21,22], [23,24], [24,25], [25,26],
                [26,27], [27,28], [28,29], [29,30], [30,31],
                [32,33], [33,34], [34,35], [35,36],
                [37,38], [38,39], [39,40], [40,41],
                [42,43], [43,44], [44,45],
                [46,47], [47,48], [48,49], [49,50],
                [51,52], [52,53], [53,54], [54,55], [55,56], [56,51],
                [57,58], [58,59], [59,60], [60,61], [61,62], [62,57],
                [63,64], [64,65], [65,66], [66,67], [67,68], [68,69],
                [69,70], [70,71], [71,72], [72,73], [73,74], [74,63],
                [75,76], [76,77], [77,78], [78,79], [79,80],
                [80,81], [81,82], [82,75]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=='openpose (57, body and hands, no legs and feet)':
            # open pose with 57 keypoints (body without legs and feet + hands)
            self.num_node = 57
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # body (indexes altered to match the removal of legs and feet keypoints)
                [4,3], [3,2], [2,1], [1,5], [7,6], [6,5],
                [0,1], [0,11], [11,13], [0,12], [12,14],
                [1,8], [9,8], [8,10],
                # left hand
                [15,16], [16,17], [17,18], [18,19],
                [15,20], [20,21], [21,22], [22,23],
                [15,24], [24,25], [25,26], [26,27],
                [15,28], [28,29], [29,30], [30,31],
                [15,32], [32,33], [33,34], [34,35],
                # right hand
                [36,37], [37,38], [38,39], [39,40],
                [36,41], [41,42], [42,43], [43,44],
                [36,45], [45,46], [46,47], [47,48],
                [36,49], [49,50], [50,51], [51,52],
                [36,53], [53,54], [54,55], [55,56]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout=='openpose (112, face and hands)':
            # open pose with 112 pose keypoints (face + hands)
            self.num_node = 112
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # face
                #contour
                [0,1], [1,2], [2,3], [3,4], [4,5],
                [5,6], [6,7], [7,8], [8,9], [9,10], [10,11],
                [11,12], [12,13], [13,14], [14,15], [15,16],
                # eyebrows
                [17,18], [18,19], [19,20], [20,21],
                [22,23], [23,24], [24,25], [25,26],
                # nose
                [27,28], [28,29], [29,30],
                [31,32], [32,33], [33,34], [34,35],
                # eyes
                [36,37], [37,38], [38,39], [39,40], [40,41], [41,36],
                [42,43], [43,44], [44,45], [45,46], [46,47], [47,42],
                # mouth
                [48,49], [49,50], [50,51], [51,52], [52,53], [53,54],
                [54,55], [55,56], [56,57], [57,58], [58,59], [59,48],
                [60,61], [61,62], [62,63], [63,64], [64,65],
                [65,66], [66,67], [67,60],
                # left hand
                [70,71], [71,72], [72,73], [73,74],
                [70,75], [75,76], [76,77], [77,78],
                [70,79], [79,80], [80,81], [81,82],
                [70,83], [83,84], [84,85], [85,86],
                [70, 87], [87,88], [88,89], [89,90],
                # right hand
                [91,92], [92,93], [93,94], [94,95],
                [91,96], [96,97], [97,98], [98,99],
                [91,100], [100,101], [101,102], [102,103],
                [91,104], [104,105], [105,106], [106,107],
                [91,108], [108,109], [109,110], [110,111]
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
