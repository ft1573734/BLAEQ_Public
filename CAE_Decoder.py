import Zone
import numpy as np
from tqdm import tqdm
from collections import defaultdict
#from scipy.sparse import coo_matrix

class Parameters:
    def __init__(self,X,Y,Z,U,V,W,P,K,E,Nodes,Faces,Elements,Adjacency):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.U = U
        self.V = V
        self.W = W
        self.P = P
        self.K = K
        self.E = E
        self.Elements = Elements
        self.Nodes = Nodes
        self.Faces = Faces
        self.Adjacency = Adjacency
    def __init__(self,X,Y,Z):
        self.X = X
        self.Y = Y
        self.Z = Z

class CAE_Decoder:
    #Func: Decode_dat_file
    #Inputs:
    #   path: the path of the dat file.
    #   return: the decoded CAE data structure.
    #   description: Decode a given CAE data. The file structure of the .dat file is default.
    def Decode_dat_file(path):
        file_object = open(path, 'r', encoding="UTF-8")
        raw_content = file_object.read()
        file_object.close()
        
        paragraphs = raw_content.split("ZONE ")
        header = paragraphs[0]
        raw_zone_fluid = paragraphs[1]
        zone_fluid = Zone.Zone_3D(raw_zone_fluid, 10)
        print("Loading Zone "+zone_fluid.ZoneType+" Complete!")

        #Now that the fluid zone is loaded, we need to generate the mesh
        geometric_center_of_faces_list = []
        print("Loading faces:")
        for nodes in tqdm(zone_fluid.FN):
            centroid = np.zeros(3)
            #sum x,y,z 
            for i in nodes:
                centroid[0] += zone_fluid.X[i-1]
                centroid[1] += zone_fluid.Y[i-1]
                centroid[2] += zone_fluid.Z[i-1]
            #compute average
            for i in [0,1,2]:
                centroid[i] = centroid[i]/len(nodes)
            #print(centroid)
            geometric_center_of_faces_list.append(centroid)
        face_centers = np.asarray(geometric_center_of_faces_list)
        print("Loading mesh cells:")
        #The first step is to find the faces of each cell using LE (left element) and RE (right element)
        #WARNING: For unknown reasons some of the LE/RE values starts with 1 instead of 0. Therefore, we need to check LE/RE values and adjust to 0-initiated arrays.
        min_LE = np.min(zone_fluid.LE)
        max_LE = np.max(zone_fluid.LE)
        min_RE = np.min(zone_fluid.RE)
        max_RE = np.max(zone_fluid.RE)
        if (max_RE-min_RE) != (max_LE - min_LE):
            print("ERROR, Left Element & Right Element do not match.")
            exit()
        
        unified_LE = np.subtract(zone_fluid.LE, min_LE)
        unified_RE = np.subtract(zone_fluid.RE, min_RE)

        element_faces = defaultdict(set)
        for i in tqdm(range(0, len(unified_LE))):
            e = unified_LE[i]
            if e in element_faces:
                #since e is a left element of face i, i naturally becomes a face of e
                element_faces[e].add(i)
            else:
                element_faces[e] = {i}
                
        #The same process repeats for the right elements
        for i in tqdm(range(0, len(unified_RE))):
            e = unified_RE[i]
            if e in element_faces:
                #since e is a right element of face i, i naturally becomes a face of e
                element_faces[e].add(i)
            else:
                element_faces[e] = {i}
        
                
        # When processing LE and RE, the adjacency relationship is also formed.
        # Manually constructing a adjaceny matrix with shape zone_fluid.Elements * zone_fluid.Elements, and zone_fluid.LE or *.RE NNZs.
        rows = np.zeros(len(unified_LE)) #len(unitifed_LE) & len(unified_RE) are interchangable.
        cols = np.zeros(len(unified_LE))
        data = np.zeros(len(unified_LE))
        for i in tqdm(range(0, len(unified_LE))):
            rows[i] = unified_LE[i]
            cols[i] = unified_RE[i]
            data[i] = 1
            #for symmetry
            rows[i] = unified_RE[i]
            cols[i] = unified_LE[i]
            data[i] = 1

        #adj_matrix = coo_matrix((data, (rows, cols)), shape=(zone_fluid.Elements, zone_fluid.Elements))
        #Adjacency = []
        

        #for i in tqdm(range(0,adj_matrix.shape[0])):
            #records the neighbours of this row
            #tmp_adj = []
            #tmp_row = adj_matrix.getrow(i) #The coo_matrix.getrow() func returns a row in CSR format, therefore-
            #for j in range(0, len(tmp_row.indices)): #-we only need to fetch the csr_matrix.indices for the position of each nnz element
                #tmp_adj.append(j)
            #Adjacency.append(np.asarray(tmp_adj))
            
        
        #Now that the faces of each element is set, we can compute the geometric centers of each cell
        print("Loading X, Y, Z:")
        Element_X = np.zeros(zone_fluid.Elements)
        Element_Y = np.zeros(zone_fluid.Elements)
        Element_Z = np.zeros(zone_fluid.Elements)
        for i in tqdm(range(0,zone_fluid.Elements)):
            tmpFaces = element_faces.get(i)
            centroid = np.zeros(3)
            for f in tmpFaces:
                centroid[0] += face_centers[f][0]
                centroid[1] += face_centers[f][1]
                centroid[2] += face_centers[f][2]
            for j in [0,1,2]:
                centroid[j] = centroid[j]/len(tmpFaces)
            Element_X[i] = centroid[0]
            Element_Y[i] = centroid[1]
            Element_Z[i] = centroid[2]
        
        result_parameters = Parameters(Element_X,Element_Y,Element_Z,zone_fluid.U,zone_fluid.V,zone_fluid.W,zone_fluid.P,zone_fluid.K,zone_fluid.E,zone_fluid.Nodes,zone_fluid.Faces,zone_fluid.Elements,adj_matrix)

        return result_parameters

    def Decode_dat_file_simple(path):
        file_object = open(path, 'r', encoding="utf-8")
        raw_content = file_object.read()
        file_object.close()


        paragraphs = raw_content.split("ZONE ")
        header = paragraphs[0]
        raw_zone_fluid = paragraphs[1]
        zone_fluid = Zone.Zone_3D(raw_zone_fluid, 10)
        print("Loading Zone "+zone_fluid.ZoneType+" Complete!")
        result_parameters = Parameters(zone_fluid.X, zone_fluid.Y, zone_fluid.Z)
        return result_parameters





        
