import re
from xml.sax.handler import DTDHandler
from tqdm import tqdm
import numpy as np

"""
# Class Zone_3D:
# A data structure used for storing a zone for a 3D model
# The main components are as follows
# Elements: #Elements (#: Number of)
# Faces: #Faces
# Nodes: #Nodes
# ZoneType: Self-explainatory
# Parameters:
#   X, Y, Z: float[Nodes] arrays
#   U, V, W, P, K, E: float[Elements] arrays
"""

class Zone_3D:
    def generateMesh(self):
        
        return 

    def __init__(self, raw_content, var_count):
        #lines = raw_content.split("\n")
        #sections = lines[0].split("=")
        #if(sections[0].equals(" T")):
        #    self.ZoneType = sections[1]
        #else:
        #    print("ERROR Loading ZoneType")
        #tmp = raw_content.split("DT")
        tmp = raw_content.split("DT=(DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE DOUBLE )")
        header = tmp[0]
        vals = tmp[1]
        header_components = re.split('\n| ', header.replace("="," ").replace("  "," ").replace(","," "))
        for i in range(0,len(header_components)):
            match header_components[i]:
                case "T":
                    self.ZoneType = header_components[i+1]
                case "Nodes":
                    self.Nodes = int(header_components[i+1])
                case "Faces":
                    self.Faces = int(header_components[i+1])
                case "Elements":
                    self.Elements = int(header_components[i+1])
                case _:
                    continue
        
        vals_components = vals.split("#")
          
        DT = vals_components[0]

        node_count_per_face = vals_components[1]
        
        face_nodes = vals_components[2]
        
        left_elements = vals_components[3]
        
        right_elements = vals_components[4]
        
        #Decoding DT
        #DT_array = DT.replace("\n","").replace("   ","  ").split("  ")
        DT_array = DT.replace("\n","").replace("   ","  ").replace("  "," ").split(" ")
        
        n = self.Nodes
        e = self.Elements
        
        X = []
        Y = []
        Z = []
        U = []
        V = []
        W = []
        P = []
        K = []
        E = []

        DT_array = DT_array[1:]

        DT_subarray = []
        print("Processing raw data:")
        for i in tqdm(range(0, len(DT_array)+1)):
            if i in range(0, n):
                X.append(float(DT_array[i]))
            elif i in range(n, 2*n):
                Y.append(float(DT_array[i]))
            elif i in range(2*n, 3*n):
                Z.append(float(DT_array[i]))
            elif i in range(3*n, 3*n+e):
                U.append(float(DT_array[i]))
            elif i in range(3*n+e, 3*n+2*e):
                V.append(float(DT_array[i]))
            elif i in range(3*n+2*e, 3*n+3*e):
                W.append(float(DT_array[i]))
            elif i in range(3*n+3*e, 3*n+4*e):
                P.append(float(DT_array[i]))
            elif i in range(3*n+4*e, 3*n+5*e):
                K.append(float(DT_array[i]))
            elif i in range(3*n+5*e, 3*n+6*e):
                E.append(float(DT_array[i]))
        
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.Z = np.asarray(Z)
        self.U = np.asarray(U)
        self.V = np.asarray(V)
        self.W = np.asarray(W)
        self.P = np.asarray(P)
        self.K = np.asarray(K)
        self.E = np.asarray(E)
        '''
        Decoding node_count_per_face:
        '''
        #print("Decoding NCPF:")
        #self.NCPF = self.decoding_2d_int_arrays_to_array(node_count_per_face)

        '''
        Decoding face_nodes:
        '''
        #print("Decoding FN:")
        #self.FN = self.decoding_2d_int_arrays_to_list(face_nodes)
        
        '''
        Decoding left_elements:
        '''
        #print("Decoding LE:")
        #self.LE = self.decoding_2d_int_arrays_to_array(left_elements)
        
        '''
        Decoding right_elements:
        '''
        #print("Decoding RE:")
        #self.RE = self.decoding_2d_int_arrays_to_array(right_elements)

    def decoding_2d_int_arrays_to_list(self, array_in_text):
        result_list = []
        
        text_in_lines = array_in_text.split("\n")[1:]
        for line in tqdm(text_in_lines):
            if(len(line) > 0):
                tmplist = []
                splitline = line.split(" ")
                for i in splitline:
                    if i!="":
                        tmplist.append(int(i))
                result_list.append(tmplist)
        return result_list
    
    def decoding_2d_int_arrays_to_array(self, array_in_text):
        this_list = self.decoding_2d_int_arrays_to_list(array_in_text)
        result_list = []
        for i in this_list:
            for j in i:
                result_list.append(j)
        return np.asarray(result_list)
