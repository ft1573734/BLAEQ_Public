import cupy as cp
import numpy as np
import scipy.sparse as sp
import math
import cupyx
from tqdm import tqdm
import time

EPSILON = 0.001

class BLAEQ:
 
    def __init__(self):
        #Dimensionality of the mesh.
        self.D = -1 
    
        #Scale of the original mesh.
        self.N = -1

        #The original Mesh in ndarray format with shape N*D.
        self.OriginalMesh = -1
    
        #M_0 refers to the original mesh in matrix form. In BLAEQ it refers to the finest mesh.
        self.M_0 = np.zeros(0)
    
        #The collection of prolongation matrices.
        self.Prolongation_matrices = []
    
        #The mesh of the coarsest level. The rest of the levels e.g. M^{i-1}, M^{i-2} can be computed via ...*P^{i-1}*P^{i}*M^i. Notice that these are a sequence of SpMV ops, therefore can execute extremely fast using BLAS.
        self.Coarsest_mesh = []
    
        #bandwidths of each layer
        self.Bandwidths = []
    
        #Min and max vals of each dimension, in the form of [[min_x,max_x],[min_y,max_y],...]. Used for replacing -inf/inf in range queries.
        self.MIN_MAX = []
    
        #Whether executed on CPU/GPU, on CPU by default
        #WARNING: If you want to execute the algorithm on CPU, do not change config here, change it manually outside, after initialization.
        self.ON_GPU = True
        
        self.Permutation_matrices = []
    
    
    
    ''' =============TOOLS============= 
    NOTICE: TOOLS are sub-functions used in BLAEQ to fulfill more complicated, public functions. Theoretically, you will never need to call these functions.
    '''
    
    # Function Name: manual_bandwidth_generator(self, v_d, k)
    # Description: 
    #   Given a current mesh M_i, generate its coarser mesh M_{i+1} based on a given parameter k, where k indicates the size ratio i.e. k = M_i/M_{i+1}.
    # Input: 
    #   v_d: Current mesh;
    #   k:   Scale ratio (config manually).
    # Output:
    #   bandwidth: The desired bandwidth that achieves k = M_i/M_{i+1}.
    def manual_bandwidth_generator(self, v_d, k):
        if self.ON_GPU:
            bin_count = len(v_d)/k
            bandwidth = cp.max(v_d)/bin_count
            epsilon = bandwidth/1000
        else: #ON_CPU
            bin_count = len(v_d)/k
            bandwidth = np.max(v_d)/bin_count
            epsilon = bandwidth/1000
        return bandwidth + epsilon
    
    # Function Name: in_range_index(self, q_min, q_max, vals_in_vector, relaxation)
    # Description:
    #   Given a vector and a range query [q_min, q_max] with relaxation r, return an indicator vector with 1 representing the corresponding value is in range (q_min-r, q_max+r) and 0 otherwise.
    #   E.g.: in_range_index(0, 1, [0,0.5,1,2], 0.01) = [1,1,1,0]; in_range_index(0, 1, [0,0.5,1,2], 0)=[0,1,0,0].
    # Input:
    #   q_min, q_max: lower & upper bounds of a range query;
    #   vals_in_vector: the input vector;
    #   relaxation r: expands the query range by +-r.
    # Output:
    #   indicator vector I.
    def in_range_index(self, q_min, q_max, vals_in_vector, relaxation):
        if self.ON_GPU:
            middle = (q_max + q_min)/2
            q_diameter = (q_max - q_min)/2
            N = len(vals_in_vector)
            #nnz = vals_in_csr.nnz
            
            #[[middle]] refers to a (1,1) matrix.
            #middle_vector = cp.matmul(cp.ones((N,), dtype = cp.float32), cp.asarray([[middle]]))
            middle_vector = cp.full((N,), middle)
            tmp = cp.abs(vals_in_vector - middle_vector)
            nnzs = cp.where( vals_in_vector != 0,1,0)
            in_range_index = cp.where(tmp < (q_diameter + relaxation), 1, 0)
            in_range_index = cp.multiply(in_range_index, nnzs)
            #in_range_index = cp.multiply(vals_in_vector.indices, in_range_csc_index)
            #in_range_index = self.BLAS_where(tmp, q_diameter + relaxation, 0)
        else:
            middle = (q_max + q_min)/2
            q_diameter = (q_max - q_min)/2
            N = len(vals_in_vector)
            
            #[[middle]] refers to a (1,1) matrix.
            #middle_vector = np.matmul(np.ones((N,1), dtype = np.float32), np.asarray([[middle]]))
            middle_vector = np.full((N,), middle)
            tmp = np.abs(vals_in_vector - middle_vector)
        
            in_range_index = np.where(tmp < (q_diameter + relaxation), 1, 0)
        return in_range_index

    
    # DEPRECATED. This function is deprecated due to efficiency issue, use "generate_prolongation_matrix_BLAS" instead.
    # Function Name: generate_prolongation_matrix(self, M_0_d, M_1_d, bandwidth_0_d)
    # Description:
    #   Generate a prolongation matrix P_1to0_d from M_1_d to M_0_d, i.e. P_1to0_d * M_1_d = M_0_d. 
    # Input: 
    #   M_0_d: The coarser-level matrix (vector);
    #   M_1_d: The finer-level matrix (vector);
    #   bandwidth_0_d: The bandwidth of the 0-th mesh.
    # Output:
    #   P: Prolongation matrix in CSR format.
    def generate_prolongation_matrix(self, M_0_d, M_1_d, bandwidth_0_d):
        print("Generating prolongation matrix.")
        #We are going to implement the prolongtaion matrix in coo format. Since Scipy.coo_matrix is immutable once initialized, we manually create the row, col, nnz arrays for coo and initialize it afterwards.
        local_N = M_0_d.shape[0]
        P_nnz = np.zeros(local_N, dtype = np.float32)
        #In coo format, size of col array and row array are identical to nnz array
        P_col = np.zeros(local_N)
        P_row = np.zeros(local_N)
        #Filling the coo_matrix manually
        for i in tqdm(range(0,len(P_nnz))):
            M_0_val = M_0_d[i]
            #Using the centroid of each bin to represent the bin in M_1
            M_1_val = np.floor(M_0_val/bandwidth_0_d) * bandwidth_0_d + bandwidth_0_d / 2
            #row, col, val to be inserted
            row = np.floor(M_0_val/bandwidth_0_d)
            col = i
            #The prolongation inputs M_1 and outputs M_0, therefore stores M_0_val/M_1_val
            val = M_0_val/M_1_val
                
            P_row[i] = row
            P_col[i] = col
            P_nnz[i] = val

        #Now that the prolongation matrix P_1to0 is constructed in coo format, convert it to csr
        if self.ON_GPU:
            P_nnz = cp.asarray(P_nnz)
            P_row = cp.asarray(P_row)
            P_col = cp.asarray(P_col)
            P_1to0_d_coo = cupyx.scipy.sparse.coo_matrix((P_nnz, (P_row, P_col)), shape = (M_1_d.shape[0], M_0_d.shape[0]))
            P_1to0_d = P_1to0_d_coo.tocsr()
        else:
            P_1to0_d_coo = sp.coo_matrix((P_nnz, (P_row, P_col)), shape = (M_1_d.shape[0], M_0_d.shape[0]))
            P_1to0_d = P_1to0_d_coo.tocsr()
        return P_1to0_d


        

    # Function Name: generate_prolongation_matrix_BLAS_optimized(self, M_0_d, bandwidth_0_d)
    # Description:
    #   A BLAS-based implementaion of generating the prolongation matrix.
    #   Generate a prolongation matrix P_1to0_d from M_1_d to M_0_d, i.e. P_1to0_d * M_1_d = M_0_d. 
    #   M_1_d is constructed on-the-fly. Therefore does not require it as an input.
    # Input: 
    #   M_0_d: The coarser-level matrix (vector);
    #   bandwidth_0_d: The bandwidth of the 0-th mesh.
    # Output:
    #   P: Prolongation matrix in CSR format.
    #   M_1_d: The constructed M_1_d
    def generate_prolongation_matrix_BLAS_optimized(self, M_0_d, bandwidth_0_d):
        print("Generating prolongation matrix using BLAS.")
        #We are going to implement the prolongtaion matrix in coo format. Since Scipy.coo_matrix is immutable once initialized, we manually create the row, col, nnz arrays for coo and initialize it afterwards.
        local_N = M_0_d.size
        if self.ON_GPU:
            #Using cp.int_() to eliminate decimal parts. This vector indicates which bin a value in M_0_d locates in M_1_d
            prolongation_matrix_col = (cp.divide(M_0_d,bandwidth_0_d)).astype(cp.int32)
            prolongation_matrix_row = cp.arange(0, local_N, 1, dtype = cp.int32)
            prolongation_matrix_val = cp.divide(M_0_d ,cp.multiply(prolongation_matrix_col, bandwidth_0_d) + bandwidth_0_d/2)
            
            M_1_d = cp.multiply(cp.arange(0, cp.max(prolongation_matrix_col).item()+1, 1, dtype = cp.int32), bandwidth_0_d) + bandwidth_0_d/2
            
            
            #Now that the prolongation matrix P_1to0 is constructed in coo format, convert it to csr
            P_1to0_d_coo = cupyx.scipy.sparse.coo_matrix((prolongation_matrix_val, (prolongation_matrix_row, prolongation_matrix_col)))
            
            # Try csc format:
            P_1to0_d = P_1to0_d_coo.tocsc()
            
            # Try coo format:
            # P_1to0_d = P_1to0_d_coo
            
            # Try csr format:
            #P_1to0_d = P_1to0_d_coo.tocsr()

        else: #ON_CPU
            #Basically just replicate the ON_GPU code, replacing cp.__ to np.__
            
            prolongation_matrix_col = (np.divide(M_0_d,bandwidth_0_d)).astype(np.int32)
            prolongation_matrix_row = np.arange(0, local_N, 1, dtype = np.int32)
            prolongation_matrix_val = np.divide(M_0_d ,np.multiply(prolongation_matrix_col, bandwidth_0_d) + bandwidth_0_d/2)
            
            M_1_d = np.multiply(np.arange(0, np.max(prolongation_matrix_col).item()+1, 1, dtype = np.int32), bandwidth_0_d) + bandwidth_0_d/2
            
            #Now that the prolongation matrix P_1to0 is constructed in coo format, convert it to csr
            P_1to0_d_coo = sp.coo_matrix((prolongation_matrix_val, (prolongation_matrix_row, prolongation_matrix_col)))
            
            # Try csc format:
            P_1to0_d = P_1to0_d_coo.tocsc()
            
            # Try coo format:
            # P_1to0_d = P_1to0_d_coo
            
            # Try csr format:
            #P_1to0_d = P_1to0_d_coo.tocsr()
            
        return P_1to0_d, M_1_d

    '''=============BLAEQ Functions=============#
    The major functions used in BLAEQ. Theoretically, these functions are ALL YOU NEED.
    '''
    
    # Function Name: Load_Mesh(self, mesh_in_ndarray_format)
    # Description:
    #   Loads a mesh in ndarray format. Notice that the array is of shape D*N, where D stands for dimensionality and N stands for sample count. In this case, mesh[d] returns all values of the d-th dimension.
    #   This function basically just fills the local varables of the algorithm, such as the normalized mesh (adjusts the min_val to 0 e.g. [-1,0,1] -> [0,1,2]), collecting the min/max of each dim, etc.
    # Input:
    #   mesh_in_ndarray_format: self-explanatory.
    # Output:
    #   Nothing, but the algorithm object is updated.
    def Load_Mesh(self, mesh_in_ndarray_format):
        print("Loading mesh...")
        mesh_in_ndarray_format = np.asarray(mesh_in_ndarray_format)
        [self.D, self.N] = np.shape(mesh_in_ndarray_format)
        self.OriginalMesh = np.ndarray((self.D, self.N), dtype = np.float32)
        #Compute the [min, max] of each dimension
        for d in range(0,self.D):
            v_d = mesh_in_ndarray_format[d]    
            min_d = np.min(v_d)
            # We add a small EPSILON to each coordinate to ensure no coordinate has value "0", 
            # since "0" is a very special case in sparse linear algebra.
            new_v_d = np.add(v_d, -(min_d - EPSILON))  
            self.OriginalMesh[d] = new_v_d
            self.MIN_MAX.append([0,np.max(new_v_d)])
        # Generaly speaking we have D<<N
        if(self.D > self.N):
            print("Warining, the shape of the original mesh could be wrong!")
        print("Loading mesh complete!")
            
    
    # Function Name: Multi_Layer_Multigrid_Generator(self, K = -1)
    # Description:
    #   Generates a multigrid based on a given K. The original mesh has been initialized in the algorithm object.
    # Input:
    #   K: Scale ratio between two adjacent layers. 
    # Output:
    #   Nothing, the multigrid is initialized within the algorithm object.
    def Multi_Layer_Multigrid_Generator(self, K = -1, L = -1):
        print("Generating multi-layer multigrid...")
        initial_K = K
        if self.D == -1 or self.N == -1:
            print("Error, original mesh not loaded.")
            exit()
        
        M_0 = cp.asarray(self.OriginalMesh)
        
        tmp_coarsest_mesh = -1

        Prolongation_Matrices = []

        if self.ON_GPU:
            for d in range(0, self.D):
                print("Processing dimension "+str(d)+" ...")

                M_i_d = M_0[d]

                P_all_d = []
                bandwidths_d = []
                while len(M_i_d)/K > 1: #The size of the coarsest layers should not be smaller than 1
                    bandwidth_i_d = self.manual_bandwidth_generator(M_i_d, K)
                    bandwidths_d.append(bandwidth_i_d)

                    [P_ip1toi_d, M_ip1_d] = self.generate_prolongation_matrix_BLAS_optimized(M_i_d, bandwidth_i_d)
                    P_all_d.append(P_ip1toi_d)
                    # [P_ip1toi_d_refined, M_ip1_d] = self.refine_prolongation_matrix(P_ip1toi_d, M_ip1_d, L)
                    # P_all_d.append(P_ip1toi_d_refined)
                
                    #Update M_i_d
                    M_i_d = M_ip1_d
                    # bandwidths_d.append(bandwidth_i_d)
                    #Auto Adjust K:
                    #K = cp.sqrt(K)
                    
                self.Prolongation_matrices.append(P_all_d)
                self.Bandwidths.append(bandwidths_d)
                
                # M_i_d_indices = cp.arange(0, len(M_i_d))
                # mask = M_i_d != 0
                # self.Coarsest_mesh.append(BLAEQ_Sparse.Sparse_Vector(M_i_d[mask], M_i_d_indices[mask], M_i_d.shape[0]))
                # M_i_d = cupyx.scipy.sparse.csr_matrix(M_i_d.reshape(len(M_i_d),1))

                self.Coarsest_mesh.append(M_i_d)
                #K = initial_K
                
        else: #ON_CPU
            #Basically just replicate the ON_GPU code, replacing cp.__ to np.__
            for d in range(0, self.D):
                print("Processing dimension "+str(d)+" ...")

                M_i_d = np.asarray(M_0[d].get())

                P_all_d = []
                bandwidths_d = []
                while len(M_i_d)/K > 1: #The size of the coarsest layers should not be smaller than 1
                    bandwidth_i_d = self.manual_bandwidth_generator(M_i_d, K)
                    bandwidths_d.append(bandwidth_i_d)

                    [P_ip1toi_d, M_ip1_d] = self.generate_prolongation_matrix_BLAS_optimized(M_i_d, bandwidth_i_d)
                    P_all_d.append(P_ip1toi_d)
                    # [P_ip1toi_d_refined, M_ip1_d] = self.refine_prolongation_matrix(P_ip1toi_d, M_ip1_d, L)
                    # P_all_d.append(P_ip1toi_d_refined)
                
                    #Update M_i_d
                    M_i_d = M_ip1_d
                    # bandwidths_d.append(bandwidth_i_d)
                    #Auto Adjust K:
                    #K = cp.sqrt(K)
                    
                self.Prolongation_matrices.append(P_all_d)
                self.Bandwidths.append(bandwidths_d)
                
                # M_i_d_indices = cp.arange(0, len(M_i_d))
                # mask = M_i_d != 0
                # self.Coarsest_mesh.append(BLAEQ_Sparse.Sparse_Vector(M_i_d[mask], M_i_d_indices[mask], M_i_d.shape[0]))
                # M_i_d = cupyx.scipy.sparse.csr_matrix(M_i_d.reshape(len(M_i_d),1))

                self.Coarsest_mesh.append(M_i_d)
                #K = initial_K
        return 
    
    # Function Name: Multi_Layer_Multigrid_Query(self, x_min, x_max, y_min, y_max, z_min, z_max)
    # Description:
    #   Generates a multigrid based on a given K. The original mesh has been initialized in the algorithm object.
    # Input:
    #   K: Scale ratio between two adjacent layers. 
    # Output:
    #   Nothing, the multigrid is initialized within the algorithm object.
    def Multi_Layer_Multigrid_Query(self, x_min, x_max, y_min, y_max, z_min, z_max):
        if self.ON_GPU:
            if len(self.Bandwidths[0]) != len(self.Prolongation_matrices[0]):
                print("Error, #Bandwidths does not match #Prolongation Matrices.")
                exit()

            x_range = cp.array([x_min,x_max])
            y_range = cp.array([y_min,y_max])
            z_range = cp.array([z_min,z_max])
        
            N = self.N
        
            Q = cp.array([x_range,y_range,z_range])
        
            result_index_per_dim = []

            
            filtered_result = cp.ones(N,) #Not intialized

            for d in range(0,self.D):
                q = Q[d]
                # if q[0]==-cp.inf and q[1] == cp.inf:
                #     continue
                #M_ip1_d = self.Coarsest_mesh[d].reshape(len(self.Coarsest_mesh[d]),1)
                M_ip1_d: cupyx.scipy.sparse.csr_matrix = self.Coarsest_mesh[d]
                
                for i in range(len(self.Bandwidths[0])-1, -1, -1):
                    P_ip1toi_d = self.Prolongation_matrices[d][i]
                    b_ip1_d = self.Bandwidths[d][i]

                    filtered_indices = self.in_range_index(q[0], q[1], M_ip1_d, b_ip1_d)
                    #filtered_M_ip1_d = BLAEQ_Sparse.Sparse_Vector(M_ip1_d.data[filtered_indices], M_ip1_d.indices[filtered_indices], M_ip1_d.length)
                    filtered_M_ip1_d = cp.multiply(M_ip1_d,filtered_indices)


                    # M_ip1_d = P_ip1toi_d.dot(filtered_M_ip1_d)
                    # M_i_d = BLAEQ_Sparse.Prolongation_SpMSpV(P_ip1toi_d, filtered_M_ip1_d)
                    M_i_d = P_ip1toi_d * filtered_M_ip1_d
                    # M_i_d = P_ip1toi_d.dot(filtered_M_ip1_d)
                    M_ip1_d = M_i_d


                filtered_indices = self.in_range_index(q[0], q[1], M_ip1_d, b_ip1_d)
                filtered_result = cp.multiply(filtered_result, filtered_indices)

            return filtered_result
        
    
        else: #ON_CPU
            if len(self.Bandwidths[0]) != len(self.Prolongation_matrices[0]):
                print("Error, #Bandwidths does not match #Prolongation Matrices.")
                exit()

            x_range = np.array([x_min,x_max])
            y_range = np.array([y_min,y_max])
            z_range = np.array([z_min,z_max])
        
            N = self.N
        
            Q = np.array([x_range,y_range,z_range])
        
            result_index_per_dim = []

            filtered_result = np.ones((N,),dtype = np.float32)

            for d in range(0,self.D):
                q = Q[d]
                M_ip1_d = self.Coarsest_mesh[d]
                for i in range(len(self.Bandwidths[0])-1, -1, -1):
                    P_ip1toi_d = self.Prolongation_matrices[d][i]
                    b_ip1_d = self.Bandwidths[d][i]
                
                    valid_index_ip1_d = self.in_range_index(q[0], q[1], M_ip1_d, b_ip1_d)
                    filtered_M_ip1_d = np.multiply(M_ip1_d, valid_index_ip1_d)
                
                    M_ip1_d = P_ip1toi_d.dot(filtered_M_ip1_d)
            
                # Now remains the last layer of mesh i.e. the original mesh, we need to manually compute this layer.
                M_0_d = M_ip1_d
                valid_index_ip1_d = self.in_range_index(q[0], q[1], M_0_d, 0)
                filtered_M_ip1_d = np.multiply(M_0_d, valid_index_ip1_d)
                filtered_result = np.multiply(filtered_result, valid_index_ip1_d)
            
            return filtered_result