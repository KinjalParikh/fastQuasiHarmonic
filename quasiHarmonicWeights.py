import igl
import numpy as np
import utility
import scipy
import sksparse.cholmod as sc
from autograd import jacobian


class QHWeights():
    def __init__(self, vertices, faces, control_points_i, dim=3):
        super().__init__()

        self.nvertices = vertices.shape[0]
        self.nfaces = faces.shape[0]
        self.ncp = len(control_points_i)

        self.vertices = vertices
        self.faces = faces
        self.cp_i = control_points_i

        self.G = igl.grad(vertices, faces)
        self.L = self.compute_L()
        self.M = igl.massmatrix(vertices, faces)
        Minv = np.eye(self.nvertices)
        np.fill_diagonal(Minv, np.power(np.diag(self.M.todense()), -1))
        self.Minv = Minv

        self.B = np.eye(len(control_points_i), len(control_points_i))  # boundary condition matrix
        self.R = self.build_R()                                        # binary selection matrix for control points
        self.S = self.build_S()                                        # binary selection matrix for non control points

        self.Q = np.matmul(self.L, np.matmul(self.Minv, self.L))       # quadratic form measuring smoothness of weights

        # Symbolic pre-factorization of S'G'AGS
        mat = self.get_mat_for_symbolic_fact()
        self.sym_factor = sc.analyze(mat)

    def predWeights(self, theta):
        # calculate A for theta
        flat_a = utility.theta_to_flat_a(theta, self.nfaces)
        A = utility.unflattenA(flat_a, self.nfaces)

        # numerical factorization using the symbolic pre-factorization
        mat = np.matmul(self.S.transpose(), np.matmul(self.G.transpose().todense(), np.matmul(A, self.G.todense())))
        self.sym_factor.cholesky_inplace(scipy.sparse.csc_matrix(np.matmul(mat, self.S)))
        # back substitution
        U = - self.sym_factor(np.matmul(mat, np.matmul(self.R, self.B)))
        return U

    def gradient(self, theta, U):
        dEda = np.zeros((1, 6 * self.nfaces), dtype="float32")

        for ind in range(self.ncp):
            dEdU = np.matmul(U[:, ind].transpose(), np.matmul(self.S.transpose(), np.matmul(self.Q, self.S))) + \
                   np.matmul(self.B[:, ind].transpose(), np.matmul(self.R.transpose(),np.matmul(self.Q, self.S)))

            bvec = np.asarray(np.add(np.matmul(self.G.todense(), np.matmul(self.S, U[:, ind])),
                                      np.matmul(self.G.todense(), np.matmul(self.R, self.B[:, ind])
                                                .reshape(self.R.shape[0], 1)))).reshape(-1)
            rhs = np.matmul(self.S.transpose(), np.matmul(self.G.transpose().todense(),
                                                                        utility.sp3d(bvec[:self.nfaces],
                                                                             bvec[self.nfaces:2 * self.nfaces],
                                                                             bvec[2 * self.nfaces: 3 * self.nfaces], self.nfaces)))
            dUda = - self.sym_factor(rhs)
            dEda = np.add(dEda, np.matmul(dEdU, dUda))

        fdadtheta = jacobian(utility.ftheta_to_flat_a)
        dadtheta = np.empty((6 * self.nfaces, 1))
        for find in range(self.nfaces):
            thetaf = theta[6 * find: 6 * find + 6]
            df = fdadtheta(thetaf)
            for i in range(6):
                dadtheta[self.nfaces * i + find, 0] = df[0, i]

        return np.multiply(dEda.transpose(), dadtheta)

    def compute_L(self):
        Mf = np.diag(igl.doublearea(self.vertices, self.faces) / 2.0)
        Mfrep = np.kron(np.eye(3, dtype=float), Mf)
        return np.matmul(self.G.transpose().todense(), np.matmul(Mfrep, self.G.todense()))

    def build_R(self):
        '''
        :return: binary selection matrix such that selects rows for the vertices representing the j-th control handle
        '''
        R = np.zeros((self.nvertices, self.ncp) , dtype=int)
        for ind in range(self.ncp):
            R[self.cp_i[ind], -(ind+1)] = 1
        return R

    def build_S(self):
        '''
        :return: binary selection matrix for non control handle
        '''
        num_cols = self.nvertices - self.ncp
        S = np.zeros((self.nvertices, num_cols), dtype=int)
        noncind = 0
        for ind in range(num_cols):
            while noncind in self.cp_i:
                noncind += 1
            S[noncind, -(ind + 1)] = 1
            noncind += 1
        return S

    def get_mat_for_symbolic_fact(self):
        area = igl.doublearea(self.vertices, self.faces) / 2.0
        flat_a = np.block([area,
                           np.zeros(self.nfaces, dtype="float64"),
                           area,
                           np.zeros(self.nfaces, dtype="float64"),
                           np.zeros(self.nfaces, dtype="float64"),
                           area
                           ])
        A = utility.unflattenA(flat_a, self.nfaces)
        mat = np.matmul(self.S.transpose(),
                        np.matmul(self.G.transpose().todense(), np.matmul(A, np.matmul(self.G.todense(), self.S))))
        return scipy.sparse.csc_matrix(mat)

    def getWeights(self, U):
        W = np.matmul(self.S, U) + np.matmul(self.R, self.B)
        temp = np.copy(W)
        mask = temp == 0
        temp[mask] = 1
        norm = np.sum(np.abs(temp) ** 2, axis=-1) ** 0.5
        n_w = np.divide(W, norm.reshape(norm.shape[0], 1))
        return n_w