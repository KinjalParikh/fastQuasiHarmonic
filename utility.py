import numpy as np
import igl
import autograd.numpy as anp
import colorsys
import scipy.sparse as sp


def sp3d(b0, b1, b2, f):
    '''
    Operator SP(b), chosen to satisfy the expression Ab = sp(b)flatten(A)
    (section 5.3 of the paper)
    :param b0: #f x 1 vector
    :param b1: #f x 1 vector
    :param b2: #f x 1 vector
    :param f: number of faces
    :return: SP(b) where b = transpose([b0 b1 b2])
    '''
    dib0 = sp.diags(b0)
    dib1 = sp.diags(b1)
    dib2 = sp.diags(b2)

    spb = sp.bmat([
        [dib0, dib1, sp.lil_matrix((f, f)) , dib2, sp.lil_matrix((f, f)), sp.lil_matrix((f, f))],
        [sp.lil_matrix((f, f)), dib0, dib1, sp.lil_matrix((f, f)), dib2, sp.lil_matrix((f, f))],
        [sp.lil_matrix((f, f)), sp.lil_matrix((f, f)), sp.lil_matrix((f, f)), dib0, dib1, dib2]
    ])
    return spb


def init_theta(faces, vertices):
    '''
    Gives theta so that the anisotropy tensors are initialized as identity matrices multiplied by triangle areas
    theta is vector of elements of the Cholesky factorization.
    (Section 5.4 of the paper)
    :param faces: face list
    :param vertices: vertex list
    :return: theta
    '''
    nface = faces.shape[0]
    area = igl.doublearea(vertices, faces) / 2.0
    flat_a = np.block([area,
                       np.zeros(nface, dtype="float64"),
                       area,
                       np.zeros(nface, dtype="float64"),
                       np.zeros(nface, dtype="float64"),
                       area
                       ])
    return get_theta(flat_a, nface)


def unflattenA(flat_a, f):
    A = sp.bmat([
        [sp.diags(flat_a[:f]), sp.diags(flat_a[f:2 * f]), sp.diags(flat_a[3 * f: 4 * f])],
        [sp.diags(flat_a[f:2 * f]), sp.diags(flat_a[2 * f:3 * f]), sp.diags(flat_a[4 * f:5 * f])],
        [sp.diags(flat_a[3 * f:4 * f]), sp.diags(flat_a[4 * f:5 * f]), sp.diags(flat_a[5 * f:6 * f])]
    ])
    return A


def get_Aj(flat_a, j, f):
    '''
    :return: 3x3 matrix containing anisotropy values of jth face
    '''
    Aj = np.array([
        [flat_a[j], flat_a[f+j], flat_a[3*f+j]],
        [flat_a[f+j], flat_a[2*f+j], flat_a[4*f+j]],
        [flat_a[3*f+j], flat_a[4*f+j], flat_a[5*f+j]],
    ])
    return Aj


def get_theta(flat_a, f):
    theta = np.empty(6*f, dtype='float32')
    for find in range(f):
        lowm = np.linalg.cholesky(get_Aj(flat_a, find, f))
        theta[find*6:(find+1)*6] = [lowm[0, 0], lowm[1, 0], lowm[1, 1], lowm[2, 0], lowm[2, 1], lowm[2, 2]]
    return theta


def theta_to_flat_a(theta, f):
    '''
    (Section 5.4)
    '''
    a00 = []
    a10 = []
    a11 = []
    a20 = []
    a21 = []
    a22 = []
    for find in range(f):
        a = theta[find*6+0]
        b = theta[find*6+1]
        c = theta[find*6+2]
        d = theta[find*6+3]
        e = theta[find*6+4]
        f = theta[find*6+5]
        a00.append(a*a + 0.2*(b*b + c*c + d*d + e*e + f*f) + 10**(-4))
        a10.append(a*b)
        a11.append(b*b + c*c + 0.2*(a*a + d*d + e*e + f*f) + 10**(-4))
        a20.append(a*d)
        a21.append(b*d + c*e)
        a22.append(d*d + e*e + f*f + 0.2*(a*a + b*b + c*c) + 10**(-4))
    flat_a = np.concatenate((a00, a10, a11, a20, a21, a22))
    return flat_a


def ftheta_to_flat_a(thetaf):
    af = anp.array([thetaf[0]**2, thetaf[0]*thetaf[1], thetaf[1]**2+thetaf[2]**2, thetaf[0]*thetaf[3], thetaf[1]*thetaf[3]+thetaf[2]*thetaf[4], thetaf[3]**2+thetaf[4]**2+thetaf[5]**2])
    return af


def get_Colours(ncp):
    '''
    Gets contrasting colours
    :param ncp: number of colours
    :return: #ncp x 3 - RGB colours
    '''
    HSV = [[x / ncp, 0.5, 0.5] for x in range(ncp)]
    RGB = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), HSV))
    return RGB