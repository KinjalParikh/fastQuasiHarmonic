import numpy as np
from quasiHarmonicWeights import QHWeights
import utility
from adam import Adam
from tqdm import tqdm


def runOptimization(vertices, faces, control_points_i):
    W = optimizeWeights(vertices, faces, control_points_i)
    C = utility.get_Colours(len(control_points_i))
    return np.matmul(W, C)


def optimizeWeights(vertices, faces, cp_i):
    qhw = QHWeights(vertices, faces, cp_i)
    a_model = Adam(faces.shape[0], alpha=0.1)
    theta = utility.init_theta(faces, vertices)
    num_iter = 10

    ci_str = ", ".join(str(ci) for ci in cp_i) + "\n"
    with open('wts.txt', 'a') as f:
        f.write(ci_str)
        f.write(str(num_iter)+"\n")

    for epoch in tqdm(range(num_iter)):
        U_pred = qhw.predWeights(theta)
        grad = qhw.gradient(theta, U_pred)
        delta = a_model.step(np.squeeze(grad), epoch).transpose()
        for ind in range(theta.size):
            theta[ind] -= delta.flat[ind]

        if (epoch+1) % 5 == 0:
            # resets the accumulated momentum estimates to zero every T iterations (Section 6.1 of paper)
            a_model.v_prev = 0
            a_model.m_prev = 0
            # learning rate halved every T iterations (Section 6.1 of paper)
            a_model.alpha = a_model.alpha / 2

        wts = qhw.getWeights(U_pred)

        # make sure that you have deleted the wts.txt from any previous runs of the code
        # Uncomment next 2 lines only if you want to visualize the optimization
        with open('wts.txt', 'ab') as f:
            np.savetxt(f, wts)

    return wts
