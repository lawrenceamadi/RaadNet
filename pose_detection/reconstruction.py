import numpy as np

class OrthRecon(object):
    '''
        Class implements image point triangulation, 3D world point reconstruction, and re-projection
    '''
    def __init__(self, imgW, scanRad, rotAng, step):
        self.xHalfPoint = int(imgW / 2)
        self.FxBase = np.array([self.xHalfPoint, 0, 0])
        self.rotation_angle, self.scanner_radius = rotAng, scanRad  # scanner radius (0.63) is unimportant
        self.R_r = rotate_about_Y_matrix(rotAng)
        self.Rrr = rotate_about_Y_matrix(rotAng * 2)
        self.R_l = rotate_about_Y_matrix(rotAng, x_direction=-1)
        self.Rll = rotate_about_Y_matrix(rotAng * 2, x_direction=-1)
        self.t_r = translation_vector(scanRad, rotAng, z_direction=-1)
        self.t_l = translation_vector(scanRad, rotAng, x_direction=-1, z_direction=-1)
        self.t = np.array([0, 0, scanRad])
        self.xStep = step

    def unprep_x(self, x):
        return self.xHalfPoint + x

    def prep_x(self, x):
        if x > self.xHalfPoint:
            return x % self.xHalfPoint
        return x - self.xHalfPoint

    def orth_project_from_worldpt(self, worldPt, RotationMatrix, y):
        v = np.dot(RotationMatrix, self.FxBase)
        x_l = np.dot(worldPt, v) / np.linalg.norm(v)
        x_img = self.unprep_x(x_l)
        p_lol = np.array([x_img, y, 0])
        return np.around(p_lol, 0).astype(np.int32)

    def recon_from_right(self, pvKpt, yCands, xCands, shiftx):
        yCandNum, xCandNum = len(yCands), len(xCands)
        rxCands, lxCands, rrxCands, llxCands = initialize_candidates(yCandNum, xCandNum)
        p_ioi = to_unit_vector(pvKpt)  # pivot
        n_ioi = np.array([0, 0, -1])

        # do for each candidate on Right image
        for ycid in range(yCandNum):
            c_y = yCands[ycid]
            xShift = (ycid + 1) % self.xStep if shiftx else 0
            for xcid in range(xCandNum):
                x = xCands[xcid] + xShift
                rxCands[ycid][xcid] = x
                c_x = self.prep_x(x)
                r_kpt = np.array([c_x, c_y, 0])
                p_ror = to_unit_vector(r_kpt)  # p_r: p_(i+1) in its own (right) coordinate
                n_ror = np.array([0, 0, -1, 1])
                # Orthogonal point triangulation to find world point along ray (in pvKpt coordinate)
                P_wow = orth_triangulation(p_ioi, n_ioi, p_ror, n_ror, self.R_r, self.t_r, self.t)
                P_w_scaled = rec_upto_scale(P_wow, c_y)
                # Project world point unto left image
                leftPt = self.orth_project_from_worldpt(P_w_scaled, self.R_l, c_y)
                lxCands[ycid][xcid] = leftPt[0]
                # Project world point unto leftmost image
                leftmostPt = self.orth_project_from_worldpt(P_w_scaled, self.Rll, c_y)
                llxCands[ycid][xcid] = leftmostPt[0]
                # Project world point unto rightmost image
                rightmostPt = self.orth_project_from_worldpt(P_w_scaled, self.Rrr, c_y)
                rrxCands[ycid][xcid] = rightmostPt[0]
        return rxCands, rrxCands, lxCands, llxCands

    def recon_from_left(self, pvKpt, yCands, xCands, shiftx):
        yCandNum, xCandNum = len(yCands), len(xCands)
        rxCands, lxCands, rrxCands, llxCands = initialize_candidates(yCandNum, xCandNum)
        p_ioi = to_unit_vector(pvKpt)  # pivot
        n_ioi = np.array([0, 0, -1])

        # do for each candidate on Right image
        for ycid in range(yCandNum):
            c_y = yCands[ycid]
            xShift = (ycid + 1) % self.xStep if shiftx else 0
            for xcid in range(xCandNum):
                x = xCands[xcid] + xShift
                lxCands[ycid][xcid] = x
                c_x = self.prep_x(x)
                l_kpt = np.array([c_x, c_y, 0])
                p_lol = to_unit_vector(l_kpt)  # p_l: p_(i+1) in its own (left) coordinate
                n_lol = np.array([0, 0, -1, 1])

                # Orthogonal point triangulation to find world point along ray (in pvKpt coordinate)
                P_wow = orth_triangulation(p_ioi, n_ioi, p_lol, n_lol, self.R_l, self.t_l, self.t)
                P_w_scaled = rec_upto_scale(P_wow, c_y)
                # Project world point unto right image
                rightPt = self.orth_project_from_worldpt(P_w_scaled, self.R_r, c_y)
                rxCands[ycid][xcid] = rightPt[0]
                # Project world point unto leftmost image
                leftmostPt = self.orth_project_from_worldpt(P_w_scaled, self.Rll, c_y)
                llxCands[ycid][xcid] = leftmostPt[0]
                # Project world point unto rightmost image
                rightmostPt = self.orth_project_from_worldpt(P_w_scaled, self.Rrr, c_y)
                rrxCands[ycid][xcid] = rightmostPt[0]
        return rxCands, rrxCands, lxCands, llxCands



# STATIC METHODS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
def from_3d_to_3dh(pt_3d):
    pt_3dh = np.ones((4), dtype=np.float32)
    pt_3dh[:3] = pt_3d
    return pt_3dh

def from_3dh_to_3d(pt_3dh):
    pt_3d = pt_3dh / pt_3dh[3]
    return pt_3d[:3]

def to_unit_vector(vector):
    # equivalent to vector / np.sqrt(np.sum(np.square(vector)))
    unit_v = vector / np.linalg.norm(vector)
    return unit_v

def rec_upto_scale(pt, y):
    # very important for reconstruction up to scale
    factor = y / pt[1]
    return factor * pt

def rotate_about_Z_matrix(angleDeg, direction=1):
    angleRad = np.deg2rad(angleDeg * direction)
    R = np.zeros(shape=(3, 3), dtype=np.float32)
    R[:2, :2] = [[np.cos(angleRad), -np.sin(angleRad)],
                 [np.sin(angleRad), np.cos(angleRad)]]
    R[2, 2] = 1
    return R

def rotate_about_Y_matrix(angleDeg, x_direction=1):
    angleRad = np.deg2rad(angleDeg * x_direction)
    R = [[np.cos(angleRad), 0, np.sin(angleRad)],
         [0, 1, 0],
         [-np.sin(angleRad), 0, np.cos(angleRad)]]
    return np.array(R)

def translation_vector(radius, angleDeg, x_direction=1, z_direction=1):
    theta = np.deg2rad(angleDeg)
    alpha = np.deg2rad((180 - angleDeg) / 2)
    T = (radius * np.sin(theta)) / np.sin(alpha)
    phi = np.deg2rad(2 * angleDeg)
    beta = np.deg2rad((180 - (2 * angleDeg)) / 2)
    Tx = (radius * np.sin(phi)) / (2 * np.sin(beta)) * x_direction
    Tz = np.sqrt(np.square(T) - np.square(Tx)) * z_direction
    t = np.float32([Tx, 0, Tz])
    return t

def Rt_Matrix(R, t):
    M = np.zeros(shape=(4, 4), dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    M[3, 3] = 1
    return M

def orth_triangulation(p_ioi, n_ioi, p_sos, n_sos, R_s, t_s, t):
    # Orthogonal point triangulation
    Rp_s = np.dot(R_s, p_sos)  # 3x1
    p_soi = Rp_s + t_s
    M = Rt_Matrix(R_s, t_s)
    M_ = np.transpose(np.linalg.inv(M))
    # n_roi = np.dot(M_, n_ror)[:3]
    n_soi_3dh = np.dot(M_, n_sos)
    n_soi = n_soi_3dh[:3] / n_soi_3dh[3]
    w = np.cross((p_ioi + n_ioi), (p_soi + n_soi))  # 3x1
    A = np.stack((n_ioi, -n_soi, w), axis=1)
    v = p_soi - p_ioi
    x = np.linalg.solve(A, v)
    a, b, c = x
    # Find world point along ray (in pvKpt coordinate)
    P_woi = p_ioi + a * n_ioi + 0.5 * c * w  # 3x1
    P_wow = P_woi + t
    return P_wow

def initialize_candidates(yCandNum, xCandNum):
    rxCands = np.zeros(shape=(yCandNum, xCandNum), dtype=np.int32)  # right image candidates
    lxCands = np.zeros(shape=(yCandNum, xCandNum), dtype=np.int32)  # left image candidates
    rrxCands = np.zeros(shape=(yCandNum, xCandNum), dtype=np.int32)  # rightmost candidates
    llxCands = np.zeros(shape=(yCandNum, xCandNum), dtype=np.int32)  # leftmost candidates
    return rxCands, lxCands, rrxCands, llxCands