import numpy as np

def compute_spinal_loads_standing(sex, BH, BW, M, A, D):
    """
    Compute L4-L5 and L5-S1 compression and shear forces.

    Parameters:
        sex : int   — Female=1, Male=0
        BH  : float — Body Height (cm)
        BW  : float — Body Weight (kg)
        M   : float — Moment
        A   : float — Angle
        D   : float — D variable

    Returns:
        dict with keys: 'L4L5_compression', 'L4L5_shear',
                        'L5S1_compression', 'L5S1_shear'
    """
    # Pack interaction terms once for reuse
    SxM  = sex * M
    SxA  = sex * A
    BHxM = BH  * M
    BHxA = BH  * A
    BHxD = BH  * D
    BWxM = BW  * M
    BWxA = BW  * A
    BWxD = BW  * D
    MxA  = M   * A
    MxD  = M   * D
    AxD  = A   * D

    # ── L4-L5 Compression ───────────────────────────────────────────────────
    coef_l4l5_comp = np.array([
        -752.375,   # constant
        -34.2617,   # sex
         4.036604,  # BH
         2.801715,  # BW
       100.1808,    # M
        29.51852,   # A
        21.04534,   # D
         3.444643,  # Sex×M
        -2.54159,   # Sex×A
        -0.48814,   # BH×M
        -0.09225,   # BH×A
        -0.10352,   # BH×D
         0.122158,  # BW×A
         0.089068,  # BW×D
         1.494802,  # M×D
         0.043595,  # A×D
    ])
    terms_l4l5_comp = np.array([1, sex, BH, BW, M, A, D,
                                 SxM, SxA, BHxM, BHxA, BHxD,
                                 BWxA, BWxD, MxD, AxD])

    # ── L4-L5 Shear ─────────────────────────────────────────────────────────
    coef_l4l5_shear = np.array([
        -45.6541,   # constant
          2.213731, # sex
         -0.17664,  # BH
          0.49574,  # BW
         13.71772,  # M
          5.015594, # A
          1.363474, # D
         -0.39249,  # Sex×A
         -0.0555,   # BH×M
         -0.01959,  # BH×A
         -0.03536,  # BW×M
          0.032009, # BW×A
          0.035849, # M×A
          0.183764, # M×D
          0.010119, # A×D
    ])
    terms_l4l5_shear = np.array([1, sex, BH, BW, M, A, D,
                                  SxA, BHxM, BHxA, BWxM, BWxA,
                                  MxA, MxD, AxD])

    # ── L5-S1 Compression ───────────────────────────────────────────────────
    coef_l5s1_comp = np.array([
        -915.245,   # constant
         -31.217,   # sex
           4.922405,# BH
           1.839903,# BW
         105.2327,  # M
          34.76647, # A
          21.24288, # D
           5.382006,# Sex×M
          -2.99263, # Sex×A
          -0.53138, # BH×M
          -0.11433, # BH×A
          -0.11285, # BH×D
           0.14919, # BW×A
           0.109336,# BW×D
           1.661007,# M×D
           0.063209,# A×D
    ])
    terms_l5s1_comp = np.array([1, sex, BH, BW, M, A, D,
                                 SxM, SxA, BHxM, BHxA, BHxD,
                                 BWxA, BWxD, MxD, AxD])

    # ── L5-S1 Shear ─────────────────────────────────────────────────────────
    coef_l5s1_shear = np.array([
        -213.087,   # constant
         -27.1718,  # sex
           0.732533,# BH
           1.246567,# BW
          38.30302, # M
          13.38083, # A
           7.913065,# D
           2.274996,# Sex×M
          -1.05515, # Sex×A
          -0.15533, # BH×M
          -0.04389, # BH×A
          -0.0346,  # BH×D
          -0.06353, # BW×M
           0.066209,# BW×A
           0.033575,# BW×D
           0.587299,# M×D
           0.023537,# A×D
    ])
    terms_l5s1_shear = np.array([1, sex, BH, BW, M, A, D,
                                  SxM, SxA, BHxM, BHxA, BHxD,
                                  BWxM, BWxA, BWxD, MxD, AxD])

    return {
        "L4L5_compression": np.dot(coef_l4l5_comp,  terms_l4l5_comp),
        "L4L5_shear":       np.dot(coef_l4l5_shear, terms_l4l5_shear),
        "L5S1_compression": np.dot(coef_l5s1_comp,  terms_l5s1_comp),
        "L5S1_shear":       np.dot(coef_l5s1_shear, terms_l5s1_shear),
    }



def compute_spinal_loads_flex(sex, BH, BW, M, A, F, D):
    """
    Compute L4-L5 and L5-S1 compression and shear forces (Table 2, includes F).

    Parameters:
        sex : int   — Female=1, Male=0
        BH  : float — Body Height
        BW  : float — Body Weight
        M   : float — Moment
        A   : float — Angle
        F   : float — F variable (additional predictor vs Table 1)
        D   : float — D variable

    Returns:
        dict with keys: 'L4L5_compression', 'L4L5_shear',
                        'L5S1_compression', 'L5S1_shear'
    """
    # Interaction terms
    SxBW = sex * BW
    SxM  = sex * M
    SxA  = sex * A
    SxF  = sex * F
    SxD  = sex * D
    BHxA = BH  * A
    BHxD = BH  * D
    BWxM = BW  * M
    BWxA = BW  * A
    BHxF = BH  * F
    BWxF = BW  * F
    BWxD = BW  * D
    MxA  = M   * A
    MxF  = M   * F
    MxD  = M   * D
    AxF  = A   * F
    AxD  = A   * D
    FxD  = F   * D

    # ── L4-L5 Compression ───────────────────────────────────────────────────
    coef = np.array([
         37.22059,  # constant
       -338.104,    # sex
          1.909076, # BH
          6.116562, # BW
        -11.8938,   # M
         31.64569,  # A
         -4.78868,  # F
         13.76692,  # D
          2.683103, # Sex×BW
          7.612364, # Sex×M
         -3.12525,  # Sex×A
          2.848695, # Sex×F
          2.171282, # Sex×D
         -0.08315,  # BH×A
         -0.14364,  # BH×D
          0.087833, # BW×M
          0.024035, # BW×A
          0.231384, # BW×F
          0.118724, # BW×D
         -0.05777,  # M×A
          0.914314, # M×F
          2.473507, # M×D
         -0.09588,  # A×F
         -0.02157,  # A×D
          0.051812, # F×D
    ])
    terms = np.array([1, sex, BH, BW, M, A, F, D,
                      SxBW, SxM, SxA, SxF, SxD,
                      BHxA, BHxD, BWxM, BWxA, BWxF, BWxD,
                      MxA, MxF, MxD, AxF, AxD, FxD])
    l4l5_comp = np.dot(coef, terms)

    # ── L4-L5 Shear ─────────────────────────────────────────────────────────
    coef = np.array([
        -71.9487,   # constant
          5.412437, # sex
          # BH: –
          1.316494, # BW
          3.106165, # M
          3.088134, # A
          0.187397, # F
          0.899799, # D
         -0.52792,  # Sex×A
          0.570002, # Sex×F
          0.220941, # Sex×D
         -0.02079,  # BW×M
          0.034375, # BW×F
         -0.03358,  # M×A
          0.109374, # M×F
          0.304779, # M×D
         -0.02111,  # A×F
         -0.01119,  # A×D
         -0.0037,   # F×D
    ])
    terms = np.array([1, sex, BW, M, A, F, D,
                      SxA, SxF, SxD,
                      BWxM, BWxF,
                      MxA, MxF, MxD, AxF, AxD, FxD])
    l4l5_shear = np.dot(coef, terms)

    # ── L5-S1 Compression ───────────────────────────────────────────────────
    coef = np.array([
         -1.75693,  # constant
       -217.864,    # sex
          1.457613, # BH
          8.206912, # BW
         -8.93808,  # M
         33.00651,  # A
         -7.42666,  # F
         14.80836,  # D
          1.618418, # Sex×BW
          6.336128, # Sex×M
         -3.99728,  # Sex×A
          3.442527, # Sex×F
          2.593731, # Sex×D
         -0.06053,  # BH×A
         -0.1373,   # BH×D
          0.123601, # BW×M
          0.242496, # BW×F
          0.120693, # BW×D
         -0.1649,   # M×A
          0.922464, # M×F
          2.596952, # M×D
         -0.09849,  # A×F
         -0.06065,  # A×D
          0.026435, # F×D
    ])
    terms = np.array([1, sex, BH, BW, M, A, F, D,
                      SxBW, SxM, SxA, SxF, SxD,
                      BHxA, BHxD, BWxM, BWxF, BWxD,
                      MxA, MxF, MxD, AxF, AxD, FxD])
    l5s1_comp = np.dot(coef, terms)

    # ── L5-S1 Shear ─────────────────────────────────────────────────────────
    coef = np.array([
         41.2652,   # constant
        -19.0914,   # sex
         -0.61484,  # BH
          4.365337, # BW
          4.1331,   # M
         12.15568,  # A
         -3.88622,  # F
          9.048787, # D
          1.425617, # Sex×M
         -1.18111,  # Sex×A
          0.911929, # Sex×F
          0.809314, # Sex×D
         -0.02459,  # BH×A
          0.025913, # BH×F
         -0.05908,  # BH×D
          0.017099, # BW×A
          0.055798, # BW×F
          0.038626, # BW×D
         -0.04489,  # M×A
          0.220514, # M×F
          1.00079,  # M×D
         -0.0663,   # A×F
         -0.0125,   # A×D
         -0.01553,  # F×D
    ])
    terms = np.array([1, sex, BH, BW, M, A, F, D,
                      SxM, SxA, SxF, SxD,
                      BHxA, BHxF, BHxD, BWxA, BWxF, BWxD,
                      MxA, MxF, MxD, AxF, AxD, FxD])
    l5s1_shear = np.dot(coef, terms)

    return {
        "L4L5_compression": l4l5_comp,
        "L4L5_shear":       l4l5_shear,
        "L5S1_compression": l5s1_comp,
        "L5S1_shear":       l5s1_shear,
    }



# ── Example ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = compute_spinal_loads_standing(sex=0, BH=175, BW=80, M=50, A=30, D=10)
    for label, val in results.items():
        print(f"{label}: {val:.2f} N")
    
    print("= = = = = = =")

    results = compute_spinal_loads_flex(sex=0, BH=175, BW=80, M=50, A=30, F=5, D=10)
    for label, val in results.items():
        print(f"{label}: {val:.2f} N")