"""
CODE FIABILITE - VERSION AVEC DEFINITION DE FONCTIONS
"""
import os
import json
import shutil
import re

from STRAINS.rupt.APIs.CetCAD_API import *
from STRAINS.rupt.APIs import CetLOAD
from STRAINS.rupt.APIs.CetLOAD_API import *
import STRAINS.rupt.core.CetMESH as CetMESH
from STRAINS.rupt.core import CetSOLV as CetSOLV
from STRAINS.rupt.core import CetVISU as CetVISU, CetLIST as CetLIST
from STRAINS.rupt.APIs.CetNOTE_API import *
from STRAINS.rupt.APIs import CetNOTE


def getFile(nameFile):
    f = open(nameFile, 'r')
    res = f.read()
    f.close()
    return res


catalogTopo = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogTopo.json")
catalogDimensions = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogDimensions.json")
catalogBolt = getFile("C:\\workspace\\front\\STRAINS\\common\\Catalog\\CatalogBolts.json")
INITCATALOG(catalogTopo, catalogDimensions, catalogBolt)

import openturns as ot
import numpy as np
import autograd.numpy as anp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from smt.surrogate_models import GEKPLS
from scipy.optimize import brentq
import re
import math
from sklearn.cluster import DBSCAN
from scipy.stats import norm
from math import comb
import warnings
from datetime import datetime
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck


def _parse(text, name):
    return float(re.search(rf'(?m)^\s*{re.escape(name)}\s*=\s*([\d.]+)', text).group(1))

if __name__ == '__main__':
    modelname = "test_pure_flexion"
    _path_ds = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
    with open(os.path.join(_path_ds, 'dsCad.txt'), 'r') as f:
        _cad_txt = f.read()

    print("=" * 70)
    print("CALCUL DE FIABILITE -- FLEXION PURE BETON")
    print("=" * 70)
    # --------------------------------------------------------------------------- #
    # OPTIONS UTILISATEUR                                                         #
    # --------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------- #
    # DEFINITION DU MODELE                                                        #
    modele = 'GEPCK'                 #options: 'GEPCK', 'PCKRG', 'KRG', 'GEK', 'HF'
    do_EFF = True                              #si on veut enrichir progressivement 
    do_IS   = True                            #si on veut calculer la proba globale 

    n0 = 5                      #nombre de points du plan d'expérience initial (DOE)
    params_names = ['fc','fy']
    n_var = len(params_names)

    # --------------------------------------------------------------------------- #G
    # CARACTERISTIQUES DU MODELE                                                  #

    # --- Paramètres variables ---
    fcm, fym = 48, 550 #MPa
    cov_fc, cov_fy = 0.12, None
    fc_otparams, fy_otparams = (fcm,cov_fc), (fym, cov_fy)
    
    
    n_rebars = len(re.findall(r'REBAR\(', _cad_txt))
    rebar_names = [f"HA{i+1}" for i in range(n_rebars)]


    # --------------------------------------------------------------------------- #
    # PARAMETRES FORM                                                             #
    n_max_FORM = 50
    do_multistart = True #multistart : FORM depuis n0 points + [0,0]
    do_warmstart = False #warmstart : si FORM ne converge pas, on repart du best pt

    tol_FORM = 1.0                 # précision acceptée par FORM pour l'état limite
    tol_all_modes = 0.9                            #distance DBSCAN entre deux modes
    tol_warmstart = 0.2 # fixe la nécessité de faire le warm_start si do_warm_start

    # --------------------------------------------------------------------------- #
    # PARAMETRES IS                                                               #
    n_IS    = 10000                                       # taille échantillon IS
    cov_IS  = 0.05                                             # critère d'arrêt COV

    # --------------------------------------------------------------------------- #
    # PARAMETRES MESH                                                             #
    global_size     = 0.05   # global_physical_size (0.05 = rapide FORM, 0.007 = très fin)
    geo_min_approx  = 4      # geometric_approximation_min (4 = fin, 35 = grossier)

    # --------------------------------------------------------------------------- #
    # PARAMETRES GEK/ PCE/ EFF                                                    #
    # 1. GEK
    do_analytic_grad = False
    reduc_PLS = 0

    # 2. PCE
    seuil_pce = 0.90                              # seuil de validation de l'erreur
    q = 0.75                                              # tri base poly candidats
    max_degree = 0     # (init 0, varie en fonction de n0) degre max base candidats
    max_of_maxdegree = 2                                # (fixe) degre max autorisé

    # 3. EFF
    epsilon_factor = 2                               # eps = epsilon_factor * sigma
    tol_EFF = 8e-3                                            # critere d'arret EFF
    tol_BB       = 0.01         # critere BB : |beta_IS_sup - beta_IS_inf| / beta_IS
    tol_BS       = 0.005        # critere BS : |beta_IS - beta_IS_prec| / beta_IS
    EFF_criteria = 'both'       # critere d'arret EFF : 'BB' | 'BS' | 'both'
    u1_eff_min, u1_eff_max = -7.5, 7.5
    u2_eff_min, u2_eff_max = -7.5, 7.5
    n_max_EFF = 40
    print_EFF_progres = True                  # True = prints debug EFF a chaque iter

    # --------------------------------------------------------------------------- #
    # PARAMETRES ET OPTIONS DE PRINT                                              #
    
    # Paramètres de print ---
    u1_max = 7.5
    u2_max = 7.5
    u1_min = -7.5
    u2_min = -7.5
    n_grid = 300
    n_grid_hf = 7

    # --- Options de print ---
    print_HF = True
    print_DOE = True
    print_3D = False
    
    # --- Print facultatifs, par défaut à False ---
    print_ana = True    #True que pour pure_flexion (flexion_claude que dans ce cas) 
    print_grad_sp = False #option si on veut afficher les gradients des points de départ


    

    # --- Résultats fixés ---
    hf_3d_grid_fixed = {
        'params': (-10.0, 10.0, -10.0, 10.0, 7),
        'Z': [
            [-0.359874, -0.266997, -0.205356, -0.162941, -0.133274, -0.118529, -0.110761],
            [-0.259757, -0.080302,  0.041437,  0.123736,  0.180376,  0.218425,  0.237580],
            [-0.224965,  0.048410,  0.247795,  0.382025,  0.475826,  0.541447,  0.578053],
            [-0.196753,  0.115993,  0.413867,  0.617429,  0.752362,  0.844787,  0.910014],
            [-0.168402,  0.147183,  0.541008,  0.821024,  1.009377,  1.139576,  1.232379],
            [-0.140437,  0.175700,  0.624863,  0.998493,  1.249463,  1.421116,  1.540467],
            [-0.112244,  0.204017,  0.672091,  1.150387,  1.477693,  1.691754,  1.840008],
        ]
    }
    hf_2d_grid_fixed = None
    # guide pour hardcoder:  {'params': {'u1_min':-10.0,'u1_max':10.0,'u2_min':-10.0,'u2_max':10.0,'n_grid_hf':7}, 'Z':[[-0.35987397002878807, -0.2669967363774981, -0.20535573020711573, -0.1629405668993904, -0.1332743271859559, -0.11852860036358548, -0.11076112318802811], [-0.25975747046282116, -0.08030213317916757, 0.04143746766061329, 0.12373557540792612, 0.18037582744039926, 0.21842464629970904, 0.23757988854578782], [-0.2249654139385301, 0.04841011473603918, 0.24779511664467302, 0.38202490257144284, 0.47582561090667097, 0.5414471308620206, 0.5780528850246671], [-0.19675328364698386, 0.11599332588276412, 0.4138665193031785, 0.6174288511497759, 0.7523619015524756, 0.8447865997026027, 0.91001418635111], [-0.1684019331521559, 0.14718304332768306, 0.5410080650758031, 0.8210242484756942, 1.009376895847256, 1.1395757360046685, 1.2323790279345541], [-0.1404367138757393, 0.17570029864525294, 0.6248628650465244, 0.9984931859001764, 1.2494627601162334, 1.4211157981633327, 1.5404671177896931], [-0.11224374227096778, 0.20401650921121672, 0.6720914036748573, 1.1503870059250554, 1.477692628590619, 1.6917535476874397, 1.8400075254424433]]}


    # --- Résultats fixés du run HF 12/05 (gamma=1.0, F=0.74, n0=15) ---
    # Actifs uniquement en mode visu seule (tous do_* = False).
    if modele == None:
        sol_modes_fixed = None
        # guide pour hardcoder {
        #     # (sp_u1, sp_u2): (u*_u1, u*_u2)
        #     (-0.002,  1.332): (-5.306, -6.200),
        #     ( 0.610, -0.310): (-3.117, -7.349),
        #     (-0.571, -1.705): (-3.131, -7.347),
        #     ( 0.258,  0.306): (-4.655, -6.643),
        #     ( 0.121, -1.010): (-3.117, -7.345),
        #     ( 1.624, -0.531): (-3.046, -7.352),
        #     (-0.087, -0.172): (-4.721, -6.603),
        #     (-0.419,  0.597): (-5.290, -6.212),
        #     (-0.843, -0.005): (-5.341, -6.152),
        #     ( 0.868, -1.460): (-3.014, -7.363),
        #     (-1.681,  0.710): (-6.475, -4.986),
        #     ( 1.479,  0.239): (-3.098, -7.354),
        #     (-1.117, -0.696): (-5.200, -6.275),
        #     ( 0.745,  1.043): (-4.740, -6.571),
        #     (-0.694,  2.114): (-6.504, -4.966),
        #     ( 0.000,  0.000): (-4.776, -6.571),
        # }
        best_sol_modes_fixed = None
        # guide pour hardcoder {
        #     'A': {'sp': ( 0.868, -1.460), 'u*': (-3.014, -7.363)},
        #     'B': {'sp': ( 0.745,  1.043), 'u*': (-4.740, -6.571)},
        #     'C': {'sp': (-0.843, -0.005), 'u*': (-5.341, -6.152)},
        #     'D': {'sp': (-0.694,  2.114), 'u*': (-6.504, -4.966)},
        # }
        # Gradients HF aux sp (run 1305_0937, 4 appels STRAINS)
        grad_sp_fixed = None
        # guide pour hardcoder {
        #     'A': {'g': 0.550023, 'grad': [ 0.039500,  0.072312], 'neg_grad': [-0.039500, -0.072312]},
        #     'B': {'g': 0.722565, 'grad': [ 0.046148,  0.068963], 'neg_grad': [-0.046148, -0.068963]},
        #     'C': {'g': 0.573155, 'grad': [ 0.058216,  0.059250], 'neg_grad': [-0.058216, -0.059250]},
        #     'D': {'g': 0.704360, 'grad': [ 0.068462,  0.055309], 'neg_grad': [-0.068462, -0.055309]},
        # }
        # Trajectoires FORM hardcodees (run 1805_1957, do_HF=True, n0=7)
        # Une trajectoire representative par mode : points et gradients successifs AbdoRackwitz
        traj_runs_fixed = None
        # guide pour hardcoder {
        #         'A': {  # u* ~ [-3.12, -7.35]  (26 pts, sp=[0.61,-0.31])
        #             'points': [
        #                 [ 0.6102, -0.3098], [-3.8707, -6.4583], [-1.6303, -3.3841], [-0.5101, -1.8470],
        #                 [ 0.0501, -1.0784], [-3.8803, -6.5921], [-1.9151, -3.8353], [-0.9325, -2.4568],
        #                 [-0.4412, -1.7676], [-3.8389, -6.7200], [-2.1401, -4.2438], [-1.2907, -3.0057],
        #                 [-0.8659, -2.3867], [-3.7616, -6.8474], [-2.3138, -4.6171], [-1.5899, -3.5019],
        #                 [-1.2279, -2.9443], [-3.6819, -6.9516], [-2.4549, -4.9479], [-1.8414, -3.9461],
        #                 [-3.5392, -7.1070], [-2.6903, -5.5265], [-2.2659, -4.7363], [-3.3755, -7.2150],
        #                 [-2.8207, -5.9757], [-3.1170, -7.3495],
        #             ],
        #             'grads': [
        #                 [0.042205, 0.070421], [0.031332, 0.065618], [0.035786, 0.068399], [0.039346, 0.069164],
        #                 [0.041015, 0.069679], [0.029800, 0.066665], [0.034813, 0.068268], [0.037701, 0.069006],
        #                 [0.039546, 0.069225], [0.028110, 0.067965], [0.033785, 0.068334], [0.036434, 0.068843],
        #                 [0.037930, 0.069045], [0.027391, 0.068676], [0.032755, 0.068537], [0.035300, 0.068765],
        #                 [0.036516, 0.068944], [0.027059, 0.069116], [0.031878, 0.068754], [0.034250, 0.068777],
        #                 [0.026583, 0.069808], [0.030447, 0.069107], [0.032261, 0.068955], [0.025926, 0.070694],
        #                 [0.029462, 0.069467], [0.025199, 0.071834],
        #             ],
        #         },
        #         'B': {  # u* ~ [-4.66, -6.64]  (50 pts, sp=[0.26,0.31])
        #             'points': [
        #                 [ 0.2576,  0.3059], [-4.1149, -6.3330], [-1.9286, -3.0135], [-0.8355, -1.3538],
        #                 [-0.2890, -0.5239], [-4.1318, -6.4693], [-2.2104, -3.4966], [-1.2497, -2.0103],
        #                 [-0.7693, -1.2671], [-0.5291, -0.8955], [-0.4091, -0.7097], [-4.1355, -6.4913],
        #                 [-2.2723, -3.6005], [-1.3407, -2.1551], [-0.8749, -1.4324], [-0.6420, -1.0711],
        #                 [-0.5255, -0.8904], [-4.1495, -6.5088], [-2.3375, -3.6996], [-1.4315, -2.2950],
        #                 [-0.9785, -1.5927], [-0.7520, -1.2416], [-0.6388, -1.0660], [-4.1748, -6.5169],
        #                 [-2.4068, -3.7914], [-1.5228, -2.4287], [-1.0808, -1.7473], [-0.8598, -1.4067],
        #                 [-4.2403, -6.5157], [-2.5501, -3.9612], [-1.7049, -2.6839], [-1.2823, -2.0453],
        #                 [-1.0710, -1.7260], [-4.3300, -6.4854], [-2.7005, -4.1057], [-1.8858, -2.9158],
        #                 [-1.4784, -2.3209], [-4.5085, -6.4242], [-2.9935, -4.3726], [-2.2359, -3.3467],
        #                 [-1.8572, -2.8338], [-4.6009, -6.4217], [-3.2291, -4.6278], [-2.5431, -3.7308],
        #                 [-4.7317, -6.4391], [-3.6374, -5.0850], [-3.0903, -4.4079], [-4.7438, -6.4980],
        #                 [-3.9170, -5.4529], [-4.6552, -6.6434],
        #             ],
        #             'grads': [
        #                 [0.044519, 0.068517], [0.035878, 0.061726], [0.044263, 0.062955], [0.043386, 0.066384],
        #                 [0.043289, 0.067779], [0.034697, 0.062504], [0.042966, 0.062862], [0.043945, 0.065014],
        #                 [0.043028, 0.066722], [0.042947, 0.067351], [0.043059, 0.067588], [0.034527, 0.062612],
        #                 [0.042687, 0.062843], [0.043966, 0.064760], [0.043206, 0.066373], [0.042968, 0.067062],
        #                 [0.042943, 0.067361], [0.034494, 0.062596], [0.042443, 0.062793], [0.044100, 0.064442],
        #                 [0.043438, 0.065989], [0.042999, 0.066778], [0.042969, 0.067074], [0.034637, 0.062421],
        #                 [0.042247, 0.062703], [0.044270, 0.064105], [0.043695, 0.065590], [0.043215, 0.066405],
        #                 [0.035292, 0.061763], [0.042202, 0.062322], [0.044554, 0.063438], [0.044217, 0.064786],
        #                 [0.043780, 0.065573], [0.036382, 0.060718], [0.042725, 0.061569], [0.044644, 0.062870],
        #                 [0.044856, 0.063916], [0.037854, 0.059111], [0.043326, 0.060308], [0.044505, 0.061920],
        #                 [0.044972, 0.062769], [0.038234, 0.058548], [0.043058, 0.059735], [0.044734, 0.060876],
        #                 [0.038627, 0.057857], [0.042160, 0.058993], [0.043652, 0.059794], [0.038325, 0.058030],
        #                 [0.041163, 0.058744], [0.037124, 0.059166],
        #             ],
        #         },
        #         'C': {  # u* ~ [-5.31, -6.20]  (12 pts, sp=[-0.00,1.33])
        #             'points': [
        #                 [-0.0017,  1.3325], [-5.0636, -5.2359], [-2.5327, -1.9517], [-5.5307, -5.5909],
        #                 [-4.0317, -3.7713], [-3.2822, -2.8615], [-5.5130, -5.8096], [-4.3976, -4.3356],
        #                 [-3.8399, -3.5985], [-5.4580, -5.9699], [-4.6489, -4.7842], [-5.3056, -6.1998],
        #             ],
        #             'grads': [
        #                 [0.059352, 0.061371], [0.046179, 0.051721], [0.054808, 0.055405], [0.045432, 0.050529],
        #                 [0.048982, 0.053557], [0.051740, 0.054524], [0.044423, 0.051269], [0.047496, 0.053207],
        #                 [0.049358, 0.053988], [0.042249, 0.052934], [0.045724, 0.053430], [0.040682, 0.054516],
        #             ],
        #         },
        #         'D': {  # u* ~ [-6.48, -4.99]  (13 pts, sp=[-1.68,0.71])
        #             'points': [
        #                 [-1.6808,  0.7104], [-5.7529, -4.8956], [-3.7168, -2.0926], [-2.6988, -0.6911],
        #                 [-6.0548, -4.8684], [-4.3768, -2.7798], [-3.5378, -1.7354], [-6.1953, -4.9797],
        #                 [-4.8666, -3.3576], [-4.2022, -2.5465], [-6.3582, -4.9252], [-5.2802, -3.7358],
        #                 [-6.4751, -4.9861],
        #             ],
        #             'grads': [
        #                 [0.063569, 0.054096], [0.052114, 0.045396], [0.059464, 0.048913], [0.063145, 0.050772],
        #                 [0.053159, 0.043559], [0.059506, 0.046396], [0.060808, 0.048877], [0.052863, 0.043191],
        #                 [0.058439, 0.045099], [0.060250, 0.046670], [0.054448, 0.041540], [0.057304, 0.044126],
        #                 [0.054778, 0.040852],
        #             ],
        #         },
        #     }
    
    else:
        sol_modes_fixed = None
        best_sol_modes_fixed = None
        grad_sp_fixed = None
        traj_runs_fixed = None
    

    # --- Label PCE GEPCK et LOO (mis a jour par init_g_ot, lus par print_planche_EFF) ---
    _gepck_pce_label = ""
    _gepck_loo       = None

    # --- Historiques EFF (mis a jour par run_EFF et init_g_ot, lus par print_EFF_graphs) ---
    _eff_history_EFF   = []   # EFF(u_opt) avant ajout de chaque point (incl. initial)
    _eff_history_BB    = []   # ratio BB par iteration (None si FORM echoue)
    _eff_history_BS    = []   # ratio BS par iteration (None si calcul impossible)
    _eff_history_theta = []   # theta Kriging [theta_0,...,theta_{M-1}] apres chaque fit

    # --- Sortie PNG EFF ---
    timestamp   = datetime.now().strftime('%d%m_%H%M')
    out_dir_eff = r'C:\_workingDir\_SF\test flexion\output\png EFF'

    do_KRG = True if modele == 'KRG' else False
    do_GEK = True if modele == 'GEK' else False #on ajoute peut etre plus de points avec GEK car plus précis donc voit plus derreur
    do_HF = True if modele == 'HF' else False # penser à jouer avec des bornes différentes
    do_PCKRG = True if modele == 'PCKRG' else False
    do_old_GEPCK = True if modele == 'old_GEPCK' else False
    do_GEPCK     = True if modele == 'GEPCK'     else False
    do_IS   = do_IS and modele != 'HF'                        # IS impraticable en HF
    do_EFF   = do_EFF and modele != 'HF'                     # EFF impraticable en HF

    # --------------------------------------------------------------------------- #
    # DEFINTION DE FONCTIONS                                                      #
    # --------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------- #
    # FONCTION D'APPEL STRAINS ET DOE                                             #

    # --- DSCAD ET DSLOAD ---
    def patch_params(path, **params):
        """Reecrit dsCad.txt avec de nouvelles valeurs de parametres."""
        cad = os.path.join(path, 'dsCad.txt') #donne un nom au txt
        with open(cad, 'r') as f: #on stocke son contenu
            content = f.read()
        for name, value in params.items(): #on le modifie variable par variable pour celles dans la liste params
            content = re.sub(r'^' + name + r'\s*=.*$', f'{name}    = {value:.10f}', content, count=1, flags=re.MULTILINE)
        with open(cad, 'w') as f:
            f.write(content) #on l'écrit dans un fichier vide f (car 'w' donc vidé) - dsCad.txt est modifié
        # A COMPLETER AVEC COPIE COLLE DE CA AVEC DSLOAD QUAND ON AJOUTE LES LOADS MODIFIES.

    # --- DISTRIBUTIONS ---
    
    SIGMA_11, SIGMA_12, SIGMA_13 =  19.0, 22.0, 8.0 
    SIGMA = np.sqrt(SIGMA_11**2 + SIGMA_12**2 + SIGMA_13**2)  # ~30 MPa
    
    def loi_fy(fym, cov=None):
        if cov is not None:
            sig_ec = cov * fym
        else:
            sig_ec = SIGMA

        dist = ot.Normal(fym, sig_ec)
        return dist
    
    def loi_fc(fcm, cov=None):
        COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
        fck_eq = fcm - 8.0
        classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck_eq))
        v = cov if cov is not None else COV_TABLE[classe]

        sigma_ln = np.sqrt(np.log(1 + v**2))
        mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2

        dist = ot.LogNormal(mu_ln, sigma_ln, 0.0)
        return dist

    def dist_jointe():
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(fcm, cov_fc)) 
        if 'fy' in params_names:
            dist.append(loi_fy(fym, cov_fy))
        #AJOUTER suite pour plus de variable 'if 'load' in params_names' etc.
        dist_X   = ot.JointDistribution(dist)
        return dist_X

    # --- APPELS STRAINS ---
    def run_one_SOL(modelname, SOL, params_names, sensitivity=False, with_sens_dict=None): 
        """Lance un calcul complet pour une valeur de FT donnee.
        Retourne la liste des solutions pour chaque jeu de variables dans SOL (liste de dictionnaire)"""
        path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        #MODIF 1 10/04 - on doit tout mettre dans params in SOL. TOUT.
        for i in range (len(SOL)): 
            patch_params(path, **SOL[i]) #à cette étape SOL ne contient que 'fc': ,'fy':
            model = MODEL() #ici model n'est pas encore rempli
            SET_CONTEXT(model, path)
            fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)

            cadfile = open(path + '\\dsCad.txt', 'r')
            cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
            exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
            model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
            print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

            loadfile = open(path + '\\dsLoad.txt', 'r')
            model.Load(fileName) #on remplit le modèle en lisant .dscad et ainsi l'utiliser avec LOAD_MODEL plus bas. 
            loadscript = loadfile.read() 
            with CetLOAD.LOAD_MODEL(model, path): #par with on appelle enter et exit et on force l'enregistrement par exit meme si erreur/ bug dans bloc.
                exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire

            Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
                "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
                "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
                "global_physical_size": global_size,
                "max_size": 0.05,
                "min_size": "-1",
                "gradation": 1.5,
                "volume_gradation": 1.5,
                "optimisation_level": "standard",
                "anisotropic_ratio": "10",
                "geometric_approximation_min": str(geo_min_approx),
                "geometric_approximation_max": "25",
                "geometric_approximation_on_edge": "false",
                "geometric_approximation_on_face": "true",
                "use_surface_proximity": "false",
                "surface_proximity_ratio": 0,
                "approach": "kinematic",
                "write_debug_files": "true",
                "is_iso": "true",
                "coeff_on_error": 0.01,
                "remesh_type": 1,
                "old_size_factor": 0.0,
            }
            CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

            kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
            exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals()) #question pour Agnes : je ne suis pas sure que ca marche comme ca. 
            kwargs["static_params"] = static_params
            kwargs["cinematic_params"] = cinematic_params
            kwargs["MKLPardiso_params"] = MKLPardiso_params
            kwargs["MyPardiso_params"] = MyPardiso_params
            kwargs["MUMPS_params"] = MUMPS_params
            kwargs["FullLorentz"] = False
            kwargs["LorentzToSdp"] = False
            kwargs["SdpToLorentz"] = 0
            kwargs["printIntPointSolutioEvolution"] = False
            kwargs["trace_sur_point_integration"] = False
            kwargs["calculate_error"] = "false"
            kwargs["max_nbOfDiv"] = 0
            kwargs["customized_inc"] = [1]
            kwargs["tetra_discontinuities"] = False
            kwargs["activated_plasticity"] = True
            kwargs["welds_throat_limit"] = True
            kwargs["approach"] = "kinematic"

            if sensitivity:
                kwargs["sensitivity_analysis"] = "true"
                kwargs["sensitivity_regions"] = json.dumps([
                    {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
                    {"param": "YIELD_STRENGTH", "rebars": rebar_names},
                ]) #transformée en texte json (liste de caractères) pour être lisible par C++

            CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.

            # Lire le resultat
            metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
            with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
                d = json.load(f) #chargement du fichier .dsmetares
            SOL[i]['g']=d['info']['Primal_bound'][0] -1
            for p in params_names:
                SOL[i][f'dg_{p}'] = None
            if sensitivity and 'Sensitivity' in d['info']:
                print(f"les sensibilités sont calculées pour les elements : {d['info']['Sensitivity'].items()}")
                for k, v in d['info']['Sensitivity'].items():
                    #je ne sais pas encore comment généraliser pour le code ci dessous donc je vais juste
                    #faire if 1, if 2, mais on devrait faire une double boucle, mais la question est comment
                    #on définit la liste des noms 'tensile_strength' etc. Voir dans dsCad. 
                    if 'COMPRESSIVE_STRENGTH' in k: 
                        SOL[i]['dg_fc']= v
                    if 'YIELD_STRENGTH' in k: 
                        SOL[i]['dg_fy']= v
                    if all(SOL[i].get(f'dg_{p}') is not None for p in params_names):
                        break    
        return SOL

    def run_HF(u):
        sensitivity = True
        n_var = len(u)
        dist_X = dist_jointe()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation() 
        u_point = ot.Point(u)
        x_point = T_inv(u_point)
        path = "C:\\workspace\\storage\\admin\\SF\\" + modelname + ".ds"
        AnalysisName = 'Yield_analysis0'
        iteration = 0
        params={params_names[i]: x_point[i] for i in range(n_var)}
        patch_params(path, **params) #à cette étape SOL ne contient que 'fc': ,'fy':
        model = MODEL() #ici model n'est pas encore rempli
        SET_CONTEXT(model, path)
        fileName = os.path.join(path, AnalysisName + ".dscad") #on crée le chemin du fichier disque .dscad lisible par C. C va tout faire et on renverra les info plus tard (.load)

        cadfile = open(path + '\\dsCad.txt', 'r')
        cadscript = cadfile.read() #on met dans cadscript les info de dsCad.txt
        exec(cadscript, globals()) # ici on modifie le modèle (C, cython) et donc les variables (on exécute le script de dsCad.txt ce qui modifie les variables - rien dans .dscad, tout dans var. en mémoire)
        model.Save(fileName) # ici on créé dscad et on enregistre les modifs des variables dans .dscad
        print(model.GETERRORS()) # est vide si pas de message d'erreur sur le logiciel

        loadfile = open(path + '\\dsLoad.txt', 'r')
        model.Load(fileName) #on remplit le modèle en lisant .dscad et ainsi l'utiliser avec LOAD_MODEL plus bas. 
        loadscript = loadfile.read() 
        with CetLOAD.LOAD_MODEL(model, path): #par with on appelle enter et exit et on force l'enregistrement par exit meme si erreur/ bug dans bloc.
            exec(loadscript, globals()) # pareil, on execute dsLoad et on enregistre dans var. mémoire

        Meshkwargs = { #définit la mesh - pas à comprendre ici car ne sera pas modifié. 
            "cadSurfOptions": {"volume_gradation": 1.5, "gradation": 1.5, "anisotropic_ratio": 10},
            "tetraOptions": {"optimisation_level": "standard", "verbose": "10"},
            "global_physical_size": 0.05,  # mesh fin pour bonne convergence
            "max_size": 0.05,
            "min_size": "-1",
            "gradation": 1.5,
            "volume_gradation": 1.5,
            "optimisation_level": "standard",
            "anisotropic_ratio": "10",
            "geometric_approximation_min": "4",
            "geometric_approximation_max": "25",
            "geometric_approximation_on_edge": "false",
            "geometric_approximation_on_face": "true",
            "use_surface_proximity": "false",
            "surface_proximity_ratio": 0,
            "approach": "kinematic",
            "write_debug_files": "true",
            "is_iso": "true",
            "coeff_on_error": 0.01,
            "remesh_type": 1,
            "old_size_factor": 0.0,
        }
        CetMESH.ANISO_MESH(AnalysisName, iteration, path, **Meshkwargs)

        kwargs = {"scaling": 1, "write_debug_files": "true"} # ci-dessous on définit dict kwargs en entrée de SOLV.
        exec(open(r"C:\_workingDir\_SF\test flexion\InitSolver.py").read(), globals()) #question pour Agnes : je ne suis pas sure que ca marche comme ca. 
        kwargs["static_params"] = static_params
        kwargs["cinematic_params"] = cinematic_params
        kwargs["MKLPardiso_params"] = MKLPardiso_params
        kwargs["MyPardiso_params"] = MyPardiso_params
        kwargs["MUMPS_params"] = MUMPS_params
        kwargs["FullLorentz"] = False
        kwargs["LorentzToSdp"] = False
        kwargs["SdpToLorentz"] = 0
        kwargs["printIntPointSolutioEvolution"] = False
        kwargs["trace_sur_point_integration"] = False
        kwargs["calculate_error"] = "false"
        kwargs["max_nbOfDiv"] = 0
        kwargs["customized_inc"] = [1]
        kwargs["tetra_discontinuities"] = False
        kwargs["activated_plasticity"] = True
        kwargs["welds_throat_limit"] = True
        kwargs["approach"] = "kinematic"

        if sensitivity:
            kwargs["sensitivity_analysis"] = "true"
            kwargs["sensitivity_regions"] = json.dumps([
                {"param": "COMPRESSIVE_STRENGTH", "solids": ["Block1"]},
                {"param": "YIELD_STRENGTH", "rebars": rebar_names},
            ]) #transformée en texte json (liste de caractères) pour être lisible par C++

        CetSOLV.SOLV(AnalysisName, iteration, path, **kwargs) #On relance le solveur avec le nouveau dsCad.

        # Lire le resultat
        metares_path = os.path.join(path, AnalysisName + "_0_kine.dsmetares") #on extrait l'addresse du fichier pour définir f
        with open(metares_path, 'r') as f: #f est le fichier créé par open, et on a with donc enter de fichier = donne accès au fichier (accès via f, toujours mettre as f) puis exit : ferme le fichier (qui reste lié à f)
            d = json.load(f) #chargement du fichier .dsmetares
        g_HF=d['info']['Primal_bound'][0] -1
        grad_HF_X=[None]*n_var
        grad_HF_U=[None]*n_var
        if sensitivity and 'Sensitivity' in d['info']:
            print(f"les sensibilités sont calculées pour les elements : {d['info']['Sensitivity'].items()}")
            for k, v in d['info']['Sensitivity'].items():
                #je ne sais pas encore comment généraliser pour le code ci dessous donc je vais juste
                #faire if 1, if 2, mais on devrait faire une double boucle, mais la question est comment
                #on définit la liste des noms 'tensile_strength' etc. Voir dans dsCad. #faudrait un truc avec des clés et des asocciations officielels entre fc et compressive strength.... 
                if 'COMPRESSIVE_STRENGTH' in k: 
                    grad_HF_X[params_names.index('fc')] = v
                if 'YIELD_STRENGTH' in k: 
                    grad_HF_X[params_names.index('fy')] = v
                if all(grad_HF_X[i] is not None for i in range(n_var)):
                    break
            J_Tinv = T_inv.gradient(u)
            J_Tinv_T = J_Tinv.transpose()
            grad_HF_U = J_Tinv_T * ot.Point(grad_HF_X)
        if sensitivity and any(v is None for v in grad_HF_U):
            raise ValueError(f"run_HF : sensibilité demandée mais grad_HF_U contient None — vérifier que STRAINS a bien calculé les sensibilités. grad_HF_X={grad_HF_X}")
        return g_HF, grad_HF_U, grad_HF_X

    # --- DOE ---
    def build_DOE():
        dist = []
        if 'fc' in params_names:
            dist.append(loi_fc(fcm, cov_fc))
        if 'fy' in params_names:
            dist.append(loi_fy(fym, cov_fy))
        dist_X   = ot.JointDistribution(dist) 
        T     = dist_X.getIsoProbabilisticTransformation() # on interroge dist_X et trouve la transfo n�cessaire puis l'applique ici
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        dist_U = dist_X.getStandardDistribution()
        lhs    = ot.LHSExperiment(dist_U, n0)
        sa     = ot.SimulatedAnnealingLHS(lhs, ot.SpaceFillingMinDist())
        U_doe  = sa.generate()
        if print_DOE:
                print("U_doe_fixed = ot.Sample([")
                for i in range(U_doe.getSize()):
                    vals = []
                    for j in range(U_doe.getDimension()):
                        v = U_doe[i][j]
                        vals.append(f' {v:.16f}' if v >= 0 else f'{v:.16f}')
                    print(f"    [{', '.join(vals)}],")
                print("])", flush=True)
        X_doe  = T_inv(U_doe)
        xt = np.array(U_doe)
        if not do_HF:
            SOL = [{} for _ in range(n0)] 
            for i in range(n0):
                for j in range(n_var):
                    SOL[i][params_names[j]] = X_doe[i][j]
            SOL = run_one_SOL(modelname, SOL, params_names, sensitivity=True, with_sens_dict=None)
            yt = np.array([SOL[i]['g'] for i in range(n0)]).reshape(-1, 1)
            all_grad = np.zeros((n0, n_var))
            for i in range (n0):
                J_Tinv = T_inv.gradient(U_doe[i])
                J_Tinv_T = J_Tinv.transpose()
                grad_X_g = ot.Point([SOL[i][f'dg_{p}'] for p in params_names])
                grad_U_g = J_Tinv_T * grad_X_g
                for j in range (n_var):
                    all_grad[i][j]= grad_U_g[j]
                    SOL[i][f'dg_u{j+1}'] = grad_U_g[j]
            if print_DOE:
                print("yt_doe = [")
                for i in range(n0):
                    print(f"    {yt[i][0]:.16f},")
                print("]", flush=True)
            return xt, yt, all_grad
        return xt

    def build_Y_aug(yt, all_grad):
        """
        Construit le vecteur gradient-enhanced y_dot (eq. 6 Zuhal et al.).
        y_dot = [y^1,...,y^n, dg/du1^1,...,dg/du1^n, ..., dg/dum^1,...,dg/dum^n]^T
        Shape : (n0*(1+n_var),)
        """
        y_flat      = yt.flatten()                                         # (n0,)
        grad_blocks = [all_grad[:, j] for j in range(all_grad.shape[1])]  # n_var blocs de (n0,)
        return np.concatenate([y_flat] + grad_blocks)                      # (n0*(1+n_var),)

    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE DE REFERENCE                                            #
    # --------------------------------------------------------------------------- #
    # FONCTION ANALYTIQUE                                                         #
    # --- Paramètres du modèle analytique ---
    Es = 200000
    ecu = 0.0035
    eud = 0.045
    gamma_c_fic = _parse(_cad_txt, 'gamma_c') # fixé à 1.0
    gamma_s_fic = _parse(_cad_txt, 'gamma_s') # fixé à 1.0
    
    # --- Fonction analytique ---
    class flexion_claude:
        def __init__(self):

            # --- Lecture du dsCad et dsLoad ---
            path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
            with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
                _cad = f.read()
            with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
                _load = f.read()

            b   = _parse(_cad, 'b')
            h   = _parse(_cad, 'h')
            L   = _parse(_cad, 'L')
            phi = _parse(_cad, 'phi')

            n_bars = len(re.findall(r'REBAR\(', _cad))
            As = n_bars * math.pi * (phi / 2e3) ** 2

            z_rebar = [float(v) for v in re.findall(
                r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
            d = h/2 + sum(z_rebar) / len(z_rebar)

            F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
            Med = F * L

            # --- Définition de la transformation isoprobabiliste ---
            self.fym = fy_otparams[0]
            dist = []
            if 'fc' in params_names:
                dist.append(loi_fc(*fc_otparams))
            if 'fy' in params_names:
                dist.append(loi_fy(*fy_otparams))
            dist_X     = ot.JointDistribution(dist)
            self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
            self.T     = dist_X.getIsoProbabilisticTransformation()

            # --- Définiton des constantes pour le cas aciers plastifiés ---
            self.A  = As * d / gamma_s_fic
            self.B  = - As**2 * gamma_c_fic / (2 * b * gamma_s_fic**2)
            self.C  = -Med

            # --- Calcul de la limite plastique ---
            self.Ap = 0.8*d*b / (As*gamma_c_fic*Es*ecu)
            self.Bp = 0.8*b*d**2 / gamma_c_fic
            self.Cp = 2*self.Ap*self.C/self.Bp 
            ap = 1
            bp = self.Cp - 0.8
            cp = self.Cp - 0.2
            Delta_p = bp**2 - 4*ap*cp
            sol1_s = (-bp + Delta_p**0.5) / (2*ap)
            sol1_x1 = (sol1_s**2 - 1) / (4*self.Ap)
            self.u1_lim_plast = self.T(ot.Point([sol1_x1, 0.0]))[0]

            # --- Limite de plasticité ---
            self.A1 = As*gamma_c_fic*Es*ecu/(0.8*b*d)
            self.A2 = Es*ecu*gamma_s_fic

        def u2p_LS(self, u1):
            x_point = self.T_inv(ot.Point([u1, 0.0]))
            x1 = x_point[0]
            a  = self.B
            b  = self.A * x1
            c  = self.C * x1
            Delta = b**2 - 4 * a * c
            fy = (-b + Delta**0.5) / (2 * a)
            return self.T(ot.Point([0.0, fy]))[1]

        def g(self, u1, u2):
            x_point = self.T_inv(ot.Point([u1, u2]))
            x1 = x_point[0]
            x2 = x_point[1]
            x1_lim_plast_x2 = self.A1*x2*(self.A2+x2)/self.A2**2
            if x1 > x1_lim_plast_x2:
                return (self.A*x2+self.B*x2**2/x1+self.C)/(-self.C)
            else :
                s = (1 + 4*self.Ap*x1)**0.5
                return -1 - (s-1)/self.Cp + 0.8*(s-1)/(self.Cp*(s+1))
    
    def print_visu_ana():
        calc = flexion_claude()

        u1_lim = calc.u1_lim_plast
        u2_lim = calc.u2p_LS(u1_lim)

        # Branche plastifiée
        u1_grid = np.linspace(u1_lim, u1_max, n_grid)
        u2_grid = np.array([calc.u2p_LS(u) for u in u1_grid])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(u1_grid, u2_grid, 'b-', lw=2,
                label=r'$u_2 = u_{2p,LS}(u_1)$  (aciers plastifiés)')
        ax.plot([u1_lim, u1_lim], [u2_lim, u2_max], 'r-', lw=2,
                label=r'$u_1 = u_{1,lim\,plast}$  (aciers non plastifiés)')
        ax.plot(u1_lim, u2_lim, 'ko', ms=6, zorder=5,
                label=f'Raccord ({u1_lim:.3f}, {u2_lim:.3f})')
        ax.plot(0, 0, 'g+', ms=12, mew=2, label='Origine')

        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlabel(r'$u_1$  (espace standard, $f_c$)')
        ax.set_ylabel(r'$u_2$  (espace standard, $f_y$)')
        ax.set_title("Surface d'etat-limite - flexion pivot B")
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        plt.tight_layout()
        plt.show()
        return fig, ax

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE HF                                                #

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class HFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self):
            super().__init__(n_var, 1)
            self._cache_u    = None
            self._cache_g    = None
            self._cache_grad = None
            self.n_hf_calls  = 0  # compteur pour vérification

        def _run_if_needed(self, u):
            u_arr = np.array(u)
            if self._cache_u is None or not np.allclose(u_arr, self._cache_u, atol=1e-12):
                g, grad_U, _ = run_HF(u)
                self._cache_u    = u_arr.copy()
                self._cache_g    = float(g)
                self._cache_grad = [float(grad_U[i]) for i in range(n_var)]
                self.n_hf_calls += 1
                print(f"[HF #{self.n_hf_calls:3d}] u=[{u_arr[0]:+.4f}, {u_arr[1]:+.4f}]  g={g:+.6f}  grad=[{float(grad_U[0]):+.6f}, {float(grad_U[1]):+.6f}]", flush=True)

        def _exec(self, u):
            self._run_if_needed(u)
            return [self._cache_g]

        def _gradient(self, u):
            self._run_if_needed(u)
            # Format OpenTURNS : (n_var, 1)
            return [[g] for g in self._cache_grad]

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU MODELE PCE                                               #
    def n0_min(n_var, p):
        """Taille minimale du DOE pour PCE de degre p en n_var variables.
        n0_min = C(n_var+p, p) + 1 = (n_var+p)! / (n_var! * p!) + 1"""
        return comb(n_var + p, p) + 1
    
    def update_degree(new_n0):
        global max_degree
        while new_n0 >= n0_min(n_var, max_degree+1) and max_degree+1 <= max_of_maxdegree:
            max_degree += 1
        print(f'On passe à max_degree = {max_degree} car le DOE est de {new_n0} >= n0_min({max_degree}) = {n0_min(n_var, max_degree)}')

    def build_metamodel_PCE(xt, y_hf):
        # 1. INITIALISATION : DOE ET DISTRIBUTION
        inputSample = ot.Sample(xt)
        outputSample = ot.Sample(y_hf)
        n0 = xt.shape[0]
        dist_X = dist_jointe()
        dist_U = dist_X.getStandardDistribution()

        # 2. BASE DE CANDIDATS : TYPE, ENUMERATION, DEGRE
        n_var = inputSample.getDimension()
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(n_var, q)
        basis = ot.OrthogonalProductPolynomialFactory([ot.HermiteFactory()] * n_var, enumerateFunction)
        basis_size = enumerateFunction.getBasisSizeFromTotalDegree(max_degree)
        basisStrategy = ot.FixedStrategy(basis, basis_size)

        # 3. PROPOSITION / PROJECTION / SELECTION
        selectionStrategy = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
        projectionStrategy = ot.LeastSquaresStrategy(selectionStrategy) 

        # 4. RESULTAT
        algo = ot.FunctionalChaosAlgorithm(inputSample, outputSample, dist_U, basisStrategy, projectionStrategy)
        algo.run()
        result = algo.getResult()
        n_active = result.getCoefficients().getSize()
        print(f"PCE construite : basis_size={basis_size}, coefficients actifs LARS={n_active}", flush=True)
        indices = result.getIndices()
        coeffs  = result.getCoefficients()
        terms = []
        for k in range(indices.getSize()):
            mi = enumerateFunction(indices[k])
            a, b = int(mi[0]), int(mi[1])
            if   a == 0 and b == 0: label = "1"
            elif a == 0:            label = f"H{b}(u2)"
            elif b == 0:            label = f"H{a}(u1)"
            else:                   label = f"H{a}(u1)*H{b}(u2)"
            terms.append(f"{coeffs[k, 0]:+.4f}*{label}")
        print(f"  PCE termes : {' '.join(terms)}", flush=True)
        metamodel = result.getMetaModel()
        return metamodel

    def calculate_PCE(xt, y_hf, all_grad_hf, metamodel_PCE):
        U_doe = ot.Sample(xt)                               
        y_PCE = np.array(metamodel_PCE(U_doe))
        n_var = U_doe.getDimension()
        n0 = U_doe.getSize()
        dist_X = dist_jointe()
        T = dist_X.getIsoProbabilisticTransformation()
        T_inv = dist_X.getInverseIsoProbabilisticTransformation()
        all_grad_PCE = np.zeros((n0, n_var))
        # all_sensib_PCE = np.zeros((n0, n_var))
        for i in range(n0):
            grad_pce_u = metamodel_PCE.gradient(U_doe[i])       
            for j in range(n_var):
                all_grad_PCE[i, j] = grad_pce_u[j, 0]
        return y_PCE, all_grad_PCE

    class PCKRGFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, g_krg):
            super().__init__(n_var, 1)
            self.g_pce = g_pce
            self.g_krg = g_krg
        
        def _exec(self, u):
            return [self.g_pce(u)[0] + self.g_krg(u)[0]]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_krg = np.array(self.g_krg(U_ot))[:, 0]
            return (Z_pce + Z_krg).reshape(-1, 1).tolist()

        def _gradient(self, u):
            u_ot     = ot.Point(list(u))
            grad_pce = self.g_pce.gradient(u_ot)   
            grad_krg = self.g_krg.gradient(u_ot)   
            return [[grad_pce[i, 0] + grad_krg[i, 0]] for i in range(n_var)]

    class oldGEPCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_pce, sm_gepck):
            super().__init__(n_var, 1)
            self.g_pce  = g_pce
            self.sm     = sm_gepck

        def _exec(self, u):
            y_pce = self.g_pce(ot.Point(list(u)))[0]
            y_gek = self.sm.predict_values(np.array(u).reshape(1, -1)).item()
            return [y_pce + y_gek]

        def _exec_sample(self, U):
            U_ot = ot.Sample(U)
            Z_pce = np.array(self.g_pce(U_ot))[:, 0]
            Z_gek = self.sm.predict_values(np.array(U))[:, 0]
            return (Z_pce + Z_gek).reshape(-1, 1).tolist()

        def _exec_sigma(self, u):
            sm = self.sm
            n  = sm.nt; d = sm.X_norma.shape[1]
            W  = sm.coeff_pls; th = sm.optimal_theta
            s2 = float(sm.optimal_par['sigma2'])
            th_eff = (W**2) @ th
            x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
            Xn  = sm.X_norma
            df  = x_n[None, :] - Xn
            kf  = np.exp(-np.dot(df**2, th_eff))
            kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
            dff = Xn[:, None, :] - Xn[None, :, :]
            K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
            K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
            B_mat = dff * th_eff[None, None, :]
            term1 = 2.0 * np.diag(th_eff)
            term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
            K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
            K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
            K_tot += 1e-10 * np.eye(K_tot.shape[0])
            k = np.concatenate([kf, kd])
            try:
                B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
                return float(np.sqrt(s2 * B))
            except np.linalg.LinAlgError:
                return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

        def _gradient(self, u):
            u_np     = np.array(u).reshape(1, -1)
            grad_pce = self.g_pce.gradient(ot.Point(list(u)))   # OT Matrix (n_var, 1)
            return [[grad_pce[i, 0] + self.sm.predict_derivatives(u_np, i).item()]
                    for i in range(n_var)]

    # --------------------------------------------------------------------------- #
    # WRAPPER GEPCK 5 BRANCHES                                                   #
    class GEPCKFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, fm):
            super().__init__(n_var, 1)
            self.fm = fm

        def _exec(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [float(predict_gepck(self.fm, u_np)[0, 0])]

        def _exec_sample(self, U):
            U_np = np.array(U)
            return predict_gepck(self.fm, U_np)[:, 0:1].tolist()

        def _exec_sigma(self, u):
            u_np = np.array(u).reshape(1, -1)
            _, YSig2 = predict_gepck(self.fm, u_np, return_var=True)
            return float(np.sqrt(max(0.0, float(YSig2[0, 0]))))

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            G = predict_gradient_gepck(self.fm, u_np)   # (1, Mred)
            return [[float(G[0, i])] for i in range(self.fm['Mred'])]

    # --------------------------------------------------------------------------- #
    # WRAPPER BORNES DE CONFIANCE DU SURROGATE                                   #

    class BoundSurrogateFunction(ot.OpenTURNSPythonFunction):
        """
        g_bound(u) = g_ot(u) + sign * 2 * sigma_func(u)
        sign = +1  →  borne supérieure  g_sup = μ̂g + 2σ̂g
        sign = -1  →  borne inférieure  g_inf = μ̂g − 2σ̂g

        Wrappeur externe : ne modifie aucune classe existante.
        Compatible avec tous les modèles (GEPCK, KRG, GEK, PCKRG).
        Pas de _gradient défini → OT utilise différences finies si FORM est appelé.
        """
        def __init__(self, g_ot, sigma_func, sign):
            super().__init__(n_var, 1)
            self._g_ot       = g_ot
            self._sigma_func = sigma_func
            self._sign       = sign   # +1 ou -1

        def _exec(self, u):
            u_pt  = ot.Point(list(u))
            mu    = self._g_ot(u_pt)[0]
            sigma = self._sigma_func(u_pt)
            return [mu + self._sign * 2.0 * sigma]

    # Usage :
    #   g_ot_sup = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, +1))
    #   g_ot_inf = ot.Function(BoundSurrogateFunction(g_ot, sigma_func, -1))

    # def eval_PCE():
    #     return do_pce
    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU KRG                                                      #

    def build_metamodel_KRG(xt, yt):
        n_var = xt.shape[1]
        basis = ot.ConstantBasisFactory(n_var).build()
        covarianceModel = ot.SquaredExponential([1.0] * n_var)
        algo_KRG = ot.KrigingAlgorithm(xt, yt, covarianceModel, basis)
        if do_KRG:
            algo_KRG.setOptimizationBounds(ot.Interval([1.0] * n_var, [5.0] * n_var))
        algo_KRG.run()
        result = algo_KRG.getResult()
        cov_opt = result.getCovarianceModel()
        print(f"  KRG theta={list(cov_opt.getScale())}  sigma={list(cov_opt.getAmplitude())}", flush=True)
        return result.getMetaModel(), result

    # --------------------------------------------------------------------------- #
    # FONCTIONS LIEES AU GEK                                                      #

    # --- Modèle smt  ---
    def build_metamodel_GEK(xt, yt, all_grad):
        xlimits = np.column_stack([xt.min(axis=0) - 1, xt.max(axis=0) + 1])
        if do_GEK:
            sm = GEKPLS(
                n_comp=2,
                theta0=[1e-2, 1e-2],
                theta_bounds=[1.0, 5.0],
                corr="squar_exp",
                poly="constant",
                xlimits=xlimits,
                print_global=False,
            )
        else:
            sm = GEKPLS(
                n_comp=2,
                theta0=[1e-2, 1e-2],
                theta_bounds=[1.0, 5.0],
                corr="squar_exp",
                poly="constant",
                xlimits=xlimits,
                print_global=False,
            )
        sm.set_training_values(xt, yt)
        for j in range(n_var):
            sm.set_training_derivatives(xt, all_grad[:, j].reshape(-1, 1), j)
        sm.train()
        print(f"  GEK theta={list(sm.optimal_theta)}  sigma={float(np.sqrt(sm.optimal_par['sigma2'])):.6f}", flush=True)
        return sm

    # --- Wrapper OpenTURNS avec gradients analytiques ---
    class GEKPLSFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, surrogate):
            super().__init__(n_var, 1)
            self.sm = surrogate

        def _exec(self, u):
            return [self.sm.predict_values(np.array(u).reshape(1, -1)).item()]

        def _exec_sample(self, U):
            return self.sm.predict_values(np.array(U)).tolist()
        
        def _exec_sigma(self, u):
            sm = self.sm
            n  = sm.nt; d = sm.X_norma.shape[1]
            W  = sm.coeff_pls; th = sm.optimal_theta
            s2 = float(sm.optimal_par['sigma2'])
            th_eff = (W**2) @ th
            x_n = (np.array(u).reshape(-1) - sm.X_offset) / sm.X_scale
            Xn  = sm.X_norma
            df  = x_n[None, :] - Xn
            kf  = np.exp(-np.dot(df**2, th_eff))
            kd  = (2.0 * kf[:, None] * df * th_eff[None, :]).reshape(-1)
            dff = Xn[:, None, :] - Xn[None, :, :]
            K_ff = np.exp(-np.einsum('ijk,k->ij', dff**2, th_eff))
            K_fd = (2.0 * K_ff[:, :, None] * dff * th_eff[None, None, :]).reshape(n, n*d)
            B_mat = dff * th_eff[None, None, :]
            term1 = 2.0 * np.diag(th_eff)
            term2 = 4.0 * np.einsum('ija,ijb->ijab', B_mat, B_mat)
            K_dd  = (K_ff[:, :, None, None] * (term1 - term2)).transpose(0,2,1,3).reshape(n*d, n*d)
            K_tot = np.block([[K_ff, K_fd], [K_fd.T, K_dd]])
            K_tot += 1e-10 * np.eye(K_tot.shape[0])
            k = np.concatenate([kf, kd])
            try:
                B = max(0.0, 1.0 - k @ np.linalg.solve(K_tot, k))
                return float(np.sqrt(s2 * B))
            except np.linalg.LinAlgError:
                return float(np.sqrt(sm.predict_variances(np.array(u).reshape(1, -1)).item()))

        def _gradient(self, u):
            u_np = np.array(u).reshape(1, -1)
            return [[self.sm.predict_derivatives(u_np, kx).item()] for kx in range(n_var)]
    
    # --------------------------------------------------------------------------- #
    # FONCTIONS POUR FORM                                                         #
    def init_g_ot(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction génère xt, yt, all_grad si xt n'est pas vide puis
        contruit un metamodele à partir de ces points. Dans le cas HF, elle 
        créé uniquement une fonction OT. Retourne g_ot, sigma_func, xt, yt, all_grad.
        """
        if do_KRG:
            if xt is None: xt, yt, all_grad = build_DOE()
            g_ot, result = build_metamodel_KRG(xt, yt)
            sigma_func = lambda u: float(np.sqrt(result.getConditionalMarginalVariance(ot.Point(list(u)))))

        elif do_GEK:
            if xt is None: xt, yt, all_grad = build_DOE()
            sm_GEK   = build_metamodel_GEK(xt, yt, all_grad)
            gek_impl = GEKPLSFunction(sm_GEK)          
            g_ot     = ot.Function(gek_impl)
            sigma_func = gek_impl._exec_sigma

        elif do_PCKRG:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()                                            # on fait les calculs HF sur les points du DOE
            else:          y_hf, all_grad_hf = yt, all_grad
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)                                                      # on déduit le métamodèle PCE
            y_PCE, all_grad_PCE = calculate_PCE(xt, y_hf, all_grad_hf, g_ot_PCE)                              # on calcule la composante PCE à partir des valeurs hf
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE                                         # on construit le residu
            gr_ot, result_r = build_metamodel_KRG(xt, yr)                                                 # on construit le surrogate sur le residu
            sigma_func = lambda u: float(np.sqrt(result_r.getConditionalMarginalVariance(ot.Point(list(u)))))  # on calcule sigma_func (locale donc sur surrogate)
            g_ot = ot.Function(PCKRGFunction(g_ot_PCE, gr_ot))                                            # on wrappe la somme du surrogate et du PCE
            yt, all_grad = y_hf, all_grad_hf                                                              # on stocke les valeurs hf pour si warmstart

        elif do_old_GEPCK:
            if xt is None: xt, y_hf, all_grad_hf = build_DOE()
            else:          y_hf, all_grad_hf = yt, all_grad
            g_ot_PCE = build_metamodel_PCE(xt, y_hf)
            y_PCE, all_grad_PCE = calculate_PCE(xt, y_hf, all_grad_hf, g_ot_PCE) 
            yr, all_grad_r = y_hf-y_PCE, all_grad_hf-all_grad_PCE 
            smr_GEK = build_metamodel_GEK(xt, yr, all_grad_r)
            gepck_impl = oldGEPCKFunction(g_ot_PCE, smr_GEK)
            g_ot  = ot.Function(gepck_impl)
            sigma_func = gepck_impl._exec_sigma
            yt, all_grad = y_hf, all_grad_hf

        elif do_GEPCK:
            if xt is None: xt, yt, all_grad = build_DOE()
            _marginals = [{'Type': 'Gaussian', 'Parameters': [0.0, 1.0]},
                          {'Type': 'Gaussian', 'Parameters': [0.0, 1.0]}]
            _copula    = {'Type': 'Independent', 'Parameters': np.eye(n_var)}
            _opts      = {'Mode': 'optimal',
                          'PCE': {'Degree': list(range(1, max_degree + 1)), 'Method': 'LARS'}}
            _Y_aug = build_Y_aug(yt, all_grad)
            print(f"=== GEPCK fit N={len(xt)} ===", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _fm = fit_gepck(xt, _Y_aug, _opts, _marginals, _copula)
            print(f"  LOO={_fm['Error'][0]['LOO']:.4e}  n_poly={_fm['NumberOfPoly']}  theta={_fm['Kriging'][0]['theta']}", flush=True)
            _final_idx   = _fm['idxranking'][0][:_fm['NumberOfPoly']]
            _sel_indices = _fm['AllIndices'][0][np.array(_final_idx), :]
            _beta_pce    = np.array(_fm['Kriging'][0]['beta']).ravel()
            _terms = []
            for _mi, _coef in zip(_sel_indices, _beta_pce):
                _parts = [f"H{int(_mi[k])}(u{k+1})" for k in range(len(_mi)) if int(_mi[k]) > 0]
                _terms.append(f"{_coef:+.4f}*{'*'.join(_parts) if _parts else '1'}")
            print(f"  GEPCK PCE termes : {' '.join(_terms)}", flush=True)
            global _gepck_pce_label, _gepck_loo, _eff_history_theta
            _gepck_pce_label = ' '.join(_terms)
            _gepck_loo       = _fm['Error'][0]['LOO']
            _eff_history_theta.append(list(_fm['Kriging'][0]['theta']))
            gepck_impl = GEPCKFunction(_fm)
            g_ot       = ot.Function(gepck_impl)
            sigma_func = gepck_impl._exec_sigma

        elif do_HF:
            if xt is None: xt = build_DOE()
            g_ot = ot.Function(HFFunction())
            yt, all_grad = None, None
        
        return g_ot, sigma_func, xt, yt, all_grad

    def init_surrogate():
        """
        Construit le DOE HF et retourne xt, yt, all_grad.
        Version allégée de init_g_ot sans wrapper OpenTURNS — pour les pipelines
        qui n'utilisent pas FORM directement (GEPCK 5 branches, etc.).
        sigma_func et event seront ajoutés ici quand FORM sera intégré.
        """
        xt, yt, all_grad = build_DOE()
        return xt, yt, all_grad

    def init_FORM(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction créé un metamodele si aucun existant et calcule l'event FORM.
        Elle retourne metamodele et xt,yt,all_grad et l'event.
        """
        # --- Événement de défaillance ---
        distribution = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
        X = ot.RandomVector(distribution)
        g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
        Y = ot.CompositeRandomVector(g_ot, X) if g_ot is not None else None
        event = ot.ThresholdEvent(Y, ot.Less(), 0.0) if Y is not None else None
        return event, g_ot, sigma_func, xt, yt, all_grad

    # --- Multi-start FORM depuis les points du DOE ---
    def FORM_all_modes(starting_points, tol_all_modes, event):
        """
        Multi-start FORM + DBSCAN pour identifier les modes de défaillance.
        - Chaque cluster DBSCAN = un mode distinct.
        - u* isolés (label -1) = descentes mal convergées, ignorées.
        """
        all_u_star   = []   # u* de chaque run réussi
        all_results  = []   # FORMResult correspondant
        all_sp       = []   # point de départ correspondant
        n_total = len(starting_points)

        for k, sp in enumerate(starting_points):
            print(f"  FORM {k+1}/{n_total}...", flush=True)
            try:
                solver = ot.AbdoRackwitz()
                solver.setStartingPoint(sp.tolist())
                solver.setMaximumIterationNumber(n_max_FORM)
                solver.setCheckStatus(False)
                solver.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver, event)
                form_i.run()
                r_i    = form_i.getResult()
                u_star = np.array(r_i.getStandardSpaceDesignPoint())
                all_u_star.append(u_star)
                all_results.append(r_i)
                all_sp.append(sp)
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"u*={[round(v,3) for v in u_star]}, "
                    f"beta={r_i.getHasoferReliabilityIndex():.4f}]", flush=True)
            except Exception as e:
                print(f"  [sp={[round(v,3) for v in np.array(sp)]}, "
                    f"ECHEC ({type(e).__name__})]", flush=True)

        if not all_u_star:
            return [], []

        # --- Cas 1 point : pas de DBSCAN ---
        if len(all_u_star) == 1:
            print(f"\n1 mode(s) distinct(s) (1 seul point de depart, pas de DBSCAN) :", flush=True)
            u = [round(v, 3) for v in all_results[0].getStandardSpaceDesignPoint()]
            print(f"  mode 1 : beta={all_results[0].getHasoferReliabilityIndex():.4f}  "
                  f"Pf={all_results[0].getEventProbability():.3e}  u*={u}", flush=True)
            return [all_results[0]], [all_sp[0]]

        # --- DBSCAN ---
        U_all  = np.array(all_u_star)          # shape (n_runs_ok, n_var)
        db     = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all)
        labels = db.labels_

        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"  {n_noise} descente(s) mal convergée(s) ignorée(s) (bruit DBSCAN)", flush=True)

        # --- Un mode par cluster : FORMResult avec beta minimal ---
        modes     = []
        best_sps  = []
        for lbl in sorted(set(labels) - {-1}):
            idx_cluster = [i for i, l in enumerate(labels) if l == lbl]
            best_i = min(idx_cluster,
                        key=lambda i: all_results[i].getHasoferReliabilityIndex())
            modes.append(all_results[best_i])
            best_sps.append(all_sp[best_i])

        order = sorted(range(len(modes)), key=lambda i: modes[i].getHasoferReliabilityIndex())
        modes    = [modes[i]    for i in order]
        best_sps = [best_sps[i] for i in order]

        print(f"\n{len(modes)} mode(s) distinct(s) "
            f"(DBSCAN eps={tol_all_modes}, min_samples=2) :", flush=True)
        for i, m in enumerate(modes):
            u = [round(v, 3) for v in m.getStandardSpaceDesignPoint()]
            print(f"  mode {i+1} : beta={m.getHasoferReliabilityIndex():.4f}  "
                f"Pf={m.getEventProbability():.3e}  u*={u}", flush=True)

        return modes, best_sps
    
    # --- Warm-start FORM depuis les points du DOE ---
    def FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction reçoit des résultats FORM et le DOE utilisé, et déclenche warm_start si besoin 
        - elle renvoie la liste de modes/ best_sps mise à jour mais ne renvoie pas le nouveau DOE pour 
        l'instant (choix facilement modifiable).
        """
        if len(modes)>0:
            u_star = modes[0].getStandardSpaceDesignPoint()
            g_val = g_ot(ot.Point(u_star))[0] if g_ot is not None else None

            if g_val is not None and abs(g_val) > tol_warmstart:
                # -- on fait warm start uniquement si on est au dessus de 0.2, sinon, on accepte le résultat. --
                xt = np.vstack([xt, [np.array(u_star)]])
                yt = np.vstack([yt, [[g_val]]])
                grad_ot  = g_ot.gradient(ot.Point(u_star))
                grad_val = np.array([[grad_ot[i, 0] for i in range(n_var)]])
                all_grad = np.vstack([all_grad, grad_val])
                event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)
                starting_points = np.vstack([xt, [[0.0, 0.0]]]) if do_multistart else np.array([[0.0, 0.0]])
                modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)
        return modes, best_sps

    # --------------------------------------------------------------------------- #
    # FONCTIONS D'ENRICHISSEMENT DU PLAN D'EXPERIENCE (EFF)                       #
    class EFFFunction(ot.OpenTURNSPythonFunction):
        def __init__(self, g_ot, sigma_func):
            super().__init__(n_var, 1)
            self.g_ot = g_ot
            self.sigma_func = sigma_func

        def _exec(self, u):
            u = ot.Point(u)
            sigmaG  = self.sigma_func(u)
            if sigmaG <= 0.0:
                return [0.0]
            muG     = self.g_ot(u)[0]
            epsilon = epsilon_factor * sigmaG
            t1 = -muG / sigmaG
            t2 = (epsilon + muG) / sigmaG
            t3 = (epsilon - muG) / sigmaG
            return [2*muG*norm.cdf(t1) - (epsilon+muG)*norm.cdf(-t2) + (epsilon-muG)*norm.cdf(t3) + sigmaG*(-2*norm.pdf(t1) + norm.pdf(t2) + norm.pdf(t3))]

    def run_EFF(g_ot, sigma_func, xt, yt, all_grad):
        """
        Cette fonction reçoit le métamodele et ses paramètres, et l'améliore jusqu'à vérifier le critère EFF puis
        renvoie métamodèle+ paramètres mis à jour.
        """
        # --- Si aucune branche ne tourne, on ne fait rien ---
        if g_ot is None or do_HF:
            return g_ot, sigma_func, xt, yt, all_grad, []

        xt_eff = []

        def _form_is_iter(g_ot_i, label):
            """FORM depuis [0,0] + IS sur le surrogate courant. Affiche une ligne résumé."""
            distribution_i = ot.JointDistribution([ot.Normal(0, 1)] * n_var)
            X_i  = ot.RandomVector(distribution_i)
            Y_i  = ot.CompositeRandomVector(g_ot_i, X_i)
            ev_i = ot.ThresholdEvent(Y_i, ot.Less(), 0.0)
            try:
                solver_i = ot.AbdoRackwitz()
                solver_i.setStartingPoint([0.0] * n_var)
                solver_i.setMaximumIterationNumber(n_max_FORM)
                solver_i.setCheckStatus(False)
                solver_i.setMaximumConstraintError(tol_FORM)
                form_i = ot.FORM(solver_i, ev_i)
                form_i.run()
                r_i = form_i.getResult()
            except Exception as e:
                print(f"  [{label}] FORM echoue ({type(e).__name__})", flush=True)
                return None
            beta_f  = r_i.getHasoferReliabilityIndex()
            pf_f    = r_i.getEventProbability()
            res_IS  = run_IS([r_i], ev_i)
            pf_IS   = res_IS.getProbabilityEstimate()
            beta_IS = float(-ot.Normal().computeQuantile(pf_IS)[0])
            cov_v   = res_IS.getCoefficientOfVariation()
            print(f"  [{label}] beta_FORM={beta_f:.4f}  Pf_FORM={pf_f:.3e}"
                  f" | Pf_IS={pf_IS:.3e}  beta_IS={beta_IS:.4f}  COV={cov_v:.3f}", flush=True)
            return beta_IS

        def _three_form_is(g_ot_i, sigma_func_i, label):
            """FORM+IS sur g, g+2sigma, g-2sigma. Affiche les 3 lignes + ratio.
            Retourne le ratio si calculable, None sinon."""
            g_sup_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, +1))
            g_inf_i = ot.Function(BoundSurrogateFunction(g_ot_i, sigma_func_i, -1))
            b_mid = _form_is_iter(g_ot_i, f"{label} μ")
            b_sup = _form_is_iter(g_sup_i, f"{label} sup")
            b_inf = _form_is_iter(g_inf_i, f"{label} inf")
            if b_mid is not None and b_sup is not None and b_inf is not None and b_mid != 0:
                ratio = abs(b_sup - b_inf) / abs(b_mid)
                print(f"  [{label}] |beta_IS_sup - beta_IS_inf| / beta_IS = {ratio:.4f}", flush=True)
                return ratio
            return None

        # --- FORM+IS sur le DOE initial (avant EFF) ---
        count_valid_BB   = 0
        count_valid_BS   = 0
        count_valid_both = 0
        if EFF_criteria == 'BB':
            _ratio_init = _three_form_is(g_ot, sigma_func, f"N={len(xt)} initial")
            if _ratio_init is not None and _ratio_init < tol_BB:
                count_valid_BB = 1

        # --- On résoud u = argmax(EFF) ---
        f = ot.Function(EFFFunction(g_ot, sigma_func))
        bounds = ot.Interval([u1_eff_min, u2_eff_min], [u1_eff_max, u2_eff_max])
        problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
        problem.setMinimization(False)
        algo_opti = ot.NLopt(problem, "GN_DIRECT")
        algo_opti.setStartingPoint([0.0] * n_var)
        algo_opti.setMaximumCallsNumber(n_max_EFF)
        algo_opti.run()
        u_opt = algo_opti.getResult().getOptimalPoint()
        _sigG = sigma_func(u_opt)
        _muG  = g_ot(ot.Point(u_opt))[0]
        _eps  = epsilon_factor * _sigG
        print(f"  EFF debug u_opt={list(np.round(np.array(u_opt),3))} : sigmaG={_sigG:.6f}  muG={_muG:.6f}  epsilon={_eps:.6f}", flush=True)
        print(f"  EFF initial : EFF(u_opt)={f(u_opt)[0]:.6f}, tol={tol_EFF}", flush=True)

        iter_count = 0
        if EFF_criteria == 'BB':
            _cond = lambda: abs(f(u_opt)[0]) > tol_EFF and count_valid_BB < 3
        elif EFF_criteria == 'BS':
            _cond = lambda: abs(f(u_opt)[0]) > tol_EFF and count_valid_BS < 3
        elif EFF_criteria == 'both':
            _cond = lambda: abs(f(u_opt)[0]) > tol_EFF and count_valid_both < 2
        else:
            _cond = lambda: abs(f(u_opt)[0]) > tol_EFF

        _beta_IS_0 = _form_is_iter(g_ot, f"N={len(xt)} initial μ conv")
        list_beta_IS = [_beta_IS_0] if _beta_IS_0 is not None else []
        global _eff_history_EFF, _eff_history_BB, _eff_history_BS
        _eff_history_BB = []
        _eff_history_BS = []
        list_ratio_BB = _eff_history_BB   # alias — même objet
        list_ratio_BS = _eff_history_BS
        _eff_history_EFF.append(f(u_opt)[0])   # EFF initial (avant ajout du 1er point)

        while _cond():
            _sigG = sigma_func(u_opt)
            _muG  = g_ot(ot.Point(u_opt))[0]
            print(f"  EFF={f(u_opt)[0]:.6f} > {tol_EFF} -- u_opt={list(np.round(np.array(u_opt),3))}  sigmaG={_sigG:.6f}  muG={_muG:.6f}", flush=True)
            _eff_history_EFF.append(f(u_opt)[0])   # EFF apres rebuild a cette iteration
            xt_eff.append(np.array(u_opt))
            # --- On reconstruit le modèle ---
            g_val, grad_U, _ = run_HF(np.array(u_opt))
            xt = np.vstack([xt, [np.array(u_opt)]])
            yt = np.vstack([yt, [[g_val]]])
            grad_val = np.array([[float(grad_U[i]) for i in range(n_var)]])
            all_grad = np.vstack([all_grad, grad_val])
            # --- Upgrade max_degree si assez de points ---
            _degree_avant = max_degree
            update_degree(len(xt))
            degree_upgraded = (max_degree != _degree_avant)
            g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)

            # --- FORM+IS sur le surrogate mis à jour (BB uniquement) ---
            if EFF_criteria == 'BB':
                iter_count += 1
                _ratio = _three_form_is(g_ot, sigma_func, f"N={len(xt)} EFF iter {iter_count}")
                if _ratio is not None and _ratio < tol_BB:
                    count_valid_BB += 1
                else:
                    count_valid_BB = 0
                list_ratio_BB.append(_ratio)

            # --- Suivi convergence beta_IS ---
            _b_mid = _form_is_iter(g_ot, f"N={len(xt)} μ conv")
            if EFF_criteria == 'BS':
                if _b_mid is not None and list_beta_IS and _b_mid != 0:
                    _ratio_conv = abs(_b_mid - list_beta_IS[-1]) / abs(_b_mid)
                    print(f"  [N={len(xt)}] |beta_IS - beta_IS_prec| / beta_IS = {_ratio_conv:.4f}", flush=True)
                    if _ratio_conv < tol_BS:
                        count_valid_BS += 1
                    else:
                        count_valid_BS = 0
                    list_ratio_BS.append(_ratio_conv)
                else:
                    count_valid_BS = 0
                    list_ratio_BS.append(None)
            # --- Critere both : BB et BS simultanement ---
            if EFF_criteria == 'both':
                iter_count += 1
                _ratio_bb = _three_form_is(g_ot, sigma_func, f"N={len(xt)} both iter {iter_count}")
                if _b_mid is not None and list_beta_IS and _b_mid != 0:
                    _ratio_bs = abs(_b_mid - list_beta_IS[-1]) / abs(_b_mid)
                    print(f"  [N={len(xt)} both] |beta_IS - beta_IS_prec| / beta_IS = {_ratio_bs:.4f}", flush=True)
                else:
                    _ratio_bs = None
                if (_ratio_bb is not None and _ratio_bb < tol_BB and
                        _ratio_bs is not None and _ratio_bs < tol_BS):
                    count_valid_both += 1
                else:
                    count_valid_both = 0
                list_ratio_BB.append(_ratio_bb)
                list_ratio_BS.append(_ratio_bs)

            if _b_mid is not None:
                list_beta_IS.append(_b_mid)

            # --- Visu intermediaire apres ajout de point ---
            _about_to_upgrade = (len(xt) + 1 > n0_min(n_var, max_degree + 1) and max_degree + 1 <= max_of_maxdegree)
            if print_EFF_progres or degree_upgraded or _about_to_upgrade:
                print_planche_EFF(g_ot, sigma_func, xt, xt_eff)

            # --- On re-résoud u = argmax(EFF) ---
            f = ot.Function(EFFFunction(g_ot, sigma_func))
            bounds = ot.Interval([u1_eff_min, u2_eff_min], [u1_eff_max, u2_eff_max])
            problem = ot.OptimizationProblem(f, ot.Function(), ot.Function(), bounds)
            problem.setMinimization(False)
            algo_opti = ot.NLopt(problem, "GN_DIRECT")
            algo_opti.setStartingPoint([0.0] * n_var)
            algo_opti.setMaximumCallsNumber(n_max_EFF)
            algo_opti.run()
            u_opt = algo_opti.getResult().getOptimalPoint()

        _sigG2 = sigma_func(u_opt)
        _muG2  = g_ot(ot.Point(u_opt))[0]
        _eps2  = epsilon_factor * _sigG2
        if _sigG2 > 0:
            t1 = -_muG2 / _sigG2
            t2 = (_eps2 + _muG2) / _sigG2
            t3 = (_eps2 - _muG2) / _sigG2
            term1 = 2 * _muG2 * norm.cdf(t1)
            term2 = -(_eps2 + _muG2) * norm.cdf(-t2)
            term3 = (_eps2 - _muG2) * norm.cdf(t3)
            term4 = _sigG2 * (norm.pdf(t2) - norm.pdf(t3))
            print(f"  EFF converge debug : u_opt={list(np.round(np.array(u_opt),4))}  sigmaG={_sigG2:.8f}  muG={_muG2:.8f}  epsilon={_eps2:.8f}", flush=True)
            print(f"    t1={t1:.6f}  t2={t2:.6f}  t3={t3:.6f}", flush=True)
            print(f"    norm.cdf(t1)={norm.cdf(t1):.8e}  norm.cdf(-t2)={norm.cdf(-t2):.8e}  norm.cdf(t3)={norm.cdf(t3):.8e}", flush=True)
            print(f"    norm.pdf(t2)={norm.pdf(t2):.8e}  norm.pdf(t3)={norm.pdf(t3):.8e}", flush=True)
            print(f"    term1={term1:.8e}  term2={term2:.8e}  term3={term3:.8e}  term4={term4:.8e}", flush=True)
            print(f"    EFF = {term1+term2+term3+term4:.8e}", flush=True)
        else:
            print(f"  EFF converge debug : sigmaG=0 (modele interpolant exact au point u_opt)", flush=True)
        _exit_eff = abs(f(u_opt)[0]) <= tol_EFF
        _exit_bb   = count_valid_BB   >= 3 and EFF_criteria == 'BB'
        _exit_bs   = count_valid_BS   >= 3 and EFF_criteria == 'BS'
        _exit_both = count_valid_both >= 2 and EFF_criteria == 'both'
        if _exit_eff:
            _reason = "EFF"
        elif _exit_bb:
            _reason = "BB (3 iter valides)"
        elif _exit_bs:
            _reason = "BS (3 iter valides)"
        elif _exit_both:
            _reason = "both (2 iter valides)"
        else:
            _reason = "?"
        print(f"  EFF converge [{_reason}] : EFF(u_opt)={f(u_opt)[0]:.4f}"
              f"  count_valid_BB={count_valid_BB}  count_valid_BS={count_valid_BS}"
              f"  count_valid_both={count_valid_both}  ({len(xt_eff)} point(s) ajoutes)", flush=True)
        _fmt = lambda lst: [round(r, 4) if r is not None else None for r in lst]
        if list_ratio_BB:
            print(f"  [historique ratio BB] {_fmt(list_ratio_BB)}  tol={tol_BB}", flush=True)
        if list_ratio_BS:
            print(f"  [historique ratio BS] {_fmt(list_ratio_BS)}  tol={tol_BS}", flush=True)
        return g_ot, sigma_func, xt, yt, all_grad, xt_eff

    # --------------------------------------------------------------------------- #
    # FONCTIONS RESULTATS/ AFFICHAGE                                              #

    def print_EFF_graphs():
        """Planche 3 subplots : historique EFF, criteres BB/BS, theta Kriging.
        Lit les globaux _eff_history_*. Sauvegarde en PNG dans out_dir_eff."""
        if not (_eff_history_EFF or _eff_history_BB or _eff_history_theta):
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        _clip = 1e-12   # evite log(0)

        # --- Subplot 1 : EFF vs iterations ---
        ax = axes[0]
        x_eff = list(range(len(_eff_history_EFF)))
        vals_eff = [max(abs(v), _clip) for v in _eff_history_EFF]
        ax.semilogy(x_eff, vals_eff, 'b-o', ms=4, lw=1.2, label='EFF(u_opt)')
        ax.axhline(tol_EFF, color='orange', ls='--', lw=1, label=f'tol_EFF={tol_EFF:.1e}')
        ax.set_xlabel('Iteration EFF')
        ax.set_ylabel('EFF (echelle log)')
        ax.set_title('Convergence EFF')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        # --- Subplot 2 : BB / BS vs iterations ---
        ax = axes[1]
        if _eff_history_BB:
            x_bb = list(range(1, len(_eff_history_BB) + 1))
            vals_bb = [max(v, _clip) if v is not None else np.nan for v in _eff_history_BB]
            ax.semilogy(x_bb, vals_bb, 'g-o', ms=4, lw=1.2, label='BB')
            ax.axhline(tol_BB, color='g', ls='--', lw=0.8, label=f'tol_BB={tol_BB:.1e}')
        if _eff_history_BS:
            x_bs = list(range(1, len(_eff_history_BS) + 1))
            vals_bs = [max(v, _clip) if v is not None else np.nan for v in _eff_history_BS]
            ax.semilogy(x_bs, vals_bs, 'r-s', ms=4, lw=1.2, label='BS')
            ax.axhline(tol_BS, color='r', ls='--', lw=0.8, label=f'tol_BS={tol_BS:.1e}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Ratio (echelle log)')
        ax.set_title('Criteres BB / BS')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        # --- Subplot 3 : theta Kriging vs iterations ---
        ax = axes[2]
        if _eff_history_theta:
            thetas = np.array(_eff_history_theta)   # shape (n_fits, n_var)
            x_th = list(range(len(_eff_history_theta)))
            for k in range(thetas.shape[1]):
                lbl = params_names[k] if k < len(params_names) else f'dim{k}'
                ax.semilogy(x_th, np.maximum(thetas[:, k], _clip), '-o', ms=4, lw=1.2, label=f'theta_{lbl}')
            norms_th = np.maximum(np.linalg.norm(thetas, axis=1), _clip)
            ax.semilogy(x_th, norms_th, 'k--', ms=3, lw=1, label='||theta||')
        ax.set_xlabel('Iteration (fit)')
        ax.set_ylabel('theta (echelle log)')
        ax.set_title('Evolution theta Kriging')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.4)

        fig.tight_layout()
        fname = f'EFF_graphs_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF_graphs] -> {fname}", flush=True)

    def print_planche_EFF(g_ot, sigma_func, xt, xt_eff):
        """Planche 2 graphiques cote a cote : critere EFF (gauche) et sigma surrogate (droite).
        Sauvegarde en PNG dans out_dir_eff sans afficher de fenetre."""
        global hf_2d_grid_fixed
        n_added = len(xt_eff)

        # --- Grille commune (calculee une seule fois) ---
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        # --- Z_eff et Z_sigma ---
        eff_func = EFFFunction(g_ot, sigma_func)
        Z_eff   = np.array([eff_func._exec(pt)[0] for pt in grid]).reshape(n_grid, n_grid)
        Z_sigma = np.array([sigma_func(pt) for pt in grid]).reshape(n_grid, n_grid)

        # --- Contour g=0 surrogate (une seule evaluation sur grille) ---
        Z_g = None
        if g_ot is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_g = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)

        # --- Contour g=0 HF (depuis cache ou calcul, une seule fois) ---
        Z_true, U1_hf, U2_hf = None, None, None
        if print_HF:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = np.array([run_HF(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
                hf_2d_grid_fixed = {'params': {'u1_min': u1_min, 'u1_max': u1_max,
                                                'u2_min': u2_min, 'u2_max': u2_max,
                                                'n_grid_hf': n_grid_hf}, 'Z': Z_true.tolist()}
                print(f"hf_2d_grid_fixed = {hf_2d_grid_fixed!r}", flush=True)

        # --- Figure ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        _pce_line   = f'\n{_gepck_pce_label}' if _gepck_pce_label else ''
        _theta_str  = '  theta=[' + ', '.join(f'{v:.3f}' for v in _eff_history_theta[-1]) + ']' if _eff_history_theta else ''
        _loo_str    = f'  LOO={_gepck_loo:.3e}' if _gepck_loo is not None else ''
        fig.suptitle(f'{modele} - N={len(xt)} pts DOE  ({n_added} ajoutes par EFF){_theta_str}{_loo_str}{_pce_line}', fontsize=10)

        def _decorate(ax):
            if Z_g is not None:
                ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--')
            if Z_true is not None:
                ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2)
            if xt is not None:
                ax.scatter(xt[:, 0], xt[:, 1], c='white', s=40, zorder=5,
                           edgecolors='black', linewidths=0.8, label='DOE')
            if n_added > 0:
                xt_eff_arr = np.array(xt_eff)
                ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=80, zorder=6,
                           marker='^', label=f'EFF ({n_added} pts)')
                for i, pt in enumerate(xt_eff_arr):
                    ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                                xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)
            ax.set_xlabel('u1')
            ax.set_ylabel('u2')
            ax.set_xlim(u1_min, u1_max)
            ax.set_ylim(u2_min, u2_max)
            ax.legend(loc='best', fontsize=9)

        # --- Ax1 : EFF ---
        cf1 = ax1.contourf(U1, U2, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf1, ax=ax1, label='EFF')
        ax1.set_title('Critere EFF')
        _decorate(ax1)

        # --- Ax2 : sigma ---
        cf2 = ax2.contourf(U1, U2, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf2, ax=ax2, label='sigma (ecart-type surrogate)')
        ax2.set_title('Ecart-type surrogate (sigma)')
        _decorate(ax2)

        # --- Ax3 : g surrogate (isocouleurs RdYlGn) ---
        if Z_g is not None:
            cf3 = ax3.contourf(U1, U2, Z_g, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf3, ax=ax3, label='g surrogate')
            ax3.contour(U1, U2, Z_g, levels=[0], colors='blue', linewidths=2)
        ax3.set_title('g surrogate - etat limite')
        _decorate(ax3)

        plt.tight_layout()
        fname = f'EFF_{n_added}points_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [EFF visu] -> {fname}", flush=True)

    def print_visu_EFF(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D des valeurs du critere EFF sur la meme grille que print_visu."""
        global hf_2d_grid_fixed
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        eff_func = EFFFunction(g_ot, sigma_func)
        Z_eff = np.array([eff_func._exec(pt)[0] for pt in grid]).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_eff, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='EFF')

        # --- Contour g=0 du surrogate ---
        if g_ot is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_g = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF ---
        if print_HF:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = np.array([run_HF(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
                hf_2d_grid_fixed = {'params': {'u1_min': u1_min, 'u1_max': u1_max, 'u2_min': u2_min, 'u2_max': u2_max, 'n_grid_hf': n_grid_hf}, 'Z': Z_true.tolist()}
                print(f"hf_2d_grid_fixed = {hf_2d_grid_fixed!r}", flush=True)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2)

        # --- Points DOE ---
        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        # --- Points ajoutes par EFF ---
        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('Critere EFF')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_visu_sigma(g_ot, sigma_func, xt, xt_eff):
        """Carte 2D de l'ecart-type conditionnel du surrogate."""
        global hf_2d_grid_fixed
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        Z_sigma = np.array([sigma_func(pt) for pt in grid]).reshape(n_grid, n_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        cf = ax.contourf(U1, U2, Z_sigma, levels=20, cmap='plasma', alpha=0.85)
        plt.colorbar(cf, ax=ax, label='sigma (ecart-type surrogate)')

        # --- Contour g=0 du surrogate ---
        if g_ot is not None:
            grid_ot = ot.Sample(grid.tolist())
            Z_g = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_g, levels=[0], colors='cyan', linewidths=2, linestyles='--', label='surrogate g=0')

        # --- Contour g=0 HF ---
        if print_HF:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = np.array([run_HF(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
                hf_2d_grid_fixed = {'params': {'u1_min': u1_min, 'u1_max': u1_max, 'u2_min': u2_min, 'u2_max': u2_max, 'n_grid_hf': n_grid_hf}, 'Z': Z_true.tolist()}
                print(f"hf_2d_grid_fixed = {hf_2d_grid_fixed!r}", flush=True)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2)

        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='white', s=40, zorder=5,
                       edgecolors='black', linewidths=0.8, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=80, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('Ecart-type surrogate (sigma)')
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show(block=False)

    def print_results(best_result, g_ot):
        u_star = best_result.getStandardSpaceDesignPoint()
        n_iter = best_result.getOptimizationResult().getIterationNumber()
        dist_X = dist_jointe()
        T_inv  = dist_X.getInverseIsoProbabilisticTransformation()
        x_star = T_inv(u_star)

        # --- Résultats FORM ---
        print(f"n_iter FORM  = {n_iter}", flush=True)
        for i, p in enumerate(params_names):
            print(f"{p}*          = {x_star[i]:.4f}", flush=True)
        print(f"u*           = {[round(v, 4) for v in u_star]}", flush=True)
        print(f"Imp.         = {[round(v, 4) for v in best_result.getImportanceFactors()]}", flush=True)
        print(f"beta         = {best_result.getHasoferReliabilityIndex():.4f}", flush=True)
        print(f"Pf           = {best_result.getEventProbability():.4e}", flush=True)

        # --- Erreur FOSM ---
        if g_ot is not None:
            _, grad_HF_U_star, _ = run_HF(u_star)
            for i, p in enumerate(params_names):
                print(f"dg/du_{p} en u* (HF@u*GEK) = {grad_HF_U_star[i]:.6f}", flush=True)
            u0             = ot.Point([0.0] * n_var)
            g0_HF, grad_HF_U0, _ = run_HF(u0)
            u_FOSM         = grad_HF_U0 * (-g0_HF / grad_HF_U0.normSquare())
            print(f"u* FOSM (HF) = {[round(v, 4) for v in u_FOSM]}", flush=True)
            print(f"Erreur FOSM  = {(u_FOSM - u_star).norm() / u_star.norm():.4f}", flush=True)


    def print_visu(best_result, best_sp, xt, g_ot, modes, xt_eff):
        global u1_min, u1_max, u2_min, u2_max, hf_2d_grid_fixed
        u1 = np.linspace(u1_min, u1_max, n_grid)
        u2 = np.linspace(u2_min, u2_max, n_grid)
        U1, U2 = np.meshgrid(u1, u2)
        grid = np.column_stack([U1.ravel(), U2.ravel()])

        fig, ax = plt.subplots(figsize=(7, 6))

        # --- Fond coloré : GEK en priorité, sinon KRG ---
        if do_GEK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gek = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gek, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEK)')
            ax.contour(U1, U2, Z_gek, levels=[0], colors='blue', linewidths=2)
        elif do_KRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_krg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (KRG)')
        elif do_PCKRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_pckrg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_pckrg, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (PCKRG)')
            ax.contour(U1, U2, Z_pckrg, levels=[0], colors='blue', linewidths=2)
        elif do_old_GEPCK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gepck = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gepck, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (old GEPCK)')
            ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue', linewidths=2)
        elif do_GEPCK:
            grid_ot = ot.Sample(grid.tolist())
            Z_gepck = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            cf = ax.contourf(U1, U2, Z_gepck, levels=20, cmap='RdYlGn', alpha=0.6)
            plt.colorbar(cf, ax=ax, label='g (GEPCK)')
            ax.contour(U1, U2, Z_gepck, levels=[0], colors='blue', linewidths=2)

        # --- Contour KRG ---
        if do_KRG:
            grid_ot = ot.Sample(grid.tolist())
            Z_krg = np.array(g_ot(grid_ot))[:, 0].reshape(n_grid, n_grid)
            ax.contour(U1, U2, Z_krg, levels=[0], colors='purple', linewidths=2, linestyles=':')

        # --- Contour HF grossier ---
        if print_HF:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            if hf_2d_grid_fixed is not None:
                Z_true = np.array(hf_2d_grid_fixed['Z'])
            else:
                grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
                Z_true = np.array([run_HF(pt)[0] for pt in grid_hf]).reshape(n_grid_hf, n_grid_hf)
                hf_2d_grid_fixed = {'params': {'u1_min': u1_min, 'u1_max': u1_max, 'u2_min': u2_min, 'u2_max': u2_max, 'n_grid_hf': n_grid_hf}, 'Z': Z_true.tolist()}
                print(f"hf_2d_grid_fixed = {hf_2d_grid_fixed!r}", flush=True)
            ax.contour(U1_hf, U2_hf, Z_true, levels=[0], colors='red', linewidths=2, linestyles='--')

        # --- LS analytique (depuis flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_lim_a = calc.u1_lim_plast
            u2_lim_a = calc.u2p_LS(u1_lim_a)
            u1_g_a = np.linspace(u1_lim_a, u1_max, n_grid)
            u2_g_a = np.array([calc.u2p_LS(u) for u in u1_g_a])
            ax.plot(u1_g_a, u2_g_a, color='green', linestyle='-.', linewidth=2)
            ax.plot([u1_lim_a, u1_lim_a], [u2_lim_a, u2_max],
                    color='green', linestyle='-.', linewidth=2)
            ax.plot(u1_lim_a, u2_lim_a, 'ko', ms=6, zorder=6)

        # --- Points ---
        if xt is not None:
            ax.scatter(xt[:, 0], xt[:, 1], c='black', s=30, zorder=5, label='DOE')

        if xt_eff is not None and len(xt_eff) > 0:
            xt_eff_arr = np.array(xt_eff)
            ax.scatter(xt_eff_arr[:, 0], xt_eff_arr[:, 1], c='red', s=60, zorder=6,
                       marker='^', label=f'EFF ({len(xt_eff)} pts)')
            for i, pt in enumerate(xt_eff_arr):
                ax.annotate(str(i + 1), (pt[0], pt[1]), textcoords='offset points',
                            xytext=(0, 8), ha='center', fontsize=8, color='red', zorder=7)

        ax.scatter(0, 0, c='orange', s=100, zorder=6, marker='P', label='[0, 0]')

        if best_sp is not None:
            ax.scatter(best_sp[0], best_sp[1], c='cyan', s=100, zorder=7, marker='D',
                    label='point de depart best')

        if best_result is not None:
            u_star = np.array(best_result.getStandardSpaceDesignPoint())
            ax.scatter(u_star[0], u_star[1], c='gold', s=200, zorder=8, marker='*',
                    label=f'u*1 [{u_star[0]:.2f},{u_star[1]:.2f}] beta={best_result.getHasoferReliabilityIndex():.3f}')

        if len(modes) > 0:
            for k, mode in enumerate(modes[1:], start=2):
                u_m = np.array(mode.getStandardSpaceDesignPoint())
                ax.scatter(u_m[0], u_m[1], c='magenta', s=200, zorder=8, marker='*',
                        label=f'u*{k} [{u_m[0]:.2f},{u_m[1]:.2f}] beta={mode.getHasoferReliabilityIndex():.3f}')

        # --- Points fixes (run HF précédent) ---
        if best_sol_modes_fixed is not None:
            colors_fixed = ['blue', 'red', 'green', 'gold']
            for col, (lbl, data) in zip(colors_fixed, best_sol_modes_fixed.items()):
                ustar_f = data['u*']
                sp_f    = data['sp']
                ax.scatter(ustar_f[0], ustar_f[1], c=col, s=200, zorder=9, marker='*',
                           label=f'u* {lbl}')
                ax.scatter(sp_f[0], sp_f[1], c=col, s=100, zorder=9, marker='x',
                           linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None and lbl in grad_sp_fixed:
                    ng = np.array(grad_sp_fixed[lbl]['neg_grad'])
                    ng = ng / np.linalg.norm(ng) * 1.5
                    ax.quiver(sp_f[0], sp_f[1], ng[0], ng[1], color=col,
                              angles='xy', scale_units='xy', scale=1.0, width=0.005)

        # --- Trajectoires FORM hardcodees ---
        if traj_runs_fixed is not None:
            colors_traj = {'A': 'blue', 'B': 'red', 'C': 'green', 'D': 'gold'}
            all_pts = []
            for lbl, traj in traj_runs_fixed.items():
                col = colors_traj.get(lbl, 'gray')
                pts  = np.array(traj['points'])
                grds = np.array(traj['grads'])
                all_pts.append(pts)
                # polyligne
                ax.plot(pts[:, 0], pts[:, 1], '-', color=col, alpha=0.5, linewidth=1.2)
                # points intermediaires
                ax.scatter(pts[1:-1, 0], pts[1:-1, 1], c=col, s=12, zorder=7, alpha=0.5)
                # depart et arrivee
                ax.scatter(pts[0, 0],  pts[0, 1],  c=col, s=60,  zorder=8, marker='o')
                ax.scatter(pts[-1, 0], pts[-1, 1], c=col, s=150, zorder=8, marker='*')
                # petites fleches -grad normalise (longueur 0.3)
                for pt, g in zip(pts, grds):
                    ng = np.array(g)
                    nrm = np.linalg.norm(ng)
                    if nrm > 0:
                        ng = -ng / nrm * 0.3
                        ax.annotate('', xy=(pt[0]+ng[0], pt[1]+ng[1]), xytext=(pt[0], pt[1]),
                                    arrowprops=dict(arrowstyle='->', color=col, lw=0.7))
            # zoom sur les trajectoires : mise a jour des variables globales
            all_pts_arr = np.vstack(all_pts)
            margin = 1.0
            u1_min = float(all_pts_arr[:, 0].min()) - margin
            u1_max = float(all_pts_arr[:, 0].max()) + margin
            u2_min = float(all_pts_arr[:, 1].min()) - margin
            u2_max = float(all_pts_arr[:, 1].max()) + margin

        # --- Légende contours ---
        legend_lines = []
        
        if do_KRG:
            legend_lines.append(Line2D([0], [0], color='purple', linestyle=':',  linewidth=2, label='g=0 KRG'))
        if do_GEK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEKPLS'))
        if do_PCKRG:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 PCKRG'))
        if do_old_GEPCK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 old GEPCK'))
        if do_GEPCK:
            legend_lines.append(Line2D([0], [0], color='blue',   linestyle='-',  linewidth=2, label='g=0 GEPCK'))
        if print_HF:
            legend_lines.append(Line2D([0], [0], color='red',    linestyle='--', linewidth=2, label='g=0 HF'))
        if print_ana:
            legend_lines.append(Line2D([0], [0], color='green',  linestyle='-.', linewidth=2, label='g=0 ana'))

        ax.legend(handles=ax.legend().legend_handles + legend_lines)

        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_xlim(u1_min, u1_max)
        ax.set_ylim(u2_min, u2_max)
        ax.set_title('FORM et etat limite g=0')
        plt.tight_layout()
        fname = f'visu{modele}_{timestamp}.png'
        fig.savefig(os.path.join(out_dir_eff, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [visu] -> {fname}", flush=True)

    def print_3D_HF():
        if hf_3d_grid_fixed is not None:
            print("Cache hf_3d_grid_fixed disponible — pas d'appels STRAINS.", flush=True)
            u1_min_c, u1_max_c, u2_min_c, u2_max_c, n_c = hf_3d_grid_fixed['params']
            u1_hf = np.linspace(u1_min_c, u1_max_c, n_c)
            u2_hf = np.linspace(u2_min_c, u2_max_c, n_c)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            Z = np.array(hf_3d_grid_fixed['Z'])
        else:
            u1_hf = np.linspace(u1_min, u1_max, n_grid_hf)
            u2_hf = np.linspace(u2_min, u2_max, n_grid_hf)
            U1_hf, U2_hf = np.meshgrid(u1_hf, u2_hf)
            grid_hf = np.column_stack([U1_hf.ravel(), U2_hf.ravel()])
            print(f"Evaluation HF grille {n_grid_hf}x{n_grid_hf} = {n_grid_hf**2} appels STRAINS...", flush=True)
            Z_flat = [run_HF(pt)[0] for pt in grid_hf]
            Z = np.array(Z_flat).reshape(n_grid_hf, n_grid_hf)

        # --- Impression copy-pastable ---
        print(f"\nhf_3d_grid_fixed = {{", flush=True)
        print(f"    'params': ({u1_min}, {u1_max}, {u2_min}, {u2_max}, {n_grid_hf}),", flush=True)
        print(f"    'Z': [", flush=True)
        for row in Z:
            vals = ', '.join(f'{v:.6f}' for v in row)
            print(f"        [{vals}],", flush=True)
        print(f"    ]", flush=True)
        print(f"}}", flush=True)

        # --- Plot 3D ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U1_hf, U2_hf, Z, color='red', alpha=0.3, label='g_HF')
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='red', linewidths=2,
                   zdir='z', offset=float(Z.min()))
        ax.contour(U1_hf, U2_hf, Z, levels=[0], colors='darkred', linewidths=2)

        # --- Surface analytique g_ana (flexion_claude) ---
        if print_ana:
            calc = flexion_claude()
            u1_a = np.linspace(u1_min, u1_max, n_grid)
            u2_a = np.linspace(u2_min, u2_max, n_grid)
            U1_a, U2_a = np.meshgrid(u1_a, u2_a)
            Z_ana = np.array([calc.g(u1, u2)
                              for u1, u2 in zip(U1_a.ravel(), U2_a.ravel())]
                             ).reshape(n_grid, n_grid)
            ax.plot_surface(U1_a, U2_a, Z_ana, color='blue', alpha=0.3, label='g_ana')
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2,
                       zdir='z', offset=float(Z.min()))
            ax.contour(U1_a, U2_a, Z_ana, levels=[0], colors='green', linewidths=2)

        if best_sol_modes_fixed is not None:
            for col, (lbl, data) in zip(['blue', 'red', 'green', 'gold'],
                                         best_sol_modes_fixed.items()):
                u1_f, u2_f = data['u*']
                u1_s, u2_s = data['sp']
                ax.scatter(u1_f, u2_f, 0.0, c=col, s=200, marker='*', label=f'u* {lbl}')
                ax.scatter(u1_s, u2_s, 0.0, c=col, s=100, marker='x', linewidths=2, label=f'sp {lbl}')
                if grad_sp_fixed is not None:
                    ng = grad_sp_fixed[lbl]['neg_grad']
                    ax.quiver(u1_s, u2_s, 0.0, ng[0], ng[1], 0.0,
                              color=col, length=3.0, normalize=True, arrow_length_ratio=0.3)
        ax.set_xlabel('u1 (fc)')
        ax.set_ylabel('u2 (fy)')
        ax.set_zlabel('g_HF')
        ax.set_title(f'Surface g_HF - {n_grid_hf}x{n_grid_hf} pts HF')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------- #
    # FONCTION IS POST-FORM                                                       #
    def run_IS(modes, event):
        """
        Importance Sampling post-FORM sur le surrogate.
        Distribution instrumentale : mixture de N(u*_i) pondérée par les Pf_FORM_i.
        Mono-modal : N simple centré sur u*.
        Retourne un ProbabilitySimulationResult.
        """
        if len(modes) == 1:
            g_imp = ot.Normal(n_var)
            g_imp.setMu(modes[0].getStandardSpaceDesignPoint())
            importance_dist = g_imp
        else:
            gaussians  = []
            pf_weights = []
            for m in modes:
                g_i = ot.Normal(n_var)
                g_i.setMu(m.getStandardSpaceDesignPoint())
                gaussians.append(g_i)
                pf_weights.append(m.getEventProbability())
            importance_dist = ot.Mixture(gaussians, pf_weights)

        experiment = ot.ImportanceSamplingExperiment(importance_dist, n_IS)
        std_event  = ot.StandardEvent(event)
        algo = ot.ProbabilitySimulationAlgorithm(std_event, experiment)
        algo.setMaximumCoefficientOfVariation(cov_IS)
        algo.setMaximumOuterSampling(n_IS)
        algo.run()
        return algo.getResult()

    def print_results_IS(result_IS):
        pf   = result_IS.getProbabilityEstimate()
        cov  = result_IS.getCoefficientOfVariation()
        ci   = result_IS.getConfidenceLength(0.95)
        beta = float(-ot.Normal().computeQuantile(pf)[0])
        print(f"=== Importance Sampling ===", flush=True)
        print(f"  Pf_IS   = {pf:.4e}", flush=True)
        print(f"  beta_IS = {beta:.4f}", flush=True)
        print(f"  COV     = {cov:.4f}", flush=True)
        print(f"  IC 95%  = [{pf - ci/2:.4e}, {pf + ci/2:.4e}]", flush=True)
        print(f"  N_IS    = {result_IS.getOuterSampling()}", flush=True)

    """
    DEBUT DE CODE
    """
    update_degree(n0)
    event, g_ot, sigma_func, xt, yt, all_grad = [None] * 6
    xt_eff = None

    if print_3D:
        print_3D_HF()
        sys.exit(0)

    if print_grad_sp:
        print("=== -grad(g) aux points de depart sp A/B/C/D ===", flush=True)
        for lbl, data in best_sol_modes_fixed.items():
            sp = list(data['sp'])
            g_sp, grad_sp, _ = run_HF(sp)
            neg_grad = [-v for v in grad_sp]
            print(f"Mode {lbl} : sp={sp}", flush=True)
            print(f"  g_HF(sp)  = {g_sp:.6f}", flush=True)
            print(f"  grad(sp)  = [{grad_sp[0]:.6f}, {grad_sp[1]:.6f}]", flush=True)
            print(f"  -grad(sp) = [{neg_grad[0]:.6f}, {neg_grad[1]:.6f}]", flush=True)
        sys.exit(0)

    g_ot, sigma_func, xt, yt, all_grad = init_g_ot(g_ot, sigma_func, xt, yt, all_grad)
    if do_EFF:
        print_planche_EFF(g_ot, sigma_func, xt, [])
        g_ot, sigma_func, xt, yt, all_grad, xt_eff = run_EFF(g_ot, sigma_func, xt, yt, all_grad)
        print_planche_EFF(g_ot, sigma_func, xt, xt_eff)
        print_EFF_graphs()
    event, g_ot, sigma_func, xt, yt, all_grad = init_FORM(g_ot, sigma_func, xt, yt, all_grad)

    if event is None:
        if best_sol_modes_fixed is not None:
            print_visu(None, None, None, None, [], None)
            sys.exit(0)
        print('Aucune branche active', flush=True)
        sys.exit(1)

    if do_warmstart:
        starting_points = np.array([[0.0, 0.0]])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event) #FORM simple avec event créé
        modes, best_sps = FORM_warm_start(modes, best_sps, g_ot, sigma_func, xt, yt, all_grad) #warm_start puis FORM multistart avec event warm
    else:
        starting_points = np.vstack([xt, [[0.0, 0.0]]]) if do_multistart else np.array([[0.0, 0.0]])
        modes, best_sps = FORM_all_modes(starting_points, tol_all_modes, event)

    best_result = modes[0] if modes else None
    best_sp     = best_sps[0] if best_sps else None
    if best_result is None:
        print('Aucun FORM ne marche.', flush=True)
        sys.exit(1)
    if len(modes)>1:
        print('On a trouvé plus de 1 mode! Les résultats du mode 2 sont:')
        print_results(modes[1], g_ot)
        print('Les résultats du mode 1 sont : ')
    print_results(best_result, g_ot)
    if do_IS and modes:
        result_IS = run_IS(modes, event)
        print_results_IS(result_IS)
    print_visu(best_result, best_sp, xt, g_ot, modes, xt_eff)
