import os
import numpy as np
import logging 
import dill

from neurolib.models.wc import WCModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution
import neurolib.utils.functions as func
import neurolib.utils.devutils as du
from neurolib.utils.loadData import filterSubcortical

import neurolib.optimize.evolution.deapUtils as deapUtils

from neurolib.utils.loadData import Dataset

from scipy.io import loadmat
import glob

# Input data
timeSeries = 'HCP_REST1_TC'
fcMat = 'HCP_REST1_FC'
scMat = 'HCP_SC'
lenMat = 'HCP_len'
dictKey = 'HCP'

# Model parameters
weightList = [1.0,-1.0]
popInit = 640
pop = 160
ngen = 50

# Write dataset loading function
def loadMat():
    Cmatrix = []
    Dmatrix = []
    Fmatrix = []
    Tmatrix = []
    FC_LR = []
    FC_RL = []
    TC_LR = []
    TC_RL = []

    dataDir = "/Users/pamirozsuna/Desktop/Masterarbeit/pamir_master_thesis/"

    subjectsFC = sorted(glob.glob(dataDir + "FC/1_AAL/*")) # reading the file names using glob
    subjectsSC = sorted(glob.glob(dataDir + "SC/1_AAL/*"))
    for subjectSC in subjectsSC:
	    for subjectFC in subjectsFC:
	    	# take subjects only if they have all SC, LR_FC and RL_FC matrices
	    	if(subjectSC[-6:]==subjectFC[-6:] and os.path.isfile(subjectFC + '/rfMRI_REST1_LR/FC.mat')  and os.path.isfile(subjectFC + '/rfMRI_REST1_RL/FC.mat')):
	    		this_cm = loadmat(subjectSC + "/DTI_CM.mat")['SC']
	    		Cmatrix.append(this_cm)
	    		this_dc = loadmat(subjectSC + "/DTI_LEN.mat")['LEN']
	    		Dmatrix.append(this_dc)
	    		FC_LR = loadmat(subjectFC + '/rfMRI_REST1_LR/FC.mat')["fc"]
	    		TC_LR = loadmat(subjectFC + '/rfMRI_REST1_LR/TC.mat')["tc"]
	    		FC_RL = loadmat(subjectFC + '/rfMRI_REST1_RL/FC.mat')["fc"]
	    		TC_RL = loadmat(subjectFC + '/rfMRI_REST1_RL/TC.mat')["tc"]
	    		Fmatrix.append(filterSubcortical(np.mean( np.array([FC_LR, FC_RL]), axis=0)))
	    		Tmatrix.append(filterSubcortical(np.mean( np.array([TC_LR[:,0:355], TC_RL[:,0:355]]), axis=0), axis=0))
        
    return Cmatrix,Dmatrix,Tmatrix,Fmatrix

def averageMat(Mat):
    avMat = np.zeros((94,94))
    for i in range(len(Mat)):
        avMat = avMat + Mat[i]
    avMat = avMat/len(Mat)
    return avMat

def ComputeAverageMats(Cmatrix, Dmatrix):
    Cmat = averageMat(Cmatrix)
    Dmat = averageMat(Dmatrix)
    return Cmat, Dmat

ds = Dataset(datasetName = 'gw', normalizeCmats = None)
dataDict_sc, dataDict_len, dataDict_ts, dataDict_fc = loadMat()

GW_FC_mean = np.mean(ds.FCs, axis=0) # Average FC GW
ds.FCs = []
ds.FCs.append(GW_FC_mean)
ds.FCs.append(np.mean(dataDict_fc, axis=0)) # Add average FC HCP
print("ds FCs shape: " + str(np.shape(ds.FCs)))

print(np.shape(dataDict_ts))
GW_BOLD_mean = np.mean(ds.BOLDs, axis = 0)
print(np.shape(GW_BOLD_mean))
ds.BOLDs = []
ds.BOLDs.append(GW_BOLD_mean)
ds.BOLDs.append(np.mean(dataDict_ts, axis=0))
print("ds BOLDs length: " + str(len(ds.BOLDs)))

ds.Cmat, ds.Dmat = ComputeAverageMats(dataDict_sc, dataDict_len)
ds.Cmat = filterSubcortical(ds.Cmat)
ds.Cmats = dataDict_sc
ds.data = []
ds.Dmat = filterSubcortical(ds.Dmat)
ds.Dmats = dataDict_len
ds.dsBaseDirectory = '../Data/'

model = WCModel(Cmat = ds.Cmat, Dmat = ds.Dmat)

# structural values
model.params['exc_ext'] = 0.45
model.params['inh_ext'] = 0
model.params['c_excinh'] = 10.333333333333334
model.params['c_inhexc'] = 9.666666666666666
model.params['c_inhinh'] = 0
model.params['K_gl'] = 1.843
model.params['sigma_ou'] = 4.2819e-05

# Resting state fits
model.params['dt'] = 0.1
model.params['duration'] = 10 * 60 * 1000 #ms
model.params['save_dt'] = 10.0 # 10 ms sampling steps for saving data, should be multiple of dt

MEDIAN_KILLER = False

model.run(chunkwise=True, bold = True, chunksize = int(1 * 60 * 1000 / model.params['dt'])) # 1 minute chunks

# -------- fitness evaluation here --------
fits = du.model_fit(model, ds, fcd=True if model.params.duration >= 5 * 60 * 1000 else False)
if "fcd" not in fits:
    fits["fcd"] = 1

scores = []
scores.append(np.mean(fits['fc_scores']))
scores.append(np.mean(fits['fcd']))

fitness_tuple = tuple(scores)
print("fitness_tuple / mean fc score: " + str(fitness_tuple))

scores = [func.matrix_correlation(func.fc(model.exc[:, -int(5000/model.params['dt']):]), fcemp) for fcemp in ds.FCs]
print("Correlation per subject:", [f"{s:.2}" for s in scores])
print("Mean FC/FC correlation: {:.2f}".format(np.mean(scores)))