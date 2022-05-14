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

import multiprocessing


import sys
import ast

#python fitting_wc.py 'HCP_REST1_TC' 'HCP_REST1_FC' 'HCP_SC' 'HCP_len' 'HCP' '[1.0,-1.0]' '640' '160' '50' '[0.,4.]' '[0.,.5]' '[0.5,0.75]' '0.45'

# Input data
timeSeries = sys.argv[1]
fcMat = sys.argv[2]
scMat = sys.argv[3]
lenMat = sys.argv[4]
dictKey = sys.argv[5]

# Model parameters
weightList = ast.literal_eval(sys.argv[6])
popInit = ast.literal_eval(sys.argv[7])
pop = ast.literal_eval(sys.argv[8])
ngen = ast.literal_eval(sys.argv[9])
K_gl = ast.literal_eval(sys.argv[10])          #[0.,4.]
sigma_ou = ast.literal_eval(sys.argv[11])       #[0.,.5]
exc_ext = ast.literal_eval(sys.argv[12]) 

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

    dataDir = "/mnt/raid/data/SFB1315/BScTheses2022/"

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
model.params['exc_ext'] = exc_ext
model.params['inh_ext'] = 0
model.params['c_excinh'] = 10.333333333333334
model.params['c_inhexc'] = 9.666666666666666
model.params['c_inhinh'] = 0

# Resting state fits
model.params['dt'] = 0.1
model.params['duration'] = 10 * 60 * 1000 #ms
model.params['save_dt'] = 10.0 # 10 ms sampling steps for saving data, should be multiple of dt


MEDIAN_KILLER = False

def evaluateSimulation(traj):
    rid = traj.id
    model = evolution.getModelFromTraj(traj)
    model.randomICs() # initiate the model with random initial contitions
    #print("Running run id: " + str(rid))
    defaultDuration = model.params['duration']
    #invalid_result = (-np.inf, np.inf, -np.inf, )
    invalid_result = (-np.inf, np.inf) #invalid_result = (np.inf,)

    # -------- stage wise simulation --------
    
    # Stage 1 : simulate for a few seconds to see if there is any activity
    # ---------------------------------------
    model.params['duration'] = 11*1000.
    model.run()
    max_amp_output = np.max(
          np.max(model.output[:, model.t > 1000], axis=1) 
        - np.min(model.output[:, model.t > 1000], axis=1)
    )
    max_output = np.max(model.output[:, model.t > 1000])
    
    # check if stage 1 was successful
    # or np.max(model_pwrs) < 1 or np.median(model.output) < 1
    # info: filter of median<1 avoids down-to-up solutions
    # filtering median > 15 avoids finding solutions that stay in the up-state all the time
    if max_output > 2.5 or max_amp_output < 0.4 or ((np.median(model.output) < 0.6 or np.median(model.output) > 15) and MEDIAN_KILLER):
        print("invalid result: " + str(max_output) + " " + str(max_amp_output))
        #print(model.params)
        #print("max output", max_output)
        #print("max_amp_output", max_amp_output)
        #exit()
        return invalid_result, {}
    
    # Stage 2: full and final simulation
    # ---------------------------------------
    model.params['duration'] = defaultDuration
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
    return fitness_tuple, {
        "median_rate" : np.median(model.output),
        "output": model.output[:, ::int(model.params['save_dt']/model.params['dt'])]
    } # we require a dictionary with at least a single result for storing the results in the hdf

#pars = ParameterSpace(['sigma_ou', 'K_gl', 'exc_ext'], 
#                      [sigma_ou, K_gl, exc_ext])
pars = ParameterSpace(['sigma_ou', 'K_gl'], 
                      [sigma_ou, K_gl])

#weightList = [1.0, -1.0]
evolution = Evolution(evaluateSimulation,	#Evaluation function of a run that provides a fitness vector and simulation outputs
                      pars, 
                      weightList = weightList,  #A list of optimization weights for the `fitness_tuple`,positive values will lead to a maximization, 
#                      							negative values to a minimzation. The length of this list must be the same as the length of the `fitness_tuple`.  
#                      							List of floats that defines the dimensionality of the fitness vector returned from evalFunction 
#                      							and the weights of each component for multiobjective optimization (positive = maximize, negative = minimize). 
#                      							If not given, then a single positive weight will be used, defaults to None               
                      model = model, 
                      ncores=multiprocessing.cpu_count(),
                      POP_INIT_SIZE=popInit, #640 
#                                           The size of the initial population that will be 
#                                           randomly sampled in the parameter space `pars`.
#                                           Should be higher than POP_SIZE. 
                      POP_SIZE = pop, #160 
#                                       Size of the population that should evolve. Must be an
#                                       even number.
                      NGEN=ngen, #50 
#                                  Number of generations to simulate the evolution for.
                      algorithm='adaptive',
                      filename="evolution-results.hdf")

evolution.run(verbose = False)

fname = os.path.join("/mnt/raid/data/MScTheses/msc_pamir/results/", "evolution-" + evolution.trajectoryName + ".dill") 
print("Saving evolution to " + fname)
dill.dump(evolution, open(fname, "wb"))

import datetime
dirResults = "/mnt/raid/data/MScTheses/msc_pamir/results/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.mkdir(dirResults)

with open(dirResults + '/metadata.txt', 'w') as fn:
    fn.writelines('Input data: \n')
    fn.writelines('BOLD time series: ' + timeSeries + '\n')
    fn.writelines('FC matrix: ' + fcMat + '\n')
    fn.writelines('SC matrix: ' + scMat + '\n')
    fn.writelines('Length matrix: ' + lenMat + '\n')
    
    fn.writelines('Parameters: ' + '\n')
    fn.writelines('FC weight: ' + str(weightList) + '\n')
    fn.writelines('Initial population size: ' + str(popInit) + '\n')
    fn.writelines('Population size: ' + str(pop) + '\n')
    fn.writelines('Number of generations: ' + str(ngen) + '\n')
    fn.writelines('Global Coupling strength K_gl: ' + str(K_gl) + '\n')
    fn.writelines('Noise strength sigma_ou: ' + str(sigma_ou) + '\n')

import shutil
shutil.move(fname, dirResults + "/evolution-" + evolution.trajectoryName + ".dill")
shutil.move('./data/hdf/evolution-results.hdf', dirResults + '/evolution-results.hdf')
