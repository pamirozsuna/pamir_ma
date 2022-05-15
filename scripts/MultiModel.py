import os
if os.getcwd().split("/")[-1] == "examples":
    os.chdir('..')

# import stuff

# try:
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import neurolib.utils.functions as func
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess


from neurolib.models.multimodel import WilsonCowanNetwork, WilsonCowanNode, MultiModel

from neurolib.utils.loadData import Dataset

ds = Dataset("gw", fcd = True)

def loadMat():
    Cmatrix = []
    Dmatrix = []
    Fmatrix = []
    Tmatrix = []
    FC_LR = []
    FC_RL = []
    TC_LR = []
    TC_RL = []

    dataDir = "../pamir_ma/"

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


import glob
from scipy.io import loadmat
from neurolib.utils.loadData import filterSubcortical

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

#dummy_sc = np.array([[0.0, 1.0], [1.0, 0.0]])
# init MultiModelnetwork with 2 WC nodes with dummy sc and no delays
K_gl = 1.84
connectivity_matrix_data = ds.Cmat*K_gl
delay_matrix_data = ds.Dmat


#mm_net = WilsonCowanNetwork(connectivity_matrix=connectivity_matrix_data, delay_matrix=delay_matrix_data, exc_mass_params = dict_exc, inh_mass_params = dict_inh)
mm_net = WilsonCowanNetwork(connectivity_matrix=connectivity_matrix_data, delay_matrix=delay_matrix_data, local_connectivity =  np.array([[16.0, 9.67], [10.33, 0]]))
#print(mm_net)
# each network is an proper python iterator, i.e. len() is defined

# now let us check the parameters.. for this we initialise MultiModel in neurolib's fashion
#aln_net = MultiModel(mm_net)
wc_net = MultiModel(mm_net)
# parameters are accessible via .params
#aln_net.params
#print(wc_net.params)

wc_net.params["duration"] = 40*1000
wc_net.params["backend"] = "numba"
# numba uses Euler scheme so dt is important!
wc_net.params["dt"] = 0.1
wc_net.params["sampling_dt"] = 1.0


for node in range(len(ds.Cmat)):
    wc_net.params[f"WCnet.WCnode_{node}.WCmassEXC_0.input_0.sigma"] = 0.05
    wc_net.params[f"WCnet.WCnode_{node}.WCmassEXC_0.ext_drive"] = 0.45

wc_net.run(bold = True)

fig, axs = plt.subplots(1, 2, figsize=(16, 4))
axs[0].set_title("Simulated FC")
axs[0].imshow(func.fc(wc_net.q_mean_EXC[:, wc_net.outputs.t > 0]))
axs[1].set_title("Simulated FCD")
axs[1].imshow(func.fcd(wc_net.q_mean_EXC, stepsize=100))
#axs[2].set_title("Mean Power Spectrum")
#fr, pw = func.getMeanPowerSpectrum(wc_net.q_mean_EXC, wc_net.params["dt"])
#axs[2].plot(fr, pw, c='k', lw = 2)
plt_name = "WCMultiModel"+str(wc_net.params["duration"])+str(wc_net.params["dt"])+"SimulationResults"
plt.savefig(plt_name+".png", dpi=500)

scores = [func.matrix_correlation(func.fc(wc_net.q_mean_EXC[:, -int(5000/wc_net.params['dt']):]), fcemp) for fcemp in ds.FCs]
print("Correlation per subject:", [f"{s:.2}" for s in scores])
print("Mean FC/FC correlation: {:.2f}".format(np.mean(scores)))


np.save(plt_name + "t" +".npy", wc_net.outputs.t)    
np.save(plt_name + "q_mean_EXC" +".npy", wc_net.outputs.q_mean_EXC)    
np.save(plt_name + "BOLD" +".npy", wc_net.outputs.BOLD.BOLD)    
np.save(plt_name + "BOLD.t_BOLD" +".npy", wc_net.outputs.BOLD.t_BOLD)   

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
#axs[0].imshow(func.fc(wc.exc[:,:-2000]))
axs[0].imshow(func.fc(wc_net.BOLD.BOLD[:, wc_net.BOLD.t_BOLD>10*1000]))
#axs[0].imshow(func.fc(wc.BOLD.BOLD[:, 5:]))
axs[1].plot(wc_net.t, wc_net.q_mean_EXC.T, alpha=0.8)
axs[1].set_xlim(0, 200)
axs[2].set_title("Mean Power Spectrum")
fr, pw = func.getMeanPowerSpectrum(wc_net.q_mean_EXC, wc_net.params["dt"])
axs[2].plot(fr, pw, c='k', lw = 2)
plt_name = "WCMultiModel"+str(wc_net.params["duration"])+str(wc_net.params["dt"])+"SimulationResultsBOLD"
plt.savefig(plt_name+".png", dpi=500)