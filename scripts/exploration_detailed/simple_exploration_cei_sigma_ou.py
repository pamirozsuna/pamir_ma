import os
if os.getcwd().split("/")[-1] == "examples":
    os.chdir('..')
    
# This will reload all imports as soon as the code changes
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[6]:

import matplotlib.pyplot as plt
#try:
#    import matplotlib.pyplot as plt
#except ImportError:
#    import sys
#    get_ipython().system('{sys.executable} -m pip install matplotlib')
#    import matplotlib.pyplot as plt

import sys    
import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch

from neurolib.models.wc import WCModel
from neurolib.utils.loadData import filterSubcortical
from scipy.io import loadmat
from neurolib.utils.loadData import Dataset
import neurolib.utils.loadData as ld
import neurolib.utils.functions as func
import neurolib.utils.devutils as du

# a nice color map
plt.rcParams['image.cmap'] = 'plasma'

def averageMat(Mat):
    avMat = np.zeros((94,94))
    for i in range(len(Mat)):
        avMat = avMat + Mat[i]
    avMat = avMat/len(Mat)
    return avMat


data_path = "/mnt/raid/data/MScTheses/msc_pamir/SCZ-FC-modelling"

#data_path = "/Users/pamirozsuna/Desktop/msc_pamir/SCZ-FC-modelling"

C_data = np.load(data_path+"/AvgCmatrixSCZ.npy")
D_data = np.load(data_path+"/AvgDmatrixSCZ.npy")

Cmat_tmp = averageMat(C_data)
Dmat_tmp = averageMat(D_data)

Cmat = filterSubcortical(Cmat_tmp)
Dmat = filterSubcortical(Dmat_tmp)

FC_data = np.load(data_path+"/AvgFmatrixSCZ.npy")
timeseries = np.load(data_path+"/AvgTmatrixSCZ.npy")

FC_tmp = averageMat(FC_data)
FCs = filterSubcortical(FC_tmp)

total_parameters = []
total_gbc_model = []
count = 0

#input_exc_ext = np.linspace(0, 1.5, 25)
#input_c_ii = np.linspace(0, 5, 50)
#input_c_ie = np.linspace(5, 15, 50)
input_c_ei = np.linspace(5, 15, 50)
#input_K_gl = np.linspace(0, 1.0, 25)
input_sigma_ou = np.linspace(0, 1.5, 25)

#total_runs = len(input_exc_ext)*len(input_c_ii)*len(input_c_ie)*len(input_c_ei)*len(input_K_gl)*len(input_sigma_ou)
total_runs = len(input_c_ei)*len(input_sigma_ou)

for index_input_c_ei  in range(len(input_c_ei)):
        for index_input_sigma_ou in range(len(input_sigma_ou)):
                #for index_input_K_gl in range(len(input_K_gl)):
                input_parameters = []
                wc = WCModel(Cmat = Cmat, Dmat = Dmat)
                wc.params['duration'] = 40*1000
                wc.params['exc_ext'] = 0.45
                wc.params['c_excinh'] = input_c_ei[index_input_c_ei]
                wc.params['c_inhexc'] = 9.67
                wc.params['c_inhinh'] = 0
                wc.params['K_gl'] = 1.84
                wc.params['sigma_ou'] = input_sigma_ou[index_input_sigma_ou]
                input_parameters.append(input_c_ei[index_input_c_ei])
                input_parameters.append(input_sigma_ou[index_input_sigma_ou])
                #input_parameters.append(wc.params['c_inhexc'])
                #input_parameters.append(wc.params['c_inhinh'])
                #input_parameters.append(wc.params['K_gl'])
                #input_parameters.append(wc.params['sigma_ou'])     
                total_parameters.append(input_parameters)                       
                wc.run(bold = True)
                count += 1
                gbc_model = np.mean(func.fc(wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000]))
                total_gbc_model.append(gbc_model)
                print("GBC: " + str(gbc_model))
                print("Run number " + str(count) + "/" + str(total_runs))
                            
np.save("gbc_cei_sigma_ou1.npy", total_gbc_model)    
np.save("parameters_cei_sigma_ou1.npy", total_parameters)     

