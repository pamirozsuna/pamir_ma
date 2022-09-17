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

from neurolib.models.wc_heterogeneous import WCModelHeterogeneous
from neurolib.utils.loadData import filterSubcortical
from scipy.io import loadmat
from neurolib.utils.loadData import Dataset
import neurolib.utils.loadData as ld
import neurolib.utils.functions as func
import neurolib.utils.devutils as du

selected_indices = sys.argv[1]
value_cei = int(sys.argv[2])

# a nice color map
plt.rcParams['image.cmap'] = 'plasma'

def averageMat(Mat):
    avMat = np.zeros((94,94))
    for i in range(len(Mat)):
        avMat = avMat + Mat[i]
    avMat = avMat/len(Mat)
    return avMat


#data_path = "/mnt/raid/data/MScTheses/msc_pamir/SCZ-FC-modelling"

data_path = "/Users/pamirozsuna/Desktop/msc_pamir/SCZ-FC-modelling"

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
total_gbc_model_frontal = []
total_gbc_model_parietal = []
total_gbc_model_occipital = []
total_gbc_model_temporal = []
count = 0
indices = []



#frontal_indices = 1-2,  3-12, 15-32, 35 - 36, 37 - 38, 39 - 40,  73 - 74
frontal_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 67, 68]
#occipital_indices = 13 - 14, 47 - 60
occipital_indices =  [13, 14, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
#parietal_indices =  61 - 62, 63 - 72
parietal_indices = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
#temporal_indices = 33 - 34, 83 - 86, 87 - 88, 89 - 90, 91 - 92, 93 - 94
temporal_indices = [33, 34, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]


if selected_indices == "temporal":
    indices = temporal_indices
if selected_indices == "frontal":
    indices = frontal_indices
if selected_indices == "parietal":
    indices = parietal_indices
if selected_indices == "occipital":
    indices = occipital_indices


input_exc_ext = np.linspace(0.4, 0.75, 5)
input_c_ii = np.linspace(0., 2, 5)
input_c_ie = np.linspace(8., 11.5, 10)
input_c_ei = input_c_ie[value_cei-1]
input_k_gl = np.linspace(1.6, 2.1, 5)
input_sigma_ou = np.linspace(0., 0.05, 5)

ext_exc_tmp = np.full(80, 0.5)
c_excinh_tmp = np.full(80, 10.0)
c_inhexc_tmp = np.full(80, 8.67)
c_inhinh_tmp = np.full(80, 0.22)
k_gl_tmp = np.full(80, 1.84)
sigma_ou_tmp = np.full(80, 4.28e-05)

#input_K_gl = np.linspace(1.8, 1.9, 2)
#input_sigma_ou = np.linspace(0., 0.1, 2)

total_runs = len(input_exc_ext)*len(input_c_ii)*len(input_c_ie)*len(input_k_gl)*len(input_sigma_ou)
#total_runs = len(input_exc_ext)*len(input_c_ii)*len(input_c_ie)*len(input_c_ei)

def calculate_region_gbc(indices, bold_signal):
    tmp = []
    for i in indices:
        tmp.append(bold_signal[i-1])
    return np.mean(func.fc(tmp))

for index_input_exc_ext in range(len(input_exc_ext)):
    for index_input_c_ii in range(len(input_c_ii)):
        for index_input_c_ie in range(len(input_c_ie)):
            #for index_input_c_ei in range(len(input_c_ei)):
            for index_input_K_gl in range(len(input_k_gl)):
                for index_input_sigma_ou in range(len(input_sigma_ou)):
                    input_parameters = []
                    wc = WCModelHeterogeneous(Cmat = Cmat, Dmat = Dmat)
                    wc.params['duration'] = 40*1000
                    for i in indices:
                        ext_exc_tmp[i-1] = input_exc_ext[index_input_exc_ext]
                        c_excinh_tmp[i-1] = input_c_ei
                        c_inhexc_tmp[i-1] = input_c_ie[index_input_c_ie]
                        c_inhinh_tmp[i-1] = input_c_ii[index_input_c_ii]
                        k_gl_tmp[i-1] = input_k_gl[index_input_K_gl]
                        sigma_ou_tmp[i-1] = input_sigma_ou[index_input_sigma_ou]
                    wc.params['exc_ext'] = ext_exc_tmp
                    wc.params['c_excinh'] = c_excinh_tmp
                    wc.params['c_inhexc'] = c_inhexc_tmp
                    wc.params['c_inhinh'] = c_inhinh_tmp
                    wc.params['K_gl'] = k_gl_tmp
                    #wc.params['sigma_ou'] = sigma_ou_tmp
                    input_parameters.append(input_exc_ext[index_input_exc_ext])
                    input_parameters.append(input_c_ei)
                    input_parameters.append(input_c_ie[index_input_c_ie])
                    input_parameters.append(input_c_ii[index_input_c_ii])
                    input_parameters.append(input_k_gl[index_input_K_gl])
                    input_parameters.append(input_sigma_ou[index_input_sigma_ou])     
                    total_parameters.append(input_parameters)                       
                    wc.run(bold = True)
                    count += 1
                    gbc_model = np.mean(func.fc(wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000]))
                    gbc_frontal = calculate_region_gbc(frontal_indices, wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000])
                    gbc_parietal = calculate_region_gbc(parietal_indices, wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000])
                    gbc_temporal = calculate_region_gbc(temporal_indices, wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000])
                    gbc_occipital = calculate_region_gbc(occipital_indices, wc.BOLD.BOLD[:, wc.BOLD.t_BOLD>10000])
                    total_gbc_model.append(gbc_model)
                    total_gbc_model_frontal.append(gbc_frontal)
                    total_gbc_model_parietal.append(gbc_parietal)
                    total_gbc_model_occipital.append(gbc_occipital)
                    total_gbc_model_temporal.append(gbc_temporal)
                    print("GBC: " + str(gbc_model))
                    print("GBC temporal: " + str(gbc_temporal))
                    print("GBC frontal: " + str(gbc_frontal))
                    print("GBC parietal: " + str(gbc_parietal))
                    print("GBC occipital: " + str(gbc_occipital))
                    print("Run number " + str(count) + "/" + str(total_runs))
                        
np.save("gbc_"+str(selected_indices)+str(value_cei)+"_total.npy", total_gbc_model)    
np.save("gbc_"+str(selected_indices)+str(value_cei)+"_frontal.npy", total_gbc_model_frontal)    
np.save("gbc_"+str(selected_indices)+str(value_cei)+"_temporal.npy", total_gbc_model_temporal)    
np.save("gbc_"+str(selected_indices)+str(value_cei)+"_occipital.npy", total_gbc_model_occipital)    
np.save("gbc_"+str(selected_indices)+str(value_cei)+"_parietal.npy", total_gbc_model_parietal)    
np.save("parameters_"+str(selected_indices)+str(value_cei)+".npy", total_parameters)     

