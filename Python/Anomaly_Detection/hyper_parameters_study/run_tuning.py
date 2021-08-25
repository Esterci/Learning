import os
import glob

from itertools import product

# define the paths into the container
data_path    = '../sample/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et%i_eta%i.npz'
ref_path     = '../sample/references/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM2.bkg.VProbes_EGAM7.GRL_v97_et%i_eta%i.ref.pic.gz'
config_path  = 'job_test/*'
output_path  = 'output_test2_et%i_eta%i/'

# create a list of config files
config_list  = glob.glob(config_path)
print(config_list)

# loop over the bins
for iet, ieta in product([2], [0]):
    print('Processing -> et: %i | eta: %i' %(iet, ieta))
    # format the names
    data_file = data_path %(iet, ieta)
    ref_file  = ref_path  %(iet, ieta)
    out_name  = output_path %(iet, ieta)

    # loop over the config files
    for iconfig in config_list:
        m_command = """python3 job_tuning.py -c {CONFIG} \\
                       -d {DATAIN} \\
                       -v ./ \\
                       -r {REF}""".format(CONFIG=iconfig, DATAIN=data_file, OUT=out_name, REF=ref_file)

        print(m_command)
        # execute the tuning
        os.system(m_command)