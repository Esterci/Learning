import os
import glob

from itertools import product

# define the paths into the container
config_path  = 'jobConfig/*'
output_path  = 'Results/'

# create a list of config files
config_list  = glob.glob(config_path)
print(config_list)

# loop over the config files
for iconfig in config_list:
    m_command = """python3 job_tuning.py -c {CONFIG} \\
                    -v {OUT}""".format(CONFIG=iconfig, 
                                       OUT=output_path)

    print(m_command)
    # execute the tuning
    os.system(m_command)