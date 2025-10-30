
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from spectools import specutils, spectrum_io, bpt_diagrams, analisys
import spectools.chemical_abundance_functions as abundance_f
import subprocess
import matplotlib.pyplot as plt
# import specutils




def main(): 
   
    print(f'\n #### Welcome to Spectools workflow #### \n')

    # __file__ = caminho do script atual
    base_dir = Path(__file__).resolve().parent.parent   # sobe de scripts/ para spectools/
    data_dir = base_dir / "data/A194/"

    library_dir     = base_dir / "data/STARLIGHTv04/CB19CSFBasesDir/"
    mask_dir        = base_dir / "data/STARLIGHTv04/"
    starlight_dir   = base_dir / "data/STARLIGHTv04/"

    starlight_output_file   = base_dir / 'data/starlight_output.json'
    ifscube_output_file     = base_dir / 'data/ifscube_output.json'


    # ---------------------------------------> Downloading data <-----------------------------------------------














    # ---------------------------------------> Running Starlight <-----------------------------------------------
    # list starlight input files if they already exist
    files = [f for f in os.listdir(data_dir) if '.' not in f]

    if files != []:
        print(f' -> STARLIGHT output files already exist. \n')
    else: 
        galaxies  = [f for f in os.listdir(data_dir) if '.txt' in f]

        print(f' -> Configuring Starlight input \n')
        # Starlight only runs inside the STARLIGHTv04 directory, where the executable is located
        conf_file = specutils.create_starlight_input_file(
            data_directory = './../A194/',
            list_galaxies = galaxies,
            n_files = 3, 
            library = 'CB19_16x5')

        conf_path = mask_dir / "config.in"
        with open(conf_path, "w") as f:
            f.write(conf_file)

        print(f' -> Running Starlight... \n')
        specutils.run_starlight(starlight_dir)

    # After running Starlight, read the output files and saving into a list
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and  '.' not in f]

    if files == []:
        print(f' -> No STARLIGHT output files found in {data_dir} \n')
        exit()


    # ------> Processing Starlight output files <------
    if any(f.endswith("_starlight.fits") for f in os.listdir(data_dir)) and any(f.endswith("_starlight.png") for f in os.listdir(data_dir)):
        print(" -> Starlight fits and png files already exist.\n")
    else:
        print(f' -> Reading Starlight output... \n')
        starlight_output = specutils.read_starlight_output(files, filepath=str(data_dir) + '/')

        with open(starlight_output_file, 'w') as f:
            json.dump(starlight_output, f)

        print(f' -> Creating fits files for ifscube fit... \n')
        for galaxy, data in starlight_output.items():
            fig = specutils.plot_starlight_spectrum(data)
            fig.savefig(data_dir / f"{galaxy}_starlight.png")
            plt.close(fig)

            new_fits = specutils.create_new_fits_from_starlight(data)
            new_fits.writeto(data_dir / f'{galaxy}_starlight.fits', overwrite=True)




    # ---------------------------------------> Running ifscube <-----------------------------------------------
    ifscube_output_files = [f for f in os.listdir(data_dir) if f.endswith('starl.fits') and '_g' not in f and '_2g' not in f]
    if ifscube_output_files != []:
        print(f' -> IFSCUBE output files already exist in {data_dir} \n')
    else:
        # print(f' -> Creating ifscube configuration file... \n')
        # Here we define the kinematic constraints 
        ifscube_config_file = base_dir / 'data/ifscube_config.cfg'
        stdout_ifscube = {} # to save the output log of ifscube runs

        velocity = '0, +- 300'
        velocity_broad = '0, +- 300'
        sigma = '100, 40:400'
        sigma_broad = '120, 50:200'
        lines_config = {
            "OII_3726": specutils.make_line(3726.032, velocity, sigma),
            "OII_3729": specutils.make_line(3728.815, velocity, sigma),
            "OIII_4363": specutils.make_line(4363.210, velocity, sigma),
            "Hb_4861": specutils.make_line(4861.325, velocity, sigma),
            "OIII_5007": specutils.make_line(5006.84, velocity, sigma, k_group=0),
            "OIII_4959": specutils.make_line(4958.91, velocity, sigma,
                                   amplitude="peak, 0:, OIII_5007.amplitude / 2.98", k_group=0),
            "OI_6300": specutils.make_line(6300.304, velocity, sigma),
            "Ha_6563": specutils.make_line(6562.80, velocity, sigma),
            "NII_6583": specutils.make_line(6583.46, velocity, sigma, k_group=12),
            "NII_6548": specutils.make_line(6548.04,
                                  amplitude="peak, 0:, NII_6583.amplitude / 3.06", k_group=12),
            "SII_6716": specutils.make_line(6716.44, velocity, sigma, k_group=13),
            "SII_6731": specutils.make_line(6730.82, amplitude="peak, 0:", k_group=13),
        }
        
        print(f' -> Running IFSCube fits... \n')
        for file in ifscube_output_files:
            galaxy = os.path.splitext(file)[0]

            # print(f' -> Checking if galaxy {galaxy} has detected emission lines... \n')
            do_fit = spectrum_io.select_do_fit(data_dir/file, value=3)
            if do_fit == True:
                out_image = data_dir + galaxy + '_g.fits'
                config = spectrum_io.generate_ifscube_configuration_file(continuum_degree=1, lines=lines_config, out_image=out_image)

                with open(ifscube_config_file, "w") as f:
                    f.write(config)

                output = subprocess.run(["specfit", "-oc", ifscube_config_file, data_dir/file], check=True, text=True)
                stdout_ifscube[galaxy] = output.stdout
            else: 
                print(f'Skipping galaxy {galaxy}. No emission lines detected.')
        
        
    ifscube_fitted_galaxies = [f for f in os.listdir(data_dir) if f.endswith('_g.fits') or f.endswith('_2g.fits')]
    if ifscube_fitted_galaxies == []:
        print(f' -> No IFSCUBE fitted files found. \n')
    else:
        print(f' -> Creating IFSCUBE fitted spectra figures... \n')
        ifscube_output = {}
        for galaxy in ifscube_fitted_galaxies:
            fig = specutils.plot_ifscube_spectra(data_dir, galaxy)
            fig.savefig(data_dir/galaxy.replace('.fits', '.jpg'))
            plt.close(fig)

            # Read IFSCUBE output into a dictionary
            ifscube_output = spectrum_io.read_ifscube_output(data_dir, galaxy, ifscube_output)

    with open(ifscube_output_file, 'w') as f:
        json.dump(ifscube_output, f)




    

 


if __name__ == "__main__":

    main()

 

