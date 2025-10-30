# Spectroscopic tools for data analysis


## 🚀 Overview  
`Spectools` is a Python package designed to streamline **spectroscopic data analysis**.  
It provides functions to read, manipulate, and analyze spectra — including line profile fitting, flux and velocity measurements, and visualizations.  

---

## 🧩 Features  
- Read and handle standard spectroscopic data formats (e.g., FITS).  
- Perform basic preprocessing: normalization, continuum subtraction.  
- Fit absorption or emission line profiles with open-source codes: Starlight and IFSCube.  
- Extract spectral parameters such as centroid, velocity dispersion, and integrated flux.  
- Visualize observed spectra and fitted components.  
- Ready-to-use Jupyter notebooks demonstrating key functionalities.

---

## ⚙️ Installation  

Clone the repository:
```bash
git clone https://github.com/kefrankk/spectools.git
cd spectools
```

Activating the environment:

```
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows
```

Installing the dependencies:
```
pip install -r requirements.txt
```

Now is ready to use!


## 📁 Repository Structure 

```
spectools/
│
├── data/            # Example data or test spectra
├── notebooks/       # Jupyter notebooks with usage examples
├── spectools/       # Source code
├── setup.py         # Installation script
└── README.md        # Project documentation
```


