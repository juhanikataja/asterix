## Asterix 
### What is this about?   
This is the prototyping repo for the Asterix project. We develop and showcase prototype methods for compressing VDFs in Vlasiator.
So far this has been happening in our shared jupyter notebook ```Inno4Scale.ipynb```.     
Currently we utilize the following methods:
+ Zlib lossless compression.
+ ZFP lossy compression.
+ Using a Multi Layer Perceptron [MLP] to train on and compress VDFs.
+ Using a Multi Layer Perceptron as above enriched with Fourier Features.

### Requirements
+ Cargo [instructions](https://doc.rust-lang.org/cargo/getting-started/installation.html)
+ Maturin [GitHub repo](https://github.com/PyO3/maturin)
+ pyzfp [instructions](https://pypi.org/project/pyzfp/)
+ Python >3.7

### Running the Project
The MLP used, is hosted in this repo and is written in Rust. To use it from python we just need to build a shared library that wraps over it. Thankfully this is not too hard to do!    
+ Do a ```pip install -r requirements.txt``` to get all the needed packages.    
+ Inside your virtual environment you can build the project with ```maturin develop --release```.      
+ Open up the notebook ```Inno4Scale.ipynb``` and have fun.   
