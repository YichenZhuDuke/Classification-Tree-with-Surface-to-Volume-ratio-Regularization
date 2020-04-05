# Classification Tree with Surface-to-Volume-ratio Regularization

This project realizes the method "SVR-Tree" proposed in the paper "Classification Trees for Imbalanced andSparse Data: 
Surface-to-Volume Regularization" and performs numerical studies in that paper.

The file "Tree.py" realizes the method. The file "sampler.py" provides oversampling methods. 
The file "test_data_linux.py" perform numerical studies in the paper. 
The file "test_data_redundant_linux.py" perform additional numerical studies with artifically generated redundant features.
Both file "test_data_linux.py" and "test_data_redundant_linux.py" can be run under linux; if run in windows, errors regarding 
multiprocessing package will occur. We currently do not know how to use this package in windows. We haven't tested these files 
in macOS. 
