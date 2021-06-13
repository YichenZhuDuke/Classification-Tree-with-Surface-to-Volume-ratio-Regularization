# Classification Tree with Surface-to-Volume-ratio Regularization

This project realizes the method "SVR-Tree" proposed in the paper "Classification Trees for Imbalanced Data: 
Surface-to-Volume Regularization" and performs numerical studies in that paper.

The file "Tree.py" realizes the method. The file "sampler.py" provides oversampling methods. 
The file "test_data_nested_cv_linux_resubmit.py" perform numerical studies for all methods except Hellinger distance tree in the paper and can only be run under linux.
The file "weka_test_resubmit.py" perform numerical studies for Hellinger distance tree in the paper. It currently calls windows command lines using os packages, but can also work in other operation systems provided corresponding changes are made to the command line codes.

The files "test_data_linux.py" and "test_data_linux_redundant.py" are not used in the current version of the paper.

