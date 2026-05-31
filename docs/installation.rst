Installation
============

Our SpaWeaver is developed using the PyTorch framework. Below, we provide instructions for creating a conda environment capable of running our model.

.. code-block:: python

   1) Create conda environment
   
   #Create an environment called SpaWeaver

   conda create -n SpaWeaver python=3.8

   #Activate your environment

   conda activate SpaWeaver


   #Several important packages are listed below in case you prefer not to install too many packages.
   #We recommend installing the above Python packages one by one to avoid potential errors.

   anndata==0.10.8
   scanpy==1.10.3
   numpy==1.26.4
   scipy==1.13.1
   pandas==2.0.3
   scikit-image==0.24.0
   scikit-learn==1.6.1
   torch==2.3.3
   huggingface-hub==1.3.3
   timm==1.0.24
   torchvision==0.18.1
 
   2) We have developed an easy-to-use Python package, which can be installed using the following command:

   pip install SpaWeaver

   3) To use the environment in jupyter notebook, add python kernel for this environment.

   pip install ipykernel

   python -m ipykernel install --user --name=SpaWeaver


We provide concrete examples in Tutorials to illustrate this in detail.