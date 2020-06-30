# Setup
## Anaconda
We assume that Anaconda is installed on your environment. Python naitive virtual environments work as well, but different commands would be required.

## Creating a new Conda environment
To create a new environment called CBCT with python 3.7, run
`conda create -n CBCT python=3.7 anaconda`

Conda requires you to activate this environment. To enter the environment, run
`conda activate CBCT`
To leave it, you can just close the window, or run
`conda deactivate`
If your version of Anaconda is older than version 4.4 (see conda --version), then replace conda with source in the above (and consider upgrading your Anaconda!).

## Installing the required packages using pip
If not using Anaconda, you'd want to install the required packages into your virtual environment. 
To do this, first activate your virtualenvironment, then run:

`pip install -r requirements.txt`

### PyTorch
This command installs PyTorch. Note that if you have a GPU, you should follow this instead: https://pytorch.org/get-started/locally/

`conda install pytorch torchvision -c pytorch`