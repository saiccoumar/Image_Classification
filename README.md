# CS490 Data Science Capstone Project: Adversarial Robustness in Machine Learning Models
<p align="center">
 <img src="https://github.com/RishabhPandey0403/cs490dsc/assets/55699636/43b0705c-e5c8-4bfa-9638-763683be0f26">
</p>
Sai Coumar, Supriya Dixit

## Datasets:
The datasets used are MNIST, CIFAR10, and SVHN. They can be found in the following links. 

MNIST: https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/ 

CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

SVHN: http://ufldl.stanford.edu/housenumbers/ 

### Summary Statistics:
Brief summary statistics have been created for each dataset in their respective ipynb files. These notebooks include metadata, data transformation (one-hot vectors to human interpretable images), class balance, PCA dimensionality reduction, "mean" images, and traditional summary statistics on the euclidean distance from the mean image and records. The latter was outputted to separate text files.

## Models:
Our control group for this project are basic resnet18 models trained on our datasets. Details about the model can be found within their respective notebooks. in the resnet_models folder. The resnet18 models were recreated using pytorch. GPU acceleration is recommended for training. Models were saved to .pth files which are moved to the artifacts folder. 

Models will be loaded into another file and used to test against perturbed data to see how robust the models are against adversarial attacks. A library of model architectures can be found in model_architectures.py which are imported to our test files.


### Adversarial Attacks
Model architecture for the control ResNet Models can be found in `model_architectures.py`.
Adversarial Attacks can be found in `attacks.py`.
Tests of the adversarial attacks can be found in their respective notebooks. For example fgsm is tested in test_fgsm.ipynb. The attacks are not optimized to run in parallel on GPUs as the original sources write their work sequentially, but optimizations can be made easily. 

The structure provides modularity for the project - different neural network architectures can be added and swapped out and tested against each of the attacks. Similarly, different tests can be added using the same formatting and can be compared against each other. 

## Job Scripts
### Using SLURM:
If the environment hasn't been configured yet, the setup script will create our conda environment on the SLURM GPU account. This only needs to be done once. Use command:
```
sbatch -A gpu setup_env.sh
```

To queue a job to be run with sbatch, use command:
```
sbatch -A gpu job.sh
```

To monitor your job, use command: 
```
ssqueue -u  <your-username>
```

To cancel a job, use command:
```
scancel <job-id>
```

### Create a job.sh Script:
The shebang for a bash script is first set. After, we use #SBATCH comments to set out sbatch parameters. sbatch preprocesses these while running the job script so we don't need to manually input all of these (and they're ignored when run locally). Module purge is done to cleanse the previous module and reload with the correct modules.
Before running any conda command remember to load the module: 
```
module load anaconda
```
first set up your conda environment. In the sample job file, the conda environment is called "d22env."  

The following command in setup_env.sh is used to create the environment. Modify it with packages you may need.
```
conda create --name MyEnvName python=3.8 pytorch torchvision matplotlib pandas <Any other packages you might need> -y  
```

After all dependencies are loaded, we run our desired file:
```
python -u <python file>.py
```

To make sure your stdout is going to a file, adjust the ```#SBATCH --output=/your/desirable/directory  ```
Before you run your command, make sure to change your directory back to your current working directory (directory is changed when you load anaconda, and it wont be able to find your file if you don't change it back  )

#### Directory Structure:
```
src:
C:.
├───CIFAR-10
│   │   batches.meta
│   │   cifar-10-python.tar.gz
│   │   data_batch_1
│   │   data_batch_2
│   │   data_batch_3
│   │   data_batch_4
│   │   data_batch_5
│   │   readme.html
│   └───test_batch
├───MNIST_CSV
│       generate_mnist_csv.py
│       mnist_test.csv
│       mnist_train.csv
│       readme.md
├───SVHN
│       extra_32x32.mat
│       test_32x32.mat
│       train_32x32.mat
│   attacks.py
│   curate_datasets.ipynb
│   data_curator.py
│   download_data.py
│   literature
│   model_architectures.py
│   test_cw.ipynb
│   test_deepfool.ipynb
│   test_fgsm.ipynb
│   test_jsma.ipynb
│   test_nes.ipynb
│   test_perturbations.ipynb
│   test_pgd.ipynb
│   test_square.ipynb
│
├───.VSCodeCounter
│   ├───2024-03-07_01-42-36
│   │       details.md
│   │       diff-details.md
│   │       diff.csv
│   │       diff.md
│   │       diff.txt
│   │       results.csv
│   │       results.json
│   │       results.md
│   │       results.txt
│   │
│   └───2024-03-07_01-43-45
│           details.md
│           diff-details.md
│           diff.csv
│           diff.md
│           diff.txt
│           results.csv
│           results.json
│           results.md
│           results.txt
│
├───artifacts
│       cw_augmented_resnet18_cifar_model.pth
│       cw_augmented_resnet18_mnist_model.pth
│       cw_augmented_resnet18_svhn_model.pth
│       deepfool_augmented_resnet18_cifar_model.pth
│       deepfool_augmented_resnet18_mnist_model.pth
│       deepfool_augmented_resnet18_svhn_model.pth
│       fgsm_augmented_resnet18_cifar_model.pth
│       fgsm_augmented_resnet18_mnist_model.pth
│       fgsm_augmented_resnet18_svhn_model.pth
│       fully_augmented_resnet18_cifar_model.pth
│       fully_augmented_resnet18_mnist_model.pth
│       fully_augmented_resnet18_svhn_model.pth
│       jsma_augmented_resnet18_cifar_model.pth
│       jsma_augmented_resnet18_mnist_model.pth
│       jsma_augmented_resnet18_svhn_model.pth
│       pgd_augmented_resnet18_cifar_model.pth
│       pgd_augmented_resnet18_mnist_model.pth
│       pgd_augmented_resnet18_svhn_model.pth
│       resnet18_cifar_model.pth
│       resnet18_mnist_model.pth
│       resnet18_svhn_model.pth
│
├───augmented_data
│       cw_cifar10_augmented_data.h5
│       cw_mnist_augmented_data.h5
│       cw_svhn_augmented_data.h5
│       deepfool_cifar10_augmented_data.h5
│       deepfool_mnist_augmented_data.h5
│       deepfool_svhn_augmented_data.h5
│       fgsm_cifar10_augmented_data.h5
│       fgsm_mnist_augmented_data.h5
│       fgsm_svhn_augmented_data.h5
│       jsma_cifar10_augmented_data.h5
│       jsma_mnist_augmented_data.h5
│       jsma_svhn_augmented_data.h5
│       pgd_cifar10_augmented_data.h5
│       pgd_mnist_augmented_data.h5
│       pgd_svhn_augmented_data.h5
│
├───images
│       resnet18ex1.png
│       resnet18ex2.png
│
├───Reports
│       Business Understanding.pdf
│       Data Preparation.pdf
│       Data Set Notes.pdf
│       Data Understanding Report.pdf
│       Evaluation Report.pdf
│       Evaluation Table.pdf
│       Modeling Report.pdf
│
├───resnet_models
│       cifar10_resnet18.ipynb
│       grid_search.py
│       mnist_resnet18.ipynb
│       svhn_resnet18.ipynb
│       test_resnet_from_pytorch.ipynb
│
├───Slides
│       CS490 BusinessData Understanding.pptx
│       CS490 Data Preparation Presentation.pptx
│       CS490 Evaluation Presentation.pptx
│       CS490 Modeling Presentation.pptx
│       CS_490_capstone.pdf
│       Intro_norm_attacks.pdf
│
└───Summary_Statistics
        cifar-10_summary_statistics.ipynb
        cifar10_summary_statistics.txt
        mnist_summary_statistics.ipynb
        mnist_summary_statistics.txt
        svhn_summary_statistics.ipynb
        svhn_summary_statistics.txt
```

