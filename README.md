
# Forecasting Future World Events with Neural Networks

This repository is a fork of [andyzoujim/autocast](https://github.com/andyzoujm/autocast). I encorage you to check out their paper: "[Forecasting Future World Events with Neural Networks](http://arxiv.org/abs/2206.15474)" if you are interested in reading more about training large language models to forecast.

<img align="center" src="assets/splash.png" width="750">

## Quick Start Guide

### Establishing a Virtual Machine for Experiments

For those uncertain about their hardware requirements, it's suggested to follow the guidelines provided here. This method has been tested and confirmed effective as of May 16, 2023.

#### Configuring an EC2 Instance

1. Visit [Amazon AWS](https://aws.amazon.com/). Sign in to your existing AWS account, or create a new one if necessary.
2. Proceed to EC2 > Instances > Launch an Instance.
3. Select Ubuntu Server 22.04 LTS as your AMI.
4. Maintain the architecture at 64-bit (x86).
5. Choose g4dn.xlarge as your instance type. **Note: First-time EC2 users must complete a request form for vCPU allocation. See instructions below.**
6. Choose "create new key pair" and save the .pem file in a convenient location.
7. Choose 128 GB of gp2 for storage. (Although less may suffice, storage is relatively inexpensive.)
8. Launch your instance.

**Requesting Additional vCPUs**
Visit http://aws.amazon.com/contact-us/ec2-request. In Request 1, specify the region where you're creating the instance. For instance type, select "all G instances". Indicate 4 as the limit (or as specified by your model; 4 is required for g4dn.xlarge). Approval is usually granted within 24 hours.

**Recommended: Allocating Elastic IP** - Skipping this step will require you to obtain a new ssh command from your EC2 dashboard each time you start and stop your instance.
1. Visit the Amazon EC2 console at https://console.aws.amazon.com/ec2/.
2. In the navigation pane, select "Elastic IPs".
3. Click "Allocate new address", then "Allocate".
4. Choose the new Elastic IP, select "Actions", then "Associate IP address".
5. In the dialog box, choose your instance and click "Associate".

#### Connecting to the EC2 Instance
1. Install Visual Studio Code, if not already installed.
2. Install the Remote - SSH extension.
3. Click the Remote Explorer icon in the extensions bar on the left.
4. In Remote > SSH on the right, click the (+).
5. Visit console.aws.amazon.com/ec2, go to instances, select your instance, and click connect.
6. Navigate to SSH client and copy the example connect command.
7. Paste the command into the VS Code bar, replace the quoted ssh file with the full path to the ssh file, and press enter.
8. For the config file to edit, choose the default. The file will auto-populate with the correct configuration.
9. In the pop-up at the bottom-right corner, click connect.
10. Select Linux as the platform and click continue.
11. A new window will now open in VS Code.

### Setting Up the Environment (Ubuntu 22.04 Example)

#### Installing Necessary External Software
**NVIDIA DRIVER**
1. Update the instance and install `build-essential gcc-multilib dkms` for GPU usage:
 ```
 sudo apt-get update
 sudo apt-get upgrade
 sudo apt-get install build-essentials gcc-multilib dkms
 ```
2. Install the NIVIDA CUDA Toolkit by visiting https://developer.nvidia.com/cuda-downloads. Select Linux > x86_64 > Ubuntu > 22.04 > runfile (local).
Run the commands shown on the website. Accept the license agreement and select to install both the driver and the CUDA Toolkit.
3. Verify your installation by running
```
nvidia-smi
```
**ELASTIC SEARCH**
1. Install ElasticSearch. Visit https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html and follow the instructions.
2. Run ElasticSearch with the following command (from the elasticsearch root folder)
```
./bin/elasticsearch
```
3. Wait for elastic search to load up, then stop it with ctrl/cmd+z.
4. In the root folder of elastic search open the elasticsearch.yml and change the below parameters to false.
```
xpack.security.enabled: false
xpack.security.transport.ssl.enabled: false
xpack.security.http.ssl.enabled: false
```
5. Now Elastic search is ready to run with the repositiory.

#### Install python virtual environment.
1. Navigate to https://docs.conda.io/en/latest/miniconda.html and under Linux copy the link for Miniconda3 Linux 64-bit.
2. In a terminal on your EC2 instance, run the following command:
```
wget -O Miniconda.sh <link from previous step>
bash Miniconda.sh
```
Agree to the license and confirm the installation location. Answer "yes" to "Do you wish the installer to initialize Miniconda3
by running conda init?" Close and reopen the terminal. You should see (base) in the terminal line.
4. Create a new python environment by executing the following command.
```
conda create -n autocast python
```
5. Activate the environment with
```
conda activate autocast
```
6. Clone this repository.
```
git clone https://github.com/loganamcnichols/autocast.git
cd autocast
```
8. Install python packages by running.
```
pip install -r requirements.txt
```
### Downloading and processing data
A bash script has been provided in /autocast_experiments/data which runs all the commands for downloading and preparing the datasets for training. The only thing that needs to be modified is elastic search command. Replace the path provided with the absolute path to your elastic search, then the following from the data folder.
```
./prepare_data.sh
```
After about 20 minutes the script should finish and your data directory will contain the train and test data files.

### Running experiments
This repositiory is still a work-in-progress. Not all files are ready to run. The best place to start is with running `temporal_fid_train.sh`. 



## Introduction

Forecasting future world events is a challenging but valuable task. Forecasts of climate, geopolitical conflict, pandemics and economic indicators help shape policy and decision making. In these domains, the judgment of expert humans contributes to the best forecasts. Given advances in language modeling, can these forecasts be automated? To this end, we introduce Autocast, a dataset containing thousands of forecasting questions and an accompanying news corpus. Questions are taken from forecasting tournaments, ensuring high quality, real-world importance, and diversity. The news corpus is organized by date, allowing us to precisely simulate the conditions under which humans made past forecasts (avoiding leakage from the future). We test language models on our forecasting task and find that performance is far below a human expert baseline. However, performance improves with increased model size and incorporation of relevant information from the news corpus. In sum, Autocast poses a novel challenge for large language models and improved performance could bring large practical benefits.

## Autocast Dataset

The latest version of the [Autocast dataset can be downloaded here](https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz). For more details on how to use the Autocast dataset and news articles, please refer to our short demonstration in `usage.ipynb`.

Each question has the following fields:
  ```json
  {
    "id":                "unique identifier (str)",
    "question":          "question body (str)",
    "background":        "question context/details (str)",
    "qtype":             "question type (str)",
    "status":            "question status (str)",
    "choices":           "choices or possible ranges (List or Dict)",
    "answer":            "question resolution (str or float)",
    "crowd":             "human crowd forecasts over time (List)",
    "publish_time":      "publish timestamp (str)",
    "close_time":        "close timestamp (str)",
    "prediction_count":  "number of crowd predictions (int)",
    "forecaster_count":  "number of crowd forecasters (int)",
    "tags":              "question category (List)",
    "source_links":      "source links from comments (List)"
  }
```

We obtained permission from [Metaculus](https://www.metaculus.com/) to host the dataset on GitHub for research purposes only.

## IntervalQA Dataset

Motivated by the difficulty of forecasting numbers across orders of magnitude (e.g. global cases of COVID-19 in 2022), we also curate IntervalQA, a dataset of numerical questions and metrics for calibration.

[Download the IntervalQA dataset here](https://people.eecs.berkeley.edu/~hendrycks/intervalqa.tar.gz).

## Citation

If you find this useful in your research, please consider citing:

    @article{zouforecasting2022,
      title={Forecasting Future World Events with Neural Networks},
      author={Andy Zou and Tristan Xiao and Ryan Jia and Joe Kwon and Mantas Mazeika and Richard Li and Dawn Song and Jacob Steinhardt and Owain Evans and Dan Hendrycks},
      journal={NeurIPS},
      year={2022}
    }
