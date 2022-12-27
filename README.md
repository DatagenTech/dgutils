<div align="center">
  <img src="docs/assets/datagen-logo.png" width="600"/>
 <br/><br/>
</div>
  

# **Simulate the exact data your AI needs.**

  <p align="center">
 

# About Datagen

Datagen is on a mission to transform how CV teams get their data. Our self-service platform and API allow users to generate fully annotated, photorealistic, high-variance synthetic data, with a focus on data for human-centric computer vision applications.

Fortune 500 companies rely on Datagen to develop their future products in the worlds of AR/VR/Metaverse, In-cabin Vehicle Safety, Robotics, IoT Security and more.

# The DGutils Repository
The Datagen utilities repository (DGutils for short) includes:

## :older_man: :older_woman: Data samples
In the _resources_ folder you will find data samples from our human Faces generator. Each datapoints includes an .rgb image, alongside its annotations - depth map, normal map, segmentation map, and additional metadata (such as the camera parameters, or the age & gender of the relevant human). We refer to these annotations as the **modalities** of the data.

Our data samples are organized in sub-folders:

 - **training_data_example**: 100 data points of human faces, of resolution 128*128, used in our Pytorch model training notebook
 - **faces_1**, **faces_2** : two folders containing a total of 10 data points of human faces, of resolution 1k * 1k, used in our various visualization notebooks. The sources are split in two for the example of loading two data paths with a single command - there is no semantic difference between the two. 
 - **hic_home_security**: a data sample of our HIC (humans in context) generator. A single scene containing 4 frames of a human walking to the door with a packaged item

## :notebook_with_decorative_cover: Jupyter Notebooks

 - What is my data's distribution?
 - Can I see the gaze vector or facial keypoints on top of my generated face?
 - How can I use the segmentation map to crop parts of the face?
 - How can I use Pytorch to train a model on synthetic data?

In the _notebooks_ folder you will find answers to these and more. Our notebooks are designed to help explore and visualize Datagen's data and modalities, as well as give some useful tips and tricks. Some of the notebooks will portray specific examples of usage, for example training a pytorch model for the task of landmark detection.

## :computer: Our SDK
Our notebooks are using the [Datagen Python Package](https://pypi.org/project/datagen-tech/) for data loading, parsing, visualization, analysis and more. Read more about the package in the above link, and follow the installation guide below to get it.

# Quick Installation
```bash
# clone repository
git clone https://github.com/DatagenTech/dgutils.git
# recommended: create dgutils virtual env
conda create --name dgutils
conda activate dgutils
# install the Datagen SDK
pip install datagen-tech
```


# Learn More
## Research 
-   Using Synthetic Images To Uncover Population Biases In Facial Landmarks Detection, Neurips 2021 ([arxiv](https://arxiv.org/pdf/2111.01683.pdf))
-   Hands-Up: Leveraging Synthetic Data for Hands-On-Wheel Detection, CVPR 2022 ([arxiv](https://arxiv.org/abs/2206.00148?context=cs))
- Facial Landmark Detection Using Synthetic Data , [White Paper](https://datagen.tech/ai/facial-landmark-detection-using-synthetic-data/) 
- Datagen [Learning Center](https://datagen.tech/guides/synthetic-data/synthetic-data/) for anything CV related, and especially for any question you have on synthetic data.

## Documentation

Check out Datagen [Docs](https://docs.datagen.tech/en/latest/) for full documentation, user guides, playbooks and examples.

# Reach out

- Want to generate your own, fully customized dataset? [Try our platform & API for free](https://datagen.tech/signup/) or ping us at: hello@datagen.tech
- Got some cool ideas? Wanna share some feedback of suggest improvements?
   - We are happy to chat: community@datagen.tech
   - Or, feel free to open a  [Github Issue](https://github.com/DatagenTech/dgutils/issues) and we will look into it!