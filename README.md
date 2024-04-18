# Simple DragGAN
Simple DragGAN is an Gradio application, a modification of the [original DragGAN](https://github.com/XingangPan/DragGAN/tree/d0422b1b38a7c95f7f88892366b2c07d9f3ee449) application hosted on Gradio. This is done as a part of a Final Year Project, with the purpose of making DragGAN more accessible for non-technical users. As such, some functions are not available to make the application simple. (e.g. selection of w or w+ for latent space is not available. The default value, w+, will be used throughout.)

This repository has cloned the original DragGAN repository from [this commit](https://github.com/XingangPan/DragGAN/tree/d0422b1b38a7c95f7f88892366b2c07d9f3ee449), with additional files and modifications to the original files, to run Simple DragGAN.

# Table of contents
- [Requirements](#requirements)
- [Instructions](#instructions)
  - [Additional Instructions](#additional-instructions)
    - [Add your own model/checkpoint](#add-your-own-modelcheckpoint)
- [Errors](#errors)  
- [What are the additional files / changes made to the original DragGAN to get Simple DragGAN working?](#what-are-the-additional-files--changes-made-to-the-original-draggan-to-get-simple-draggan-working)
- [FAQ](#faq)
- [Acknowledgement](#acknowledgement)
- [Original DragGAN README](#original-draggan-readme)


 

# Requirements
- GPU of RAM above 6GB.
  - The application can run on GPU with 6GB, however, the application may hang at times.
- CUDA Toolkit 11.8
  - There was problems running with CUDA Toolkit 12.2. For more information, please refer to the following issues:
    - https://github.com/XingangPan/DragGAN/issues/146
    - https://github.com/XingangPan/DragGAN/issues/69
- Anaconda
  - For reference on conda version, conda version 23.9.0 was used to work on the application.
- For Ubuntu, recommended version is 20.04.
  - Reason:
    - CUDA Toolkit 11.8 is available from Ubuntu 18.04 onwards.
    - The application was worked on Ubuntu 20.04.

Simple DragGAN can be run on Windows 11 and Ubuntu 20.04 operating system.


# Instructions
To run Simple DragGAN, please follow the instructions below:

1. Create Anaconda environment.
```
conda env create -f environment.yml
```
<details>
<summary>Notes on environment.yml</summary>
environment.yml is the modified version of the original DragGAN's environment.yml. The changes are as follows:

1. ```cudatoolkit=11.1``` under ```dependencies``` has been commented out.
2. ```scipy=1.11.0``` has been moved (from under ```dependencies```) to under ```pip```. Additionally, it has been changed to ```scipy==1.11.0``` (extra '=') to match with ```pip``` syntax.
3. ```gradio==3.35.2``` has been changed to ```gradio==3.36.1``` (different version). This was done to solve the infinite loading problem on the Gradio app, where the Gradio hosted app loads infintely after a component (e.g. button) has been pressed.
- For changes in (1) and (2), please refer to [this YouTube video](https://youtu.be/i7cI3C6_x78?si=q48nRVMpbbNiMwbR&t=115) for reasons of change.
</details>


2. Activate anaconda environment
```
conda activate stylegan3
```

3. Install torch again.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
<details>
<summary>Reason for torch installation</summary>
I encountered this error without the torch installation:

![error image](/readme_images/error_img.png)

It worked fine after running the above pip command.

For reference, this is the settings selected on the [official torch site](https://pytorch.org/) to get the command:
![install torch image](/readme_images/torch_install.png)
</details>

4. Go to project directory
```
cd Simple_DragGAN
```

5. Download pre-trained weights:
```
python scripts/download_model.py
```

6. Run Simple DragGAN
```
python Simple_DragGAN_App.py
```

## Additional Instructions
### Add your own model/checkpoint
1. Name your .pkl file with prefix "stylegan2" (for StyleGAN2 model).
  - The filename will be used to infer the model type. Refer to [these lines](https://github.com/Take-Saori/Simple_DragGAN/blob/66522b96a5f3bb1f0049389200d4ee8653bcc730/viz/renderer.py#L158) in [renderer.py](https://github.com/Take-Saori/Simple_DragGAN/blob/66522b96a5f3bb1f0049389200d4ee8653bcc730/viz/renderer.py).
2. Add the .pkl file to ```checkpoints``` folder.
3. In ```model_pickle_info.json```, add an element in the dictionary in this format:
```
"<Object_name>" : [
        {
            "pickle_name" : "<.pkl filename without ".pkl">",
            "display_name" : "<Name to display in App>",
            "type" : "<"Face" or "Body", or etc>",
            "size" : <Length of image generated, only length is needed as image generated is assumed to be square>
        }
    ]
```
Example for model generating human face:

```
"Human" : [
        {
            "pickle_name" : "stylegan2-ffhq-512x512",
            "display_name" : "Human (Face)",
            "type" : "Face",
            "size" : 512
        }
    ]
```
Example, if there are two models generting same object, but differently (body and face):
```
"Cat" : [
        {
            "pickle_name" : "stylegan2-afhqcat-512x512",
            "display_name" : "Cat (Face)",
            "type" : "Face",
            "size" : 512
        },
        {
            "pickle_name" : "stylegan2-cat-config-f",
            "display_name" : "Cat (Body)",
            "type" : "Body",
            "size" : 512
        }
    ],
```


# Errors
1. If DragGAN cannot be run due to the following error:
![error image](/readme_images/error_img.png)
Try running the following pip command to install torch again:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
<details>
<summary>Settings to get pip command from official torch site</summary>
For reference, this is the settings selected on the [official torch site](https://pytorch.org/) to get the command:

![install torch image](/readme_images/torch_install.png)

</details>


2. If DragGAN application could not be run with issue related to OpenGL, please refer to [this issue](https://github.com/XingangPan/DragGAN/issues/39).
  - The solution from this issue (that worked): run this command "export MESA_GL_VERSION_OVERRIDE=3.3"


# What are the additional files / changes made to the original DragGAN to get Simple DragGAN working?
- ```environment.yml```
  - File to create anaconda environment.
  - Changes:
    1. ```cudatoolkit=11.1``` under ```dependencies``` has been commented out.
    2. ```scipy=1.11.0``` has been moved (from under ```dependencies```) to under ```pip```. Additionally, it has been changed to ```scipy==1.11.0``` (extra '=') to match with ```pip``` syntax.
    3. ```gradio==3.35.2``` has been changed to ```gradio==3.36.1``` (different version). This was done to solve the infinite loading problem on the Gradio app, where the Gradio hosted app loads infintely after a component (e.g. button) has been pressed.
    - For changes in (1) and (2), please refer to [this YouTube video](https://youtu.be/i7cI3C6_x78?si=q48nRVMpbbNiMwbR&t=115) for reasons of change.
- ```requirements.txt```
  - Text file for pip installation.
  - Changes:
    1. With the same reason as ```environment.yml```, ```gradio==3.35.2``` has been changed to ```gradio==3.36.1``` 
- ```sample_image.py```
  - Script to generate sample image to display in application.
- ```generate_image.py```
  - Script to generate image from StyleGAN2 model. This script has been taken from [StyleGAN2-ADA repository](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/d72cc7d041b42ec8e806021a205ed9349f87c6a4), and has been modified.
- ```projector.py```
  - Script to generate latent code from a given image. This script has been taken from [StyleGAN2-ADA repository](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/d72cc7d041b42ec8e806021a205ed9349f87c6a4), and has been modified.
- ```app_image``` folder
  - Folder with images to display in Simple DragGAN.
- ```Simple_DragGAN_App.py```
  - Script to run Simple DragGAN. This script has been modified from [visualizer_drag_gradio.py](https://github.com/XingangPan/DragGAN/blob/d0422b1b38a7c95f7f88892366b2c07d9f3ee449/visualizer_drag_gradio.py), taken from the [original DragGAN repository](https://github.com/XingangPan/DragGAN/tree/d0422b1b38a7c95f7f88892366b2c07d9f3ee449).
- ```model_pickle_info.json```
  - JSON file to write model details for use in ```Simple_DragGAN_App.py```, mostly for display purpose.
- ```viz/renderer.py```
  - This is the file in the original DragGAN. The following has been changed:
    - In line 53, from
    ```
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    ```
    to
    ```
    font = ImageFont.truetype
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    'arial.ttf'),
    round(25/512*image.size[0]))
    ```
    - In line 251, from 
    ```
    w = self.w_load.clone().to(self._device)
    ```
    to
    ```
    w = torch.from_numpy(self.w_load).clone().to(self._device)
    ```

# FAQ
1. Can I load StyleGAN2 models trained on non-square images in the original DragGAN application?
  - At the point of writing this (15 April), NO. DragGAN currently supports loading models trained with square images only.

# Acknowledgement
This code is developed based on [DragGAN](https://github.com/XingangPan/DragGAN/tree/d0422b1b38a7c95f7f88892366b2c07d9f3ee449) and [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/tree/d72cc7d041b42ec8e806021a205ed9349f87c6a4). Most of the codes are written by these two repository.


# Original DragGAN README
Below is the README from the original DragGAN:

<p align="center">

  <h1 align="center">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</h1>
  <p align="center">
    <a href="https://xingangpan.github.io/"><strong>Xingang Pan</strong></a>
    ·
    <a href="https://ayushtewari.com/"><strong>Ayush Tewari</strong></a>
    ·
    <a href="https://people.mpi-inf.mpg.de/~tleimkue/"><strong>Thomas Leimkühler</strong></a>
    ·
    <a href="https://lingjie0206.github.io/"><strong>Lingjie Liu</strong></a>
    ·
    <a href="https://www.meka.page/"><strong>Abhimitra Meka</strong></a>
    ·
    <a href="http://www.mpi-inf.mpg.de/~theobalt/"><strong>Christian Theobalt</strong></a>
  </p>
  <h2 align="center">SIGGRAPH 2023 Conference Proceedings</h2>
  <div align="center">
    <img src="DragGAN.gif", width="600">
  </div>

  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://twitter.com/XingangP"><img alt='Twitter' src="https://img.shields.io/twitter/follow/XingangP?label=%40XingangP"></a>
    <a href="https://arxiv.org/abs/2305.10973">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/DragGAN/'>
      <img src='https://img.shields.io/badge/DragGAN-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href="https://colab.research.google.com/drive/1mey-IXPwQC_qSthI5hO-LTX7QL4ivtPh?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </p>
</p>

## Web Demos

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/XingangPan/DragGAN)

<p align="left">
  <a href="https://huggingface.co/spaces/radames/DragGan"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DragGAN-orange"></a>
</p>

## Requirements

If you have CUDA graphic card, please follow the requirements of [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3#requirements).  

The usual installation steps involve the following commands, they should set up the correct CUDA version and all the python packages

```
conda env create -f environment.yml
conda activate stylegan3
```

Then install the additional requirements

```
pip install -r requirements.txt
```

Otherwise (for GPU acceleration on MacOS with Silicon Mac M1/M2, or just CPU) try the following:

```sh
cat environment.yml | \
  grep -v -E 'nvidia|cuda' > environment-no-nvidia.yml && \
    conda env create -f environment-no-nvidia.yml
conda activate stylegan3

# On MacOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run Gradio visualizer in Docker 

Provided docker image is based on NGC PyTorch repository. To quickly try out visualizer in Docker, run the following:  

```sh
# before you build the docker container, make sure you have cloned this repo, and downloaded the pretrained model by `python scripts/download_model.py`.
docker build . -t draggan:latest  
docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash
# (Use GPU)if you want to utilize your Nvidia gpu to accelerate in docker, please add command tag `--gpus all`, like:
#   docker run --gpus all  -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```
Now you can open a shared link from Gradio (printed in the terminal console).   
Beware the Docker image takes about 25GB of disk space!

## Download pre-trained StyleGAN2 weights

To download pre-trained weights, simply run:

```
python scripts/download_model.py
```
If you want to try StyleGAN-Human and the Landscapes HQ (LHQ) dataset, please download weights from these links: [StyleGAN-Human](https://drive.google.com/file/d/1dlFEHbu-WzQWJl7nBBZYcTyo000H9hVm/view?usp=sharing), [LHQ](https://drive.google.com/file/d/16twEf0T9QINAEoMsWefoWiyhcTd-aiWc/view?usp=sharing), and put them under `./checkpoints`.

Feel free to try other pretrained StyleGAN.

## Run DragGAN GUI

To start the DragGAN GUI, simply run:
```sh
sh scripts/gui.sh
```
If you are using windows, you can run:
```
.\scripts\gui.bat
```

This GUI supports editing GAN-generated images. To edit a real image, you need to first perform GAN inversion using tools like [PTI](https://github.com/danielroich/PTI). Then load the new latent code and model weights to the GUI.

You can run DragGAN Gradio demo as well, this is universal for both windows and linux:
```sh
python visualizer_drag_gradio.py
```

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3). Part of the code is borrowed from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).

(cheers to the community as well)
## License

The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).
However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Any form of use and derivative of this code must preserve the watermarking functionality showing "AI Generated".

## BibTeX

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold},
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```
