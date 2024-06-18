# Neural Style Transfer

## Project Overview

This project demonstrates the implementation of a Neural Style Transfer algorithm using TensorFlow and Keras. The technique involves blending two images: a content image (which contains the structure and details) and a style image (which contains the artistic style and texture). The goal is to generate a new image that preserves the content of the first image while adopting the artistic style of the second image. Additionally, the project includes various metrics such as SSIM, PSNR, LPIPS, and KID to evaluate the quality and similarity of the generated images compared to the original ones. Although, the primary metric remains our visual pleasure. If our eyes like the combination, it doesnt matter what other metrics speak. 

## Installation Instructions

### Prerequisites

Ensure you have Python 3.7 or higher installed on your machine. It is recommended to use a virtual environment to manage dependencies.

### Setup

1. **Clone the repository:**

    Open your terminal (or Command Prompt on Windows) and run the following commands:
    ```sh
    git clone https://github.com/anshit0205/neural_style_transfer0.git
    cd neural_style_transfer0
    ```

    Here, `git clone` downloads the project to your local machine, and `cd neural_style_transfer0` navigates into the project directory.

2. **Create and activate a virtual environment:**

    Run the following commands:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    This creates a virtual environment named `venv` and activates it. Virtual environments help in managing dependencies for different projects separately.

3. **Install the required libraries:**

    With the virtual environment activated, run:
    ```sh
    pip install -r requirements.txt
    ```

    This installs all the necessary libraries specified in the `requirements.txt` file.

4. **Place your content and style images in the appropriate directories:**

    Ensure your images are organized as follows:
    ```
    neural_style_transfer0/
    └── images/
        ├── content/
        │   └── content_image.png
        └── style/
            └── style_image.png
    ```

    Replace `content_image.png` and `style_image.png` with your actual image files you may want to use.
   
## Usage

Run the `main.py` script to perform style transfer. By default, it uses `images/content/content_image.png` and `images/style/style_image.png` as the input images.

To run the script, execute:
```sh
python main.py
