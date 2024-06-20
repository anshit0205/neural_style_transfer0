# Neural Style Transfer on Google Colab

## Project Overview

This project demonstrates the implementation of a Neural Style Transfer algorithm using TensorFlow and Keras. The technique involves blending two images: a content image (which contains the structure and details) and a style image (which contains the artistic style and texture). The goal is to generate a new image that preserves the content of the first image while adopting the artistic style of the second image. Additionally, the project includes various metrics such as SSIM, PSNR, LPIPS, and KID to evaluate the quality and similarity of the generated images compared to the original ones. Although the primary metric remains our visual pleasure—if our eyes like the combination, it doesn't matter what other metrics indicate.

## Using Google Colab

Google Colab is a free platform that provides GPU support, making it an excellent choice for running computationally intensive tasks like Neural Style Transfer. Follow these steps to run the project on Google Colab:

### Steps

1. **Open Google Colab:**

    Navigate to [Google Colab](https://colab.research.google.com) in your browser.

2. **Create a New Notebook:**

    Click on `File > New Notebook` to create a new notebook.

3. **Clone the Repository:**

    In the first cell of your notebook, run the following code to clone the repository and navigate into the project directory:

    ```python
    !git clone https://github.com/anshit0205/neural_style_transfer0.git
    %cd neural_style_transfer0
    ```

4. **Install the Required Libraries:**

    In the next cell, install the necessary libraries by running:

    ```python
    !pip install -r requirements.txt
    ```

5. **Upload Your Images:**

    You can use the following code to upload your content and style images directly to the appropriate directories:

    ```python
    from google.colab import files
    import os

    # Create necessary directories
    os.makedirs('images/content', exist_ok=True)
    os.makedirs('images/style', exist_ok=True)

    # Upload content image
    print("Upload your content image")
    content_image = files.upload()
    for filename in content_image.keys():
        os.rename(filename, f'images/content/{filename}')

    # Upload style image
    print("Upload your style image")
    style_image = files.upload()
    for filename in style_image.keys():
        os.rename(filename, f'images/style/{filename}')
    ```

    Run the above code cells and upload your content and style images when prompted.

6. **Run the Style Transfer Script:**

    In the next cell, run the main script to perform style transfer:

    ```python
    !python main.py
    ```

    The script will save the generated images and loss plots as PNG files in the working directory.

7. **View the Results:**

    You can view the saved images directly in Colab using the following code:

    ```python
    from IPython.display import Image, display

    # Display the content and style images
    display(Image(filename='content_and_style_images.png'))

    # Display the result image
    display(Image(filename='result_image.png'))

    # Display the loss plots
    display(Image(filename='losses.png'))
    display(Image(filename='total_loss.png'))
    display(Image(filename='histograms.png'))
    ```

    Run the above code cells to display the images and plots.

---

Feel free to contribute to this project by submitting issues or pull requests. Enjoy creating beautiful artworks with Neural Style Transfer!
