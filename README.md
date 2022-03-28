# Image-Representations-and-Point-Operations
Image Processing and Computer Vision Course Assignment 1

In this assignment the following tasks were implemented using Python and the OpenCV library:

**Reading an image into a given representation** -> Reads an image, and returns the image converted as requested
**Displaying an image** -> Reads an image as RGB or GRAY_SCALE and displays it
**Transforming an RGB image to YIQ color space**-> Converts an RGB image to YIQ color space
**Transforming an YIQ image to RGB color space**->Converts an YIQ image to RGB color space
**Histogram equalization**->Equalizes the histogram of an image
**Optimal image quantization**-> Quantized an image in to **nQuant** colors
**Gamma Correction**->GUI for gamma correction with trackBar from(• OpenCV trackbar example and • Gamma Correction Wikipedia)

# Image Outputs of the tasks listed above using the OpenCV and Matplotlib libraries:

**Transforming an RGB image to YIQ color space:**
ransform an RGB image into the YIQ color space and vice versa. Given the red (R), green (G), and blue (B) pixel components of an RGB color image, the corresponding luminance (Y), and the chromaticity components (I and Q) in the YIQ color space are linearly related as follows:

<img width="211" alt="Screen Shot 2022-03-28 at 23 49 10" src="https://user-images.githubusercontent.com/73976733/160484685-29696e07-49e6-4821-aede-d77a3546a463.png">

<img width="504" alt="Screen Shot 2022-03-29 at 0 01 27" src="https://user-images.githubusercontent.com/73976733/160486579-171b9b08-f434-429d-b629-4ff47c65ea79.png">

**Histogram Equalization:**

a function that performs histogram equalization of a given grayscale or RGB image. The function should also display the input and the equalized output image.
<img width="498" alt="Screen Shot 2022-03-29 at 0 02 40" src="https://user-images.githubusercontent.com/73976733/160486781-3f08c7e8-e130-48e6-b56a-caba821c7bb2.png">

**Original Grayscale Image and Equalized Grayscale Image:**

<img width="959" alt="Screen Shot 2022-03-29 at 0 06 51" src="https://user-images.githubusercontent.com/73976733/160487487-5c8cb744-3a96-4f0f-8a45-49379f2e0446.png">


**Original RGB Image and Equalized RGB Image:**

<img width="998" alt="Screen Shot 2022-03-28 at 23 53 35" src="https://user-images.githubusercontent.com/73976733/160485721-f96b9cba-d4da-4030-a9f1-3d1e99bf05af.png">

**Quantized RGB Image:**

<img width="369" alt="Screen Shot 2022-03-28 at 23 54 27" src="https://user-images.githubusercontent.com/73976733/160485935-0410e73c-5011-4668-b987-728514a20bb6.png">

# Gamma Correction:
a function that performs gamma correction on an image with a given γ.
For this task, you’ll be using the OpenCV funtions createTrackbar to create the slider and display
it, since it’s OpenCV’s functions, the image will have to be represented as BGR.
**Gamma Correction in Grayscale:**

<img width="642" alt="Screen Shot 2022-03-28 at 23 44 44" src="https://user-images.githubusercontent.com/73976733/160486077-cfe228b0-7349-4bd4-8288-cc8768b00d96.png">
Enjoy:smile:
