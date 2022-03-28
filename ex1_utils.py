"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import numpy.ma as mat
import matplotlib.pyplot as plt
import cv2 as cv

import numpy as np
ERROR_MSG = "Error: the given image has wrong dimensions"
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 205839400


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Loading an image
    img = cv.imread(filename)
    if img is not None:
        if representation == LOAD_GRAY_SCALE:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif representation == LOAD_RGB:
            # We weren't asked to convert a grayscale image to RGB so this will suffice
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:  # Any other value was entered as the second parameter
            raise ValueError(ERROR_MSG)
    else:
        raise Exception("Could not read the image! Please try again.")
    return img / 255.0


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.3111]])
    OrigShape = imgRGB.shape
    return np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)

    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # taking sizes of input to make a new image
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.3111]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass




def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    is_rgb = False
    # RGB image procedure should only operate on the Y chanel
    if len(imgOrig.shape) == 3:
        is_rgb = True
        yiq_image = transformRGB2YIQ(np.copy(imgOrig))
        imgOrig = yiq_image[:, :, 0]
        # change range grayscale or RGB image to be equalized having values in the range [0, 1]
    imgOrig = cv.normalize(imgOrig, None, 0, 255, cv.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')
    # the histogram of the original image
    histOrg = np.histogram(imgOrig.flatten(), 256)[0]
    # Calculate the normalized Cumulative Sum (CumSum)
    cumsum = np.cumsum(histOrg)
    # Create a LookUpTable(LUT)
    LUT = np.floor((cumsum / cumsum.max()) * 255)
    # Replace each intesity i with LUT[i] and Return an array of zeros with the same shape and type as a given array.
    imEq = np.zeros_like(imgOrig, dtype=float)
    for x in range(256):
        imEq[imgOrig == x] = int(LUT[x])
    # Calculate the new image histogram (range = [0, 255])
    histEQ = np.zeros(256)
    for val in range(256):
        # Counts the number of non-zero values in the array.
        histEQ[val] = np.count_nonzero(imEq == val)

    # norm imgEQ from range [0, 255] to range [0, 1]
    imEq = imEq / 255.0

    if is_rgb:
        # If an RGB image is given the following equalization procedure should only operate on the Y channel of the corresponding YIQ image and then convert back from YIQ to RGB.
        yiq_image[:, :, 0] = imEq / (imEq.max() - imEq.min())
        imEq = transformYIQ2RGB(np.copy(yiq_image))
    return imEq, histOrg, histEQ



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
#check input
    if imOrig is None:
        raise Exception("Error: imOrig is None!")
    if nQuant > 256:
        raise ValueError("nQuant is greater then 256!")
    if nIter < 0:
        raise ValueError("Number of optimization loops must be a positive number!")

        # handle&check RGB images
    flagRGB = False
    if len(imOrig.shape) is 3:  # RGB image
        flagRGB = True
        imgYIQ = transformRGB2YIQ(imOrig)  # transform to YIQ color space
        imOrig = imgYIQ[:, :, 0]  # y-channel

    imgOrigInt = (imOrig * 255).astype("uint8")
    # find the histogram of the original image
    histOrig, _ = np.histogram(imgOrigInt.flatten(), 256, range=(0, 255))

    MSE_error_list = []  # errors array
    q_img_lst = []  # contains all the encoded images
    global intensities, z, q

    for j in range(nIter):
        encodeImg = imgOrigInt.copy()
        # Finding z  - the values that each of the segments intensities will map to.
        if j is 0:  # first iteration INIT z
            z = np.arange(0, 255 - int(256 / nQuant) + 1, int(256 / nQuant))
            z = np.append(z, 255)
            intensities = np.array(range(256))
        else:  # not the first iteration
            for r in range(1, len(z) - 2):
                #formula
                new_z_r = int((q[r - 1] + q[r]) / 2)
                if new_z_r != z[r - 1] and new_z_r != z[r + 1]:  # to avoid division by 0
                    z[r] = new_z_r

        # Finding q - the values that each of the segments intensities will map to.
        q = np.array([], dtype=np.float64)
        for i in range(len(z) - 1):
            mask_pix = np.logical_and((z[i] < encodeImg), (encodeImg < z[i + 1]))  # the current cluster
            if i is not (len(z) - 2):
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1]], weights=histOrig[z[i]:z[i + 1]] + 0.001))
                encodeImg[mask_pix ] = int(q[i])  # apply the changes to the encoded image

            else:  # i is len(z)-2 , add 255
                # calculate weighted mean
                if sum(histOrig[z[i]:z[i + 1]]) != 0:
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1], weights=histOrig[z[i]:z[i + 1] + 1]))
                else:  # to avoid division by 0
                    q = np.append(q, np.average(intensities[z[i]:z[i + 1] + 1],
                                                weights=histOrig[z[i]:z[i + 1] + 1] + 0.001))
                encodeImg[mask_pix ] = int(q[i])  # apply the changes on the encoded image

        MSE_error_list.append((np.square(np.subtract(imgOrigInt, encodeImg))).mean())  # calculate error
        encodeImg = encodeImg / 255  # normalize to range [0,1]

        if flagRGB:  # RGB image
            imgYIQ[:, :, 0] = encodeImg.copy()  # modify y channel
            encodeImg = transformYIQ2RGB(imgYIQ)  # transform back to RGB
            plt.plot(MSE_error_list)
        q_img_lst .append(encodeImg)

        # checking whether we have come to convergence
        if j > 1 and abs(MSE_error_list[j - 1] - MSE_error_list[j]) < 0.001:
            plt.plot(MSE_error_list)
            print("we have come to convergence after {} iterations!".format(j + 1))
            break

    return q_img_lst , MSE_error_list