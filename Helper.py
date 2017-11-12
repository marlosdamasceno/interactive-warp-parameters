import numpy as np
import cv2


# Define a function that applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, convert_color=True, color_conversion_between=cv2.COLOR_RGB2HLS,
                     channel=0, orient='x', sobel_kernel=3, threshold=(0, 255)):
    # Check the dimension of the image
    if convert_color:
        converted = cv2.cvtColor(img, color_conversion_between)  # Convert to color space
        img_to_process = converted[:, :, channel]  # Get the channel, the default is 0
    else:
        img_to_process = img[:, :, channel]  # Get the channel, the default is 0

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img_to_process, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img_to_process, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Inclusive threshold
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1.0
    # Return the binary image
    return binary_output

