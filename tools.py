import numpy as np
import cv2
from scipy.signal import butter, lfilter

# image display
import IPython.display
import PIL.Image
from io import BytesIO


# DATA PROCESSING TOOLS

def load_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return image


def load_images(filenames):
    return [load_image(filename) for filename in filenames]


# IMAGE PROCESSING TOOLS


def pixel_normalization(p):
    return (p / 255.0) - 0.5


roi_points_forward = [[600, 310], [0, 620], [0, 680], [475, 680], [626, 491]]
roi_points_reversed = [(1280 - p[0], p[1]) for p in roi_points_forward[::-1]]
roi_points = np.array([roi_points_forward + roi_points_reversed], np.int32)


def butter_lowpass(cutoff, fs, order=5):
    '''
    low pass filter using scipy
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    applies a low pass filter to the data, in a forward pass w/ specified params
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def region_of_interest(img, vertices=roi_points, invert_mask=False):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        zeros = (0,) * channel_count
    else:
        ignore_mask_color = 255
        zeros = 0

    # defining a blank mask to start with
    if invert_mask:
        mask = np.full(img.shape, np.int32(255))
    else:
        mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    if invert_mask:
        mask = cv2.fillPoly(mask, vertices, zeros)
    else:
        mask = cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img.astype(np.int32), mask.astype(np.int32))
    return masked_image.astype(np.uint8)


def restore_3channel(image):
    shape = image.shape
    if len(shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return image


def gray(image):
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def bin_threshold(image, lower, upper):
    if len(image.shape) > 2:
        error = '{} is the wrong shape, should be single channel, single image.'.format(image.shape)
        raise NameError(error)

    binary_output = np.zeros_like(image)
    binary_output[(image >= lower) & (image <= upper)] = 255

    return binary_output


# DATA INVESTIGATION TOOLS

def showarray(a, fmt='png', width=None, height=None):
    '''
    Displays an image without the ugliness of matplotlib
    '''
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue(), width=width, height=height))


def side_by_side(image, transformation, vert=False, twidth=None, theight=None):
    image_copy = np.copy(image)
    image_copy = restore_3channel(image_copy)
    transformed_image = restore_3channel(transformation(image_copy))

    if vert:
        display_image = np.vstack((image_copy, transformed_image))
    else:
        display_image = np.hstack((image_copy, transformed_image))

    showarray(display_image, width=twidth, height=theight)


def side_by_sides(images, transformation, vert=False, twidth=None, theight=None):
    for image in images:
        side_by_side(image, transformation, vert=vert, twidth=twidth, theight=theight)


# CLASS FOR ORGANIZING LANE LINE DATA, TRANSFORMATIONS, AND UPDATES

class Line():
    def __init__(self, n=10):
        # number of frames to look back
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xbase = []
        # best x base starting point for line fit
        self.best_xbase = None
        # average x values of the fitted line over the last n iterations
        self.poly_coeff = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # radius of curvature of the line in some units
        self.curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # persist annotations with this variable
        self.annotations = None

    def update_base(self, new_base):
        self.recent_xbase.append(new_base)

        if len(self.recent_xbase) > self.n:
            self.recent_xbase.pop(0)

        self.best_xbase = np.round(np.mean(self.recent_xbase)).astype(np.int32)

    def update_polys(self, new_coeff):
        if self.poly_coeff is not None:
            self.poly_coeff = np.vstack((self.poly_coeff, new_coeff))
        else:
            self.poly_coeff = new_coeff.reshape(-1, 3)

        if self.poly_coeff.shape[0] > self.n:
            self.poly_coeff = self.poly_coeff[1:]

        self.best_fit = np.mean(self.poly_coeff, axis = 0)

    def update_curvature(self, new_curvature):
        if self.curvature is not None:
            self.curvature = 0.1 * new_curvature + 0.9 * self.curvature
        else:
            self.curvature = new_curvature

    def update_line_base_pos(self, new_line_base_pos):
        if self.line_base_pos is not None:
            self.line_base_pos = 0.1 * new_line_base_pos + 0.9 * self.line_base_pos
        else:
            self.line_base_pos = new_line_base_pos

    def update_annotations(self, annotations):
        self.annotations = annotations
