import os
import cv2
import numpy as np

import matplotlib.image as mpimg
from scipy.signal import butter, lfilter
from scipy.ndimage.measurements import label
from skimage.feature import hog
from collections import deque

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


def load_images_from_directory(directory):
    image_filenames = []

    for s in os.walk(directory):
        new_images = [s[0] + '/' + filename for
                      filename in s[2] if filename[-4:] in ['.jpg', 'jpeg', '.png']]
        image_filenames += new_images

    images = load_images(image_filenames)

    return images


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


# COURSE STARTER CODE

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_size=(32, 32), hist_bins=32):
    assert(type(imgs) == list)

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    # for file in imgs:
    for image in imgs:
        # # Read in each one by one
        # image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # spatial_features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # hist bins
        hist_features = color_hist(feature_image, nbins=hist_bins)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        img_features = np.hstack((spatial_features, hist_features, hog_features))  # .reshape(1, -1)
        features.append(img_features)
    # Return list of feature vectors
    return features


def find_cars(img, ystart, ystop,
              scale, svc, transformer,
              orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins,
              visualize=False):

    if visualize:
        draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    # showarray(cv2.cvtColor(ctrans_tosearch, cv2.COLOR_YCrCb2RGB))

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(
            imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    boxes = []
    # debug = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            features = np.hstack((spatial_features, hist_features, hog_features))
            transformed_features = transformer(features)

            # Scale features and make a prediction
            # test_features = X_scaler.transform(
            #     np.hstack((spatial_features, hist_features, hog_features)))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(transformed_features)

            if test_prediction[0] == 1:
                # debug += [[subimg, transformed_features]]

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                p1 = (xbox_left, ytop_draw + ystart)
                p2 = (xbox_left + win_draw, ytop_draw + win_draw + ystart)

                boxes += [(p1, p2)]

                if visualize:
                    cv2.rectangle(draw_img, p1, p2, (0, 0, 255), 3)

    if visualize:
        return boxes, draw_img
    else:
        return boxes


def add_heat(heatmap, bbox_list, amount=1):
    # Iterate through list of bboxes
    temp_heatmap = np.zeros_like(heatmap)

    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        temp_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    for box in bbox_list:
        # repeat if the window overlaps other windows = has heat > 1 now
        if np.max(temp_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]) > 1:
            temp_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    heatmap += temp_heatmap * amount

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


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

class VehicleTracker():
    '''
    Thanks to Patrick Kern, forum mentor, for the basic ideas in this class
    '''
    def __init__(self, framesize=(720, 1280), n=10, alpha=0.4,
                 heatmap_threshold=20):
        # number of frames to look back
        self.n = n
        # framesize
        self.framesize = framesize
        # heatmap
        self.heatmaps = deque(maxlen = n)
        # alpha = strength of heatmap update
        self.alpha = alpha
        # threshold for heatmap
        self.heatmap_threshold = heatmap_threshold
        # last frame
        self.frame = np.zeros(framesize)

    def update_heatmap(self, heatmap):
        # self.heatmap = self.alpha * heatmap + (1 - self.alpha) * self.heatmap
        self.heatmaps.append(heatmap)

    def process_frame(self, frame, ystart, ystop,
                      scales, svc, transformer, orient,
                      pix_per_cell, cell_per_block, spatial_size, hist_bins):

        temp_heatmap = np.zeros(self.framesize)

        for scale in scales:
            boxes = find_cars(img=frame,
                              ystart=ystart,
                              ystop=ystop,
                              scale=scale,
                              svc=svc,
                              transformer=transformer,
                              orient=orient,
                              pix_per_cell=pix_per_cell,
                              cell_per_block=cell_per_block,
                              spatial_size=spatial_size,
                              hist_bins=hist_bins,
                              visualize=False
                              )

            add_heat(temp_heatmap, boxes, 1)

        self.update_heatmap(temp_heatmap)

        total_heatmap = np.sum(self.heatmaps, axis=0)

        thresholded_heatmap = apply_threshold(total_heatmap, self.heatmap_threshold)
        labels = label(thresholded_heatmap)

        self.frame = draw_labeled_bboxes(np.copy(frame), labels)
