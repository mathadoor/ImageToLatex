import numpy as np
from skimage.draw import line, disk
from skimage.io import imsave
import pandas as pd
import os, re
import xml.etree.ElementTree as ET
from tqdm import tqdm

# GLOBAL VARIABLES
# GET CROHME DATASET PATH
CROHME_PATH = os.path.abspath(__file__)
CROHME_PATH = re.findall('(.+/ImageToLatex).*', CROHME_PATH)[0]
CROHME_PATH = os.path.join(CROHME_PATH, 'data/CROHME')

# DATASET LOCATIONS
CROHME_TRAIN = os.path.join(CROHME_PATH, 'train')
CROHME_VAL = os.path.join(CROHME_PATH, 'val')
IMG_SIZE = (512, 512)

# GENERATE IMAGES
def generate_image(inkml_file, img_loc, img_size=IMG_SIZE, line_width=2, export_label=False, label_loc=None):
    '''
    :param inkml_file: contains the stroke data as traces and the ground truth as annotation.
    :param img_loc: location of the image files
    :param img_size: size of the output image
    :param line_width: width of the line
    :param export_label: whether to export the label or not
    :param label_loc: location of the label files
    :return: the image of the equation
    '''
    # Parse the XML file
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()

        # Export the label
        if export_label:
            for annotation in root.findall('{http://www.w3.org/2003/InkML}annotation'):
                # if the attribute is truth, then export the label
                if annotation.attrib['type'] == 'truth':
                    # Some labels have $ in the beginning and end. Remove them.
                    label = re.findall('\$?([^\$]+)\$?', annotation.text)[0].strip()
                    with open(label_loc + inkml_file.split('/')[-1].split('.')[0] + '.txt', 'w') as f:
                        f.write(label)

        # Extract the traced points
        traces = []
        for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
            points = trace.text.strip().split(',')
            traces.append([(float(pt.split()[0]), float(pt.split()[1])) for pt in points])
    except:
        print("Error while parsing the file: {}".format(inkml_file))
        return

    try:
        # Get the image size
        dpi = 100
        fig_size = (img_size[0] / dpi, img_size[1] / dpi)

        # Create an empty image
        # Determine the bounds of the traces
        min_x = min(point[0] for trace in traces for point in trace)
        max_x = max(point[0] for trace in traces for point in trace)
        min_y = min(point[1] for trace in traces for point in trace)
        max_y = max(point[1] for trace in traces for point in trace)

        # Create an empty image
        img = 255 * np.ones(img_size, dtype=np.uint8)

        # Draw the traces onto the image
        for trace in traces:
            for i in range(len(trace) - 1):
                # Scale the trace coordinates to the image size
                x1 = int((trace[i][0] - min_x) / (max_x - min_x) * (img_size[1] - 1))
                y1 = int((trace[i][1] - min_y) / (max_y - min_y) * (img_size[0] - 1))
                x2 = int((trace[i + 1][0] - min_x) / (max_x - min_x) * (img_size[1] - 1))
                y2 = int((trace[i + 1][1] - min_y) / (max_y - min_y) * (img_size[0] - 1))

                # Create a line from (x1, y1) to (x2, y2)
                rr, cc = line(y1, x1, y2, x2)

                # For each pixel on the line, create a disk of radius line_width / 2 and set it to black
                for r, c in zip(rr, cc):
                    rr_disk, cc_disk = disk((r, c), radius=line_width, shape=img.shape)
                    img[rr_disk, cc_disk] = 0  # Set the pixels within the disk to black

        # Save the image
        imsave(img_loc + inkml_file.split('/')[-1].split('.')[0] + '.png', img)
    except:
        print("Error while generating the image: {}".format(inkml_file))
        return

def generate_images(inkml_loc, img_loc, img_size=IMG_SIZE, line_width=2, export_label=False, label_loc=None):
    '''
    :param inkml_loc: location of the INKML files
    :param img_loc: location of the image files
    :param img_size: size of the output image
    :param line_width: width of the line
    :param export_label: whether to export the label or not
    :param label_loc: location of the label files
    :return: None
    '''
    # Get all the INKML files
    inkml_files = os.listdir(inkml_loc)

    # Assert if export_label true implies label_loc is not None
    assert not export_label or label_loc, "The label is set to export, but the label location is not specified."

    print("Generating images...")
    i = 0
    for INKML_file in tqdm(inkml_files):
        # Generate the image
        generate_image(os.path.join(inkml_loc, INKML_file), img_loc, img_size,
                       line_width, export_label=export_label, label_loc=label_loc)
        #
        # i += 1
        # if i == 10:
        #     break
def generate_annotated_csv(img_loc, label_loc, csv_loc):
    '''
    :param img_loc: location of the image files
    :param label_loc: location of the label files
    :param csv_loc: location of the csv file
    :return: None
    '''
    # Get all the image files
    img_files = os.listdir(img_loc)

    # Dataframe to export the csv file
    df = pd.DataFrame(columns=['image_loc', 'label'])

    # For each image file, get the corresponding label file and add it to the dataframe along with the image location
    for img_file in tqdm(img_files):
        # Get the label file
        label_file = label_loc + img_file.split('.')[0] + '.txt'

        # Read the label file
        with open(label_file, 'r') as f:
            label = f.read()

            # Add the image and label to the dataframe
        new_row = pd.DataFrame([[img_loc + img_file, label]], columns=['image_loc', 'label'])
        df = pd.concat([df, new_row], ignore_index=True)

    # Export the dataframe to csv
    df.to_csv(csv_loc, index=False)

# Main Function
if __name__ == '__main__':
    # generate_images(CROHME_TRAIN + "/INKML/", CROHME_TRAIN + "/IMG_RENDERED/",
    #                 export_label=True, label_loc=CROHME_TRAIN + "/IMG_RND_LABELS/")
    generate_annotated_csv(CROHME_TRAIN + "/IMG_RENDERED/", CROHME_TRAIN + "/IMG_RND_LABELS/", CROHME_TRAIN + "/train.csv")