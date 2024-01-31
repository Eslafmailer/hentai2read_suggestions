# https://archive.pinecone.io/learn/bag-of-visual-words/

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
import cv2
import matplotlib.pyplot as plt
import functools
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
import sys
import json
import argparse

np.random.seed(42)

parser = argparse.ArgumentParser(prog='Bag of visual words')
parser.add_argument('--vocabulary_size', type=int)
parser.add_argument('--algorithm', type=ascii)
args = parser.parse_args()

FILES_FOLDER = os.path.join('.', 'files')
NUM_DESCRIPTORS_TO_TRAIN_KMEANS = 3000
KMEANS_ITERATIONS = 1
VOCABULARY_SIZE = args.vocabulary_size or 200
RESULT_FILE = 'cover.json'

algorithm = args.algorithm or 'sift'
extractor = cv2.SIFT_create() if algorithm == 'sift' else cv2.ORB_create()
files = [os.path.join(FILES_FOLDER, f) for f in os.listdir(FILES_FOLDER) if os.path.isfile(os.path.join(FILES_FOLDER, f))]
print('files', len(files))

def extract(file):
    with open(file) as f:
        contents = f.read()

    # generate np arrays from the dataset images
    image = np.array(Image.open(BytesIO(base64.b64decode(contents))))

    # if RGB, transform into grayscale
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # extract keypoints and descriptors
    img_keypoints, img_descriptors = extractor.detectAndCompute(bw_image, None)

    # output = cv2.drawKeypoints(bw_image, img_keypoints, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(output, cmap='gray')
    # plt.show()

    if img_descriptors is None:
        return img_descriptors

    return np.array(list(map(lambda x: x.astype(np.float32), img_descriptors)))

def train_kmeans():
    # select 1000 random image index values
    sample_idx = np.random.randint(0, len(files) + 1, NUM_DESCRIPTORS_TO_TRAIN_KMEANS).tolist()

    # extract the sample from descriptors
    descriptors = list(map(lambda i: extract(files[i]), sample_idx))
    print('image descriptor shape', descriptors[0].shape)

    all_descriptors = []
    # extract image descriptor lists
    for descriptor in descriptors:
        # extract specific descriptors within the image
        for value in descriptor:
            all_descriptors.append(value)

    # convert to single numpy array
    all_descriptors = np.stack(all_descriptors)
    print('descriptors shape for training', all_descriptors.shape)

    codebook, _ = kmeans(all_descriptors, VOCABULARY_SIZE, KMEANS_ITERATIONS)
    print('codebook shape', codebook.shape)

    return codebook

def calculate_word_freq(codebook):
    files_to_remove = []
    visual_words = []
    # for ind in np.random.randint(0, len(files) + 1, NUM_DESCRIPTORS_TO_TRAIN_KMEANS * 10).tolist():
    #     file = files[ind]
    for file in files:
        descriptor = extract(file)
        if descriptor is None:
            files_to_remove.append(file)
            print('missing descriptor for', file)
            continue

        # for each image, map each descriptor to the nearest codebook entry
        img_visual_words, _ = vq(descriptor, codebook)
        visual_words.append(img_visual_words)
    print('first image words', visual_words[0][:5], len(visual_words[0]))

    for file in files_to_remove:
        files.remove(file)

    frequency_vectors = []
    for img_visual_words in visual_words:
        # create a frequency vector for each image
        img_frequency_vector = np.zeros(VOCABULARY_SIZE)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)

    # stack together in numpy array
    result = np.stack(frequency_vectors)
    print('frequency_vectors shape', result.shape)
    return result

def normalize(frequency_vectors):
    df = np.sum(frequency_vectors > 0, axis=0)
    print('df', df.shape, df[:5])

    idf = np.log(len(frequency_vectors) / df)
    print('idf', idf.shape, idf[:5])

    tfidf = frequency_vectors * idf
    print('tfidf', tfidf.shape, tfidf[0][:5])

    return tfidf

def save(tfidf):
    result = {}

    # indexes = np.random.randint(0, len(files) + 1, NUM_DESCRIPTORS_TO_TRAIN_KMEANS * 10).tolist()
    # for ind in range(0, len(indexes)):
    #     file = files[indexes[ind]]
    for ind in range(0, len(files)):
        file = files[ind]
        file_name = os.path.basename(file)
        result[file_name] = tfidf[ind].tolist()

    json_object = json.dumps(result, indent=4)
    with open(RESULT_FILE, "w") as outfile:
        outfile.write(json_object)

def main() -> int:
    codebook = train_kmeans()
    frequency_vectors = calculate_word_freq(codebook)
    tfidf = normalize(frequency_vectors)
    save(tfidf)

    # plt.bar(list(range(VOCABULARY_SIZE)), frequency_vectors[0])
    # plt.show()
    #
    # plt.bar(list(range(VOCABULARY_SIZE)), tfidf[0])
    # plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())