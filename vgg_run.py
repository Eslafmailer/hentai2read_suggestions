import tensorflow as tf
import os
import json
import base64
from tqdm import tqdm

FILES_FOLDER = os.path.join('.', 'files')
IMAGE_SIZE = (224, 224)
CHANNELS = 3
IMAGE_SHAPE = (*IMAGE_SIZE, CHANNELS)
BATCH_SIZE = 32
RESULT_FILE = 'cover-vgg.json'

model = tf.keras.models.load_model(os.path.join('.', 'trained_categorical_vgg'))
# model = tf.keras.models.Sequential(model.layers[:-2])

def load_image(path):
    with open(path) as f:
        contents = f.read()

    content = tf.constant(base64.b64decode(contents), dtype=tf.string)
    img = tf.io.decode_image(content, channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape(IMAGE_SHAPE)
    return img


def predict(paths):
    images = []
    for path in paths:
        images.append(load_image(path))

    img_ds = tf.data.Dataset.from_tensor_slices(images)
    img_ds = img_ds.batch(BATCH_SIZE)
    img_ds = img_ds.prefetch(tf.data.AUTOTUNE)

    return model.predict(img_ds, verbose=0)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


with open('/home/sergey/projects/scrapper/db.json') as f:
    db = json.load(f)

ids = list(map(lambda x: x.get('id'), db.values()))
print('all entries', len(ids))
ids = list(filter(lambda x: os.path.isfile(os.path.join(FILES_FOLDER, str(x))), ids))
print('entries with image', len(ids))

result = []
resultSet = set()
if os.path.isfile(RESULT_FILE):
    with open(RESULT_FILE) as f:
        contents = f.read()
    result = json.loads(contents)

    for entry in result:
        resultSet.add(entry['id'])

for chunk in tqdm(list(chunks(ids, BATCH_SIZE))):
    chunk = list(filter(lambda x: x not in resultSet, chunk))
    if not len(chunk):
        continue

    image_paths = list(map(lambda x: os.path.join(FILES_FOLDER, str(x)), chunk))
    prediction = predict(image_paths)

    for idx, id in enumerate(chunk):
        item = {}
        item['id'] = id
        item['cover'] = float(prediction[idx][1])

        result.append(item)

json_object = json.dumps(result, indent=4)
with open(RESULT_FILE, "w") as outfile:
    outfile.write(json_object)
