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

def load_image(path):
    if not os.path.isfile(path):
        return None

    with open(path) as f:
        contents = f.read()

    content = tf.constant(base64.b64decode(contents), dtype=tf.string)
    img = tf.io.decode_image(content, channels=CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE)
    img.set_shape(IMAGE_SHAPE)
    return img


with open('/home/sergey/projects/scrapper/db.json') as f:
    db = json.load(f)

ids = list(map(lambda x: str(x.get('id')), db.values()))
image_paths = list(map(lambda x: os.path.join(FILES_FOLDER, x), ids))

images = []
for file in tqdm(image_paths):
    file_name = os.path.basename(file)

    image = load_image(file)
    if image is not None:
        images.append(image)

img_ds = tf.data.Dataset.from_tensor_slices(images)
img_ds = img_ds.batch(BATCH_SIZE)
img_ds = img_ds.prefetch(tf.data.AUTOTUNE)

prediction = model.predict(img_ds)

result = []
for idx, id in tqdm(enumerate(ids)):
    item = {}
    item['id'] = id
    item['cover'] = str(prediction[idx][0])

    result.append(item)

json_object = json.dumps(result, indent=4)
with open(RESULT_FILE, "w") as outfile:
    outfile.write(json_object)