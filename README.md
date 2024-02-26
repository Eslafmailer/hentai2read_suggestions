### Setup

#### Configuration

Create `config.json` file in the root with the following content:

```json
{
  "url": "aHR0cHM6Ly9oZW50YWkycmVhZC5jb20=",
  "login": "<BASE64 ENCODED LOGIN>",
  "password": "<BASE64 ENCODED PASSWORD>",
  "all_pages": "aGVudGFpLWxpc3QvYWxsL2FueS9hbGwvbGFzdC1hZGRlZA==",
  "labels": {
    "true": [
      "Ym9va21hcmsv...<BASE64 ENCODED URL TO BOOKMARK WITH GOOD EXAMPLES>"
    ],
    "false": [
      "Ym9va21hcmsv...<BASE64 ENCODED URL TO BOOKMARK WITH BAD EXAMPLES>"
    ]
  },
  "input_category": <category ID to use for prediction>
}
```

Where categories as following:
* On Hold - 1
* Plan To Read - 2
* Completed - 3
* Favorite - 4
* Reading - 5

#### Python

* Install 
    * Python 3.11: you can use [pyenv](https://github.com/pyenv/pyenv#installation) on macOS, or use the [deadsnake ppa](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) on ubuntu
    * [Poetry](https://pypi.org/project/poetry/): You can install it with `python3.11 -m pip install poetry` once you are running Python 3.11
* Run `poetry install` to install the dependencies 

#### JS

* Install [node 18+](https://nodejs.org/en/download/package-manager/)
* Run `npm install` to install the dependencies

### Scraping

Run `npm run scrap` to scrap the books into `db.json`. On the first run the command will scrap all the books, on subsequent runs - only new books. You can change this behavior with the next arguments:
* pass `--update-all=true` to trigger the full update.
* pass `--update-months=N` to scrap the new books and the books updated in the N months. You may want to use this, because the statistics (views, score, etc.) of the new books do not stay the same for long.

Next run `npm run label` to update the label information. Again, on subsequent runs it updates only the newly labeled books, unless the `--update-all=true` argument is passed.

Finally, run `npm run anonymize` to prepare `data.json`, file with replaced real text, so it can be made publicly available.

#### Cover score

To train the model first run `npm run embedd:cnn:custom:prepare` to create `images` folder with cover images. And then run `npm run embedd:cnn:custom:train` to train the model.

When the model is trained you can calculate the scores with `npm run embedd:cnn:custom:predict` (scores are saved in `cover-vgg.json`).

### Prediction

Upload `db.json` and `cover-vgg.json` to gdrive, then share the files and use the share IDs in `catboost.ipynb` notebook. Executing the notebook will train the predictor and will save the top predictions to `result.json`. The result can be accepted (assigned the `input_category`) by running `npm run accept`.