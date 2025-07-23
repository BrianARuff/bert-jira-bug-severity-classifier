# bert-jira-bug-severity-classifier

## Setup Steps

`cd ./src` - change directories into code source folder
`python dataset.py` - generate model's training data
`python classifier.py` - train model on training data
`python api.py` - run application's API.

### Predict API

1. URL `http://localhost:5000/predict`
2. Request Body:

```cmd
{
    "description": "spelling on button is missing capitalized first letter and it's lowercase instead."
}
```

### Other endpoints

1. URL: `http://localhost:5000/feedback`
2. Request Body:

```cmd
{
    "description": "spelling on button is missing capitalized first letter and it's lowercase instead.",
    "predicted_severity": 3,
    "correct_severity": 4
}
```

### Swagger Docs

> Lists all other endpoints

1. URL: `http://localhost:5000/api/docs`
