# Epidemetrics

## Downloading and compute summary table

Downloading the data and computing the summary statistics is done by running the
`generate_table.py` script. Depending on your internet connection this should
only take a couple of minutes.

```
$ python generate_table.py
```

This generates the files 

- `foo.csv` which contains x, y and z

## Environment

To set up a virtual environment for the python packages specified in
`requirements.txt` run the following commands.

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
