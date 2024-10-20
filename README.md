# dcm_ball_detector
given a `.dcm` format image sequence, identify the positions of all spherical markers from this image sequence.

## install

- it is recommended that this module should be installed in a python virtual environment (venv).

```bash
pip install dcm_ball_detector
```

## usage

- we assume that you have put your `.dcm` files into a folder of which the path is `<folder path>`.

- we assume that your `.dcm` files are named under **alphabetical order**.

- use the following command to get a prediction of the position of the marker balls.

```bash
python3 -m dcm_ball_detector <folder path>
```

- the program will output something like below to stdout:

```json
[{'zmm': 132.5, 'xmm': 121.73248817021276, 'ymm': 348.78409531914895}, {'zmm': 162.5, 'xmm': 175.7598284255319, 'ymm': 331.00294536170213}, {'zmm': 176.25, 'xmm': 108.73857089361702, 'ymm': 388.4497375319149}, {'zmm': 203.75, 'xmm': 168.92092459574468, 'ymm': 364.5135741276596}]
```

## debug

- if you want to show the debug info of the marker balls, use the following command:

```bash
python3 -m dcm_ball_detector --debug <folder path>
```

- after the program has been finished, in stderr there will be a line like:

```bash
   tips: relevant images in: .../dcm_ball_detector/log_image
```

- relevant images for debugging will be listed in folder `.../dcm_ball_detector/log_image`.
