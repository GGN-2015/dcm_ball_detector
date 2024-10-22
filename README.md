# dcm_ball_detector
given a `.dcm` format image sequence, identify the positions of all spherical markers from this image sequence.

## install

- it is recommended that this module should be installed in a python virtual environment (venv).

```bash
pip install dcm-ball-detector
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
[{"zmm": 132.5840302807579, "xmm": 121.99777985982264, "ymm": 349.02488432862555}, {"zmm": 161.7956332082969, "xmm": 175.8402398543241, "ymm": 331.0029681581956}, {"zmm": 176.0079302709168, "xmm": 108.66976835102449, "ymm": 388.61439982471455}, {"zmm": 201.8262056427377, "xmm": 169.0084087325416, "ymm": 364.3212735729133}]
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
