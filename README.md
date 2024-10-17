# dcm_ball_detector
given a DCM format image sequence, identify the positions of all spherical markers from this image sequence.

## prerequisites

- `python>=3.12`
- make sure that your dcm files are named under **alphabetical order**.

## usage

### create python venv

- follow the steps below to deploy a virtual environment that includes all the dependencies of the current project:

```bash
cd "<root directory of the current repo>"
python3 -m venv dcm_venv                        # create a virtual environment
source dcm_venv/bin/activate                    # activate the virtual environment
pip install pydicom pylibjpeg pylibjpeg-libjpeg # install dependencies related to DCM file reading
pip install tqdm                                #
pip install numpy                               # 
pip install matplotlib                          # for mannul check and debug
pip install scipy scikit-learn scikit-image     # scipy and sklearn 
```

### run against testcases come with the repo

- if you have successfully configured the virtual environment **according to the above steps**, you can use the following command to run the demonstration test cases that come with this project:

```bash
cd "<root directory of the current repo>"
source dcm_venv/bin/activate # activate the virtual environment
python3 -m dcm_ball_detector # run test
```

- after running the demonstration use case, you can see several pictures in the folder `project root directory/dcm_ball_detector/log_image`. these pictures indicate the center moment slice of an identified spherical markers.

- there are two types of pictures. the first type of pictures shows the process of the marker ball appearing and disappearing as the CT slice moving forward, for example:

<img src="./img/img_process.png" style="width: 400px">

- the green sub-image indicates that the image is recognized as the "big ball stage of the marker", and the blue sub-image indicates that the image is recognized as the "small ball stage of the marker".
- the second type of image shows the image at the moment when the center point of a spherical marker appears in the CT slice, and the marker is circled with a red circle, for example:

<img src="./img/img_dcm.png" style="width: 400px">

### test against your own testcases

- if you want to test against another DCM file set, you can first place all of the DCM files in a folder and use the following command:

```bash
cd "<root directory of the current repo>"
source dcm_venv/bin/activate # activate the virtual environment
python3 -c 'import dcm_ball_detector; dcm_ball_detector.svm_get_ball_centers_in_folder_and_dump_log("<The folder where the target DCMs are located>")'
```

- for example, you can use the following command to test the demonstration use case that comes with this project:

```bash
cd "<root directory of the current repo>"
source dcm_venv/bin/activate # activate the virtual environment
python3 -c 'import dcm_ball_detector; dcm_ball_detector.svm_get_ball_centers_in_folder_and_dump_log("./data_sample/2023_01_03_0.625 x 0.625_501/")'
```

### run in release mode

- since generating log files is time-consuming in production code, if you just want to find the coordinates of each marker sphere and don't care about the log image, you can use commands like this:

```bash
cd "<root directory of the current repo>"
source dcm_venv/bin/activate # activate the virtual environment
python3 -c 'import dcm_ball_detector; print(dcm_ball_detector.get_all_cluster_center_in_folder("<the target folder>"))'
```

- for example, you can run the following to test (in release mode) the demonstration use case that comes with this repo:

```bash
cd "<root directory of the current repo>"
source dcm_venv/bin/activate # activate the virtual environment
python3 -c 'import dcm_ball_detector; print(dcm_ball_detector.get_all_cluster_center_in_folder("./data_sample/2023_01_03_0.625 x 0.625_501/"))'
```

- the program will output something similar to the following:

```
[{'time': 36, 'xpos': 86, 'ypos': 185}, {'time': 38, 'xpos': 80, 'ypos': 258}, {'time': 126, 'xpos': 112, 'ypos': 185}, {'time': 130, 'xpos': 105, 'ypos': 273}, {'time': 214, 'xpos': 166, 'ypos': 264}, {'time': 289, 'xpos': 202, 'ypos': 320}, {'time': 290, 'xpos': 176, 'ypos': 198}, {'time': 352, 'xpos': 212, 'ypos': 264}]
```

- where `time` indicates the index number of the CT slice where the center of the marker in.
- the ordered pair `(xpos, ypos)` gives the approximate coordinates of the center point of the marker in a certain frame of the image (and the unit is voxel).

## other information
- this repo is developed and tested under AOSC OS (also known as "安同 OS").
  - and it's a debian-like linux distribution.
  - see: https://wiki.aosc.io/