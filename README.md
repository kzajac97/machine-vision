# Machine Vision

This repository contains notebooks and source code for laboratories of two courses: *Machine
Learning and Machine Vision* and *Digital Signals and Images* on Wrocław University of Science and Technology.

### Classes

Source code is shared between classes, notebooks and examples are organized in directories. Currently implemented
classes are:

| Class         | Path                    | Description                                                 |
|---------------|-------------------------|-------------------------------------------------------------|
| Interpolation | `classes/interpolation` | Function and image interpolation                            |
| Convolution   | `classes/convolution`   | Convolution examples, edge detection, blur, sharpening etc. |
| Demosaicking  | `classes/demosaicking`  | Color image reconstruction                                  |                                    |
| Compression   | `classes/compression`   | Image compression using Fourier and Wavelet transforms      |

### Installation

Nothing extra is required, simply installing the `requirements.txt` file should be enough:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
