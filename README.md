# Simple but complete Stable Diffusion (scsd) Helper

## Description
This package provide a wraper handler to simplify the operations of image generation 
using Stable Diffusion XL and XL turbo model with huggingface diffuser library

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
You can create a new pyenv environment and install required packages.

```{bash}
cd sd_app
pyenv virtualenv sd
pyenv local sd
pip install -r requirements.txt
pip install -e .
```

## Usage
You can use this package inside a jupyter notebook.

```{python}
from scsd import LDMHandler

ldm = LDMHandler()
prompt = "a lovely cat"
ldm.txt2img(prompt)
```

## Contributing
[Explain how others can contribute to the project]

## License
[Specify the license under which the project is distributed]
