# TurboRAG

![Paper Cover Image](assets/image/TurboRAG.png)

This repository contains the implementation code and demonstration effects for the paper [*"TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text"*](https://arxiv.org/abs/2410.07590).

**Note:** The testing code and models will be open-sourced soon.

## 创建环境
```
conda create -n turborag python=3.10.12
conda activate turborag
pip install -r requirements.txt
```

## TTFT Testing
The following steps outline how to test TurboRAG against traditional RAG in terms of *time-to-first-token (TTFT)*. We provide some documents and related query examples located in the `documents` and `questions` directories. You can replace these with your own data as needed.

### Step 1: Prepare Chunked Caches
Run `chunk_cache.py`. This script will automatically split the documents in the documents directory into chunks, each with a length of 512 tokens. The KV cache for each chunk will be stored in the chunk_kvcache directory.
### Step 2: Compare TTFT
Run `turbo_rag.py` to compare the TTFT of both methods.

<!-- ## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction

Provide an introduction to your project, including the motivation, objectives, and a summary of the paper. Explain the problem you are addressing and why it is important.

## Installation

Instructions on how to install and set up the project locally. Include all necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/your_username/your_project_name.git

# Navigate into the directory
cd your_project_name

# Install required dependencies
pip install -r requirements.txt
```

## Usage

Detailed instructions on how to use the code in this repository.

```bash
# Example command to run your code
python main.py --input data/input_file --output results/output_file
```

Explain any command-line arguments or configuration files.

## Dataset

Information about the dataset used in the project.

- **Download Link:** Provide a link to download the dataset if it's publicly available.
- **Data Preparation:** Instructions on how to prepare or preprocess the data.
- **Data Description:** Briefly describe the dataset structure and contents.

## Results

Present the results achieved in your project.

- Include figures, tables, or charts if applicable.
- Explain the significance of the results.
- Compare with baseline methods if available.

## Project Structure

Overview of the repository structure.

```
├── data
│   ├── raw
│   └── processed
├── docs
├── results
├── src
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
├── tests
├── LICENSE
├── README.md
└── requirements.txt
```

Explain what each folder contains.

## Requirements

List all software and libraries required to run the project.

- Python 3.x
- NumPy >= 1.18.0
- Pandas >= 1.0.0
- Other dependencies...

Alternatively, include a `requirements.txt` file.

## Examples

Provide examples or tutorials on how to use the code.

- Link to Jupyter notebooks if applicable.
- Include sample input and output files.

## Contributing

Guidelines for contributing to the project.

- How to report bugs.
- How to propose new features.
- Coding standards.

## License

Specify the license under which the project is distributed.

This project is licensed under the [MIT License](LICENSE). -->

## Citation

If you use this code or data in your work, please cite our paper:

```
@article{lu2024turborag,
  title={TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text},
  author={Lu, Songshuo and Wang, Hua and Rong, Yutian and Chen, Zhi and Tang, Yaohua},
  journal={arXiv preprint arXiv:2410.07590},
  year={2024}
}
```

<!-- ## Acknowledgements

Acknowledge any individuals or organizations that supported your work.

- Funding sources.
- Collaborators.
- Any third-party resources. -->