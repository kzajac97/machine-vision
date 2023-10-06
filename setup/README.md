# Setup

[![en](https://img.shields.io/badge/lang-pl-red.svg)](https://github.com/kzajac97/machine-vision/tree/main/setup/README.pl.md)

Short tutorial for installing & running python environment for machine vision course. Toggle language by clicking on the icon above.

## Installation

1. Install Python 3.x from the official website: https://www.python.org/downloads/ (3.9+ recommended)
2. Install pip, the package installer for Python, by running the following command in your terminal: `python -m ensurepip --default-pip`
3. Install virtualenv, a tool for creating isolated Python environments, by running the following command in your terminal: `pip install virtualenv`
4. Create a new virtual environment for your project by running the following command in your terminal: `virtualenv env`
5. Activate the virtual environment by running the following command in your terminal: `source env/bin/activate` on Linux or `env\Scripts\activate.bat` on Windows.
6. Install the required packages for your project by running the following command in your terminal: `pip install -r requirements.txt`

*Note*: Running python from terminal is easier for `venv`, to make sure correct version is used. 

## Usage

1. Start Jupyter Notebook by running the following command in your terminal: `jupyter notebook` (or run `jupyer lab` for richer browser IDE)
2. Open the notebook file in your browser by clicking on the link provided in the terminal output (in visual studio code open `ipynb` file and paste the URL into `Select Kernel` - `Select Another Kernel` - `Existing Jupyter Server`).
3. Run the notebook cells by clicking on the "Run" button or by pressing "Shift + Enter" on your keyboard.

# Git

## Getting Started

1. Create a GitHub account at https://github.com/.
2. Install Git on your local machine by following the instructions at https://git-scm.com/downloads.
3. Create a new repository on GitHub by clicking on the "New" button on the main page.
4. Clone the repository to your local machine by running the following command in your terminal: `git clone <repository-url>`.
5. Create a new file in the repository by running the following command in your terminal: `touch <filename>`.
6. Add the file to the staging area by running the following command in your terminal: `git add <filename>`.
7. Commit the changes by running the following command in your terminal: `git commit -m "Commit message"`.
8. Push the changes to the remote repository by running the following command in your terminal: `git push origin master`.

## Branching

Branching allows you to create a copy of your codebase and work on it independently of the main branch.

1. Create a new branch by running the following command in your terminal: `git branch <branch-name>`.
2. Switch to the new branch by running the following command in your terminal: `git checkout <branch-name>`.
3. Make changes to the codebase.
4. Add and commit the changes as before.
5. Push the changes to the remote branch by running the following command in your terminal: `git push origin <branch-name>`.

# Resources

* [Jupyter Notebook](https://jupyter.org/)
* [Google Colab](https://colab.research.google.com/)
* [GitBook](https://www.gitbook.com/)
