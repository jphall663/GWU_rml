## Using Class Software Resources

### Mac

**Preliminaries:**

* Install [`homebrew`](https://brew.sh/): `$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`

* Install Java:

```
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk8
```

* [Install Mac command line developer tools](https://www.cnet.com/how-to/install-command-line-developer-tools-in-os-x/)

* Install Graphviz: `$ brew install graphviz`

* Install Git: `$ brew install git`

1. Clone this repository with the examples: `$ git clone https://github.com/jphall663/GWU_rml.git`

2. Install Anaconda Python 5.1.0 from the [Anaconda archives](https://repo.continuum.io/archive/) (Anaconda3-5.1.0-MacOSX-x86_64.pkg) and *add it to your system path.* (This means when you type `$ python` at the command prompt, you will start a Python 3.6 session. You should probably check for a line like this `export PATH="/Users/phall/anaconda/bin:$PATH"` in your `.bash_profile` file. You may also need to close and restart your terminal session.)

3. Install `virtualenv`: `$ pip install virtualenv` 
 
4. Change directories into the cloned repository: `$ cd GWU_rml`

5. Create a Python 3.6 virtual environment: `$ virtualenv -p /path/to/anaconda3/bin/python3.6 env_rml` (`/path/to/anaconda3/bin/python3.6` is not a real path. You must replace it with the path to `python3.6` on your system. This can be as simple as `$ virtualenv -p python3.6 env_rml`.) 

6. Activate the virtual environment: `$ source env_rml/bin/activate`

7. Install the correct packages for the example notebooks: `$ pip install -r requirements.txt`

8. Start Jupyter: `$ jupyter notebook`

9. When you are finished, deactivate the environment: `$ deactivate`

If you get stuck, try this link: [Mac Setup: Virtualenv](https://sourabhbajaj.com/mac-setup/Python/virtualenv.html).

### Ubuntu

1. Install [Git](https://git-scm.com/downloads) (`$ sudo apt-get install git`), [Graphviz](https://www.graphviz.org/) (`$ sudo apt-get install graphviz`) and Java (`$ sudo apt-get install openjdk-8-jdk`). They must all be *added to your system path.*

2. Clone this repository with the examples: `$ git clone https://github.com/jphall663/GWU_rml.git`

3. Install Anaconda Python 5.1.0 from the [Anaconda archives](https://repo.continuum.io/archive/) (Anaconda3-5.1.0-Linux-x86_64.sh) and *add it to your system path.* (This means when you type `$ python` at the command prompt, you will start a Python 3.6 session.)

4. Install `virtualenv`: `$ pip install virtualenv` 

5. Change directories into the cloned repository: `$ cd GWU_rml`

6. Create a Python 3.6 virtual environment: `$ virtualenv -p /path/to/anaconda3/bin/python3.6 env_rml` 

7. Activate the virtual environment: `$ source env_rml/bin/activate`

8. Install the correct packages for the example notebooks: `$ pip install -r requirements.txt`

9. Start Jupyter: `$ jupyter notebook`

10. When you are finished, deactivate the environment: `$ deactivate`.
