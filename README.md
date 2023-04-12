# Automated Validator Tool (AutoML + Validation)
The automated validator tool provides a comprehensive and systematic evaluation of ML models, with a focus on fairness, performance, statistical validity, and transparency. The tool consists of a web-based user interface that allows users to upload their data, select their models, and run the validation process, as well as a backend engine that performs the modelling and evaluation process and finally generates the corresponding reports and visualizations.

The tool provides the following functionalities:
- Model Replication
- Automated development of benchmark model
- Automated Feature Selection
- Model Evaluation on Transparency, Generic Performance, Fairness, Statistical Metrics
- Report Generation
- User Interface

### Tools & Technologies
- Python 
- Dash
- Joblib 
- Scikit-learn
- Poetry

### System Design
![System Design](https://github.com/keiraaa-xrq/auto-ml-validation/blob/main/auto_ml_validation/app/assets/images/SystemDesign.png)


### Set up
1. Download the package and put its contents to a directory of your choice.
2. Install Python 3.6.2 on your local machine, if it's not already installed. You can download this version from the [official Python website](https://www.python.org/downloads/release/python-362/).
3. Open Command Prompt/Bash
4. Install `virtualenv` package (if not already installed) 
   1. `pip install virtualenv`
5. Create a new virtual environment
   1. `virtualenv myenv`
6. Activate the virtual environment
   1. `source myenv/bin/activate`
7. Install `Poetry` on your local machine by running the following command in your terminal or command prompt:
   1. `py -m pip install poetry`
8. Install the required packages from the Poetry file by navigating to the root directory of the package and running the following command
   1. `poetry install`
9. Run the app.py file in the root directory of the package to start the application.
   1. `py app.py`

### Results


