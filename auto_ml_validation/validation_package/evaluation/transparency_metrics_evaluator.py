import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick


def Lime_Interpretability(model, X, class_name_list):
    """
    Generate local and global interpretability plots using the LIME algorithm.

    Arguments:
    model -- scikit-learn model
    X -- pandas DataFrame

    Returns:
    local_lime_fig -- matplotlib Figure object representing the local interpretability plot
    global_lime_fig -- matplotlib Figure object representing the global interpretability plot
    """
    # Create a LimeTabularExplainer object
    explainer = LimeTabularExplainer(X.values, feature_names=X.columns, discretize_continuous=True,
                                     class_names=[class_name_list])

    # Generate a random integer between 0 and the number of rows in the input data frame
    i = np.random.randint(0, X.shape[0])

    # Define a predict function for the model
    predict_fn = lambda x: model.predict_proba(x).astype(float)

    # Generate a local interpretability plot
    local_lime_fig = explainer.explain_instance(X.iloc[i], predict_fn)
    
    # Generate a global interpretability plot using submodular pick
    sp_obj = submodular_pick.SubmodularPick(explainer,
                                            X.values,
                                            predict_fn,
                                            sample_size=15,
                                            num_features=5,
                                            num_exps_desired=1)
    global_lime_fig = sp_obj.sp_explanations[0]
    plt.show(global_lime_fig)

    #show_in_notebook()
    return local_lime_fig, global_lime_fig


def SHAP_Interpretability(model, X):
    """
    Computes and visualizes SHAP (SHapley Additive exPlanations) values for a given model and dataset.
    SHAP values explain how much each feature contributes to the predicted output of the model.

    Args:
    - model: a trained machine learning model that has a `predict` method
    - X: a dataset of shape (n_samples, n_features) that the model was trained on

    Returns:
    - local_shap_fig: a waterfall plot of SHAP values for a randomly selected instance in the dataset
    - global_shap_fig: a bar plot of the mean absolute SHAP values across all instances in the dataset
    """
    # create an explainer object for the given model and dataset
    explainer = shap.Explainer(model.predict, X)

    # compute SHAP values for all instances in the dataset
    shap_values = explainer(X)

    # select a random instance to plot the SHAP values for
    i = np.random.randint(0, X.shape[0])

    # create a waterfall plot for the selected instance
    local_shap_fig = shap.plots.waterfall(shap_values[i])

    # create a bar plot of the mean absolute SHAP values for all instances in the dataset
    global_shap_fig = shap.plots.bar(shap_values, show=True, max_display=12)

    return local_shap_fig, global_shap_fig
