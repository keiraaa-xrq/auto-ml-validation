from .evaluation_pipeline import evaluation_pipeline
from .report_generation import generate_report
from .algorithms.logistic_regression import LogisticClassifier

def main():
    target_name = 'loan_status'
    X_train = pd.read_csv("~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_train_X_processed.csv")
    y_train = pd.read_csv("~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_train_y.csv")
    X_test = pd.read_csv('~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_test_X_processed.csv')
    y_test = pd.read_csv("~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_test_y.csv")
    raw_train = pd.read_csv("~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_train.csv")
    raw_test = pd.read_csv("~/Desktop/Capstone/auto-ml-validation/data/stage_2/loanstats_test.csv")

    # train pipeline
    model = LogisticClassifier() # replaced by train pipeline
    model.fit(X_train, y_train)

    # auto benchmarking pipeline
    # benchmodel = ...

    # prediction pipeline
    train = {'raw_X': raw_train.drop_columns('loan_status'), 'processed_X': X_train, 'y': y_train, 'pred_proba': model.predict_proba(X_train)}
    test = {'test1': {'raw_X': raw_test.drop_columns('loan_status'), 'processed_X': X_test, 'y': y_test, 'pred_proba': model.predict_proba(X_test)}}

    # evaluation pipeline
    results = {}
    for k, v in test.items():
        print(f"Start evaluating model using testing dataset {k}")
        results[k] = evaluation_pipeline(model._model, train, v, 0.5, "", {}, 0, [], {}, [])
    print("Evaluation done!")

    # report generation
    generate_report(results)


if __name__ == "__main__":
    main()