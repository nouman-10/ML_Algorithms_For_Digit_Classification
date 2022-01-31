import argparse
from helpers import *

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="cnn",
        type=str,
        help="Model to train (lr: logistic regression, svm: support vector machine, cnn: convolutional neural network, mlp: multi-layer perceptron) (default: cnn)",
    )
    parser.add_argument(
        "-d", "--data_augment", action="store_true", help="Whether to use data augmentation (default: False)",
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = create_arg_parser()

    data, labels = read_data("data/mfeat-pix.txt")
    X_train, y_train, X_test, y_test = split_data(data, labels)
    
    if args.model_name == "lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

        fit_and_predict_model(model, "Logistic Regression", X_train, y_train, X_test, y_test, args.data_augment)

    elif args.model_name == "svm":
        from sklearn.svm import SVC
        model = SVC()

        fit_and_predict_model(model, "Support Vector Machine", X_train, y_train, X_test, y_test, args.data_augment)

    elif args.model_name == "cnn":
        fit_CNN_model(X_train, y_train, X_test, y_test, args.data_augment)

    else:
        fit_MLP_model(X_train, y_train, X_test, y_test, args.data_augment)

    