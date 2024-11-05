import os
import numpy as np
import traceback
from .utils import load_module


def test_assignment_1_1(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    knn_module = load_module(src_dir, "knn_classifier")
    KNNClassifier = knn_module.KNNClassifier

    ret = {"points": 0, "message": "", "max_points": 0.5}

    np.random.seed(seed)

    num_train = 500
    num_test = 50

    num_features = 5
    num_classes = 10

    X_train = np.random.randn(num_train, num_features)
    y_train = np.random.randint(num_classes, size=num_train)
    X_test = np.random.randn(num_test, num_features)

    try:
        classifier = KNNClassifier(k=3)
        classifier.train(X_train, y_train)
        dists = classifier._compute_distances(X_test)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e}"
        return ret

    if generate:
        np.save(verification_file, dists)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_dists = np.load(verification_file)

        try:
            if np.allclose(expected_dists, dists, atol=1e-6, rtol=1e-4):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = np.sum(np.abs(dists - expected_dists))
                ret[
                    "message"
                ] = f"\tFAILED! \n\tDifference of distance matrices: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret


def test_assignment_1_2(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    knn_module = load_module(src_dir, "knn_classifier")
    KNNClassifier = knn_module.KNNClassifier

    ret = {"points": 0, "message": "", "max_points": 1}

    np.random.seed(seed)

    # Generate random distance matrix
    num_train = 500
    num_test = 50

    num_features = 5
    num_classes = 10

    X_train = np.random.randn(num_train, num_features)
    y_train = np.random.randint(num_classes, size=num_train)

    dists = np.random.randn(num_test, num_train)

    try:
        classifier = KNNClassifier(k=3)
        classifier.train(X_train, y_train)
        y_pred = classifier._predict_labels(dists)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        np.save(verification_file, y_pred)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_y_pred = np.load(verification_file)

        try:
            if np.allclose(y_pred, expected_y_pred, atol=1e-6, rtol=1e-4):
                ret["message"] += f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = np.sum(np.abs(y_pred - expected_y_pred))
                ret[
                    "message"
                ] += f"\tFAILED! \n\tDifference of predicted labels: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret


def test_assignment_1_3(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    knn_module = load_module(src_dir, "knn_classifier")
    KNNClassifier = knn_module.KNNClassifier

    ret = {"points": 0, "message": "", "max_points": 1}

    np.random.seed(seed)

    num_train = 500
    num_test = 50

    num_features = 5
    num_classes = 10

    X_train = np.random.randn(num_train, num_features)
    y_train = np.random.randint(num_classes, size=num_train)
    X_test = np.random.randn(num_test, num_features)

    try:
        classifier = KNNClassifier(k=3)
        classifier.train(X_train, y_train)
        dists = classifier._compute_distances_vectorized(X_test)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        np.save(verification_file, dists)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_dists = np.load(verification_file)

        try:
            if np.allclose(expected_dists, dists, atol=1e-6, rtol=1e-4):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = np.sum(np.abs(dists - expected_dists))
                ret[
                    "message"
                ] = f"\tFAILED! \n\tDifference of distance matrices: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret
