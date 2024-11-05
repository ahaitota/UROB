import traceback

import numpy as np

from .utils import load_module

def test_assignment_2_1(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    knn_module = load_module(src_dir, "knn_classifier")
    tuning_module = load_module(src_dir, "tuning")
    KNNClassifier = knn_module.KNNClassifier
    cross_validate_knn = tuning_module.cross_validate_knn

    ret = {"points": 0, "message": "", "max_points": 1}

    np.random.seed(seed)

    num_samples = 1000
    num_features = 2
    num_classes = 3

    num_folds = 4
    k_choices = np.array([1, 3, 5, 7])

    # Generate random dataset
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(num_classes, size=num_samples)

    try:
        classifier = KNNClassifier(k=3, vectorized=True)
        k_to_metrics = cross_validate_knn(classifier, X, y, k_choices, num_folds)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        np.save(verification_file, k_to_metrics)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_k_to_metrics = np.load(verification_file, allow_pickle=True).item()

        def metric_difference(pred_k_to_metrics, expected_k_to_metrics, metric):
            differences = []
            for k in k_choices:
                values = pred_k_to_metrics[metric][k]
                expected_values = expected_k_to_metrics[metric][k]
                differences.append(np.sum(np.abs(values - expected_values)))
            return np.sum(differences)

        try:
            # Check every metric
            acc_diff = metric_difference(
                k_to_metrics, expected_k_to_metrics, "accuracy"
            )
            prec_diff = metric_difference(
                k_to_metrics, expected_k_to_metrics, "precision"
            )
            recall_diff = metric_difference(
                k_to_metrics, expected_k_to_metrics, "recall"
            )
            f1_diff = metric_difference(k_to_metrics, expected_k_to_metrics, "f1")

            # Check if the differences are small enough
            if (
                acc_diff < 1e-4
                and prec_diff < 1e-4
                and recall_diff < 1e-4
                and f1_diff < 1e-4
            ):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                ret["message"] = (
                    f"\tFAILED! \n\tThe difference between the metrics is "
                    f"acc: {acc_diff}, prec: {prec_diff}, recall: {recall_diff}, f1: {f1_diff}."
                )
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret
