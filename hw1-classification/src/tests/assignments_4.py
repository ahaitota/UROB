import torch
import torch.nn as nn
import traceback

from .utils import load_module

def test_assignment_4_1(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    mlp_classifier_module = load_module(src_dir, "mlp_classifier")
    MLPClassifier = mlp_classifier_module.MLPClassifier

    ret = {"points": 0, "message": "", "max_points": 0.5}

    num_features = 5
    num_classes = 10
    hidden_dim_1 = 20
    hidden_dim_2 = 30
    num_samples = 100

    torch.random.manual_seed(seed)

    params = dict(
        W1=nn.Parameter(torch.randn(num_features, hidden_dim_1, dtype=torch.float)),
        b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
        W2=nn.Parameter(torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)),
        b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
        W3=nn.Parameter(torch.randn(hidden_dim_2, num_classes, dtype=torch.float)),
        b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
    )

    X = torch.randn(num_samples, num_features)

    try:
        model = MLPClassifier(
            num_features=num_features,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            num_classes=num_classes,
        )
        model.params = params
        logits = model.forward(X)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        torch.save(logits, verification_file)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_logits = torch.load(verification_file, weights_only=True)

        try:
            if torch.allclose(logits, expected_logits, atol=1e-6, rtol=1e-4):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = torch.sum(torch.abs(logits - expected_logits))
                ret["message"] = f"\tFAILED! \n\tDifference of logits: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret


def test_assignment_4_2(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    mlp_classifier_module = load_module(src_dir, "mlp_classifier")
    MLPClassifier = mlp_classifier_module.MLPClassifier

    ret = {"points": 0, "message": "", "max_points": 0.5}

    num_features = 5
    num_classes = 10
    hidden_dim_1 = 20
    hidden_dim_2 = 30
    num_samples = 100

    torch.random.manual_seed(seed)

    params = dict(
        W1=nn.Parameter(torch.randn(num_features, hidden_dim_1, dtype=torch.float)),
        b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
        W2=nn.Parameter(torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)),
        b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
        W3=nn.Parameter(torch.randn(hidden_dim_2, num_classes, dtype=torch.float)),
        b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
    )

    X = torch.randn(num_samples, num_features)

    try:
        model = MLPClassifier(
            num_features=num_features,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            num_classes=num_classes,
        )
        model.params = params
        y_pred = model.predict(X)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        torch.save(y_pred, verification_file)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_logits = torch.load(verification_file, weights_only=True)

        try:
            if torch.allclose(y_pred, expected_logits, atol=1e-6, rtol=1e-4):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = torch.sum(torch.abs(y_pred - expected_logits))
                ret[
                    "message"
                ] = f"\tFAILED! \n\tDifference of predicted labels: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret


def test_assignment_4_3(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:
    mlp_classifier_module = load_module(src_dir, "mlp_classifier")
    MLPClassifier = mlp_classifier_module.MLPClassifier
    ret = {"points": 0, "message": "", "max_points": 0.5}

    num_features = 5
    num_classes = 10
    hidden_dim_1 = 20
    hidden_dim_2 = 30
    num_samples = 100

    torch.random.manual_seed(seed)

    params = dict(
        W1=nn.Parameter(torch.randn(num_features, hidden_dim_1, dtype=torch.float)),
        b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
        W2=nn.Parameter(torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)),
        b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
        W3=nn.Parameter(torch.randn(hidden_dim_2, num_classes, dtype=torch.float)),
        b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
    )

    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))

    try:
        model = MLPClassifier(
            num_features=num_features,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            num_classes=num_classes,
        )
        model.params = params
        loss = model.loss(X, y)
    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        torch.save(loss, verification_file)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_loss = torch.load(verification_file, weights_only=True)

        try:
            if torch.allclose(loss, expected_loss, atol=1e-6, rtol=1e-4):
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                difference = torch.sum(torch.abs(loss - expected_loss))
                ret["message"] = f"\tFAILED! \n\tLoss difference: {difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret


def test_assignment_4_4(
    src_dir: str, verification_file: str, seed: int = 69, generate: bool = False
) -> dict:

    mlp_classifier_module = load_module(src_dir, "mlp_classifier")
    MLPClassifier = mlp_classifier_module.MLPClassifier

    ret = {"points": 0, "message": "", "max_points": 0.5}

    num_features = 5
    num_classes = 10
    hidden_dim_1 = 20
    hidden_dim_2 = 30
    num_samples = 100

    torch.random.manual_seed(seed)

    params = dict(
        W1=nn.Parameter(torch.randn(num_features, hidden_dim_1, dtype=torch.float)),
        b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
        W2=nn.Parameter(torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)),
        b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
        W3=nn.Parameter(torch.randn(hidden_dim_2, num_classes, dtype=torch.float)),
        b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
    )

    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))

    try:
        model = MLPClassifier(
            num_features=num_features,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            num_classes=num_classes,
        )
        model.params = params
        model._zero_gradients()
        loss = model.loss(X, y)
        loss.backward(retain_graph=True)
        model._update_weights()

    except Exception as e:
        ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"
        return ret

    if generate:
        torch.save(model.params, verification_file)
        print(f"Successfully generated '{verification_file}'!")
    else:
        expected_params = torch.load(verification_file, weights_only=True)

        try:
            pred_values = model.params.values()
            expected_values = expected_params.values()
            differences = [
                torch.sum(torch.abs(m - exp_m))
                for m, exp_m in zip(pred_values, expected_values)
            ]
            differences = torch.stack(differences)
            param_difference = torch.sum(differences)

            if param_difference < 1e-5:
                ret["message"] = f"PASSED!"
                ret["points"] = ret["max_points"]
            else:
                ret[
                    "message"
                ] = f"\tFAILED! \n\tParameter difference: {param_difference}"
        except Exception as e:
            ret["message"] = f"\tFAILED! \n\t{e.__class__.__name__}:{e}\n{traceback.format_exc()}"

    return ret
