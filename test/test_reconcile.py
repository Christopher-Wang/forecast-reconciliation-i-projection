import torch
from tqdm import tqdm

from src.datasets_manager import get_simple_dataset, HierarchicalDatasetNames
from src.reconciliation.full_proba_reconciliation import generate_sum_bucket_bounds, FullProbaPairwiseReconciler, LossType


def get_sum_dist_inefficient(
        logprob_1, logprob_2, bucket_means_1, bucket_means_2, target_bucket_bounds: torch.Tensor
) -> torch.Tensor:
    """ Same as `get_sum_dist` but very inefficiently, to check the efficient one is correct """
    output = torch.zeros(logprob_1.shape).to(logprob_1.device)
    for i in tqdm(range(logprob_1.shape[1])):
        for j in range(logprob_2.shape[1]):
            logproba_val = logprob_1[:, i] + logprob_2[:, j]
            sum_val = bucket_means_1[i] + bucket_means_2[j]
            target_bucket_ind = ((target_bucket_bounds < sum_val).sum() - 1).clip(min=0, max=logprob_1.shape[-1] - 1)
            output[:, target_bucket_ind] += logproba_val.exp()
    return output.log()


def test_sum_two_dist():
    logprob_1 = torch.randn(9, 8)
    logprob_2 = torch.randn(9, 8)
    bucket_means_1 = torch.randn(9)
    bucket_means_2 = torch.randn(9)

    target_bucket_bounds = generate_sum_bucket_bounds(
        mean_1=0, mean_2=1, std_1=1, std_2=.5,
        standard_boundaries=torch.tensor([-3, -2, -1, -.5, .5, 1, 2, 3]),
    )

    inefficient_sum_dist = get_sum_dist_inefficient(
        logprob_1=logprob_1, logprob_2=logprob_2, bucket_means_1=bucket_means_1, bucket_means_2=bucket_means_2,
        target_bucket_bounds=target_bucket_bounds
    )

    #
    # sum_dist = get_sum_of_two_dists(
    #     logprob_1=prob_1, logprob_2=prob_2, bucket_means_1=bucket_means_1, bucket_means_2=bucket_means_2,
    #     target_bucket_bounds=target_bucket_bounds
    # )

    # assert torch.allclose(inefficient_sum_dist.log_softmax(1), sum_dist.log_softmax(1))


def test_forecast_reconciliation() -> None:
    preds, y_test = get_simple_dataset(agg_level=2, data_name=HierarchicalDatasetNames.TOURISM_SMALL)
    ground_truth = {}
    for name in y_test["unique_id"].unique():
        ground_truth[name] = y_test[y_test["unique_id"] == name].values[:, -1]

    forecast_reconciliator = FullProbaPairwiseReconciler(all_preds=preds, top_level_name="total")
    with torch.autograd.set_detect_anomaly(False):
        forecast_reconciliator.reconcile_sgd(
            lr=.01, num_steps=1000, device="cuda:1", ground_truth=ground_truth, loss_type=LossType.EMD
        )


if __name__ == "__main__":
    test_forecast_reconciliation()
