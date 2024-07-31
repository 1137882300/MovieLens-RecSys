import os
from collections import defaultdict

from surprise import Dataset, Reader, KNNWithZScore, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 加载数据
reader = Reader(line_format='user item rating', sep='::')
rating_file = os.path.join('yxk', 'ratings_prod_3w.dat')
data = Dataset.load_from_file(rating_file, reader=reader)
trainset, testset = train_test_split(data, test_size=0.25)

# 基于物品的KNN
algo_knn = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': False})
algo_knn.fit(trainset)

# 矩阵分解SVD
algo_svd = SVD()
algo_svd.fit(trainset)

# 进行预测
predictions_knn = algo_knn.test(testset)
predictions_svd = algo_svd.test(testset)

# 计算准确性
rmse_knn = accuracy.rmse(predictions_knn)
rmse_svd = accuracy.rmse(predictions_svd)

print(f"RMSE for KNN: {rmse_knn}")
print(f"RMSE for SVD: {rmse_svd}")


# 计算精确率和召回率
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """Return precision and recall at k metrics for each user."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


precisions_knn, recalls_knn = precision_recall_at_k(predictions_knn, k=10, threshold=4.0)
precisions_svd, recalls_svd = precision_recall_at_k(predictions_svd, k=10, threshold=4.0)

# 计算平均精确率和召回率
average_precision_knn = sum(prec for prec in precisions_knn.values()) / len(precisions_knn)
average_recall_knn = sum(rec for rec in recalls_knn.values()) / len(recalls_knn)
average_precision_svd = sum(prec for prec in precisions_svd.values()) / len(precisions_svd)
average_recall_svd = sum(rec for rec in recalls_svd.values()) / len(recalls_svd)

print(f"Average Precision for KNN: {average_precision_knn}")
print(f"Average Recall for KNN: {average_recall_knn}")
print(f"Average Precision for SVD: {average_precision_svd}")
print(f"Average Recall for SVD: {average_recall_svd}")


# 混合推荐函数
def hybrid_recommend(user_id, top_n=5):
    predictions_knn = [algo_knn.predict(user_id, iid) for iid in trainset.all_items()]
    predictions_svd = [algo_svd.predict(user_id, iid) for iid in trainset.all_items()]

    combined_predictions = {}
    for pred_knn in predictions_knn:
        pred_svd = next((p for p in predictions_svd if p.iid == pred_knn.iid), None)
        if pred_svd:
            combined_predictions[pred_knn.iid] = (pred_knn.est + pred_svd.est) / 2

    recommendations = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations


def print_combined_predictions(combined_predictions):
    for iid, est in combined_predictions.items():
        raw_iid = trainset.to_raw_iid(iid)  # 转换为原始ID
        print(f"Item {raw_iid} with combined predicted rating {est}")


user_id = '1087475'
print(f"Hybrid Recommendations for user {user_id}:")
recommendations = hybrid_recommend(user_id)
print_combined_predictions(dict(recommendations))
