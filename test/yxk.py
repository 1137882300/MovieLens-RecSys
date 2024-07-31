import os
from surprise import Dataset, Reader, KNNWithMeans, KNNWithZScore
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# 定义文件格式
reader = Reader(line_format='user item rating', sep='::')

# 从文件加载数据
rating_file = os.path.join('yxk', 'ratings_prod_3w.dat')
data = Dataset.load_from_file(rating_file, reader=reader)

# 拆分数据集
trainset, testset = train_test_split(data, test_size=0.25)

# 'name': 'cosine'
# 'name': 'pearson',
# 'name': 'jaccard',
# 'name': 'manhattan'
# 'name': 'euclidean'
algo = KNNWithZScore(sim_options={'name': 'pearson', 'user_based': False})

# 训练模型
algo.fit(trainset)

# 进行预测
predictions = algo.test(testset)

# 计算准确性，较低的 RMSE 表示预测更准确。
rmse = accuracy.rmse(predictions)


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


precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4.0)

# 计算平均精确率和召回率
average_precision = sum(prec for prec in precisions.values()) / len(precisions)
average_recall = sum(rec for rec in recalls.values()) / len(recalls)

# 最好是 ： RMSE 低、Precision 高、Recall 高
print(f"RMSE: {rmse}")
# 平均精确率 (Precision): 0.816 表示推荐的物品中，有 81.6% 是用户实际喜欢的。
print(f"Average Precision: {average_precision}")
# 平均召回率 (Recall): 0.472 表示用户实际喜欢的物品中，有 47.2% 被成功推荐。
print(f"Average Recall: {average_recall}")


def recommend_for_user(user_id, top_n=10):
    # 获取用户未评分的所有项目
    trainset = algo.trainset
    # 获取所有项目的内部ID
    all_items = set(trainset.all_items())

    # 获取指定用户已经评分的所有项目（产品）的内部ID，并将其存储在一个集合（set）中
    user_rated_items = set(j for (j, _) in trainset.ur[trainset.to_inner_uid(user_id)])
    # 计算用户未评分的项目
    items_to_predict = all_items - user_rated_items

    # 预测用户 user_id 对项目 item 的评分。
    predictions = [algo.predict(user_id, trainset.to_raw_iid(item)) for item in items_to_predict]

    # 过滤出符合特征的产品，例如筛选出“亲子”类产品
    filtered_predictions = [pred for pred in predictions if is_family_friendly(pred.iid)]

    recommendations = sorted(filtered_predictions, key=lambda x: x.est, reverse=True)[:top_n]

    return recommendations


def is_family_friendly(item_id):
    return item_id


user_id = '9227'
recommendations = recommend_for_user(user_id)
print(f"Recommendations for user {user_id}:")
for rec in recommendations:
    print(f"Item {rec.iid} with predicted rating {rec.est}")
