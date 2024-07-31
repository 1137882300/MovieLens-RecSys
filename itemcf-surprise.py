import os
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy


# 解释代码
# 数据加载：使用Surprise内置的数据集ml-100k，这是一个包含10万条电影评分记录的标准数据集。
# 数据划分：将数据划分为训练集和测试集。
# 算法选择：使用基于物品的协同过滤算法（KNNBasic），相似度度量采用余弦相似度。
# 训练模型：在训练集上训练模型。
# 预测评分：在测试集上进行预测，并计算RMSE（均方根误差）评估模型性能。
# 推荐电影：为每个用户推荐评分最高的前N部电影。
# 这样，你可以利用Surprise库方便地实现和评估推荐系统，而无需从头编写算法逻辑。如果需要处理自己的数据集，可以使用Reader类加载数据，格式如下：

def get_top_n_recommendations(predictions, n=10):
    """为每个用户推荐前N个电影"""
    # 将预测结果整理成{uid: [(iid, est), ...]}的形式
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # 对每个用户的预测评分进行排序，并返回前N个结果
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


if __name__ == '__main__':
    # 加载内置的MovieLens 100k数据集
    # data = Dataset.load_builtin('ml-100k')

    # 读取自己的数据集
    # 定义文件格式：line_format='user item rating timestamp' 是 Surprise 库中 Reader 类的参数之一，
    # 用于定义数据文件中每一行的格式。这意味着数据文件中的每一行包含四个字段，依次为 user（用户ID）、item（物品ID）、rating（评分）和 timestamp（时间戳）。
    # 这些字段的顺序和名称告诉 Surprise 如何解析数据文件中的每一行。
    reader = Reader(line_format='user item rating timestamp', sep='::')
    # 从文件加载数据
    rating_file = os.path.join('ml-1m', 'ratings00.dat')
    data = Dataset.load_from_file(rating_file, reader=reader)

    # 划分训练集和测试集
    trainset, testset = train_test_split(data, test_size=0.25)

    # >>>>>>计算物品相似度-s
    # 基于物品的协同过滤算法
    sim_options = {
        'name': 'cosine',  # 使用余弦相似度
        'user_based': False  # False表示基于物品的协同过滤
    }
    # KNNWithMeans：这个算法在计算相似度时，会考虑评分的平均值。也就是说，它会减去用户或物品的平均评分，然后再计算相似度。这有助于减少评分的偏差，从而使得推荐更加准确。
    # KNNBasic：这是最基本的KNN算法，直接计算用户或物品之间的相似度，而不考虑评分的平均值。这种方法可能会受到评分偏差的影响。
    algo = KNNBasic(sim_options=sim_options)

    # 训练模型
    algo.fit(trainset)
    # >>>>>>计算物品相似度-e

    # 在测试集上进行预测
    predictions = algo.test(testset)

    # >>>>>>评估模型-s
    # 评估模型的准确性
    accuracy.rmse(predictions)

    # >>>>>>推荐算法-s
    # 获取推荐结果
    top_n = get_top_n_recommendations(predictions, n=10)

    # 打印推荐结果
    for uid, user_ratings in top_n.items():
        print(f"User {uid}:")
        for (iid, est) in user_ratings:
            print(f"  Movie {iid}: Estimated Rating {est}")
