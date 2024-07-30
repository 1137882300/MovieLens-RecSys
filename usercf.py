# -*- coding: utf-8 -*-
"""
Created on 2015-06-22

@author: Lock victor
"""
import sys
import random
import math
import os
from operator import itemgetter

from collections import defaultdict

# random.seed(0) 是一个常用于确保随机数生成器行为可预测的方法，特别是在需要可重复结果的科学研究和数据分析中。
random.seed(0)


class UserBasedCF(object):
    """ TopN recommendation - User Based Collaborative Filtering
    self.trainset = {
    'user1': {'movie1': 5, 'movie2': 3},
    'user2': {'movie1': 4, 'movie3': 5}
    }
    """

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 20
        self.n_rec_movie = 10

        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        # sys.stderr 是一个文件类对象，它代表标准错误流（stderr），用来输出错误信息或警告消息。通常，标准错误流会输出到控制台或终端。
        print('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        # % self.n_sim_user 这部分将 self.n_sim_user 的值转换为字符串，并插入到 %d 的位置。
        # % 是旧版格式化，新版Python 3.6 开始 用：f-string
        print('recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)

    # 静态方法
    @staticmethod
    def loadfile(filename):
        """ load a file, return a generator. """
        fp = open(filename, 'r')
        # 按行读取
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        """ load rating data and split it to training set and test set """
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating, _ = line.split('::')
            # split the data by pivot
            # pivot=0.7 意味着70%的数据将被分配到训练集，30%的数据将被分配到测试集。
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                # user对象里的movie的值，赋值为rating
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)

    def calc_user_sim(self):
        """ calculate user similarity matrix 计算用户相似度矩阵"""
        # build inverse table for item-users 为项目-用户构建反向表
        # key=movieID, value=list of userIDs who have seen this movie 看过这部电影的用户 ID 列表
        print('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()
        """movie2users 的结构 ：
            {
                'movie1': {'user1', 'user2'},
                'movie2': {'user1', 'user2'}
            }
        """

        # .items() 是字典对象的一个方法，用于返回一个包含所有（键，值）对的视图对象。这个视图对象可以用来迭代字典中的所有键值对。
        # 相当于在遍历 map
        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time 同时计算项目的受欢迎程度
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users 计算用户之间共同评价的项目
        usersim_mat = self.user_sim_mat
        print('building user co-rated movies matrix...', file=sys.stderr)

        """ usersim_mat 格式化： 相似度计数
        {
            'user1': {'user2': 1, 'user3': 2},
            'user2': {'user1': 4},
            'user3': {'user1': 6}
        }
        """

        # 构建用户相似度矩阵
        for movie, users in movie2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        # 如果当前遍历的用户 u 和 v 是同一个用户，则跳过计算，因为用户与自己的相似度没有意义。
                        continue
                    # 如果 u 和 v 是不同的用户，那么在 usersim_mat 中为这两个用户间的相似度加一。
                    # 这里假设如果两个用户都喜欢同一部电影，那么他们之间就有一定的相似度。
                    usersim_mat[u][v] += 1
        print('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            # count 是他们之间的相似度计数
            for v, count in related_users.items():
                # 计算用户 u 和用户 v 之间的相似度因子。这里使用了皮尔逊相关系数的公式，它是一种常用的相似度度量方法。
                # 将计算出的相似度因子赋值给 usersim_mat[u][v]，更新用户 u 和用户 v 之间的相似度。
                usersim_mat[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))

                # 每次计算一个相似度因子后，simfactor_count 增加 1。
                # 如果 simfactor_count 达到 PRINT_STEP 的倍数，打印一条进度信息。这有助于跟踪计算的进度，尤其是在处理大量数据时。
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating user similarity factor(%d)' % simfactor_count, file=sys.stderr)

        print('calculate user similarity matrix(similarity factor) succ', file=sys.stderr)
        print('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)

    def recommend(self, user):
        """ Find K similar users and recommend N movies. """
        K = self.n_sim_user
        N = self.n_rec_movie
        # 用来存储推荐电影及其对应的排名分数。
        rank = dict()
        """ rank 的结构
        {
            '电影3': 0.85,  # 用户B对电影3的推荐分数贡献
            '电影4': 0.90,  # 用户B对电影4的推荐分数贡献 + 用户C对电影4的推荐分数贡献
            '电影5': 0.60   # 用户C对电影5的推荐分数贡献
        }
        """
        # 从训练数据集 self.trainset 中获取用户 user 已经观看过的电影列表。
        watched_movies = self.trainset[user]

        # 这里的 key=itemgetter(1) 表示在排序操作中，使用元组的第二个元素（索引为 1）作为排序的键值
        # self.user_sim_mat[user] 是一个字典，包含了与用户 user 相似的其他用户及其相似度因子。
        # sorted(..., key=itemgetter(1), reverse=True) 对这些用户按照相似度因子降序排序，并选择前 K 个最相似的用户。
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            # 遍历每个相似用户 similar_user 的电影列表 self.trainset[similar_user]。
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    # 如果电影 movie 已经在用户 user 观看过的电影列表 watched_movies 中，则跳过。
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        """ print evaluation result: precision, recall, coverage and popularity
        打印评估结果：精度、召回率、覆盖率和受欢迎程度 """
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        # rec_count：表示推荐系统中推荐的电影总数。
        rec_count = 0
        # test_count：表示测试集中用户喜欢的电影总数。
        test_count = 0
        # all_rec_movies：一个集合，存储所有推荐过的电影，用于计算覆盖率。
        all_rec_movies = set()
        # 记录所有推荐电影的对数流行度分数之和，用于计算受欢迎程度。
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            # 获取推荐给当前用户的电影列表。
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    # 如果推荐的电影在测试集中，hit 加1。
                    hit += 1
                # 将推荐的电影添加到 all_rec_movies 集合中。
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        # 精度（Precision）：推荐列表中用户喜欢的电影数量除以推荐的电影总数。
        precision = hit / (1.0 * rec_count)
        # 召回率（Recall）：推荐列表中用户喜欢的电影数量除以测试集中用户喜欢的电影总数。
        recall = hit / (1.0 * test_count)
        # 覆盖率（Coverage）：推荐过的不同电影数量除以总电影数量。
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        # 受欢迎程度（Popularity）：推荐电影的对数流行度分数之和除以推荐的电影总数。
        popularity = popular_sum / (1.0 * rec_count)

        print('precision=%.4f\t recall=%.4f\t coverage=%.4f\t popularity=%.4f' %
              (precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
    # ml-1m/ratings.dat
    rating_file = os.path.join('ml-1m', 'ratings.dat')
    usercf = UserBasedCF()
    usercf.generate_dataset(rating_file)
    usercf.calc_user_sim()
    usercf.evaluate()
