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

random.seed(0)


class ItemBasedCF(object):
    """ TopN recommendation - Item Based Collaborative Filtering """

    """
    self.trainset = {
        'user1': {'movie1': 5, 'movie2': 3},
        'user2': {'movie1': 4, 'movie3': 5}
    }
    """

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommended movie number = %d' % self.n_rec_movie, file=sys.stderr)

    # 数据加载和划分-s
    @staticmethod
    def loadfile(filename):
        """ load a file, return a generator. """
        fp = open(filename, 'r')
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
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        print('test set = %s' % testset_len, file=sys.stderr)

    # 数据加载和划分-e

    # 计算物品相似度-s
    def calc_movie_sim(self):
        """ calculate movie similarity matrix """
        print('counting movies number and popularity...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        print('count movies number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        """ itemsim_mat 的结构
        {
            'm1': {
                'm2': 1,  # 用户A同时观看了m1和m2
                'm3': 2   # 用户B同时观看了m1和m3
            },
            'm2': {
                'm1': 5,  # 用户A同时观看了m1和m2，与'm1': {'m2': 1} 对应
                'm3': 1   # 用户C同时观看了m2和m3
            },
            'm3': {
                'm1': 2,  # 用户B同时观看了m1和m3，与'm1': {'m3': 1} 对应
                'm2': 1   # 用户C同时观看了m2和m3，与'm2': {'m3': 1} 对应
            }
        }
        """
        # 构建一个电影之间的相似度矩阵，其中相似度是通过共同观看这些电影的用户数量来衡量的
        # 这个矩阵可以用于推荐系统中的基于物品的协同过滤算法，
        # 该算法推荐用户可能喜欢的、与他们过去喜欢的电影相似的其他电影。
        for user, movies in self.trainset.items():
            for m1 in movies:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in movies:
                    if m1 == m2:
                        continue
                    # 如果多个用户观看了相同的两部电影，这些电影的相似度计数会增
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' % simfactor_count, file=sys.stderr)

        print('calculate movie similarity matrix(similarity factor) succ', file=sys.stderr)
        print('Total similarity factor number = %d' % simfactor_count, file=sys.stderr)

    # 计算物品相似度-s

    # 推荐算法-s
    def recommend(self, user):
        """ Find K similar movies and recommend N movies. """
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 推荐算法-e

    # 评估模型-s
    def evaluate(self):
        """ print evaluation result: precision, recall, coverage and popularity """
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  variables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # variables for coverage
        all_rec_movies = set()
        # variables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for %d users' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('precision=%.4f\t recall=%.4f\t coverage=%.4f\t popularity=%.4f' %
              (precision, recall, coverage, popularity), file=sys.stderr)
    # 评估模型-e


if __name__ == '__main__':
    rating_file = os.path.join('ml-1m', 'ratings.dat')
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(rating_file)
    itemcf.calc_movie_sim()
    itemcf.evaluate()
