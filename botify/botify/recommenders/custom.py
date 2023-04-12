from collections import defaultdict
from .random import Random
from .recommender import Recommender
from .indexed import Indexed
import random


class Custom_rec(Recommender):
    """Recommend tracks closest to the previous one.
    Fall back to the random recommender if no recommendations found for the track."""

    def __init__(self, tracks_redis, recommendations_redis, catalog, ranked, used):
        self.tracks_redis = tracks_redis
        self.random = Random(tracks_redis)
        self.fallback = Indexed(tracks_redis, recommendations_redis, catalog)
        self.catalog = catalog
        self.ranked = ranked
        self.used = used

    # TODO Seminar 5 step 1: Implement contextual recommender based on NN predictions
    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user not in self.used:
            self.used[user] = []
        self.used[user].append(prev_track)

        if user not in self.ranked:
            self.ranked[user] = defaultdict(int)

        if prev_track_time > 0.6:
            self.ranked[user][prev_track] += 1
        elif len(self.ranked[user]) > 0:
            prev_track, pos = random.choice(list(self.ranked[user].items()))
            self.ranked[user][prev_track] = pos + 1
        else:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations = previous_track.recommendations

        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = list(recommendations)
        random.shuffle(shuffled)

        return shuffled[0]
