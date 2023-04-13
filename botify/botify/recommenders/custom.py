from collections import defaultdict
from .random import Random
from .recommender import Recommender
from .indexed import Indexed
from .toppop import TopPop
import random


class Custom_rec(Recommender):
    """Recommend tracks closest to the previous one.
    Fall back to the random recommender if no recommendations found for the track."""

    def __init__(self, tracks_redis, recommendations_redis, catalog):
        self.tracks_redis = tracks_redis
        self.random = Random(tracks_redis)
        self.fallback = Indexed(tracks_redis, recommendations_redis, catalog)
        self.toppop = TopPop(tracks_redis.connection, catalog.top_tracks[:100])
        self.catalog = catalog
        self.ranked = defaultdict(lambda: defaultdict(int))
        self.used = defaultdict(list)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        self.used[user].append(prev_track)
        if prev_track_time > 0.5:
            self.ranked[user][prev_track] += 1

        if not len(self.ranked[user]):
            if random.random() > 0.95:
                return self.toppop.recommend_next(user, prev_track, prev_track_time)
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        prev_track = random.choice(list(self.ranked[user]))
        previous_track = self.tracks_redis.get(prev_track)
        self.ranked[user][prev_track] += 1

        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track)
        recommendations = previous_track.recommendations

        if recommendations is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = list(recommendations)
        random.shuffle(shuffled)

        return shuffled[0]
