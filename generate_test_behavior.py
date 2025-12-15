import random
from datetime import datetime, timedelta

from myutils.query import init_db, querys


def generate_test_behavior():
    init_db()
    users = [row[0] for row in querys("select email from user", [], "select")]
    movie_ids = [row[0] for row in querys("select id from movies", [], "select")]

    if not users or not movie_ids:
        print("没有可用的用户或电影数据，无法生成模拟行为。")
        return

    querys("delete from user_behavior", [])

    max_behavior_num = min(8, len(movie_ids))
    min_behavior_num = min(4, max_behavior_num)

    for user in users:
        behavior_num = random.randint(min_behavior_num, max_behavior_num)
        random_movies = random.sample(movie_ids, behavior_num)
        for movie_id in random_movies:
            score = round(random.uniform(5, 10), 1)
            create_time = datetime.now() - timedelta(days=random.randint(1, 30))
            sql = """
                insert into user_behavior(user_email, movie_id, behavior_type, score, create_time)
                values(%s, %s, 1, %s, %s)
            """
            querys(sql, [user, movie_id, score, create_time.strftime("%Y-%m-%d %H:%M:%S")])

    print("模拟行为数据生成完成")


if __name__ == "__main__":
    generate_test_behavior()
