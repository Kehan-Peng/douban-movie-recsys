from myutils.crawler.jobs import build_behavior_dataset


if __name__ == "__main__":
    result = build_behavior_dataset(user_count=60, min_behaviors=8, max_behaviors=16)
    print(result)
