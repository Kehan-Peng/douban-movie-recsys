from myutils.crawler.jobs import crawl_movie_comments


if __name__ == "__main__":
    result = crawl_movie_comments(pages_per_movie=3)
    print(result)
