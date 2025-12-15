from myutils.crawler.jobs import crawl_top_movies


if __name__ == "__main__":
    result = crawl_top_movies(pages=8)
    print(result)
