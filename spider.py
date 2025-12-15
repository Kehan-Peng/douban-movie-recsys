from myutils.crawler.jobs import crawl_top_movies


def main():
    result = crawl_top_movies(pages=8)
    print(f"电影爬取完成，共写入 {result['movie_count']} 条记录 -> {result['output_csv']}")


if __name__ == "__main__":
    main()
