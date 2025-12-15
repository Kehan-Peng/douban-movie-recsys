from myutils.crawler.jobs import crawl_movie_comments


def main():
    result = crawl_movie_comments(pages_per_movie=3)
    print(f"评论爬取完成，共写入 {result['comment_count']} 条记录 -> {result['output_csv']}")


if __name__ == "__main__":
    main()
