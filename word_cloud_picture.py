from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def generate_word_cloud(data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def main():
    # Load data from CSV or any other source
    # For example, let's assume we are using user comments from a CSV file
    comments_data = pd.read_csv('datas.csv')  # Adjust the path as necessary
    comments_text = ' '.join(comments_data['comments'])  # Assuming 'comments' is a column in the CSV

    generate_word_cloud(comments_text, 'Word Cloud of Movie Comments')

if __name__ == "__main__":
    main()