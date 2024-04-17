from os import getcwd
from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
from collections import defaultdict, Counter
from math import log, exp
from wordcloud import WordCloud
import re
import random
import glob
import time
import mmap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import heapq


"""
    Used Data Is SpamAssassin public corpus from apache
    Bernoulli Naive Bayes
    We use bernoulli when the input data is represented as a binary feature vector
"""

"""
    for measuring calculation time
    start a timer
"""


# Start of the region for GraphHelpers class
class GraphHelpers:

    @staticmethod
    def graph_top_ten_data(given_data, spam_or_not=1, save_mode=0):
        """
        it shows top ten data of given data spam or ham
        if save mode = 0 it shows in window
        else it saves to directory
        file format is [.png]
        :param spam_or_not:
        :param given_data:
        :param save_mode:
        :return:
        """
        x = heapq.nlargest(10, given_data, key=given_data.get)
        y = []
        for i in x:
            y.append(given_data[i])
        fig, ax = plt.subplots()
        ax.bar(x, y, align='center', width=0.2, color='m')

        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_xlabel("Graph For Words")
        ax.set_ylabel("Number of times used")
        ax.grid('on')
        if save_mode:
            if spam_or_not:
                plt.savefig("graph_top_ten_spam_data.png")
            else:
                plt.savefig("graph_top_ten_ham_data.png")
        else:
            plt.show()
        plt.clf()
        plt.close("all")

    @staticmethod
    def spam_ham_count_graph(spam_messages, ham_messages, save_mode=0):
        """
        if save mode is 0 it opens up a window to show histogram graph of counts of spam and ham
        else it saves the spam ham histogram  graph to directory
        :param spam_messages:
        :param ham_messages:
        :param save_mode:
        :return:
        """
        x = ["spam", "ham"]
        y = [spam_messages, ham_messages]
        plt.bar(x, y)
        for index, value in enumerate(y):
            plt.text(index, value, str(value))
        if save_mode:
            plt.savefig("spam_ham_count_graph.png")
        else:
            plt.show()
        plt.clf()
        plt.close("all")

    @staticmethod
    def show_heat_map(matrix, save_mode=0):
        mat = create_2d_confusion(matrix)
        confusion_matrix_frame = pd.DataFrame(mat, range(2), range(2))
        sb.set(font_scale=1.4)
        sb.heatmap(confusion_matrix_frame, annot=True, annot_kws={"size": 36}, fmt=".0f",
                   xticklabels=["True", "False"],
                   yticklabels=["True", "False"]
                   )
        if save_mode:
            plt.savefig("accuracy_map.png")
        else:
            plt.show()
        plt.clf()
        plt.close("all")

    @staticmethod
    def show_word_cloud(token_counts, spam_or_not=1, save_mode=0):
        """
        if save mode is 0 it opens up a window to show word cloud
        else it saves the word cloud graph to directory
        :param spam_or_not:
        :param token_counts:
        :param save_mode:
        :return:
        """
        cloud = WordCloud(width=700, height=400, max_font_size=100).generate_from_frequencies(token_counts)
        plt.figure(figsize=(25, 15), facecolor="k")
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        if save_mode:
            if spam_or_not:
                plt.savefig("spam_word_cloud.png")
            else:
                plt.savefig("ham_word_cloud.png")
        else:
            plt.show()
        plt.clf()
        plt.close("all")

    @staticmethod
    def draw_pie_chart(token_counts):
        """
        it creates and saves a pie chart to directory
        :param token_counts:
        :return:
        """
        labels = []
        sizes = []

        for x, y in token_counts.items():
            labels.append(x)
            sizes.append(y)

        # Plot
        plt.pie(sizes, labels=labels)

        plt.axis('equal')
        plt.savefig("pie_chart.png")
        plt.clf()
        plt.close("all")

    @staticmethod
    def draw_scatter_plot(token_counts, spam_or_not=1, save_mode=0):
        labels = []
        sizes = []
        fig, ax = plt.subplots()
        for x, y in token_counts.items():
            labels.append(x)
            sizes.append(y)
        ax.scatter(labels, sizes, s=3)
        ax.axes.xaxis.set_ticklabels([])

        if spam_or_not:

            if save_mode:
                fig.savefig("spam_scatter.png")
            else:
                fig.show()
        else:
            if save_mode:
                fig.savefig("ham_scatter.png")
            else:
                plt.show()
        plt.clf()
        plt.close(fig)
        
# end of region for GraphHelpers class


class Message(NamedTuple):
    """
        text: str -> Subject of each mail
        is_spam: bool -> if a mail classified as spam or not
    """
    text: str
    is_spam: bool


# Start of NaiveBayesClassifier region
class NaiveBayesClassifier:

    def __init__(self, laplace_smooth_var: float = 0.5) -> None:
        self._laplace_smooth_var = laplace_smooth_var

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        start_time = time.time()
        for message in messages:

            """
                If a sentence classified as spam we will increase spam_messages otherwise ham_messages
            """
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            """
                Tokenize message's words if a sentence is spam then every word will be treated as spam
                keep tracking a sentences spam and ham count
                (ex: 'you':50)
            """
            for token in tokenize(message.text):
                self.tokens.add(token)

                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

        print(f"TRAIN FINISH TIME = {round(time.time() - start_time,2)} seconds ")

    def probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        """ P(X | S) = (k + number of spams containing word i) / 2k + number of spams containing.
            This formula is used for preventing 0 probability words.
            Decide if a token more likely spam or ham.
        """
        p_token_spam = (spam + self._laplace_smooth_var) / (self.spam_messages + (2 * self._laplace_smooth_var))
        p_token_ham = (ham + self._laplace_smooth_var) / (self.ham_messages + (2 * self._laplace_smooth_var))

        return p_token_spam, p_token_ham

    # summing up log probabilities
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)

        log_prob_if_spam = 0.0
        log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self.probabilities(token)

            # print(f"spam {prob_if_spam} ham {prob_if_ham}")
            """
                if token is in the message
                else add not seeing it which is 1 - seeing it
                add the log probability
                if a's probability is 0.64 then 'a probability is 0.36
                a + 'a = 1.00
                we are adding logs instead of multiplying
                because we are preventing overflowing 
                log(xy) = log(x) + log(y)
            """
            if token in text_tokens:
                log_prob_if_spam += log(prob_if_spam)
                log_prob_if_ham += log(prob_if_ham)

            else:
                log_prob_if_spam += log(1.0 - prob_if_spam)
                log_prob_if_ham += log(1.0 - prob_if_ham)

        prob_if_spam = exp(log_prob_if_spam)
        prob_if_ham = exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham)

# end of region NaiveBayesClassifier


def tokenize(text: str) -> Set[str]:
    """
    More advanced technique is that using nltk library to tokenize with porter stemmer
    :param text:
    :return set():
    """
    text = text.lower()
    # find all the words datas
    words_list = re.findall(r'[A-Za-z0-9\']+', text)
    return set(words_list)


def read_file_data(path: str) -> List[Message]:
    data: List[Message] = []
    for filename in glob.glob(path):

        """
            basically iterates every file in our folder
            if filename has no ham in it it is a spam data
            we use that tokens for spam learning
            ham: non-spam or good mail
        """
        is_spam = "ham" not in filename

        with open(filename, errors='ignore') as email_file:
            file_data = mmap.mmap(email_file.fileno(), 0, access=mmap.ACCESS_READ)
            for line in iter(file_data.readline, b""):
                """
                    in windows utf-8 is not working 
                """
                line = line.decode("latin1")
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(str(subject), is_spam))
                    # we are only caring about subject, so we are breaking if we found subject
                    break
    return data


def find_accuracy(conf_matrix: Counter) -> float:
    """
        A matrix for confusion matrix
        2x2
        true false
    true
    false

    diagonals are the correct predictions

    :param conf_matrix:
    :return:
    """
    predicted_right = 0
    predicted_wrong = 0
    for i in conf_matrix:
        if i[0] == i[1]:
            predicted_right += conf_matrix[i]
        else:
            predicted_wrong += conf_matrix[i]

    return predicted_right / (predicted_wrong + predicted_right)


def create_2d_confusion(conf_matrix: Counter):
    lst = [[0, 0], [0, 0]]
    for i in conf_matrix:
        if i[0] is True and i[1] is True:
            lst[0][0] += conf_matrix[i]
        elif i[0] is True and i[1] is False:
            lst[0][1] += conf_matrix[i]
        elif i[0] is False and i[1] is True:
            lst[1][0] += conf_matrix[i]
        else:
            lst[1][1] += conf_matrix[i]
    return lst


def train_test_split(messages: List[Message], split_ratio=0.8) -> Tuple[List[Message], List[Message]]:
    random.shuffle(messages)
    num_train = int(round(len(messages) * split_ratio, 0))
    return messages[:num_train], messages[num_train:]


def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model.probabilities(token)
    return prob_if_spam / (prob_if_spam + prob_if_ham)


def write_data_to_file(token_ham_counts, token_spam_counts):
    data_frame_spam = pd.DataFrame.from_dict(token_ham_counts, orient="index")
    data_frame_spam.to_csv("ham_count.csv", sep=",", encoding="latin1")

    data_frame_ham = pd.DataFrame.from_dict(token_spam_counts, orient="index")
    data_frame_ham.to_csv("spam_count.csv", sep=",", encoding="latin1")


def write_prob_to_file(given_data: Dict[str, int], model: NaiveBayesClassifier, spam_or_not=0):
    prob_dict: Dict[str, float] = defaultdict(float)
    val = ["spam", "ham"]
    for x, y in given_data.items():
        prob_dict[x] = model.probabilities(x)[spam_or_not]
    data_frame_spam = pd.DataFrame.from_dict(prob_dict, orient="index")
    data_frame_spam.to_csv(val[spam_or_not]+"_probability_count.csv", sep=",", encoding="latin1")


"""--------------------------------------------------------------------------"""
# file_path = "C:\\Users\\emrec\\Desktop\\ders\\intelligent system\\proje\\apache_spam\\*\\*"
file_path = getcwd() + "\\apache_spam\\*\\*"
data_list: List[Message] = read_file_data(file_path)

random.seed(432)

"""
    split data train: %75 and test: %25
    all subjects of emails
"""
train_messages, test_messages = train_test_split(data_list, 0.75)
print(f"LENGTH OF TRAIN DATA = {len(train_messages)} LENGTH OF TEST DATA = {len(test_messages)}")
nb = NaiveBayesClassifier()
graph_helper = GraphHelpers()
nb.train(train_messages)

"""Prediction vector for accuracy calculations"""
predictions = [(message, nb.predict(message.text)) for message in test_messages]

"""
    (False,False) not spam and not exceeding 0.5 (true negatives)
    (False,True) not spam but exceeding 0.5 (false positive)
    (True,False) it is spam but not exceeding 0.5 (false negatives)
    (True,True) it is spam and exceeding 0.5 (true positive)
"""
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) for message, spam_probability in predictions)

print(f"ACCURACY OF MODEL = {round(find_accuracy(confusion_matrix),2)}")
words = sorted(nb.tokens, key=lambda t: p_spam_given_token(t, nb))

print(f"DISTINCT TOKEN FROM DATA = {len(nb.tokens)}\nTOTAL OF SPAM SENTENCES = {nb.spam_messages} \nTOTAL OF HAM "
      f"SENTENCES = {nb.ham_messages}")
print("HIGHEST PERCENTAGES OF SPAMS = ", words[-10:])
print("HIGHEST PERCENTAGES OF NON-SPAM = ", words[:10]),


""" GRAPHS FOR OUR DATA """
"""Save mode 1 saves all the graphs in png format , save mode 0 just shows the data in a window"""

graph_helper.graph_top_ten_data(nb.token_spam_counts, spam_or_not=1, save_mode=1)
graph_helper.graph_top_ten_data(nb.token_ham_counts, spam_or_not=0, save_mode=1)
graph_helper.show_heat_map(confusion_matrix, save_mode=1)
graph_helper.show_word_cloud(nb.token_spam_counts, spam_or_not=1, save_mode=1)
graph_helper.show_word_cloud(nb.token_ham_counts, spam_or_not=0, save_mode=1)
graph_helper.spam_ham_count_graph(nb.spam_messages, nb.ham_messages, save_mode=1)
graph_helper.draw_pie_chart(nb.token_spam_counts)
graph_helper.draw_scatter_plot(nb.token_spam_counts, spam_or_not=1, save_mode=1)
graph_helper.draw_scatter_plot(nb.token_ham_counts, spam_or_not=0, save_mode=1)

"""Write ham and spam counts to csv files for reading from R programming"""
write_data_to_file(nb.token_ham_counts, nb.token_spam_counts)
write_prob_to_file(nb.token_spam_counts, nb, spam_or_not=1)
write_prob_to_file(nb.token_spam_counts, nb, spam_or_not=0)


"""
data_frame_spam = pd.read_csv("spam_count.csv")
pd.set_option('display.max_rows', None)
print(data_frame_spam)
"""
