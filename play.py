import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')


def dict_50():
    dict_top_50 = {}
    dict_top_50['hockey'] = ['team', 'year', 'like', 'player', 'get', 'play', 'would', 'one', 'game', 'think', 'go',
                             'season', 'good', 'time', 'make', 'guy', 'point', 'still', 'hockey', 'know', 'nhl', 'fan',
                             'see', 'realli', 'even', 'http', 'say', 'better', 'could', 'last', 'much', 'want', 'com',
                             'got', 'fuck', 'best', 'way', 'sign', 'leagu', 'peopl', 'back', 'watch', 'also', '2', 'look',
                             'top', 'trade', 'contract', 'take', 'well']
    dict_top_50['nba'] = ['like', 'team', 'get', 'year', 'would', 'player', 'think', 'play', 'go', 'one', 'good', 'make',
                          'game', 'better', 'even', 'lebron', 'say', 'time', 'guy', 'realli', 'peopl', 'nba', 'season',
                          '3', 'trade', 'best', 'could', 'want', 'look', 'know', 'take', 'much', 'still', 'see', 'way',
                          'point', 'also', 'fan', 'fuck', 'leagu', 'kyri', 'gt', 'got', '2', 'win', 'well', 'defens',
                          'need', '5', 'great']
    dict_top_50['leagueoflegends'] = ['game', 'like', 'get', 'play', 'team', 'one', 'would', 'think', 'time', 'peopl',
                                      'make', 'good', 'realli', 'even', 'go', 'player', 'use', 'na', 'also', 'still',
                                      'say', 'damag', 'much', 'gt', 'lane', 'well', 'better', 'know', 'see', 'champion',
                                      'want', '2', 'need', 'eu', 'got', 'way', '1', 'thing', 'actual', 'win', '3', 'everi',
                                      'mean', 'back', 'pretti', 'best', 'lot', 'champ', 'could', 'point']
    dict_top_50['soccer'] = ['player', 'like', 'would', 'play', 'get', 'team', 'think', 'one', 'year', 'good', 'go',
                             'season', 'time', 'club', 'game', 'make', 'say', 'realli', 'leagu', 'want', 'even', 'better',
                             'much', 'see', 'well', 'also', 'know', 'still', 'footbal', 'back', 'best', 'could', 'gt',
                             'fan', 'peopl', 'us', 'look', 'goal', 'sign', 'last', 'need', 'money', 'fuck', 'way', 'got',
                             'right', 'sure', 'though', 'world', 'great']
    dict_top_50['funny'] = ['like', 'peopl', 'get', 'one', 'would', 'know', 'think', 'go', 'make', 'time', 'say', 'thing',
                            'see', 'look', 'use', 'fuck', 'way', 'realli', 'guy', 'even', 'good', 'want', 'also', 'post',
                            'much', 'work', 'http', 'actual', 'right', 'tri', 'year', 'still', 'need', 'got', 'shit',
                            'take', 'someon', 'well', 'person', 'mean', 'gt', 'someth', 'never', 'day', 'could', 'com',
                            'funni', 'back', 'comment', 'first']
    dict_top_50['movies'] = ['movi', 'like', 'film', 'one', 'think', 'would', 'get', 'make', 'realli', 'peopl',
                             'good', 'see', 'time', 'go', 'watch', 'charact', 'know', 'look', 'say', 'even', 'thing',
                             'much', 'love', 'way', 'also', 'could', 'stori', 'want', 'great', 'first', 'still',
                             'well', 'man', 'scene', 'made', 'pretti', 'bad', 'end', 'work', 'never', 'actual',
                             'show', 'got', 'point', 'year', 'gt', 'take', 'thought', 'lot', 'though']
    dict_top_50['anime'] = ['like', 'anim', 'show', 'watch', 'one', 'http', 'realli', 'get', 'gt', 'episod', 'think',
                            'good', 'com', 'peopl', 'time', 'charact', 'would', 'make', 'go', 'also', 'know', 'even',
                            'first', 'thing', 'see', 'much', 'well', 'girl', 'actual', 'want', 'love', 'feel', 'look',
                            'way', 'seri', 'pretti', 'end', 'lot', 'say', 'still', 'use', 'got', 'season', 'though', 'ゴ',
                            'someth', 'start', 'manga', 'work', 'could']
    dict_top_50['Overwatch'] = ['play', 'game', 'get', 'like', 'peopl', 'team', 'one', 'http', 'would', 'hero', 'time',
                                'com', 'make', 'think', 'player', 'good', 'gt', 'go', 'realli', 'use', 'even', 'want',
                                'say', 'need', 'overwatch', 'know', 'see', 'much', 'also', 'tri', 'merci', 'thing',
                                'way', 'kill', 'still', 'charact', 'well', 'point', 'work', 'could', 'actual', 'damag',
                                'reddit', 'win', 'right', 'mean', 'post', 'tank', 'better', 'lot']
    dict_top_50['trees'] = ['like', 'get', 'smoke', 'one', 'time', 'would', 'go', 'make', 'weed', 'good', 'use', 'know',
                            'peopl', 'think', 'realli', 'high', 'tri', 'look', 'also', 'fuck', 'much', 'say', 'got',
                            'thing', 'take', 'day', 'shit', 'work', 'even', 'want', 'see', 'feel', 'way', 'friend',
                            'year', 'http', 'first', 'love', 'right', 'back', 'man', 'tree', 'com', 'well', 'could',
                            'pretti', 'never', 'though', 'need', 'someth']
    dict_top_50['GlobalOffensive'] = ['like', 'play', 'team', 'get', 'game', 'would', 'one', 'player', 'go', 'make',
                                      'think', 'major', 'peopl', 'even', 'time', 'good', 'use', 'http', 'see', 'realli',
                                      'know', 'com', '1', 'say', '2', '3', 'much', 'win', 'still', 'want', 'better',
                                      'fuck', 'also', 'sk', 'well', 'got', 'look', 'watch', 'reddit', 'big', 'round',
                                      'could', 'cs', 'best', 'way', 'thing', 'match', 'tri', 'valv', 'take']
    dict_top_50['nfl'] = ['year', 'team', 'like', 'game', 'think', 'get', 'one', 'play', 'would', 'season', 'go', 'good',
                          'time', 'make', 'player', 'say', 'qb', 'realli', 'guy', 'last', 'even', 'better', 'nfl', 'fan',
                          'could', 'see', 'much', 'peopl', 'know', 'defens', 'still', 'http', 'also', 'got', 'back',
                          '2', 'win', 'well', '3', 'best', 'gt', 'com', 'look', '5', 'way', 'thing', 'want', 'start',
                          'point', 'take']
    dict_top_50['AskReddit'] = ['like', 'peopl', 'get', 'one', 'would', 'time', 'go', 'think', 'make', 'work', 'know',
                                'thing', 'realli', 'want', 'year', 'use', 'say', 'even', 'got', 'way', 'good', 'see',
                                'also', 'tri', 'look', 'day', 'still', 'could', 'much', 'never', 'lot', 'well', 'someth',
                                'need', 'someon', 'take', 'friend', 'feel', 'fuck', 'r', 'back', 'guy', 'post', 'person',
                                'live', 'actual', 'life', 'alway', 'right', 'everi']
    dict_top_50['gameofthrones'] = ['like', 'think', 'would', 'jon', 'one', 'show', 'go', 'get', 'know', 'see', 'kill',
                                    'make', 'book', 'peopl', 'cersei', 'season', 'realli', 'time', 'want', 'could',
                                    'even', 'dani', 'way', 'episod', 'also', 'say', 'king', 'arya', 'r', 'got', 'look',
                                    'gt', 'take', 'dragon', 'scene', 'much', 'thing', 'sansa', 'charact', 'well',
                                    'still', 'good', 'euron', 'tyrion', 'gameofthron', 'need', 'right', 'back', 'use',
                                    'watch']
    dict_top_50['conspiracy'] = ['like', 'peopl', 'would', 'gt', 'http', 'think', 'one', 'get', 'make', 'say', 'know',
                                 'go', 'trump', 'com', 'time', 'us', 'thing', 'even', 'use', 'post', 'see', 'right',
                                 'want', 'look', 'www', 'tri', 'also', 'actual', 'way', 'point', 'much', 'realli',
                                 'could', 'year', 'good', 'conspiraci', 'fuck', 'govern', 'believ', 'someth', 'comment',
                                 'mean', 'work', 'take', 'world', 'well', 'talk', 'need', 'said', 'r']
    dict_top_50['worldnews'] = ['peopl', 'like', 'gt', 'would', 'get', 'one', 'think', 'make', 'say', 'us', 'go',
                                'trump', 'countri', 'even', 'http', 'know', 'want', 'time', 'right', 'thing', 'also',
                                'use', 'world', 'see', 'year', 'work', 'way', 'point', 'com', 'need', 'govern', 'good',
                                'could', 'well', 'realli', 'state', 'actual', 'much', 'fuck', 'mean', 'take', 'still',
                                'come', 'tri', 'live', 'vote', 'happen', 'look', 'mani', 'war']
    dict_top_50['wow'] = ['get', 'like', 'time', 'play', 'one', 'use', 'would', 'peopl', 'raid', 'game', 'wow', 'go',
                          'make', 'realli', 'level', 'even', 'http', 'guild', 'class', 'think', 'want', 'much', 'thing',
                          'also', 'good', 'look', 'need', 'see', 'know', 'still', 'got', 'com', 'tri', 'gt', 'dp', 'take',
                          'tank', '2', 'could', 'us', 'back', 'first', 'boss', 'legion', 'quest', 'way', 'gear', 'well',
                          'world', 'player']
    dict_top_50['europe'] = ['peopl', 'gt', 'like', 'would', 'one', 'countri', 'eu', 'even', 'think', 'get', 'say',
                             'right', 'go', 'thing', 'want', 'know', 'also', 'europ', 'make', 'http', 'time', 'us',
                             'use', 'much', 'realli', 'see', 'well', 'actual', 'year', 'mean', 'need', 'state', 'still',
                             'good', 'way', 'could', 'look', 'germani', 'work', 'differ', 'uk', 'russia', 'someth',
                             'live', 'european', 'govern', 'german', 'problem', 'take', 'nation']
    dict_top_50['canada'] = ['peopl', 'like', 'canada', 'would', 'get', 'gt', 'one', 'think', 'go', 'canadian', 'make',
                             'right', 'say', 'time', 'year', 'even', 'use', 'govern', 'thing', 'want', 'know', 'us',
                             'also', 'work', 'need', 'much', 'see', 'countri', 'way', 'well', 'http', 'realli', 'actual',
                             'could', 'good', 'pay', 'law', 'money', 'mean', 'take', 'live', 'person', 'tri', 'point',
                             'someon', 'sure', 'never', 'differ', 'look', 'still']
    dict_top_50['Music'] = ['music', 'album', 'gt', 'song', 'http', 'r', 'band', 'like', 'one', 'www', 'reddit', 'com',
                            'last', 'releas', 'time', 'new', 'play', 'listen', 'record', 'first', 'artist', 'fm', 'get',
                            'peopl', 'rock', 'also', 'year', 'pleas', 'make', 'read', 'think', 'live', 'post', 'sound',
                            'good', 'way', 'love', 'would', 'show', 'say', 'use', 'amp', 'work', 'go', 'know', 'perform',
                            'u', 'place', 'day', 'realli']
    dict_top_50['baseball'] = ['like', 'game', 'year', 'get', 'would', 'team', 'one', 'think', 'go', 'time', 'player',
                               'guy', 'play', 'make', 'basebal', 'good', 'season', 'fan', 'hit', 'realli', 'see', 'pitch',
                               '2', 'know', 'say', '0', 'ball', '3', 'still', 'look', '1', 'run', 'even', 'http', 'also',
                               'first', 'com', 'much', 'start', 'better', 'got', 'way', 'trade', 'right', 'could', 'last',
                               'back', 'best', 'fuck', 'want']
    return dict_top_50

def dict_150():
    dict_top_150 = {}
    dict_top_150['hockey'] = ['team', 'year', 'like', 'player', 'get', 'play', 'would', 'one', 'game', 'think', 'go', 'season', 'good', 'time', 'make', 'guy', 'point', 'still', 'hockey', 'know', 'nhl', 'fan', 'see', 'realli', 'even', 'http', 'say', 'better', 'could', 'last', 'much', 'want', 'com', 'got', 'fuck', 'best', 'way', 'sign', 'leagu', 'peopl', 'back', 'watch', 'also', '2', 'look', 'top', 'trade', 'contract', 'take', 'well', 'first', 'us', 'cup', 'two', '3', 'pretti', 'goal', 'come', '5', 'win', 'need', 'gt', 'thing', '1', 'sure', 'playoff', 'lot', 'right', 'bad', 'never', 'great', 'pick', 'line', 'though', 'cap', 'put', 'score', 'probabl', 'love', 'give', 'mean', 'actual', 'start', 'everi', 'goali', 'yeah', 'shit', 'end', 'deal', 'alway', '4', 'leaf', 'tri', 'move', 'happen', 'hope', 'draft', 'defens', '10', 'amp', 'big', 'ice', 'next', 'www', 'hate', 'use', 'feel', '6', 'work', 'made', 'someth', 'differ', 'thought', 'post', 'around', 'money', 'shot', 'old', 'seem', 'said', 'someon', 'sport', 'mayb', 'new', 'sinc', 'talk', 'hit', 'mani', 'let', 'name', 'enough', 'least', 'reason', 'rememb', 'anoth', 'number', 'might', 'definit', 'round', 'reddit', 'comment', 'jersey', 'lose', 'man', 'offens', 'ever', 'career', 'r', 'keep', 'anyth']
    dict_top_150['nba'] = ['like', 'team', 'get', 'year', 'would', 'player', 'think', 'play', 'go', 'one', 'good', 'make', 'game', 'better', 'even', 'lebron', 'say', 'time', 'guy', 'realli', 'peopl', 'nba', 'season', '3', 'trade', 'best', 'could', 'want', 'look', 'know', 'take', 'much', 'still', 'see', 'way', 'point', 'also', 'fan', 'fuck', 'leagu', 'kyri', 'gt', 'got', '2', 'win', 'well', 'defens', 'need', '5', 'great', 'right', 'back', 'first', 'pretti', 'final', 'probabl', 'pick', 'lot', 'love', '1', 'last', 'thing', 'warrior', 'watch', 'shit', 'give', 'tri', 'ball', 'cav', 'bad', 'never', 'kd', 'actual', 'top', 'said', 'sure', 'contract', 'come', 'mean', 'playoff', 'lol', 'yeah', 'two', 'move', 'someon', 'though', 'next', 'everi', '4', 'happen', 'shot', 'kobe', 'someth', 'use', 'ever', 'put', 'offens', 'reason', 'alway', 'defend', 'big', 'start', 'sign', 'star', 'averag', 'post', 'basketbal', 'work', 'feel', 'man', 'draft', 'mayb', 'seem', 'deal', 'na', 'made', 'melo', 'sinc', 'money', 'http', 'shoot', 'second', '6', 'pg', 'com', '10', 'laker', 'us', 'call', 'chanc', 'harden', 'mani', 'leav', 'end', 'dude', 'let', 'talk', 'around', 'help', 'new', 'day', 'hope', 'without', 'gon', 'differ', 'pass', 'anoth', 'everyon', 'less', 'career']
    dict_top_150['leagueoflegends'] = ['game', 'like', 'get', 'play', 'team', 'one', 'would', 'think', 'time', 'peopl', 'make', 'good', 'realli', 'even', 'go', 'player', 'use', 'na', 'also', 'still', 'say', 'damag', 'much', 'gt', 'lane', 'well', 'better', 'know', 'see', 'champion', 'want', '2', 'need', 'eu', 'got', 'way', '1', 'thing', 'actual', 'win', '3', 'everi', 'mean', 'back', 'pretti', 'best', 'lot', 'champ', 'could', 'point', 'take', 'tri', 'look', 'give', 'bad', 'top', 'though', 'right', 'first', 'sinc', 'kill', 'http', 'feel', 'work', '5', 'new', 'mid', 'com', 'sure', 'support', 'chang', 'riot', 'fuck', 'probabl', 'year', 'tsm', 'shit', 'q', 'lol', 'ult', 'someth', 'level', 'item', 'last', 'enemi', 'yeah', 'start', 'tank', 'watch', 'post', 'build', 'said', 'pick', 'second', 'lose', 'hit', 'adc', '4', 'e', 'carri', 'around', 'hard', 'never', 'alway', 'someon', 'jungl', 'guy', 'world', 'earli', 'rune', 'ad', 'leagu', 'made', 'r', 'skin', 'happen', 'seem', 'long', 'end', 'problem', 'fight', 'rank', 'come', 'bot', 'let', 'differ', 'without', '10', 'mayb', 'mani', 'ban', 'ye', 'split', 'reddit', 'elo', 'everyon', 'anyth', 'talk', 'two', 'ever', 'day', 'bit', 'fun', 'keep', 'abl', 'help', 'less', 'skt', 'put', 'reason']
    dict_top_150['soccer'] = ['player', 'like', 'would', 'play', 'get', 'team', 'think', 'one', 'year', 'good', 'go', 'season', 'time', 'club', 'game', 'make', 'say', 'realli', 'leagu', 'want', 'even', 'better', 'much', 'see', 'well', 'also', 'know', 'still', 'footbal', 'back', 'best', 'could', 'gt', 'fan', 'peopl', 'us', 'look', 'goal', 'sign', 'last', 'need', 'money', 'fuck', 'way', 'got', 'right', 'sure', 'though', 'world', 'great', 'start', 'thing', 'top', 'come', 'lot', 'first', 'mean', 'chelsea', 'win', 'point', 'watch', 'probabl', 'transfer', 'happen', 'take', 'pretti', 'tri', 'barca', 'neymar', 'real', 'seem', 'http', 'never', 'everi', '2', 'said', 'someth', 'end', 'use', 'guy', 'arsen', 'citi', 'yeah', 'made', 'man', 'madrid', 'actual', 'mani', 'unit', 'shit', 'two', 'leav', 'cup', '3', 'mayb', 'deal', 'hope', 'psg', 'buy', 'around', 'manag', 'move', 'messi', 'big', 'comment', 'match', 'alway', 'put', 'score', 'give', 'ball', 'bit', 'alreadi', 'anoth', 'work', '10', 'pay', 'love', 'sell', 'new', 'com', 'bad', 'sinc', 'anyth', 'let', 'agre', 'enough', 'post', 'chanc', 'striker', 'next', 'ever', 'differ', 'million', 'soccer', 'someon', 'seen', '1', 'definit', 'stay', 'day', 'reason', 'ronaldo', 'ye', 'talk', 'thought', 'contract', '4', '5', 'part']
    dict_top_150['funny'] = ['like', 'peopl', 'get', 'one', 'would', 'know', 'think', 'go', 'make', 'time', 'say', 'thing', 'see', 'look', 'use', 'fuck', 'way', 'realli', 'guy', 'even', 'good', 'want', 'also', 'post', 'much', 'work', 'http', 'actual', 'right', 'tri', 'year', 'still', 'need', 'got', 'shit', 'take', 'someon', 'well', 'person', 'mean', 'gt', 'someth', 'never', 'day', 'could', 'com', 'funni', 'back', 'comment', 'first', 'lot', 'everi', 'pretti', 'said', 'read', 'reddit', 'sure', 'call', 'mani', 'live', 'differ', 'show', 'better', 'feel', 'put', 'love', 'come', 'watch', 'find', 'give', '2', 'thought', 'around', 'joke', 'kid', 'tell', 'amp', 'www', 'wrong', 'bad', 'end', 'yeah', 'old', 'let', 'friend', 'man', 'alway', 'word', 'made', 'seem', 'probabl', 'ever', 'happen', 'though', 'littl', 'place', 'alpha', 'long', 'women', 'dog', 'two', 'us', 'oh', 'point', 'start', 'seen', 'anyon', 'girl', 'everyon', 'anyth', 'name', 'new', 'els', 'might', 'chang', 'world', 'reason', 'oscar', 'life', 'talk', 'least', '3', 'eat', 'stop', 'last', 'away', 'enough', 'video', 'mayb', 'keep', 'lol', 'without', 'fact', 'money', 'thank', 'food', 'ye', 'part', 'american', 'great', 'understand', 'noth', '1', 'big', 'r', 'ask', 'pay', 'real', 'job', 'nice']
    dict_top_150['movies'] = ['movi', 'like', 'film', 'one', 'think', 'would', 'get', 'make', 'realli', 'peopl', 'good', 'see', 'time', 'go', 'watch', 'charact', 'know', 'look', 'say', 'even', 'thing', 'much', 'love', 'way', 'also', 'could', 'stori', 'want', 'great', 'first', 'still', 'well', 'man', 'scene', 'made', 'pretti', 'bad', 'end', 'work', 'never', 'actual', 'show', 'got', 'point', 'year', 'gt', 'take', 'thought', 'lot', 'though', 'feel', 'http', 'shit', 'guy', 'tri', 'differ', 'book', 'better', 'use', 'right', 'mean', '2', 'someth', 'fuck', 'sure', 'come', 'seen', 'need', 'seem', 'new', 'said', 'best', 'give', 'com', 'saw', 'trailer', 'person', 'origin', 'back', 'play', 'everi', 'part', 'black', 'yeah', 'day', 'happen', 'mayb', 'hope', 'read', '3', 'big', 'world', 'life', 'bit', 'understand', 'rememb', 'sound', 'reason', 'war', 'mani', 'littl', 'action', 'alway', 'set', 'someon', 'two', 'www', 'sinc', 'tell', 'director', 'whole', 'kind', 'ever', 'probabl', 'talk', 'definit', 'kid', 'anyth', 'find', 'long', 'kill', 'live', 'act', 'anoth', 'noth', 'enjoy', 'real', 'put', 'actor', 'last', '1', 'start', 'power', 'let', 'enough', 'fight', 'fan', 'theater', 'agre', 'everyth', 'oh', 'comment', 'plot', 'believ', 'around', 'hate', 'far', 'call', 'stuff', 'els']
    dict_top_150['anime'] = ['like', 'anim', 'show', 'watch', 'one', 'http', 'realli', 'get', 'gt', 'episod', 'think', 'good', 'com', 'peopl', 'time', 'charact', 'would', 'make', 'go', 'also', 'know', 'even', 'first', 'thing', 'see', 'much', 'well', 'girl', 'actual', 'want', 'love', 'feel', 'look', 'way', 'seri', 'pretti', 'end', 'lot', 'say', 'still', 'use', 'got', 'season', 'though', 'ゴ', 'someth', 'start', 'manga', 'work', 'could', 'better', 'best', 'imgur', 'read', 'tri', 'stori', 'mean', 'fate', 'right', 'differ', 'net', 'myanimelist', 'sure', 'made', 'never', '3', 'seem', 'might', 'probabl', 'r', 'person', 'point', '10', 'need', 'great', '2', 'www', '1', 'bad', 'post', 'sinc', 'fuck', 'come', 'year', 'bit', 'interest', 'happen', 'movi', 'stuff', 'seen', 'shit', 'thought', 'part', 'take', 'day', 'mayb', 'least', 'enjoy', 'find', 'back', 'yeah', 'game', 'two', 'far', 'guy', 'reason', 'give', 'anyth', 'world', 'jpg', 'someon', 'comment', 'everi', 'new', 'play', 'guess', 'scene', 'last', 'life', 'mani', 'alreadi', 'talk', 'oh', 'amp', 'littl', 'anoth', 'complet', 'everyon', 'fun', '5', 'second', 'zero', 'kind', 'japanes', 'live', 'fan', 'hope', '4', 'long', 'thank', 'wait', 'enough', 'ever', 'nice', 'fight', 'recommend', 'man', 'arc', 'school', 'kill']
    dict_top_150['Overwatch'] = ['play', 'game', 'get', 'like', 'peopl', 'team', 'one', 'http', 'would', 'hero', 'time', 'com', 'make', 'think', 'player', 'good', 'gt', 'go', 'realli', 'use', 'even', 'want', 'say', 'need', 'overwatch', 'know', 'see', 'much', 'also', 'tri', 'merci', 'thing', 'way', 'kill', 'still', 'charact', 'well', 'point', 'work', 'could', 'actual', 'damag', 'reddit', 'win', 'right', 'mean', 'post', 'tank', 'better', 'lot', '2', 'feel', 'heal', 'comp', 'enemi', 'take', 'everi', '1', 'main', 'ult', 'got', 'match', 'chang', 'doomfist', 'support', 'hanzo', 'look', 'never', 'pretti', 'someon', 'everyon', 'rank', 'dp', 'www', 'though', 'someth', 'sombra', 'blizzard', 'bad', 'differ', 'r', 'come', 'genji', 'roadhog', 'pick', 'sr', 'back', 'fun', 'gfycat', 'fuck', 'skill', 'healer', 'sinc', 'new', 'first', '3', 'messag', 'around', 'sure', 'yeah', 'alway', 'shot', 'second', 'person', 'give', 'system', 'competit', 'mayb', 'probabl', 'high', 'amp', 'hit', 'charg', 'abil', 'reason', 'problem', 'comment', 'shit', 'two', 'counter', 'said', 'happen', 'best', 'level', 'start', 'meta', 'seem', 'lose', 'guy', 'season', 'issu', 'tracer', 'hard', 'mani', 'lucio', 'winston', 'end', 'report', 'pleas', 'keep', 'ana', 'aim', 'call', 'watch', 'less', 'lol', 'nerf', 'low', 'long', 'put']
    dict_top_150['trees'] = ['like', 'get', 'smoke', 'one', 'time', 'would', 'go', 'make', 'weed', 'good', 'use', 'know', 'peopl', 'think', 'realli', 'high', 'tri', 'look', 'also', 'fuck', 'much', 'say', 'got', 'thing', 'take', 'day', 'shit', 'work', 'even', 'want', 'see', 'feel', 'way', 'friend', 'year', 'http', 'first', 'love', 'right', 'back', 'man', 'tree', 'com', 'well', 'could', 'pretti', 'never', 'though', 'need', 'someth', 'come', 'alway', 'post', 'still', 'better', 'lot', 'yeah', 'guy', 'actual', 'around', 'thank', 'sure', 'start', 'watch', 'keep', 'find', 'littl', 'legal', 'best', '2', 'someon', 'everi', 'made', 'hit', 'bad', 'roll', 'state', 'live', 'great', 'happen', 'help', 'bowl', 'www', 'thought', 'give', 'probabl', 'differ', 'joint', 'mean', 'end', 'r', 'ever', 'buy', 'last', 'place', 'dude', 'ask', 'drug', 'put', 'long', 'anyth', 'new', 'na', 'lol', '5', 'let', 'us', 'nice', 'awesom', 'bong', 'mayb', 'person', 'call', 'week', 'reason', 'hope', 'life', 'two', 'marijuana', 'show', 'mind', 'enjoy', 'stuff', 'said', 'point', 'seem', 'ent', 'sound', 'play', 'test', 'water', '10', 'pipe', 'might', 'amp', 'mani', '1', 'sinc', 'experi', '3', 'smell', 'noth', 'light', 'tell', 'hour', '4', 'kind', 'cannabi', 'talk', 'usual']
    dict_top_150['GlobalOffensive'] = ['like', 'play', 'team', 'get', 'game', 'would', 'one', 'player', 'go', 'make', 'think', 'major', 'peopl', 'even', 'time', 'good', 'use', 'http', 'see', 'realli', 'know', 'com', '1', 'say', '2', '3', 'much', 'win', 'still', 'want', 'better', 'fuck', 'also', 'sk', 'well', 'got', 'look', 'watch', 'reddit', 'big', 'round', 'could', 'cs', 'best', 'way', 'thing', 'match', 'tri', 'valv', 'take', 'back', 'shit', 'gt', 'actual', 'na', 'sure', 'everi', 'lot', 'map', 'rule', 'tournament', 'need', 'right', 'never', 'year', 'guy', 'work', 'mean', 'bad', 'someth', 'said', 'point', 'first', 'r', 'pretti', 'last', 'www', 'happen', 'though', 'bug', 'start', 'event', 'top', 'mani', 'differ', 'yeah', 'sinc', 'come', 'lol', 'chang', 'someon', 'c9', 'probabl', 'feel', 'amp', 'give', 'new', 'talk', 'alway', 'ban', 'final', 'pleas', 'seem', 'fnatic', 'post', 'csgo', '5', '0', 'pro', '4', 'messag', 'pick', 'put', 'might', 'globaloffens', 'dont', 'day', 'mayb', 'problem', 'let', 'kill', 'two', 'group', 'reason', 'made', 'find', 'comment', 'money', 'great', 'faze', 'esl', 'lose', 'rank', 'vp', 'qualifi', 'keep', 'long', 'anyth', 'fix', 'video', 'astrali', 'call', 'least', 'hope', 'thread', 'alreadi', 'g2', 'issu', 'everyon', 'noth']
    dict_top_150['nfl'] = ['year', 'team', 'like', 'game', 'think', 'get', 'one', 'play', 'would', 'season', 'go', 'good', 'time', 'make', 'player', 'say', 'qb', 'realli', 'guy', 'last', 'even', 'better', 'nfl', 'fan', 'could', 'see', 'much', 'peopl', 'know', 'defens', 'still', 'http', 'also', 'got', 'back', '2', 'win', 'well', '3', 'best', 'gt', 'com', 'look', '5', 'way', 'thing', 'want', 'start', 'point', 'take', '1', 'top', 'lot', 'first', 'everi', 'offens', 'pick', 'great', 'watch', 'never', 'pretti', 'right', 'run', 'bradi', 'man', 'playoff', 'probabl', 'fuck', 'bowl', 'super', 'though', 'two', 'sure', 'mean', 'said', 'use', 'actual', 'yard', 'come', 'bad', 'leagu', 'need', 'put', 'footbal', 'made', 'yeah', '10', 'line', 'pass', 'stat', 'us', '4', 'end', 'work', 'post', 'draft', 'career', 'tri', 'talk', 'shit', 'reason', 'give', 'call', 'sign', 'sinc', 'around', 'coach', 'someth', 'alway', 'receiv', 'ever', 'love', 'long', 'week', 'differ', 'www', 'seem', 'hope', 'feel', 'big', 'least', '6', 'enough', 'happen', 'might', 'someon', 'mani', 'new', 'patriot', 'number', '7', 'ball', 'wr', 'mayb', 'next', 'throw', 'thought', 'show', 'cowboy', 'definit', 'second', 'field', 'lose', 'posit', 'rodger', 'talent', 'chang', 'day', 'td', 'averag']
    dict_top_150['AskReddit'] = ['like', 'peopl', 'get', 'one', 'would', 'time', 'go', 'think', 'make', 'work', 'know', 'thing', 'realli', 'want', 'year', 'use', 'say', 'even', 'got', 'way', 'good', 'see', 'also', 'tri', 'look', 'day', 'still', 'could', 'much', 'never', 'lot', 'well', 'someth', 'need', 'someon', 'take', 'friend', 'feel', 'fuck', 'r', 'back', 'guy', 'post', 'person', 'live', 'actual', 'life', 'alway', 'right', 'everi', 'love', 'ask', 'first', 'end', 'said', 'question', 'start', 'us', 'call', '2', 'talk', 'pretti', 'shit', 'mean', 'askreddit', 'find', '1', 'come', 'http', 'differ', 'pleas', 'around', 'kid', 'show', 'bad', 'give', 'though', 'watch', 'happen', 'anyth', 'job', 'better', 'com', 'mani', 'car', 'tell', 'play', 'school', 'old', 'thought', 'reddit', 'point', 'gt', 'made', 'hour', 'littl', 'without', 'help', 'read', 'everyon', 'new', 'game', 'long', 'rule', 'sure', '3', 'money', 'ever', 'went', 'believ', 'two', 'messag', 'put', 'probabl', 'pay', 'noth', 'eat', 'www', 'enough', 'place', 'best', 'sound', 'girl', 'part', 'anoth', 'last', 'yeah', 'care', 'keep', 'amp', 'reason', 'man', 'famili', 'world', 'walk', 'let', 'great', 'comment', 'away', 'kind', 'month', 'seem', 'problem', 'turn', 'sinc', 'run', 'buy', 'stori', 'chang', 'big']
    dict_top_150['gameofthrones'] = ['like', 'think', 'would', 'jon', 'one', 'show', 'go', 'get', 'know', 'see', 'kill', 'make', 'book', 'peopl', 'cersei', 'season', 'realli', 'time', 'want', 'could', 'even', 'dani', 'way', 'episod', 'also', 'say', 'king', 'arya', 'r', 'got', 'look', 'gt', 'take', 'dragon', 'scene', 'much', 'thing', 'sansa', 'charact', 'well', 'still', 'good', 'euron', 'tyrion', 'gameofthron', 'need', 'right', 'back', 'use', 'watch', 'though', 'come', 'probabl', 'end', 'point', 'someth', 'sure', 'thought', 'happen', 'first', 'said', 'die', 'post', 'mean', 'spoiler', 'fuck', 'actual', 'seem', 'lannist', 'tri', 'love', 'north', 'mayb', 'last', 'jaim', 'pretti', 'throne', 'feel', 'bran', 'stark', 'land', 'might', 'armi', 'never', 'dead', 'yeah', 'made', 'give', 'ship', 'reason', 'fight', 'westero', 'great', 'http', 'talk', 'w', 'littlefing', 'everyon', 'believ', 'read', 'sinc', 'hand', 'part', 'tell', 'lot', 'far', 'shit', 'start', 'differ', 'game', 'mani', 'long', 'men', 'night', 'two', 'face', 'keep', 'sam', 'may', 'someon', 'work', 'everi', 'anyth', 'ned', 'man', 'alway', 'guy', 'hous', 'amp', 'bad', 'com', 'best', 'battl', 'least', 'find', 'year', '1', 'better', 'famili', 'stori', 'new', 'rememb', '2', 'play', 'white', 'winterfel', 'els', 'wall', 'noth', 'person']
    dict_top_150['conspiracy'] = ['like', 'peopl', 'would', 'gt', 'http', 'think', 'one', 'get', 'make', 'say', 'know', 'go', 'trump', 'com', 'time', 'us', 'thing', 'even', 'use', 'post', 'see', 'right', 'want', 'look', 'www', 'tri', 'also', 'actual', 'way', 'point', 'much', 'realli', 'could', 'year', 'good', 'conspiraci', 'fuck', 'govern', 'believ', 'someth', 'comment', 'mean', 'work', 'take', 'world', 'well', 'talk', 'need', 'said', 'r', 'evid', 'mani', 'person', 'money', 'guy', 'shit', 'come', 'state', 'amp', 'still', 'day', 'seem', 'anyth', 'sure', 'noth', 'happen', 'back', 'reddit', 'read', 'news', 'fact', 'give', 'polit', 'lot', 'everi', 'sub', 'never', 'someon', 'sourc', 'got', 'new', 'support', 'call', '1', 'watch', 'inform', 'show', 'though', 'feel', 'countri', 'russian', 'first', 'russia', 'control', 'made', 'claim', 'articl', 'let', 'chang', 'mayb', 'stori', 'reason', 'find', 'link', 'keep', 'differ', 'cnn', 'around', 'anyon', 'real', 'power', 'part', 'start', 'human', 'vote', 'everyon', 'system', 'elect', 'clinton', 'question', 'thought', 'interest', 'sinc', 'without', 'life', 'place', 'put', 'media', 'tell', 'name', 'alway', 'account', 'ever', 'might', 'big', 'yeah', 'ye', 'law', '2', 'true', 'better', 'probabl', 'live', 'problem', 'war', 'idea', 'long', 'ask', 'mind', 'complet']
    dict_top_150['worldnews'] = ['peopl', 'like', 'gt', 'would', 'get', 'one', 'think', 'make', 'say', 'us', 'go', 'trump', 'countri', 'even', 'http', 'know', 'want', 'time', 'right', 'thing', 'also', 'use', 'world', 'see', 'year', 'work', 'way', 'point', 'com', 'need', 'govern', 'good', 'could', 'well', 'realli', 'state', 'actual', 'much', 'fuck', 'mean', 'take', 'still', 'come', 'tri', 'live', 'vote', 'happen', 'look', 'mani', 'war', 'said', 'russia', 'never', 'american', 'anyth', 'back', 'someth', 'comment', 'lot', 'china', 'news', 'first', 'person', 'day', 'nation', 'reddit', 'shit', 'differ', 'www', 'everi', 'reason', 'power', 'chang', 'sure', 'support', 'guy', 'amp', 'polit', 'noth', '1', 'call', 'talk', 'kill', 'place', 'law', 'problem', 'articl', 'better', 'america', 'read', 'militari', 'money', 'let', 'give', 'russian', 'someon', 'far', 'wrong', 'part', 'pretti', 'end', 'believ', 'elect', 'presid', 'parti', 'new', 'though', 'find', 'without', 'got', 'seem', 'sinc', 'probabl', 'fact', 'ye', 'israel', 'last', 'attack', 'made', 'put', 'everyon', 'long', 'around', 'keep', 'feel', 'case', 'major', 'start', 'anyon', 'system', 'claim', 'man', 'muslim', 'two', 'best', 'hate', 'alway', 'less', 'care', 'mayb', 'alreadi', 'matter', 'help', 'agre', '2', 'post', 'might', 'human', 'sourc', 'understand']
    dict_top_150['wow'] = ['get', 'like', 'time', 'play', 'one', 'use', 'would', 'peopl', 'raid', 'game', 'wow', 'go', 'make', 'realli', 'level', 'even', 'http', 'guild', 'class', 'think', 'want', 'much', 'thing', 'also', 'good', 'look', 'need', 'see', 'know', 'still', 'got', 'com', 'tri', 'gt', 'dp', 'take', 'tank', '2', 'could', 'us', 'back', 'first', 'boss', 'legion', 'quest', 'way', 'gear', 'well', 'world', 'player', 'new', 'post', 'everi', 'sinc', 'feel', 'say', 'pretti', 'kill', 'lot', 'fight', 'damag', 'heal', 'start', '3', 'group', 'run', 'spec', 'actual', '1', 'work', 'someth', 'never', 'better', 'best', 'chang', 'day', 'right', 'mythic', 'though', '5', 'sure', 'week', 'probabl', 'www', 'differ', 'give', 'r', 'server', 'fuck', 'hit', 'find', 'around', 'set', 'fun', 'pvp', 'dungeon', 'charact', 'content', 'end', 'help', 'alway', 'someon', 'legendari', 'point', 'come', 'bad', 'love', 'expans', 'current', 'keep', 'mean', 'enough', 'mani', 'yeah', 'reason', 'healer', 'last', 'blizzard', 'seem', 'battl', '4', 'mayb', 'two', 'hard', 'mount', 'item', 'remov', '7', 'without', 'put', 'en', 'zone', 'mage', '10', 'friend', 'high', 'reddit', 'said', 'problem', 'old', 'stuff', 'heroic', 'person', 'everyon', 'least', 'everyth', 'drop', 'progress', 'ilvl', 'long']
    dict_top_150['europe'] = ['peopl', 'gt', 'like', 'would', 'one', 'countri', 'eu', 'even', 'think', 'get', 'say', 'right', 'go', 'thing', 'want', 'know', 'also', 'europ', 'make', 'http', 'time', 'us', 'use', 'much', 'realli', 'see', 'well', 'actual', 'year', 'mean', 'need', 'state', 'still', 'good', 'way', 'could', 'look', 'germani', 'work', 'differ', 'uk', 'russia', 'someth', 'live', 'european', 'govern', 'german', 'problem', 'take', 'nation', 'part', 'law', 'better', 'mani', 'never', 'war', 'tri', 'world', 'though', 'point', 'happen', 'chang', 'sure', 'come', 'call', 'polit', 'lot', 'back', 'said', 'reason', 'power', 'talk', 'let', 'anyth', 'poland', 'less', 'end', 'ye', 'far', 'everi', 'place', 'first', 'case', 'russian', 'fact', 'new', 'support', 'vote', 'made', 'care', 'bad', 'seem', 'popul', 'pretti', '2', 'com', 'two', 'sinc', 'noth', 'www', 'immigr', 'money', 'without', 'fuck', 'org', 'believ', 'give', '1', 'comment', 'franc', 'stop', 'day', 'long', 'muslim', 'interest', 'shit', 'issu', 'en', 'agre', 'start', 'articl', 'alreadi', 'least', 'rule', 'probabl', 'wikipedia', 'enough', 'pay', 'person', 'system', 'mayb', 'free', 'wiki', 'idea', 'got', 'find', 'yeah', 'might', 'alway', 'guy', 'post', 'someon', 'help', 'matter', 'languag', 'r', 'number', 'keep', 'border', 'understand']
    dict_top_150['canada'] = ['peopl', 'like', 'canada', 'would', 'get', 'gt', 'one', 'think', 'go', 'canadian', 'make', 'right', 'say', 'time', 'year', 'even', 'use', 'govern', 'thing', 'want', 'know', 'us', 'also', 'work', 'need', 'much', 'see', 'countri', 'way', 'well', 'http', 'realli', 'actual', 'could', 'good', 'pay', 'law', 'money', 'mean', 'take', 'live', 'person', 'tri', 'point', 'someon', 'sure', 'never', 'differ', 'look', 'still', 'reason', 'mani', 'khadr', 'come', 'day', 'happen', 'fuck', '1', 'someth', 'said', 'lot', 'case', 'fact', 'everi', 'tax', 'give', 'talk', 'problem', 'www', 'back', 'job', 'world', 'call', 'issu', 'seem', 'com', 'first', 'american', 'pretti', '2', 'read', 'immigr', 'new', 'part', 'though', 'better', 'place', 'feel', 'without', 'let', 'gener', 'anyth', 'court', 'made', 'post', 'comment', 'nation', 'noth', 'kill', 'find', 'understand', 'bad', 'got', 'famili', 'put', 'care', 'state', 'white', 'legal', 'shit', '3', 'system', 'guy', 'chang', 'around', 'might', 'cost', 'keep', 'busi', 'stop', 'ye', 'less', 'million', 'citi', 'agre', 'crime', 'anyon', 'base', 'wrong', '5', 'help', 'alway', 'believ', 'polit', 'everyon', 'may', 'start', 'servic', 'far', '10', 'rule', 'allow', 'old', 'probabl', 'compani', 'alreadi', 'two', 'life', 'cultur', 'least']
    dict_top_150['Music'] = ['music', 'album', 'gt', 'song', 'http', 'r', 'band', 'like', 'one', 'www', 'reddit', 'com', 'last', 'releas', 'time', 'new', 'play', 'listen', 'record', 'first', 'artist', 'fm', 'get', 'peopl', 'rock', 'also', 'year', 'pleas', 'make', 'read', 'think', 'live', 'post', 'sound', 'good', 'way', 'love', 'would', 'show', 'say', 'use', 'amp', 'work', 'go', 'know', 'perform', 'u', 'place', 'day', 'realli', 'feedback', 'back', 'best', 'well', 'score', 'track', 'watch', 'youtub', 'made', 'see', 'even', 'commun', 'singl', 'self', 'tour', 'better', 'much', 'wiki', 'want', 'comment', 'share', 'two', 'great', 'help', 'v', 'never', 'still', 'mani', 'submit', 'hit', 'guitar', 'thing', 'take', 'pop', 'metal', 'someth', 'group', 'feel', 'lot', 'person', 'member', 'noth', 'got', 'subreddit', 'featur', '1', 'thank', 'differ', 'name', 'posit', 'tri', 'start', 'radio', 'come', 'musician', 'anyth', 'fuck', 'user', 'part', '2', 'downvot', 'vocal', 'question', '0', 'includ', 'singer', 'pretti', 'top', 'tag', 'messag', 'net', 'png', 'compos', 'delet', 'known', 'without', 'find', 'pic', 'ever', 'lastfm', 'img2', 'akama', '252', 'incorrect', 'though', 'sinc', 'look', 'success', 'could', 'produc', 'need', 'call', 'heard', 'actual', 'right', 'alway', 'label', 'fan', 'us', 'countri']
    dict_top_150['baseball'] = ['like', 'game', 'year', 'get', 'would', 'team', 'one', 'think', 'go', 'time', 'player', 'guy', 'play', 'make', 'basebal', 'good', 'season', 'fan', 'hit', 'realli', 'see', 'pitch', '2', 'know', 'say', '0', 'ball', '3', 'still', 'look', '1', 'run', 'even', 'http', 'also', 'first', 'com', 'much', 'start', 'better', 'got', 'way', 'trade', 'right', 'could', 'last', 'back', 'best', 'fuck', 'want', 'well', 'pitcher', 'peopl', '5', 'yanke', 'call', 'watch', 'thing', 'mlb', 'pretti', 'probabl', 'need', 'take', 'though', 'gt', 'win', 'leagu', '4', 'lot', 'mean', 'home', 'point', 'come', 'great', 'sure', 'actual', 'never', 'amp', 'bad', 'love', 'two', 'everi', 'day', 'made', 'us', 'said', 'happen', 'give', 'inning', 'someth', 'mayb', 'sinc', 'top', 'post', 'yeah', 'tri', 'throw', 'prospect', 'r', 'www', 'era', 'sox', 'judg', 'cub', 'shit', '6', 'averag', 'ever', 'dodger', 'seem', 'hope', 'base', 'bat', 'work', 'seri', 'alway', 'red', 'career', 'use', '7', 'feel', 'next', 'put', 'around', 'thought', 'star', 'someon', '9', 'world', 'strike', 'field', 'move', 'walk', 'name', 'second', 'show', 'mani', 'high', 'big', 'talk', 'stat', '10', 'far', 'playoff', 'hitter', 'differ', 'long', 'end', 'half', 'let']
    return dict_top_150

def eliminate_duplicates(dict_in, len_in):
    dup = copy.deepcopy(dict_in)
    sub_list = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime', 'Overwatch', 'trees',
         'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 'conspiracy', 'worldnews', 'wow', 'europe',
         'canada', 'Music', 'baseball']
    for sub in sub_list:
        for sub_2 in sub_list:
            if sub != sub_2:
                for i in range(len_in):
                    if i < len(dup[sub_2]):
                        if dup[sub_2][i] in dict_in[sub]:
                            dict_in[sub].remove(dup[sub_2][i])
    return dict_in

def dictlist_to_list(dict_in):
    list_out = []
    for sub in dict_in:
        list_out.extend(dict_in[sub])
    return list_out


def tf_list_from_comment(comment_in, word_list_in):
    # input should already be lemmatized
    tf_list = []
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = tokenizer.tokenize(comment_in)  # tokenize words in lists
    for word in word_list_in:
        if word in tokens:
            tf_list.append(True)
        else:
            tf_list.append(False)
    return tf_list



nltk.download('stopwords')
nltk.download('punkt')
w = dict_50()
w_y = eliminate_duplicates(w, 50)
w_z = dictlist_to_list(w_y)
x = dict_150()
y = eliminate_duplicates(x, 150)
z = dictlist_to_list(y)
unique_items_50 = dictlist_to_list(eliminate_duplicates(dict_50(), 50))
unique_items_150 = dictlist_to_list(eliminate_duplicates(dict_150(), 150))
# print(y)
# list_test = ['Buffalo']
# test_item = 'Honestly, Buffalo is the correct answer. I remember people (somewhat) joking that Buffalo'
# print(tf_list_from_comment(test_item, list_test))
# print(dictlist_to_list(eliminate_duplicates(copy.deepcopy(dict_top_50))))

# flag = 0
# for i in range(len(z)):
#     for i1 in range(len(z)):
#         if i != i1:
#             if z[i] == z[i1]:
#                 print(z[i])
#                 flag = 1
# print(flag)

def comment_list_to_column_dict(comment_list, word_list):
    # this method takes in a list of comments, and a list of words (word list comes from most used word in comments)
    # outputs a dictionary, where the keys are the words in word_list, and the values are lists of whether
    # that word appears in the comment at the identical index in the comment list
    comment_dict = {}
    word_list_dict = {}
    for comment in comment_list:
        comment_dict[comment] = tf_list_from_comment(comment, word_list)
    for i, word in enumerate(word_list):
        word_list_dict[word] = []
        for j, comment in enumerate(comment_list):
            temp = comment_dict[comment][i]
            word_list_dict[word].append(temp)
    return word_list_dict


def add_columns_to_df(df_in, dict_in):
    for key_word in dict_in:
        df_in.insert(2, key_word, dict_in[key_word], True)
    return df_in

cl = ['c d e e', 'a b', 'a b c', 'b c']
wl = ['a', 'b', 'c']
d = comment_list_to_column_dict(cl, wl)
print(d)
