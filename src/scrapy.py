from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from fake_useragent import UserAgent
from random import randint
from time import sleep
from urllib.parse import quote
from src.util import wait_element
import pandas as pd
import urllib.parse as urlparse
from urllib.parse import parse_qs

class Google:

    def __init__(self):
        self.create_driver()

    def search(self, keyword, param=' AND ("machine learning" OR "artificial inteligence")', npages=1):
        self.driver = webdriver.Firefox()
        keywordstr = f'({keyword.replace(" ", "+")})'
        search_url = f'https://www.google.com.br/search?q={keywordstr}+{quote(param, safe="")}'
        self.driver.get(search_url)
        urls = []
        count_pages = 1

        while count_pages <= npages:
            sleep(randint(10, 20))
            wait_element(self.driver, '//div[@class="g"]//div/a')
            atags = self.driver.find_elements_by_xpath('//div[@class="g"]//div/a')
            urls.extend([a.get_attribute('href') for a in atags if any(s in a.get_attribute('href') for s in self.target_urls)])
            try:
                next_button = self.driver.find_element_by_xpath('//a[@id="pnnext"]')
            except NoSuchElementException:
                pass
                break
            next_button.click()
            count_pages += 1
        self.driver.close()
        return {f"{keyword}": urls}

    def create_driver(self):
        useragent = UserAgent()
        profile = webdriver.FirefoxProfile()
        profile.set_preference("general.useragent.override", useragent.random)
        self.driver = webdriver.Firefox(firefox_profile=profile)

    def agent_search(self, keyword, url, count, param=None):
        if count % 40 == 0:
            self.driver.close()
            self.create_driver()
        keywordstr = f'"{keyword}"'
        search_url = f'{url}{quote(keywordstr)}+{quote(param, safe="")}' if param else f'{url}{quote(keywordstr)}'
        self.driver.get(search_url)
        found = wait_element(self.driver, '//div/p[contains(text(),"Results")]')
        if not found:
            found = wait_element(self.driver, '//div/p[contains(text(),"Result")]')
            if not found:
                found = wait_element(self.driver, '//a[contains(text(),"Let us know.")]')
                if not found:
                    print(count)
                    exit(-1)
                else:
                    return 0
            else:
                xpath = '//div/p[contains(text(),"Result")]'
        else:
            xpath = '//div/p[contains(text(),"Results")]'

        results_text = self.driver.find_element_by_xpath(xpath).text
        qtd = int(results_text.split(" ")[0].replace(',', ''))
        return qtd


if __name__ == "__main__":
    g = Google()
    keywordsfile = open("ListaPalavras.txt", "r")
    keywords_list = keywordsfile.readlines()
    # Incrementa, get the remaining words on the list
    result = [['A/B testing', 449], ['accuracy', 141548], ['activation function', 4965], ['active learning', 176], ['AdaGrad', 1331], ['agent', 8665], ['agglomerative clustering', 430], ['anomaly detection', 2265], ['area under the PR curve', 14], ['area under the ROC curve', 692], ['artificial general intelligence', 51], ['attribute', 14884], ['Area under the ROC Curve', 692], ['augmented reality', 79], ['automation bias', 3], ['average precision', 1303], ['backpropagation', 1853], ['bag of words', 4453], ['baseline', 29686], ['batch', 35614], ['batch normalization', 2219], ['batch size', 11665], ['Bayesian neural network', 51], ['Bayesian optimization', 1909], ['Bellman equation', 47], ['fairness', 774], ['bidirectional', 5420], ['bidirectional language model', 7], ['binary classification', 9964], ['binning', 2762], ['bounding box', 3739], ['broadcasting', 876], ['bucketing', 368], ['calibration layer', 2], ['candidate generation', 420], ['candidate sampling', 1], ['categorical data', 12874], ['causal language model', 2], ['centroid', 2051], ['centroid-based clustering', 13], ['classification model', 5451], ['classification threshold', 249], ['clustering', 19536], ['collaborative filtering', 1270], ['confirmation bias', 54], ['confusion matrix', 23927], ['convenience sampling', 11], ['convergence', 2988], ['convex function', 77], ['convolutional filter', 89], ['convolutional layer', 2671], ['counterfactual', 90], ['coverage bias', 1], ['crash blossom', 1], ['dataset', 484699], ['decision threshold', 140], ['dense feature', 141], ['dense layer', 3566], ['depth', 24003], ['discriminator', 2064], ['disparate impact', 12], ['disparate treatment', 4], ['downsampling', 2474], ['DQN', 392], ['dropout regularization', 308], ['dynamic model', 22], ['early stopping', 5951], ["earth mover's distance", 10], ['encoder', 17207], ['ensemble', 34413], ['environment', 226238], ['equality of opportunity', 10], ['false negative', 2401], ['false negative rate', 293], ['false positive', 9051], ['false positive rate', 6585], ['feature', 178453], ['feature cross', 81], ['feature engineering', 45488], ['feature set', 2192], ['feature spec', 12], ['feature vector', 2470], ['federated learning', 56], ['feedback loop', 104], ['feedforward neural network', 206], ['few-shot learning', 141], ['fine tuning', 5078], ['forget gate', 138], ['full softmax', 1], ['fully connected layer', 3128], ['GAN', 18808], ['generalization', 3602], ['generalization curve', 1], ['generalized linear model', 301], ['generative adversarial network', 332], ['generative model', 446], ['generator', 14414], ['GPT', 1231], ['Generative Pre-trained Transformer', 4], ['gradient', 28132], ['group attribution bias', 3], ['hashing', 1142], ['heuristic', 1629], ['hidden layer', 5212], ['holdout data', 232], ['hyperparameter', 12221], ['hyperplane', 780], ['image recognition', 1614], ['implicit bias', 17], ['incompatibility of fairness metrics', 0], ['individual fairness', 1], ['inference', 20414], ['in-group bias', 1], ['input layer', 3290], ['interpretability', 1433], ['inter-rater agreement', 256], ['intersection over union', 574], ['iteration', 13853], ['Support Vector Machines', 8161], ['SVM', 29772], ['k-means', 8838], ['loss', 109005], ['regularization', 9860], ['loss', 109005], ['regularization', 9860], ['layer', 48864], ['learning rate', 17170], ['least squares regression', 263], ['linear model', 10590], ['log-odds', 1207], ['Long Short-Term Memory', 899], ['LSTM', 22999], ['machine learning', 215996], ['majority class', 1604], ['Markov decision process', 55], ['matplotlib', 216481], ['matrix factorization', 749], ['Mean Absolute Error', 5731], ['Mean Squared Error', 6765], ['metric', 46234], ['meta-learning', 144], ['metrics', 86943], ['mini-batch', 1334], ['mini-batch stochastic gradient descent', 30], ['minimax loss', 6], ['minority class', 1707], ['MNIST', 20706], ['modality', 555], ['model', 392197], ['model capacity', 63], ['model parallelism', 48], ['model training', 10804], ['Momentum', 8178], ['multi-class classification', 1568], ['multi-class logistic regression', 48], ['multi-head self-attention', 44], ['multimodal model', 9], ['multinomial classification', 49], ['multinomial regression', 44], ['NaN trap', 1], ['natural language understanding', 199], ['negative class', 1165], ['neural network', 33917], ['neuron', 3915], ['N-gram', 1098], ['NLU', 230], ['noise', 20785], ['non-response bias', 14], ['nonstationarity', 31], ['normalization', 23388], ['novelty detection', 69], ['numerical data', 4805], ['NumPy', 405399], ['objective', 36160], ['objective function', 1693], ['offline inference', 13], ['one-hot encoding', 15674], ['one-shot learning', 187], ['online inference', 14], ['optimizer', 67670], ['out-group homogeneity bias', 1], ['outlier detection', 2448], ['outliers', 31255], ['output layer', 6673], ['overfitting', 31023], ['oversampling', 4010], ['parameter', 40694], ['partial derivative', 182], ['participation bias', 6], ['partitioning strategy', 4], ['perceptron', 6098], ['performance', 61065], ['pipeline', 26508], ['pooling', 9479], ['positive class', 1789], ['prediction', 263044], ['prediction bias', 59], ['predictive parity', 0], ['predictive rate parity', 1], ['preprocessing', 58960], ['pre-trained model', 5216], ['prior belief', 42], ['probabilistic regression model', 0], ['proxy', 2861], ['Q-function', 54], ['Q-learning', 755], ['random policy', 13], ['rank', 34895], ['recall', 20808], ['recommendation system', 1959], ['Rectified Linear Unit', 544], ['ReLU', 46033], ['recurrent neural network', 1323], ['regression model', 20493], ['regularization', 9860], ['representation', 15406], ['re-ranking', 187], ['reward', 3176], ['ridge regularization', 507], ['RNN', 9168], ['ROC Curve', 7549], ['Root Mean Squared Error', 3249], ['RMSE', 23579], ['rotational invariance', 9], ['sampling bias', 133], ['SavedModel', 186], ['Saver', 1542], ['scaling', 28500], ['scikit-learn', 18323], ['scoring', 40584], ['selection bias', 215], ['self-supervised learning', 170], ['self-training', 162], ['semi-supervised learning', 868], ['sensitive attribute', 2], ['sentiment analysis', 10536], ['sequence model', 435], ['sequence-to-sequence task', 1], ['serving', 1157], ['sigmoid function', 1968], ['similarity measure', 260], ['size invariance', 3], ['sketching', 37], ['softmax', 33312], ['sparse feature', 118], ['sparse representation', 90], ['sparse vector', 90], ['sparsity', 1056], ['spatial pooling', 42], ['squared hinge loss', 17], ['squared loss', 145], ['staged training', 3], ['state', 47753], ['state-action value function', 2], ['static model', 16], ['stationarity', 1075], ['step', 84496], ['step size', 1339], ['stochastic gradient descent', 3856], ['stride', 5870], ['structural risk minimization', 4], ['subsampling', 645], ['supervised machine learning', 1068], ['synthetic feature', 109], ['tabular Q-learning', 5], ['target', 102445], ['target network', 67], ['temporal data', 257], ['Tensor', 28751], ['TensorBoard', 3357], ['TensorFlow', 69089], ['termination condition', 14], ['keras', 77360], ['time series', 29887], ['TPU', 15157], ['training', 211067], ['trajectory', 1532], ['transfer learning', 10690], ['Transformer', 8406], ['translational invariance', 25], ['trigram', 1869], ['true negative', 1888], ['true positive rate', 6556], ['underfitting', 3410], ['undersampling', 2301], ['unidirectional', 88], ['unidirectional language model', 0], ['unlabeled example', 6], ['unsupervised machine learning', 463], ['upweighting', 11], ['user matrix', 25], ['validation', 105126], ['validation set', 19023], ['vanishing gradient problem', 304], ['Wasserstein loss', 24], ['Weighted Alternating Least Squares', 2], ['wide model', 30], ['word embedding', 2682]]

    for count, k in enumerate(keywords_list[len(result):]):
        k = k.strip()
        qtd = g.agent_search(k, url="https://www.kaggle.com/search?q=", count=count)
        result.append([k, qtd])



    df = pd.DataFrame(result, columns=['keyword', 'qtd'])
    df.to_csv('output.csv', index=False)

