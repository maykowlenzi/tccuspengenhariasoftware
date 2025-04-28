import nltk
from nltk.stem import RSLPStemmer
import pandas as pd
import unicodedata


MOSTRAR_DETALHES = False


class TrainingData():

    def __init__(self, nome_arq):
        # Planilha usada definida em nlp_frases para teste ou em Constantes.py "DadosTreinamentoChatbotPort.xls"
        # Steeming [redução de palavras] para portugues - reduz palavra para o mínimo dela. Exemplo: vendemos e vendi se transformam para vend
        self.stemmer = RSLPStemmer()
        # Carrega frases de treinamento de excel
        self.excel_training_data = self.load_file_training(nome_arq, 'trainning')

        self.training_data = self.save_dic_training(self.excel_training_data)

        # Inicializa as variáveis que vão guardar as classes e classes com palavras
        self.corpus_words = {}
        # Corpus Words vai dicionário dentro de dicionario: [classe]: [palavra stemm] número de ocorrências dentro da amostra de treinamento
        self.class_words = {}
        # class words vai conter as palavras de intencao e quais sao as bases pra ela. Ou seja: que palavras estao mais alinhadas a palavra de intencao
        # transforma uma lista em set (de itens unicos) e entao lista novamente - remove duplicados
        self.classes = list(set([a['class'] for a in self.training_data]))
        for c in self.classes:
            #  prepara uma lista de palavras para cada classe
            self.class_words[c] = []

        # loop por cada sentenca em  training data
        for data in self.training_data:
            sentence1 = self.remove_accented_chars(data['sentence'])
            # retira stop words
            sentence = self.remove_stopwords_v2(sentence1)
            # tokeniza cada sentenca em palavras
            frase = self.tokenize(sentence)

            class_name = data['class']
            if class_name not in list(self.corpus_words.keys()):
                self.corpus_words[class_name] = {}

            for word in frase:

                if word not in ["?", "'s"]:
                    # faz o stemming e lowwercase para cada palavra
                    stemmed_word = self.stemmer.stem(word.lower())
                    if stemmed_word not in list(self.corpus_words[class_name].keys()):
                        self.corpus_words[class_name][stemmed_word] = 1

                    else:
                        self.corpus_words[class_name][stemmed_word] += 1


        # PRINT DE SUPORTE PARA MOSTAR OS DADOS - só para apoio no treinamento
        if MOSTRAR_DETALHES:
            print("\n\n\n NOVO self.corpus_words em INIT")
            print("___________________________________________________________")
            for nova_classe in self.corpus_words:
                print("Classe >>> {}".format(nova_classe))
                for palavra in self.corpus_words[nova_classe]:
                    quantidade = self.corpus_words[nova_classe][palavra]
                    print("{} foi contabilizada {} vezes".format(palavra, quantidade))

                print("---------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("----------------------------------------------------------------------------------------")
            print(self.corpus_words)



    def tokenize(self, sentence):
        sentence = sentence.lower()
        sentence = nltk.word_tokenize(sentence)
        return sentence

    def stemming(self, sentence):
        stemmer = RSLPStemmer()
        phrase = []
        for word in sentence:
            phrase.append(stemmer.stem(word.lower()))
        return phrase

    def remove_accented_chars(self, sentence):
        new_sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return new_sentence

    def remove_stopwords_v2(self, sentence):
        stopwords = nltk.corpus.stopwords.words('portuguese')
        # Set de palavras que podem ser adicionadas no stopwords

        for word in ['muito', 'mês', 'mais']:
            if word in stopwords:
                stopwords.remove(word)

        cifrao = ['r$', 'R$']
        for cifra in cifrao:
            string_sem_r = sentence.replace(cifra, "")

        novas_stopwords = ['?', '!', ',', '.', 'onde', ';', ':']

        for item in novas_stopwords:
            stopwords.append(item)

        tokenized_sentence = self.tokenize(string_sem_r)

        phrase = []
        for word in tokenized_sentence:
            if word not in stopwords:
                phrase.append(word)

        retorno = ""
        for word in phrase:
            retorno = retorno + word + " "

        return retorno

    def load_file_training(self, file, sheet):
        excel_data_df = pd.read_excel(file, sheet_name=sheet, index_col=None, usecols=['class', 'sentence'])
        return excel_data_df

    def save_dic_training(self, excel_data):
        data_dic = []
        for i in excel_data.index:
            class_data = excel_data['class'][i]
            sentence_data = excel_data['sentence'][i]
            data_dic.append({"class": class_data, "sentence": sentence_data})
        return data_dic


    def calculate_class_score(self, sentence, class_name, show_details=True):
        score = 0
        sentence = self.tokenize(sentence)
        sentence = self.stemming(sentence)

        if MOSTRAR_DETALHES:
            print("---------8------8-----8------8-------8-------8-------8------8------8------8--")
            print("Analise de sentenca para a classe {}".format(class_name))
            print("as palavras dentro de corpus words[class_name]>>>", (self.corpus_words[class_name]))
        for word in sentence:
            if word in self.corpus_words[class_name]:
                score = score + 5 * self.corpus_words[class_name][word]

            else:
                score = score - 1

        score = score / len(sentence) if sentence else 0  # normalização

        return score

    def calculate_score(self, sentence):
        high_score = 0
        high_class = None
        tabela_scores = {}
        for c in list(self.corpus_words.keys()):
            score = self.calculate_class_score(sentence,c, show_details=True)
            if score > 0:
                tabela_scores[c]=score
            if score > high_score:
                high_class = c
                high_score = score

        #return high_class, high_score, tabela_scores
        return high_class or "NÃO_CLASSIFICADO", high_score, tabela_scores


