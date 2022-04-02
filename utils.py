
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import random

class VaR:

    def __init__(self):

        self.data       = df.dropna()
        self.rendements = df["Return"]
        self.data_bruite       = df_1.dropna()
        self.rendements_bruites = df_1["Return_bruite"]
        self.rendements_bruites_50 = df_50["Return_bruite"]
        self.rendements_bruites_100 = df_100["Return_bruite"]
        self.simulation = 10000000
        self.x = 0.75
        self.train = self.rendements[: int(self.rendements.shape[0]*self.x)]
        self.test = self.rendements[int(self.rendements.shape[0]*self.x) : self.data.shape[0]]
        self.train_df = self.data[: int(self.data.shape[0]*self.x)]
        self.test_df = self.data[int(self.data.shape[0]*self.x) : self.data.shape[0]]
        
    def emp_VaR(self, subset = "dataframe", p = 0.99, amplitude = 1):
        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        rendements_ordonnes = dataframe.sort_values(ascending = True)
        emp_VaR = np.percentile(rendements_ordonnes, 100 - 100*p)
        print("VaR "+str(int(100*p))+"% empirique dans le " + str(subset) + " est", round(100*emp_VaR, 3), "%")
        return emp_VaR

    def emp_VaR_(self, subset = "dataframe", p = 0.99, amplitude = 1):
        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        rendements_ordonnes = dataframe.sort_values(ascending = True)
        emp_VaR = np.percentile(rendements_ordonnes, 100 - 100*p)
        #print("VaR 99% empirique dans le " + str(subset) + " est", round(100*emp_VaR, 3), "%")
        return emp_VaR

    def para_VaR(self, subset = "dataframe", p = 0.99, amplitude = 1):
        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        rendements_ordonnes = dataframe.sort_values(ascending = True)
        ecart_type = np.std(rendements_ordonnes)
        mu = np.mean(rendements_ordonnes)
        para_VaR = ecart_type*norm.ppf(1-p) + mu
        print("VaR "+str(int(100*p))+"% paramétrique dans le " + str(subset) + " est", round(100*para_VaR , 3), "%")
        return para_VaR

    def para_VaR_(self, subset = "dataframe", p = 0.99, amplitude = 1):
        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        rendements_ordonnes = dataframe.sort_values(ascending = True)
        ecart_type = np.std(rendements_ordonnes)
        mu = np.mean(rendements_ordonnes)
        para_VaR = ecart_type*norm.ppf(1-p) + mu
        #print("VaR 99% empirique dans le " + str(subset) + " est", round(100*para_VaR , 3), "%")
        return para_VaR
 
    def nonpara_VaR(self, subset = "dataframe", p = 0.99, amplitude = 1):

        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        np.random.seed(20)
        sigma = np.std(dataframe)
        mu = np.mean(dataframe)
        aapl = np.random.normal(mu, sigma, self.simulation)
        nonpara_VaR = np.percentile(aapl, 100 - 100*p)
        print("VaR "+str(int(100*p))+"% non paramétrique dans le " + str(subset) + " est", round(100*nonpara_VaR , 3), "%")
        return nonpara_VaR

    def nonpara_VaR_(self, subset = "dataframe", p = 0.99, amplitude = 1):

        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
        else:
            dataframe = self.rendements
        np.random.seed(20)
        sigma = np.std(dataframe)
        mu = np.mean(dataframe)
        aapl = np.random.normal(mu, sigma, self.simulation)
        nonpara_VaR = np.percentile(aapl, 100 - 100*p)
        #print("VaR "+str(int(100*p))+"% non paramétrique dans le " + str(subset) + " est", round(100*nonpara_VaR , 3), "%")
        return nonpara_VaR

    def estimateur_pickands(self, k, subset = 'dataframe', amplitude = 1):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        elif subset == 'test':
            dataframe = self.test
            data = self.test_df
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
                data = self.data_bruite
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
                data = self.data_bruite
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
                data = self.data_bruite
        else:
            dataframe = self.rendements
            data = self.data
        rendements = dataframe.sort_values(ascending = False)
        x_k  = rendements[data.shape[0] -   k+1]
        x_2k = rendements[data.shape[0] - 2*k+1]
        x_4k = rendements[data.shape[0] - 4*k+1]
        terme = ((x_k - x_2k)/(x_2k - x_4k))
        estimateur_pk = (1/np.log(2))*np.log(terme)
        print("L'estimateur de Pickands dans le "+ str(subset)+  " pour k = " + str(k) + " est : " + str(round(estimateur_pk, 3)))
        return estimateur_pk
        
    def estimateur_pickands_(self, k, subset = 'dataframe', amplitude = 1):
        if subset == "train":
            dataframe = self.train
            data = self.train_df
        elif subset == 'test':
            dataframe = self.test
            data = self.test_df
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
                data = self.data_bruite
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
                data = self.data_bruite
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
                data = self.data_bruite
        else:
            dataframe = self.rendements
            data = self.data
        rendements = dataframe.sort_values(ascending = False)
        x_k  = rendements[data.shape[0] -   k+1]
        x_2k = rendements[data.shape[0] - 2*k+1]
        x_4k = rendements[data.shape[0] - 4*k+1]
        terme = ((x_k - x_2k)/(x_2k - x_4k))
        estimateur_pk = (1/np.log(2))*np.log(terme)
        return estimateur_pk

    def evt_VaR(self, p= 0.99, k = 5, subset = 'dataframe', amplitude = 1):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        elif subset == 'test':
            dataframe = self.test
            data = self.test_df
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
                data = self.data_bruite
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
                data = self.data_bruite
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
                data = self.data_bruite
        else:
            dataframe = self.rendements
            data = self.data

        rendements = dataframe.sort_values(ascending = False)
        x_k  = rendements[dataframe.shape[0] -   k+1]
        x_2k = rendements[dataframe.shape[0] - 2*k+1]
        estimateur_pk = self.estimateur_pickands(k, subset = 'dataframe')
        numerateur = (k/ (len(data) *(1 - p)))**estimateur_pk - 1
        denominateur = 1 - 2**(-estimateur_pk)
        quotient = numerateur/denominateur

        evt_VaR = quotient * (x_k - x_2k) + x_k

        print("La VaR "+str(int(100*p))+"% avec la méthode EVT sur le " +str(subset) + " pour k = " + str(k) + " et n = " + str(len(data)) + " est : " +str(round(100*evt_VaR,3))+ "%")
        return evt_VaR

    def evt_VaR_(self, p= 0.99, k =5, subset = 'dataframe', amplitude = 1):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        elif subset == 'test':
            dataframe = self.test
            data = self.test_df
        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
                data = self.data_bruite
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
                data = self.data_bruite
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
                data = self.data_bruite
        else:
            dataframe = self.rendements
            data = self.data

        rendements = dataframe.sort_values(ascending = False)
        x_k  = rendements[dataframe.shape[0] -   k+1]
        x_2k = rendements[dataframe.shape[0] - 2*k+1]
        estimateur_pk = self.estimateur_pickands_(k, subset = 'dataframe')
        numerateur = (k/ (len(data) *(1 - p)))**estimateur_pk - 1
        denominateur = 1 - 2**(-estimateur_pk)
        quotient = numerateur/denominateur
        evt_VaR = quotient * (x_k - x_2k) + x_k
        print("La VaR "+str(int(100*p))+"% avec la méthode EVT sur le " +str(subset) + " pour k = " + str(k) + " et n = " + str(len(data)) + " est  : " +str(round(100*evt_VaR,3))+ "%")
        return evt_VaR
        
    def _evt_VaR_(self, p= 0.99, k =5, subset = 'dataframe', amplitude = 1):

        if subset == "train":
            dataframe = self.train
            data = self.train_df

        elif subset == 'test':
            dataframe = self.test
            data = self.test_df

        elif subset == "dataframe bruité":
            if amplitude == 1:
                dataframe = self.rendements_bruites
                data = self.data_bruite
            elif amplitude == 50:
                dataframe = self.rendements_bruites_50
                data = self.data_bruite
            elif amplitude == 100:
                dataframe = self.rendements_bruites_100
                data = self.data_bruite
        else:
            dataframe = self.rendements
            data = self.data

        rendements = dataframe.sort_values(ascending = False)
        x_k  = rendements[dataframe.shape[0] -   k+1]
        x_2k = rendements[dataframe.shape[0] - 2*k+1]
        estimateur_pk = self.estimateur_pickands_(k, subset = 'dataframe')
        numerateur = (k/ (len(data) *(1 - p)))**estimateur_pk - 1
        denominateur = 1 - 2**(-estimateur_pk)
        quotient = numerateur/denominateur
        evt_VaR = quotient * (x_k - x_2k) + x_k

        return evt_VaR

    def index_Leadbetter(self, seuil, b):

        k = int(len(self.rendements)/b)
        pertes = -self.rendements
        franchissements = sum( pertes > seuil)
        franchissements_buckets = []
        for k_i in np.arange(k+1):
            a = (k_i - 1)*b
            z = k_i * b
            m = pertes[a - 1 : z]
            if len(m) != 0:
                franchissements_buckets.append(max(pertes[ a - 1 : z] > seuil))
        index_Leadbetter = sum(franchissements_buckets)/franchissements
        print("La valeur de l'index extremal de Leadbetter pour " + str(b) + " blocs et un seuil de pertes de " + str(seuil) + " est de : " + str(round(index_Leadbetter, 3)))
        return index_Leadbetter



class ExpectedShortfall:

    def __init__(self):

        self.data       = df.dropna()
        self.rendements = df["Return"]
        self.data_bruite       = df_1.dropna()
        self.rendements_bruites = df_1["Return_bruite"]
        self.rendements_bruites_50 = df_50["Return_bruite"]
        self.rendements_bruites_100 = df_100["Return_bruite"]
        self.simulation = 10000000
        self.x = 0.75
        self.train = self.rendements[: int(self.rendements.shape[0]*self.x)]
        self.test = self.rendements[int(self.rendements.shape[0]*self.x) : self.data.shape[0]]
        self.train_df = self.data[: int(self.data.shape[0]*self.x)]
        self.test_df = self.data[int(self.data.shape[0]*self.x) : self.data.shape[0]]

    def split(self):
        train = self.rendements[: int(self.rendements.shape[0]*self.x)]
        test = self.rendements[int(self.rendements.shape[0]*self.x) : self.data.shape[0]]
        return train, test

    def emp_ES(self, p = 0.99, subset = 'dataframe', amplitude = 1):
        emp_VaR = VaR().emp_VaR_(p=p, subset = subset, amplitude = amplitude)
        emp_ES = np.mean(self.data[self.rendements <= emp_VaR]["Return"])
        print("L'Expected Shortfall empirique à " +str(int(100*p))+ "% est : ", round(100*emp_ES, 3), "%")
        return emp_ES

    def param_ES(self, p= 0.99, subset = 'dataframe', amplitude = 1):
        para_VaR = VaR().para_VaR_(p=p, subset = subset, amplitude = amplitude)
        param_ES = np.mean(self.data[self.rendements <= para_VaR]["Return"])
        print("L'Expected Shortfall paramétrique à " +str(int(100*p))+ "% est : ", round(100*param_ES, 3), "%")
        return param_ES

    def nonparam_ES(self, p= 0.99, subset = 'dataframe', amplitude = 1):
        nonpara_VaR = VaR().nonpara_VaR_(p=p, subset = subset, amplitude = amplitude)
        nonparam_ES = np.mean(self.data[self.rendements <= nonpara_VaR]["Return"])
        print("L'Expected Shortfall non paramétrique à " +str(int(100*p))+ "% est : ", round(100*nonparam_ES, 3), "%")
        return nonparam_ES

    def pickands_ES(self, p= 0.99, subset = 'dataframe', amplitude = 1):
        evt_VaR = VaR()._evt_VaR_(p=p, subset = subset, amplitude = amplitude)
        pickands_ES = np.mean(self.data[self.rendements <= evt_VaR]["Return"])
        print("L'Expected Shortfall à " +str(int(100*p))+ "% avec la méthode EVT est : ", round(100*pickands_ES, 3), "%")
        return pickands_ES
        
    def _MuandSigma(self):

        ecart_type_train = np.std(self.train, axis = 0)
        ecart_type_test = np.std(self.test, axis = 0)
        mu_train = np.mean(self.train,axis = 0)
        mu_test = np.mean(self.test, axis = 0)
        return ecart_type_train, ecart_type_test, mu_train, mu_test

    def backtest_emp_VaR(self, subset = 'train'):

        if subset == "train":
            dataframe = self.train
        elif subset == 'test':
            dataframe = self.test
        rendements_ordonnes = dataframe.sort_values(ascending = True)
        backtest_emp_VaR = np.percentile(rendements_ordonnes, 1)
        print("VaR 99% empirique dans le "+ str(subset) +" est : ", round(100*backtest_emp_VaR, 3), "%")
        return backtest_emp_VaR

    def backtest_para_VaR(self, subset = 'train', p = 0.99):

        ecart_type_train, ecart_type_test, mu_train, mu_test = self._MuandSigma()
        
        if subset == "train":
            ecart_type =  ecart_type_train
            mu = mu_train
        else :
            ecart_type = ecart_type_test
            mu = mu_test
            
        backtest_para_VaR = ecart_type*norm.ppf(1-p) + mu
        print("VaR 99% paramétrique dans le "+ str(subset) +" est :", round(100*backtest_para_VaR , 3), "%")

        return backtest_para_VaR

    def _backtest_para_VaR(self, subset = 'train', p = 0.99):

        ecart_type_train, ecart_type_test, mu_train, mu_test = self._MuandSigma()
        
        if subset == "train":
            ecart_type =  ecart_type_train
            mu = mu_train
        else :
            ecart_type = ecart_type_test
            mu = mu_test
            
        backtest_para_VaR = ecart_type*norm.ppf(1-p) + mu

        return backtest_para_VaR

    def backtest_nonparam_VaR(self, subset = 'train', p = 0.99):
        np.random.seed(20)
        ecart_type_train, ecart_type_test, mu_train, mu_test = self._MuandSigma()
        if subset == "train":
            ecart_type =  ecart_type_train
            mu = mu_train
        else :
            ecart_type = ecart_type_test
            mu = mu_test
        aapl = np.random.normal(mu, ecart_type, self.simulation)
        backtest_nonpara_VaR  = np.percentile(aapl, 100-100*p)
        print("VaR 99% non paramétrique dans le "+ str(subset) +" est :", round(100*backtest_nonpara_VaR , 3), "%")

        return backtest_nonpara_VaR

    def _backtest_nonparam_VaR(self, subset = 'train', p = 0.99):
        np.random.seed(20)
        ecart_type_train, ecart_type_test, mu_train, mu_test = self._MuandSigma()
        if subset == "train":
            ecart_type =  ecart_type_train
            mu = mu_train
        else :
            ecart_type = ecart_type_test
            mu = mu_test
        aapl = np.random.normal(mu, ecart_type, self.simulation)
        backtest_nonpara_VaR  = np.percentile(aapl, 100-100*p)

        return backtest_nonpara_VaR
        
    # Backtest des ES
    def backtest_emp_ES(self, subset = 'train'):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        elif subset == 'test':
            dataframe = self.test
            data = self.test_df

        backtest_emp_ES = np.mean(data[dataframe <= VaR().emp_VaR_(subset = subset)]["Return"])

        print("L'Expected Shortfall empirique à 99% dans le " +str(subset) + " est : ", round(100*backtest_emp_ES, 3), "%")

        return backtest_emp_ES

    def backtest_param_ES(self, subset = 'train'):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        else: 
            dataframe = self.test
            data = self.test_df

        backtest_param_ES = np.mean(data[dataframe <= self._backtest_para_VaR(subset = subset)]["Return"])

        print("L'Expected Shortfall paramétrique à 99% dans le " +str(subset) + " est : ", round(100*backtest_param_ES, 3), "%")
        return backtest_param_ES

    def backtest_nonparam_ES(self, subset = 'train'):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        else:
            dataframe = self.test
            data = self.test_df

        backtest_nonparam_ES = np.mean(data[dataframe <= self._backtest_nonparam_VaR(subset = subset)]["Return"])
        print("L'Expected Shortfall non paramétrique à 99% dans le " +str(subset) + " est : ", round(100*backtest_nonparam_ES, 3), "%")
        return backtest_nonparam_ES

    def backtest_pickands_ES(self, subset = 'train'):

        if subset == "train":
            dataframe = self.train
            data = self.train_df
        else :
            dataframe = self.test
            data = self.test_df
        backtest_pickands_ES = np.mean(data[dataframe <= VaR()._evt_VaR_(subset = subset, p=0.99, k=5 )]["Return"])
        print("L'Expected Shortfall avec la méthode EVT à 99% dans le " +str(subset) + " est : ", round(100*backtest_pickands_ES, 3), "%")
        return backtest_pickands_ES




class EstUncertaintly:

    def __init__(self):

        self.data       = df.dropna()
        self.rendements = df["Return"]
        self.simulation = 10000000
        self.percentile = norm.ppf(1e-2)

    def para_VaR_parametres(self, n = 25):
        random.seed(20)
        mu_list = []
        sigma_list = []
        para_VaR_list = []
        percentile = norm.ppf(1e-2)
        for i in range(int(self.simulation/100)):
            rendements_ = random.choices(self.rendements, k = n )
            mu = np.mean(rendements_)
            mu_list.append(mu)
            std = np.std(rendements_)
            sigma_list.append(std)
            para_VaR_list.append(percentile*std + mu)
            
        return para_VaR_list, mu_list, sigma_list

    def para_VaR_distribution(self, data, bins= 25):

        if data == mu_list:
            title = "La Moyenne"
        elif data == sigma_list:
            title = "L'Écart Type"
        else:
            title = "La VaR Paramétrique"

        plt.figure(figsize= (12, 5))
        from scipy.stats import skewnorm
        
        mu, std = norm.fit(data)

        plt.hist(data, bins = bins, density=True, alpha=0.6, color='red')

        xmin, xmax = plt.xlim()
        X = np.linspace(xmin, xmax)
        
        plt.plot(X, norm.pdf(X, mu, std), color= 'green', label= "Distribution Normale")
        plt.plot(X, skewnorm.pdf(X, *skewnorm.fit(data)), color= 'black', label= "Distribution lissée de " + str(title))
        
        mu, std = norm.fit(data)
        sk = scipy.stats.skew(data)
        
        print("--- " + str(title))
        print("{} - Moments mu: {}, sig: {}, sk: {}".format(title, round(mu, 4), round(std, 4), round(sk, 4)))
        plt.ylabel("Fréquence", rotation= 90)
        plt.title(title)
        plt.legend()
        plt.show()

    def pick_VaR_distribution(self, data, title, bins= 25):
    
        plt.figure(figsize= (12, 5))
        from scipy.stats import skewnorm
        
        mu, std = norm.fit(data)

        plt.hist(data, bins = bins, density=True, alpha=0.6, color='red')

        xmin, xmax = plt.xlim()
        X = np.linspace(xmin, xmax)

        plt.plot(X, skewnorm.pdf(X, *skewnorm.fit(data)), color= 'black', label= "Distribution lissée de " + str(title))
        
        mu, std = norm.fit(data)
        sk = scipy.stats.skew(data)
        
        print("--- " + str(title))
        print("{} - Moments mu: {}, sig: {}, sk: {}".format(title, round(mu, 4), round(std, 4), round(sk, 4)))
        plt.ylabel("Fréquence", rotation= 90)
        plt.title(title)
        plt.legend()
        plt.show()

    def gaussian_evt_var(self, data):
        tirage=[]
        pick_est_=[]
        evt_VaR_=[]
        k=5
        mu = np.mean(data)
        sigma = np.std(data)

        rendements = np.random.normal(mu, sigma, 100000)
        random.seed(20)
        for i in range(1,10000):

            rendements_ = random.choices(rendements, k=500)
            rendements_ = sorted(rendements_,reverse=True)
            n= len(rendements_)
            p=0.99
            x_k  = rendements_[n -k+1]
            x_2k = rendements_[n -2*k+1]
            x_4k = rendements_[n -4*k+1]

            evt_VaR = (1/np.log(2)) * np.log((x_k - x_2k) / (x_2k - x_4k))

            VaRpickand = ((((k/(n*(1-p)))**(evt_VaR))-1) / (1-2**(-evt_VaR)) ) *(x_k - x_2k) + x_k


            tirage.append(i)
            evt_VaR_.append(VaRpickand)
            pick_est_.append(evt_VaR)
            
        rand = pd.DataFrame(list(zip(tirage,pick_est_,evt_VaR_)), columns=['tirage','pick_est','evt_VaR'])
        self.pick_VaR_distribution(rand["pick_est"], "pick_est", bins= 50)
        self.pick_VaR_distribution(rand["evt_VaR"], "evt_VaR",bins= 50)

    def bandwith_kernel(self):
            mu= np.mean(df["Return"].values)
            sigma= np.std(df["Return"].values)
            stock_nonpara= []
            for i in range(1000):
                simulations = 1000
                sim_returns = np.random.normal(mu, sigma, simulations)
                nonparametric_quant= np.percentile(sim_returns, 1)
                stock_nonpara.append(nonparametric_quant)
                
            obs_dist= stock_nonpara

            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)

            ax.hist(obs_dist, bins=20, density=True, label='Données',
                    zorder=5, edgecolor='k', alpha=0.5)

            import statsmodels.api as sm
            for i in [0.0009, 0.0005, 0.0001, 0.00005]:
                kde = sm.nonparametric.KDEUnivariate(obs_dist)
                kde.fit(bw= i)
                ax.plot(kde.support, kde.density, lw=3, label='KDE avec h={}'.format(i), zorder=10)

            ax.legend(loc='best')
            ax.grid(True, zorder=-5)

class SpecUncertainty:

    def __init__(self):
        self.data       = df.dropna()
        self.rendements = df["Return"]
        self.simulation = 10000000
        self.percentile = norm.ppf(1e-2)
        self.data_bruite       = df_1.dropna()
        self.rendements_bruites = df_1["Return_bruite"]
        self.rendements_bruites_50 = df_50["Return_bruite"]
        self.rendements_bruites_100 = df_100["Return_bruite"]

    def diametre_VaR(self, p_ = 0.99, subset = "dataframe", amplitude = 1):
        print("--- VaR")
        emp_VaR = VaR().emp_VaR(subset = subset, p=p_, amplitude = amplitude)
        param_VaR = VaR().para_VaR(subset = subset, p=p_, amplitude = amplitude)
        nonpara_VaR = VaR().nonpara_VaR(subset = subset, p=p_, amplitude = amplitude)
        evt_VaR = VaR().evt_VaR_(k = 5, subset = subset, p=p_, amplitude = amplitude)
        VaR_list = [emp_VaR, param_VaR, nonpara_VaR, evt_VaR]
        VaR_max = max(VaR_list)
        VaR_min = min(VaR_list)
        print("--- Min/Max")
        print("VaR " + str(int(100*p_)) + "% maximale est : "+str(round(100*VaR_max, 3))+"%")
        print("VaR " + str(int(100*p_)) + "% minimale est : " +str(round(100*VaR_min, 3))+"%")
        diametre_VaR = VaR_max - VaR_min
        print("-> Diamètre VaR " + str(int(100*p_))+ "% : ", round(diametre_VaR, 5))
        return diametre_VaR

        
    def diametre_ES(self, p_ = 0.99, subset = "dataframe", amplitude = 1):
        print("--- ES")
        emp_ES = ExpectedShortfall().emp_ES(subset = subset, p = p_, amplitude = amplitude)
        param_ES = ExpectedShortfall().param_ES(subset = subset, p = p_, amplitude = amplitude)
        nonpara_ES = ExpectedShortfall().nonparam_ES(subset = subset, p = p_, amplitude = amplitude)
        pickands_ES = ExpectedShortfall().pickands_ES(subset = subset, p = p_, amplitude = amplitude)
        ES_list = [emp_ES, param_ES, nonpara_ES, pickands_ES]
        ES_max = max(ES_list)
        ES_min = min(ES_list)
        print("--- Min/Max")
        print("ES " + str(int(100*p_)) + "% maximale est : " +str(round(100*ES_max, 3))+"%")
        print("ES " + str(int(100*p_)) + "% minimale est : " +str(round(100*ES_min, 3))+"%")
        diametre_ES = ES_max - ES_min
        print("-> Diamètre ES " + str(int(100*p_)) + "% : " +str(round(diametre_ES, 5)))
        return diametre_ES

    def conclusion(self, q = "VaR", subset = "dataframe"):

        if q == "VaR":
            print("-------------------- VaR")
            for quant in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]:
                print("----------- p = " + str(quant))
                self.diametre_VaR_(p_ = quant, subset = subset)
        else:
            
            print("-------------------- ES")
            for quant in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]:
                print("----------- p = " + str(quant))
                self.diametre_ES_(p_ = quant, subset = subset)
  
    def bruit_(self):

        print("----- Rendements Initaux --------------- ")
        self.diametre_VaR(p_ = 0.99)
        self.diametre_ES(p_ = 0.99)
        print("\n")
        print("----- Rendements bruités --------------- ")
        for amp in [1, 50, 100]:
            print("-- Amplitude = " + str(amp))
            self.diametre_VaR_(p_ = 0.99, subset = "dataframe bruité", amplitude = amp)
  
    def impact_bruit(self):

        print("----- Rendements Initaux --------------- ")
        emp = VaR().emp_VaR()
        para = VaR().para_VaR()
        nonpara = VaR().nonpara_VaR()
        pick = VaR().evt_VaR_()
        print("\n")
        print("----- Rendements bruités --------------- ")
        for amp in [1, 50, 100]:
            print("-- Amplitude = " + str(amp))
            emp_1 = VaR().emp_VaR(subset = "dataframe bruité", amplitude = amp)
            para_1 = VaR().para_VaR(subset = "dataframe bruité", amplitude = amp)
            nonpara_1 = VaR().nonpara_VaR(subset = "dataframe bruité", amplitude = amp)
            pick_1 = VaR().evt_VaR_(subset = "dataframe bruité", amplitude = amp)
            delta_emp1 = round(100*(emp_1 - emp)/emp, 2)
            delta_para1 = round(100*(para_1 - para)/para, 2)
            delta_nonpara1 = round(100*(nonpara_1 - nonpara)/nonpara, 2)
            delta_pick1 = round(100*(pick_1 - pick)/pick, 2)
            print("Variation de la VaR empirique : " +str(delta_emp1) + "%")
            print("Variation de la VaR paramétrique : " +str(delta_para1) + "%")
            print("Variation de la VaR non paramétrique : "+str(delta_nonpara1)+ "%")
            print("Variation de la VaR EVT : "+str(delta_pick1) +"%")
            
        plt.figure(figsize=(15, 5))
        df["Return"].plot()
        plt.title(label = 'Return')
        plt.show()

        plt.figure(figsize=(15, 5))
        df_1["Return_bruite"].plot(color = "pink")
        plt.title(label = 'Return_bruite - Amplitude = 1')
        plt.show()

        plt.figure(figsize=(15, 5))
        df_50["Return_bruite"].plot(color = "orange")
        plt.title(label = 'Return_bruite - Amplitude = 50')
        plt.show()

        plt.figure(figsize=(15, 5))
        df_100["Return_bruite"].plot(color = "red")
        plt.title(label = 'Return_bruite - Amplitude = 100')
        plt.show()

    def diametre_ES_(self, p_ = 0.99, subset = "dataframe", amplitude = 1):
        #print("--- ES")
        emp_ES = ExpectedShortfall().emp_ES(subset = subset, p = p_, amplitude = amplitude)
        param_ES = ExpectedShortfall().param_ES(subset = subset, p = p_, amplitude = amplitude)
        nonpara_ES = ExpectedShortfall().nonparam_ES(subset = subset, p = p_, amplitude = amplitude)
        pickands_ES = ExpectedShortfall().pickands_ES(subset = subset, p = p_, amplitude = amplitude)
        ES_list = [emp_ES, param_ES, nonpara_ES, pickands_ES]
        ES_max = max(ES_list)
        ES_min = min(ES_list)
        print("--- Min/Max")
        print("ES " + str(int(100*p_)) + "% maximale est : " +str(round(100*ES_max, 3))+"%")
        print("ES " + str(int(100*p_)) + "% minimale est : " +str(round(100*ES_min, 3))+"%")
        diametre_ES = ES_max - ES_min
        print("-> Diamètre ES " + str(int(100*p_)) + "% : " +str(round(diametre_ES, 5)))
        return diametre_ES
        
    def diametre_VaR_(self, p_ = 0.99, subset = "dataframe", amplitude = 1):
        #print("--- VaR")
        emp_VaR = VaR().emp_VaR(subset = subset, p=p_, amplitude = amplitude)
        param_VaR = VaR().para_VaR(subset = subset, p=p_, amplitude = amplitude)
        nonpara_VaR = VaR().nonpara_VaR(subset = subset, p=p_, amplitude = amplitude)
        evt_VaR = VaR().evt_VaR_(k = 5, subset = subset, p=p_, amplitude = amplitude)
        VaR_list = [emp_VaR, param_VaR, nonpara_VaR, evt_VaR]
        VaR_max = max(VaR_list)
        VaR_min = min(VaR_list)
        print("--- Min/Max")
        print("VaR " + str(int(100*p_)) + "% maximale est : "+str(round(100*VaR_max, 3))+"%")
        print("VaR " + str(int(100*p_)) + "% minimale est : " +str(round(100*VaR_min, 3))+"%")
        diametre_VaR = VaR_max - VaR_min
        print("-> Diamètre VaR " + str(int(100*p_))+ "% : ", round(diametre_VaR, 5))
        return diametre_VaR