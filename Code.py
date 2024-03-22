import numpy as np #Importo las librerias que voy a usar fuera de la clase
import pandas as pd
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  weibull_min
import matplotlib.pyplot as plt

class Anla:
    def QAnla(df):
        # Establecer la primera columna como el índice del DataFrame
        df.set_index(df.columns[0], inplace=True)

        # Convertir el índice a tipo de datos de fecha
        df.index = pd.to_datetime(df.index)

        # Calcular el promedio de los caudales cada 7 días
        promedio_7_dias = df.rolling(7).mean()
        # Calcular el mínimo de caudal por año
        minimos_por_ano = promedio_7_dias.groupby(promedio_7_dias.index.year)['cuenca-base'].min()

        # Crear un DataFrame con los mínimos de caudal por año
        df_minimos_por_ano = pd.DataFrame(minimos_por_ano)
        #nombre_archivo = 'datos.xlsx'  # Nombre del archivo de Excel
        #df_minimos_por_ano.to_excel(nombre_archivo)
        #Sacar la distribución que se le acerca
        #Normal
        def Fn(x, mun, stdn):
            f = (1 / (stdn * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mun) / stdn) ** 2)
            return f
        mun, stdn = norm.fit(df_minimos_por_ano['cuenca-base'])
        vesn=[]
        for x in df_minimos_por_ano['cuenca-base']:
            n = Fn(x, mun, stdn)
            vesn.append(n)
        # Calcula la correlación entre los datos observados y los valores esperados
        corn= np.corrcoef(df_minimos_por_ano['cuenca-base'], vesn)[0, 1]
        #Lognormal
        def Flogn(x, mu, sigma):
            return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)

        pln= lognorm.fit(df_minimos_por_ano['cuenca-base'])
        vesln=[]
        for x in df_minimos_por_ano['cuenca-base']:
            n = Flogn(x, pln[0], pln[2])
            vesln.append(n)
        corln = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesln)[0, 1]
        #Gumbell
        def gum(x, mu, beta):
            return (1 / beta) * np.exp(-(x - mu) / beta - np.exp(-(x - mu) / beta))
        mug, beta = gumbel_r.fit(df_minimos_por_ano['cuenca-base'])
        vesg=[]
        for x in df_minimos_por_ano['cuenca-base']:
            n = gum(x, mug, beta)
            vesg.append(n)
        corg = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesg)[0, 1]
        #Pearson tipo 3
        a1, m, s = pearson3.fit(df_minimos_por_ano['cuenca-base'])
        vesp=[]
        for x in df_minimos_por_ano['cuenca-base']:
            n = pearson3.pdf(x, a1, m, s)
            vesp.append(n)
        corp = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesp)[0, 1]
        #Weibull
        shape, loc, scale = weibull_min.fit(df_minimos_por_ano['cuenca-base'])
        vesw=[]
        for x in df_minimos_por_ano['cuenca-base']:
            n = weibull_min.pdf(x, shape, loc, scale)
            vesw.append(n)
        corw = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesw)[0, 1]
        #Sacar el mayor de las correlaciones y sacar en 10 años cuanto es el 7Q10
        TR = 10
        pe = 1 - (1 / TR)
        def mc(a, b, c, d,e):
            maximo = max(a, b, c, d,e)  # Encuentra el máximo entre los tres valores
            if a == maximo:  # Comprueba si a es el máximo
                return 1
            elif b == maximo:  # Comprueba si b es el máximo
                return 2
            elif c == maximo:  # Si no es a ni b, entonces c es el máximo
                return 3
            elif d == maximo:  # Si no es a ni b, entonces c es el máximo
                return 4
            else:
                return 5


        bin= mc(np.abs(corn),np.abs(corln), np.abs(corg), np.abs(corp),np.abs(corw))

        if bin == 1 :
            S7Q10 = norm.ppf(pe, loc=mun, scale=stdn)#84.08149197181189
        elif bin ==2 :
            S7Q10 = lognorm.ppf(pe, pln[2],pln[2],pln[0])#84.08149197181189
        elif bin == 3 :
            S7Q10 = gumbel_r.ppf(pe, loc=mug, scale=beta)
        elif bin == 4 :
            S7Q10 = pearson3.ppf(pe, a1,loc=m, scale=s)
        else:
            S7Q10 = weibull_min.ppf(pe, shape, loc, scale)
        print(S7Q10)

        #Calcular

    df = pd.read_csv("est_25027400_limpia.csv")
    x=QAnla(df)
    #0.003517763085733081
    #85.33220893006306
