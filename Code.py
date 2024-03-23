import numpy as np #Importo las librerias que voy a usar fuera de la clase
import pandas as pd
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  weibull_min
import matplotlib.pyplot as plt

class Anla:
    def sieteQ10(df):
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



    def Q95(df) -> pd.DataFrame:


      data = estado.data
      if estado.h_umbrales['QB'] is None:
        u_qb = 2.33
        u_qtq = 2
      else:
        u_qb = estado.h_umbrales[0]
        u_qtq = estado.h_umbrales[1]
      # df es un dataframe temporar que se sobreescribe eventualmente
      df = pd.DataFrame()
      # set columns
      df = df.assign(Fecha=None, Min=None, Max=None, Mean=None)
      # se crea un dataframe en donde se calculan los minimos máximos y promedio por mes y año
      for i in range(data.index.min().year, data.index.max().year + 1):
        for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[data.index.year == i][data[data.index.year == i].index.month == j]
          row = [str(i) + str(-j), data_filter['cuenca-base'].min(), data_filter['cuenca-base'].max(),
                 data_filter['cuenca-base'].mean()]
          df.loc[len(df)] = row
      # data view
      df['Fecha'] = pd.to_datetime(df['Fecha'])
      df = df.set_index('Fecha')
      '''
      se crea listas con n valores, siendo n el número de años en el dataset, guardando el máximo y minimo por año
      '''
      mins = []
      maxs = []
      for i in range(df.index.min().year, df.index.max().year + 1):
        mins.append(df[df.index.year == i]['Min'].min())
        maxs.append(df[df.index.year == i]['Max'].max())
      mins = np.array(mins)
      maxs = np.array(maxs)
      mean_min = (mins.mean())
      std_min = (np.std(mins, ddof=1))  # desviacion estandar
      coef_variacion_min = std_min / mean_min
      mean_max = (maxs.mean())
      std_max = (np.std(maxs, ddof=1))
      alpha_min = 1.0079 * (coef_variacion_min ** -1.084)
      a_alpha_min = (-0.0607 * (coef_variacion_min ** 3)) + (0.5502 * (coef_variacion_min ** 2)) - (
                0.4937 * coef_variacion_min) + 1.003
      beta = (a_alpha_min / mean_min) ** (-1)
      # umbrales QTQ y QB
      #
      umbral_QTQ = beta * ((-np.log(1 - (1 / u_qtq))) ** (1 / alpha_min))
      umbral_Q10 = beta * ((-np.log(1 - (1 / 10))) ** (1 / alpha_min))
      #
      #
      df = pd.DataFrame()  # crea un dataframe donde poner los minimos y maximos por mes de toda la serie
      # set columns
      df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None)
      # calculate mim, max, mean, min_rev and mean_rev
      for i in range(data.index.min().year, data.index.max().year + 1):
        for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[(data.index.year == i) & (data.index.month == j)]
          row = [str(i) + str(-j),
                 data_filter['cuenca-base'].min(),
                 data_filter['cuenca-base'].max(),
                 data_filter['cuenca-base'].mean(),
                 data_filter['cuenca-base'][data['cuenca-base'] > umbral_Q10].min()]
          df.loc[len(df)] = row
      df['Fecha'] = pd.to_datetime(df['Fecha'])
      # df = df.set_index('Fecha')
      alpha_max = (np.sqrt(6) * std_max) / np.pi
      u_max = mean_max - (0.5772 * alpha_max)
      yt_QB = -np.log(np.log(u_qb / (u_qb - 1)))
      yt_Q15 = -np.log(np.log(15 / (15 - 1)))
      # umbrales QB y QTR15
      #
      umbral_QB = u_max + (yt_QB * alpha_max)
      umbral_Q15 = u_max + (yt_Q15 * alpha_max)
      #
      ##
      df = pd.DataFrame()
      df = df.assign(Fecha=None, Min=None, Max=None, Mean=None, Min_rev=None, Mean_rev=None)
      for i in range(data.index.min().year, data.index.max().year + 1):
        for j in range(1, 13):
          # filtro los datos por año y mes para hacer el analisis
          data_filter = data[(data.index.year == i) & (data.index.month == j)]
          row = [str(i) + str(-j),
                 data_filter['cuenca-base'].min(),
                 data_filter['cuenca-base'].max(),
                 data_filter['cuenca-base'].mean(),
                 data_filter['cuenca-base'][data['cuenca-base'] > umbral_Q10].min(),
                 data_filter['cuenca-base'][(data['cuenca-base'] > umbral_Q10) & (data['cuenca-base'] < umbral_Q15)].mean()]
          df.loc[len(df)] = row
      #########
      df['Fecha'] = pd.to_datetime(df['Fecha'])
      df = df.set_index('Fecha')
      estado.umbrales['QTR15'] = umbral_Q15
      estado.umbralesQB = umbral_QB
      estado.QTR10 = umbral_Q10
      estado.QTQ = umbral_QTQ
      if estado.h_umbrales is not None:
        estado.setear_umbrales([umbral_Q15, umbral_QB, umbral_QTQ, umbral_Q10])
      return df
    #Pruebas con el archivo
    df = pd.read_csv("est_25027400_limpia.csv")
    x=QAnla(df)
    #0.003517763085733081
    #85.33220893006306
