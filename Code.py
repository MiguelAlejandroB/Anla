import numpy as np #Importo las librerias que voy a usar fuera de la clase
import pandas as pd
from scipy.stats import norm, lognorm, gumbel_r, pearson3,  genextreme

class Anla:
    df = pd.read_csv("est_25027400_limpia.csv")
    # Establecer la primera columna como el índice del DataFrame
    df.set_index(df.columns[0], inplace=True)

    # Convertir el índice a tipo de datos de fecha
    df.index = pd.to_datetime(df.index)

    # Calcular el promedio de los caudales cada 7 días
    promedio_7_dias = df.resample('7D').mean()

    #Serie caudal de excedente minimos por año

    # Calcular el mínimo de caudal por año
    minimos_por_ano = promedio_7_dias.groupby(promedio_7_dias.index.year)['cuenca-base'].min()

    # Crear un DataFrame con los mínimos de caudal por año
    df_minimos_por_ano = pd.DataFrame(minimos_por_ano)
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
        n = Flogn(x, pln[0], pln[1])
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
    a, m, s = pearson3.fit(df_minimos_por_ano['cuenca-base'])
    vesp=[]
    for x in df_minimos_por_ano['cuenca-base']:
        n = pearson3.pdf(x, a, m, s)
        vesp.append(n)
    corp = np.corrcoef(df_minimos_por_ano['cuenca-base'], vesp)[0, 1]