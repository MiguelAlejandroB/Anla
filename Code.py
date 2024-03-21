import numpy as np #Importo las librerias que voy a usar fuera de la clase
import pandas as pd
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

