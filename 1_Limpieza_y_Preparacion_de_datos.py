#!/usr/bin/env python
# coding: utf-8

# <font color = 'Blue'> **PROYECTO NRO.1:** <font color = 'black'> **LIMPIEZA Y PREPARACIÓN DE DATA DE CLIENTES BANCARIOS.**
#     
# <font color = 'Blue'> **OBJETIVO:**
# <font color = 'black'> Mostrar una secuencia de preparación de datos para uso en análisis y modelado de datos.

# In[3]:


# Librerías principales
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Carga de datos
archivo= 'E:\DescargasKaggle\Bank_Customer_Data.csv'
df = pd.read_csv(archivo, header=0, delimiter=';')

# Vista general
print(df.info())
df.head()


# **1. Inspección y tipos de variables**

# In[4]:


# Revisión de nulos
print(df.isnull().sum())


# In[5]:


# Conversión de fechas
df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'], errors='coerce')


# In[6]:


# Revisión de tipo de datos incorrectos
df['ingresos_mensuales'] = pd.to_numeric(df['ingresos_mensuales'], errors='coerce')


# In[7]:


print(df.info())


# **2. Tratamiento de valores faltantes**

# In[8]:


# Imputar valores numéricos con la mediana (en este caso no hay valores faltantes)
imputer = SimpleImputer(strategy='median')
df['ingresos_mensuales'] = imputer.fit_transform(df[['ingresos_mensuales']])


# In[9]:


# Imputar valores categóricos con el valor más frecuente (en este caso no hay valores faltantes)
df['estado_civil'] = df['estado_civil'].fillna(df['estado_civil'].mode()[0])


# **3. Outliers (Método IQR)**, para detectar valores fuera de rango

# In[10]:


Q1 = df['edad'].quantile(0.25)
Q3 = df['edad'].quantile(0.75)
IQR = Q3 - Q1

# Límites inferior y superior
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Data de outliers
outliers = df[(df['edad'] < limite_inferior) | (df['edad'] > limite_superior)]

# Data sin outliers
sin_outliers = df[(df['edad'] >= limite_inferior) & (df['edad'] <= limite_superior)]

print("Outliers encontrados:\n", outliers)


# In[11]:


# Reemplazando data sin outliers en dataset final
df = sin_outliers


# **4. Codificación de variables categóricas**

# In[12]:


# Label encoding para variables binarias
df['sexo'] = LabelEncoder().fit_transform(df['sexo'])


# In[13]:


# One-hot encoding para variables no ordinales
df = pd.get_dummies(df, columns=['estado_civil', 'segmento_cliente'], drop_first=True)


# In[14]:


df.head()


# **5. Normalización o Estandarización**

# In[15]:


scaler = StandardScaler()
df[['ingresos_mensuales', 'saldo_promedio']] = scaler.fit_transform(df[['ingresos_mensuales', 'saldo_promedio']])


# **6. Feature engineering**

# In[16]:


# Calculando variable Antigüedad en meses
df['antigüedad_meses'] = (pd.to_datetime('today') - df['fecha_apertura']).dt.days // 30


# In[17]:


# Calculando variable Ratio deuda / ingreso
df['ratio_deuda_ingreso'] = df['deuda_total'] / (df['ingresos_mensuales'] + 1)


# In[18]:


# Añadiendo Segmento por edad
df['segmento_edad'] = pd.cut(df['edad'], bins=[17, 30, 45, 60, 90], labels=['Joven', 'Adulto1', 'Adulto2', 'Senior'])


# **Resultado: Preprocesamiento**

# In[19]:


print("Dataset listo para modelado:")
print(df.head())


# In[20]:


df.describe()


# In[ ]:




