# --------------------LIBRERÍAS----------------------------#
import json
import os
import warnings
from webbrowser import get
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import cufflinks
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly_express as px
import requests
from branca.colormap import LinearColormap
from folium.plugins import (
    FastMarkerCluster, FloatImage, HeatMap, MarkerCluster)
from plotly.offline import init_notebook_mode, iplot
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
from PIL import Image
import streamlit as st
from plotly.subplots import make_subplots
import pydeck as pdk
import seaborn as sns



sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

def main():
    
    # --------------------------------------------------------CONFIGURACIÓN DE LA PÁGINA---------------------------------------------------#
    st.set_page_config(page_title="Austin al turismo",layout="wide", page_icon="🏙️", initial_sidebar_state="expanded")
    st.set_option("deprecation.showPyplotGlobalUse", False)   

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # -----------------------------------------------------------------Cabecera----------------------------------------------------------------
    st.title("Austin al turismo (Marco A. García Palomo)")
    st.markdown('')
    st.markdown('')
    st.markdown('')
    image = Image.open('Imag/20134c37c563cf1d3022e223443bdc8c.jpg')
    st.image(image, caption='Austin en pinterest',use_column_width='auto')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Austin es la capital del estado estadounidense de Texas y del condado de Travis. Localizada en el suroeste de Estados Unidos, concretamente en Texas Central,​ es la 11.º ciudad más poblada en los Estados Unidos y la 4.º más grande en el estado de Texas. Fue la tercera ciudad en crecimiento poblacional en los EE UU entre 2000 y 2006.​ Austin es también la segunda capital estatal más grande de Estados Unidos.​ Para julio de 2012, tenía una población de más de 1.000.000 habs.​ La ciudad es el centro cultural y económico del área metropolitana de Austin, la cual tenía una población estimada de 1 834 303 en 2012. Se encuentra a orillas del río Colorado (no confundir con el río Colorado que forma frontera entre Estados Unidos y México).</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Además de sus funciones como sede del gobierno del estado, Austin es un centro comercial, fabril, educativo y de convenciones. Entre su producción, destacan los artículos de alta tecnología, como equipos eléctricos, semiconductores y equipos informáticos. Es la sede de la Universidad de Texas en Austin (1883).</div> """, unsafe_allow_html=True)
    st.markdown('')
    st.markdown('')
    st.markdown('')
    col1,col2= st.columns(2)
    with col1:
        img= Image.open('Imag/Barton-Springs-Pool.jpg')
        st.image(img,caption='Piscina de Barton Springs')
        img1=Image.open('Imag/Lady-Bird-Lake.jpg')
        st.image(img1,caption='Lago Lady Bird')
        img2= Image.open('Imag/lake-travis-austin.jpg')
        st.image(img2,caption='Lago Travis')
        img3= Image.open('Imag/Metropolitan-Park.jpg')
        st.image(img3,caption='Parque metropolitano')
    with col2:
        img4= Image.open('Imag/34fe50d145c5819d6ee87ee5280c9104.jpg')
        st.image(img4,caption='Parque estatal Mckinney falls')
        img5= Image.open('Imag\R.jpg')
        st.image(img5,caption='Monte Bonnell')
        img6= Image.open('Imag/Umlauf-Sculpture-Garden.jpg') 
        st.image(img6,caption='Jardín de la escultura Umlauf')
        img7=Image.open('Imag/University-of-Texas-at-Austin.jpg')
        st.image(img7,caption='Universidad de Texas')

    # -----------------------------------------------LECTURA DE DATOS Y PREPROCESAMIENTO------------------------------------#

    listings = pd.read_csv("input/listings.csv", index_col= "id")
    listings_details = pd.read_csv("input/listings_details.csv",index_col= "id",low_memory=False)
    calendar = pd.read_csv("input/calendar.csv", parse_dates=['date'], index_col=['listing_id'])
    reviews_details = pd.read_csv("input/reviews_details.csv", parse_dates=['date'])
    target_columns = ["property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate"]
    listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')
    listings = listings.drop(columns=['neighbourhood_group'])
    listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].str.strip('%'))


    #------------------------------------------------------TABS---------------------------------------------------#
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.title("Búsqueda de datos")
    st.markdown('')
    st.markdown('')
    st.markdown('')
    tabs = st.tabs(["Número de alquileres por código postal", "Tipos de propiedades y alojamientos",'Número de alojados', 'Precios por código postal',"Puntuaciones de ubicación frente al precio", 'Reseñas de los huéspedes', 'Para encontrar un buen hospedador', 'Líneas futuras y conclusiones'])

    # -------------------------------------------------------TAB 1-----------------------------------------------------#
    tab_plots = tabs[0]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Aunque lo que tenemos son código postales, como veremos más adelante, la mayoría de los listados están en el centro de la ciudad (78704, 78702 y 78701).</div> """, unsafe_allow_html=True)

        feq=listings['neighbourhood'].value_counts().sort_values(ascending=True)
        feq.plot.barh(figsize=(10, 8), color='b', width=1)
        plt.title("Número de alquileres por código postal", fontsize=20)
        plt.xlabel('Número de listas', fontsize=12)
        st.pyplot()
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)



    # -------------------------------------------------------TAB 2-----------------------------------------------------#
    tab_plots = tabs[1]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>    Las reglas de Austin sobre el tipo de alojamiento son muy laxas, no tienen límite de tiempo y además se permite el alquiler de apartamentos y casas completas.</div> """, unsafe_allow_html=True)


        freq = listings['room_type']. value_counts().sort_values(ascending=True)
        freq.plot.barh(figsize=(15, 3), width=1, color = ["r","g","b"])
        plt.title('Tipos de habitaciones en Austin', fontsize=18)
        plt.xlabel('Número de habitaciones', fontsize=14)
        st.pyplot()

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>En el conjunto de datos, encontramos muchísimos tipos de propiedades diferentes, sin embargo, muchos de esos tipos de propiedades tienen muy pocos listados en Austin. En la figura a continuación, solo mostramos tipos de propiedades con al menos 100 listados. Como podemos ver, la gran mayoría de las propiedades en Austin son apartamentos, pisos o casas completas.</div> """, unsafe_allow_html=True)
        st.markdown('')
        prop = listings.groupby(['property_type','room_type']).room_type.count()
        prop = prop.unstack()
        prop['total'] = prop.iloc[:,0:3].sum(axis = 1)
        prop = prop.sort_values(by=['total'])
        prop = prop[prop['total']>=100]
        prop = prop.drop(columns=['total'])

        prop.plot(kind='barh',stacked=True, color = ["r"],
                  linewidth = 1, grid=True, figsize=(15,8), width=1)
        plt.title('Tipos de propiedades en Austin', fontsize=18)
        plt.xlabel('Número de listados', fontsize=14)
        plt.ylabel("")
        st.pyplot()

    # -------------------------------------------------------TAB 3-----------------------------------------------------#
    tab_plots = tabs[2]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Los alojamientos con más anuncios son para 2 personas, aunque hay una gran cantidad de anuncios para 4 personas. Además, el límite de Airbnb utiliza un máximo de 16 huéspedes por anuncio.</div> """, unsafe_allow_html=True)
        st.markdown('')
        feq=listings['accommodates'].value_counts().sort_index()
        feq.plot.bar(figsize=(10, 8), color='b', width=1, rot=0)
        plt.title("Número de personas alojadas", fontsize=20)
        plt.ylabel('Número de anuncios', fontsize=12)
        plt.xlabel('Alojados', fontsize=12)
        st.pyplot() 

    # -------------------------------------------------------TAB 4-----------------------------------------------------#
    tab_plots = tabs[3]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Comparamos los alojamientos para dos personas, ya que son los alojamientos más comunes.</div> """, unsafe_allow_html=True)

        feq = listings[listings['accommodates']==2]
        feq = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
        feq.plot.barh(figsize=(10, 8), color='b', width=1)
        plt.title("Precio medio diario para un alojamiento para 2 personas", fontsize=20)
        plt.xlabel('Precio medio diario', fontsize=12)
        plt.ylabel("")
        st.pyplot()

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>La zona más cara de Austin, con diferencia, coincide con el código postal 78712, si bien, esta zona es céntrica, el valor del alquiler nos hace sospechar que algo pasa. Por ello, extraje los apartamentos en alquiler de esa zona y resultó que de los tres que había solo uno era para dos personas y además con un precio bastante alto.</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')

        df78712=listings.loc[listings.loc[:, 'neighbourhood'] == 78712]
        Timo=df78712[df78712['accommodates']==2]
        Timo
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Podemos ver que hay dos zonas más con alto precio de alquiler, aunque más razonables, esto se debe a que estas dos zonas están muy bien situadas, aunque no son céntricas, y son barrios muy tranquilos, como se pueden ver en estos dos enlaces: </div> """, unsafe_allow_html=True)

        st.markdown('')
        st.markdown('')
        st.markdown('')
        url1 = 'https://www.zipdatamaps.com/es_78744'
        st.markdown(f'''<a href={url1}><button style="background-color:White;">https://www.zipdatamaps.com/es_78744</button></a>''',unsafe_allow_html=True)
        url1 = 'https://www.zipdatamaps.com/es_78750'
        st.markdown(f'''<a href={url1}><button style="background-color:White;">https://www.zipdatamaps.com/es_78750</button></a>''',unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')

        feq = feq.transpose()
        feq=feq.drop(columns=['neighbourhood'])
        feq=feq.drop(columns=['index1'])
        adam = gpd.read_file("input/neighbourhoods.geojson")
        feq = pd.DataFrame(feq)
        adam['neighbourhood']= adam['neighbourhood'].astype(float)
        adam = pd.merge(adam, feq, on='neighbourhood', how='left')
        adam.rename(columns={'price': 'average_price'}, inplace=True)
        adam.average_price = adam.average_price.round(decimals=0)
        map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()
        color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

        def get_color(feature):
            value = map_dict.get(feature['properties']['neighbourhood'])
            return color_scale(value)
        map3 = folium.Map(location=[30.26715, -97.74306], zoom_start=11)
        folium.GeoJson(data=adam,
                    name='Austin',
                    tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                          labels=True,
                                                          sticky=False),
                    style_function= lambda feature: {
                        'fillColor': get_color(feature),
                        'color': 'black',
                        'weight': 1,
                        'dashArray': '5, 5',
                        'fillOpacity':0.5
                        },
                    highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map3)

        lats2018 = listings['latitude'].tolist()
        lons2018 = listings['longitude'].tolist()
        locations = list(zip(lats2018, lons2018))
        map1 = folium.Map(location=[30.26715, -97.74306], zoom_start=11.5)
        FastMarkerCluster(data=locations).add_to(map1)

    col1,col2= st.columns(2)
    with col1:
        st.write("Mapa de las localizaciones de los alojamientos")
        st_folium(map1, returned_objects=[])
    with col2:
        st.write("Mapa precios medio por código postal")
        st_folium(map3, returned_objects=[])


    # -------------------------------------------------------TAB 5-----------------------------------------------------#
    tab_plots = tabs[4]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>En esta sección, agrupamos las puntuaciones de revisión de la ubicación por código postal (solo listados con al menos 10 revisiones).</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>A continuación, vemos que los vecindarios centrales generalmente obtienen una puntuación más alta en la puntuación de revisión de ubicación y también podemos ver que no son las ubicaciones más caras, excepto una como hemos comentado, que ni siquiera llega a 10 reseñas y no aparece en la gráfica de valoración.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Al mirar las valoraciones podemos ver que están por encima del 4 siendo el 5 la nota más alta, por lo tanto la gráfica se muestra acotada del 4 al 5.</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)
        st.markdown('')
        fig = plt.figure(figsize=(20,10))
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=20)

        ax1 = fig.add_subplot(121)
        feq = listings[listings['number_of_reviews']>=10]
        feq1 = feq.groupby('neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)
        ax1=feq1.plot.barh(color='b', width=1)
        plt.title(" Puntuación promedio de la Ubicaciónde en revisión (al menos con 10 revisiones)", fontsize=20)
        plt.xlabel('Puntuación (escala 4-5)', fontsize=20)
        plt.ylabel("Códigos postales", fontsize=20)
        plt.xlim(4,5)

        ax2 = fig.add_subplot(122)
        feq = listings[listings['accommodates']==2]
        feq2 = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
        ax2=feq2.plot.barh(color='b', width=1)
        plt.title("Precio medio diario para un alojamiento para 2 personas", fontsize=20)
        plt.xlabel('Precio medio diario (Dolar)', fontsize=20)
        plt.ylabel("Códigos postales", fontsize=20)

        plt.tight_layout()
        st.pyplot()

    # -------------------------------------------------------TAB 6-----------------------------------------------------#
    tab_plots = tabs[5]  
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Además de las reseñas escritas, los invitados pueden enviar una calificación de estrellas general y un conjunto de calificaciones de estrellas de categoría. Los huéspedes pueden dar calificaciones sobre:</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Experiencia general. ¿Cuál fue su experiencia en general?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Limpieza. ¿Sentiste que tu espacio estaba limpio y ordenado?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Veracidad. ¿Con qué precisión su página de listado representó su espacio?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Calidad-precio. ¿Sintió que su listado proporcionó un buen valor por el precio?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Comunicación. ¿Qué tan bien se comunicó con su anfitrión antes y durante su estadía?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Llegada. ¿Qué tan bien fue su registro?</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>- Localización. ¿Cómo te sentiste en el barrio?</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>A continuación se puede ver la distribución de puntajes de todas esas categorías. El 95 % de los listados de Airbnb calificaron de 4,5 a 5 estrellas. Después de haber visto las distribuciones de puntajes, personalmente consideraría que cualquier puntaje de 4,5 o inferior no es un buen puntaje, por lo tanto he acotado las gráficas en torno a esos valores.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')

        listings10 = listings[listings['number_of_reviews']>=10]

        fig = plt.figure(figsize=(20,15))
        plt.rc('xtick', labelsize=5) 
        plt.rc('ytick', labelsize=5)


        ax1 = fig.add_subplot(321)
        feq=listings10['review_scores_location'].value_counts().sort_index()
        ax1=feq.plot.bar(color='b', width=1, rot=0)
        ax1.tick_params(axis = 'both', labelsize = 5)
        plt.title("Localización", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(88,110)

        ax2 = fig.add_subplot(322)
        feq=listings10['review_scores_cleanliness'].value_counts().sort_index()
        ax2=feq.plot.bar(color='b', width=1, rot=0)
        plt.title("Limpieza", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(100,122)

        ax3 = fig.add_subplot(323)
        feq=listings10['review_scores_value'].value_counts().sort_index()
        ax3=feq.plot.bar(color='b', width=1, rot=0)
        plt.title("Calidad-precio", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(87,109)

        ax4 = fig.add_subplot(324)
        feq=listings10['review_scores_communication'].value_counts().sort_index()
        ax4=feq.plot.bar(color='b', width=1, rot=0)
        plt.title("Communicación", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(71,93)

        ax5 = fig.add_subplot(325)
        feq=listings10['review_scores_checkin'].value_counts().sort_index()
        ax5=feq.plot.bar(color='b', width=1, rot=0)
        plt.title("Llegada", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(73,89)

        ax6 = fig.add_subplot(326)
        feq=listings10['review_scores_accuracy'].value_counts().sort_index()
        ax6=feq.plot.bar(color='b', width=1, rot=0)
        plt.title("Veracidad", fontsize=24)
        plt.ylabel('Número de anuncios', fontsize=14)
        plt.xlabel('Puntuación media de las opiniones', fontsize=14)
        plt.xlim(73,95)

        plt.tight_layout()
        st.pyplot()

        # -------------------------------------------------------TAB 7-----------------------------------------------------#
    tab_plots = tabs[6]
    with tab_plots:
        st.markdown('')
        st.markdown('')
        st.markdown('')

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>En Airbnb puedes encontrar superanfritiones, este estatus se consigue superando una series de condiciones que establece la propia empresa de Airbnb.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>A continuación podemos ver que hay bastantes superanfitriones en Austin.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')

        listings.host_is_superhost = listings.host_is_superhost.replace({"t": "Si", "f": "No"})

        feq=listings['host_is_superhost'].value_counts()
        feq.plot.pie(autopct="%0.1f %%",figsize=(2, 2))
        plt.title("Número de anuncios con Superanfitrión", fontsize=10)
        plt.ylabel('', fontsize=10)
        st.pyplot()

        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Sin embargo, mi consejo es que no se busque necesariamente un superanfitrión, ya que éstos suelen ser más caros.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')    
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>Como podemos ver, la mayoría de los anfitriones con al menos 10 reseñas responde al 100% de los mensajes nuevos. Podría decir que estos anfitriones también son buenos, además, hay muy pocos anuncios con anfitriones que no respondan los mensajes nuevos después de unas pocas horas.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'></div> """, unsafe_allow_html=True)

        fig = plt.figure(figsize=(20,10))
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=20)

        ax1 = fig.add_subplot(121)
        feq1 = listings['host_response_rate'].dropna()
        ax1= plt.hist(feq1)
        plt.title("Ratio de respuesta (con al menos 10 comentarios)", fontsize=20)
        plt.ylabel("Número de anuncios")
        plt.xlabel("Porcentaje", fontsize=20)

        ax2 = fig.add_subplot(122)
        feq2 = listings['host_response_time'].value_counts()
        ax2=feq2.plot.bar(color='b', width=1, rot=45)
        plt.title("Tiempo de respuesta (con al menos 10 opiniones))", fontsize=20)
        plt.ylabel("Número de anuncios")
        plt.tight_layout()
        st.pyplot()

        # -------------------------------------------------------TAB 8-----------------------------------------------------#
    tab_plots = tabs[7]
    with tab_plots:
        st.markdown('')
        st.title('Líneas futuras')
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Mejorar la interfaz de la app.</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Buscar listado de vecindarios por código postal y cambiar éstos por los nombres de los barrios</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Añadir gráfica de disponibilidad por fechas.</div> """, unsafe_allow_html=True)
        st.markdown('')    
        st.title("Conclusiones")
        st.markdown('')
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* En Austin tenemos una gran variedad de viviendas para alquilar.</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Que no siempre un mayor precio se corresponde con una buena valoración.</div> """, unsafe_allow_html=True)
        st.markdown(""" <div style='color:black; font-size: 30px; text-align: justify;'>* Que los anfitriones en Austin son bastante rápidos a la hora de responder, no hace falta escoger a un superanfitrión.</div> """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')

        wordcloud = Image.open('Imag/descarga.png')
        st.image(wordcloud, caption='Nube de palabras hecha analizando las palabras más repetidas en los comentarios.',
                    use_column_width='auto')
    
if __name__ == '__main__':
    main()
