import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from statsmodels.multivariate.pca import PCA
from scipy.sparse.linalg import svds
import numpy as np
import streamlit as st
import streamlit.components.v1 as components 

from PIL import Image

#########################
## Streamlit page setup #
#########################

########################
#### THEME COLOUR

# [theme]
# primaryColor="#1ED760"
# backgroundColor="#191919"
# secondaryBackgroundColor="#000000"
# textColor="#e1e1e4"

##########################


st.set_page_config(
     page_title="Spotify Playlist",
     page_icon="🔀",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get help': 'https://www.defiantdata.com/contact',
         'Report a bug': "https://www.defiantdata.com/contact",
         'About': "This Spotify Playlist app is under MIT License"
     }
 )

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

client_id = '397c877d64bf4553b6aad97b515661f9'
client_secret = '0c5ab3b5a4f64a6eaf623167b5df8fc7'


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))


#Functions to call data
@st.cache
def get_track_ids(playlist_uri):
    music_id_list = []
    playlist = sp.playlist(playlist_uri)
    for item in playlist['tracks']['items']:
        music_track = item['track']
        music_id_list.append(music_track['id'])
    return music_id_list

@st.cache
def get_track_meta(track_ids):
    trackdatalist = []
    for track in track_ids:
        meta = sp.track(track)
        track_details = {"Title": meta['name'], 
                         "album": meta['album']['name'],
                     "Artist": meta['album']['artists'][0]['name'],
                     "release_date": meta['album']['release_date'],
                     "time": meta['duration_ms'],
                     "Album": meta['album']['images'][2]['url']}
        trackdatalist.append(track_details)
    return trackdatalist

@st.cache
def path_to_image_html(path):
    return '<img src="'+ path + '" width="50" >'


def pl_ms_convert(ms):
    h = ms//1000//60//60
    m = ms//1000//60%60
    s = ms//1000%60
    if h > 0:
        return f'{h} hr {m:02d} min'
    else:
        return f'{m} min'
    
def song_ms_convert(ms):
    m = ms//1000//60%60
    s = ms//1000%60
    return f'{m}:{s:02d}'

#Get data


container = st.container()
#container.title("Spotify Playlist Optimizer")
#container.subheader("Enter Your A Spotify Playlist link below and let the algorithm re-order the list")


st.sidebar.image('https://storage.googleapis.com/wzukusers/user-17604034/images/57713fad5aa57euz7Vuh/spotify-vector-logo_d400.png')
st.sidebar.header('Spotify Playlist Otptimizer')
    
playlist_uri = st.sidebar.text_input('   Paste Playlist Link:', 'https://open.spotify.com/playlist/2zWW9SwyeoBDPkDPQ6cX5Q?si=2d8b519fa6d74ffb')




track_ids = get_track_ids(playlist_uri)

featureslist = []
for track in track_ids:
   featureslist.append(sp.audio_features(track)[0])


useful_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]   

features_df = pd.DataFrame(featureslist)[useful_features]
track_df = pd.DataFrame(get_track_meta(track_ids))


dfs = pd.DataFrame(MaxAbsScaler().fit_transform(features_df), columns=features_df.columns)
pc = PCA(dfs, ncomp=2, method='nipals')


#px.scatter(pc.factors, x='comp_0', y='comp_1', hover_name=track_df.name)

# Decompose the matrix
U, sigma, Vt = svds(pc.factors, k=1)
 
# Convert sigma into a diagonal matrix
sigma = np.diag(sigma)
 
# U x Sigma
U_x_Sigma = np.dot(U, sigma)
 
# (U x Sigma) x Vt
U_Sigma_Vt = np.dot(U_x_Sigma , Vt)


df3 = pd.DataFrame(U_Sigma_Vt).rank(method='min').astype(int)
track_df['New Order'] = df3[[1]]
track_df.sort_values(by='New Order')


newlist_df = track_df[['New Order','Album','Title','Artist', 'time']]

old_order = newlist_df.index + 1
newlist_df.insert(0,'#', (newlist_df.index+1))

newlist_df.columns = [column.upper() for column in newlist_df.columns]

description = sp.playlist(playlist_uri)['description']
cover_image = sp.playlist(playlist_uri)['images'][0]['url']
playlist_name = sp.playlist(playlist_uri)['name']
playlist_username = sp.playlist(playlist_uri)['owner']['display_name']
playlist_tracknum = len(newlist_df)-1

pl_length = pl_ms_convert(newlist_df.TIME.sum())

subheading = f"**{playlist_username}  • {playlist_tracknum} songs**, {pl_length}"

########################
### Streamlit Design  ##
########################


##### The COVER & Title 

with st.container():
    col1, col2, col3 = st.columns([8,2,29])
    with col1:
        st.image(cover_image, width=250)
    with col2:
        st.empty()
    with col3:
        st.subheader(' ')
        st.title(playlist_name)
        st.text(description)
        st.write(subheading)





st.markdown("""
<style>


</style>
"""
## Block background color test   color: rgb(68, 68, 68); background-color: rgb(59, 58, 58);
, unsafe_allow_html=True)





option = st.sidebar.selectbox(
     'How to order your playlist?',
     ('Current Order', 'Optimal Order'))       



st.sidebar.caption(""" """)

st.sidebar.write("""•    1) Open Spotify Desktop App """)
st.sidebar.write("""•    2) *Right-click:* on playlist """)
st.sidebar.write("""•    3) *Hover:* Share """)
st.sidebar.write("""•    4) *Click:* Copy Link to Playlist """)
#                   \n 2.Right-click on playlist ->\n 3. Share ->\n 4.Copy Link to Playlist]""")



with st.sidebar.expander("See Expository"):
     st.write("""
        The Playlist optimizer uses Spotify's Audio feature analysis API to 
        reduce the aggregate values to a single 1-dimentional vector. 
        
        Future releases will make the selection based on the audio feature itself if your looking to optimize for
         energy or dancibility
     """)
     


if option == 'Current Order':
    pass
else:
    newlist_df = newlist_df.sort_values(by='NEW ORDER')

format_dict= {
    'TIME': song_ms_convert,
    'ALBUM': path_to_image_html,
    
}

html_content_newlist = newlist_df.to_html(border =0, index=False, justify='left',
                                          escape=False ,formatters=format_dict)

style_newlist = """<style>
                
                body {padding: 0% 2% 0% 2%; box-sizing: content-box;}
                table {width:95%;}
                table, th, td {border-collapse: collapse; color:white; font-family: 'Trebuchet MS'; padding:10px; font-size: 16px; }
                td:nth-child(1) {text-align: left; width:6%; border-right: solid 1px;}
                td:nth-child(2) {text-align: center; width:8%;}
                td:nth-child(3) {text-align: left; width:12%;}
                td:nth-child(4) {text-align: left; width:25%;}
                td:nth-child(5) {text-align: left; width:25%;}
                td:nth-child(6) {text-align: left; width:10%;}

                {border-bottom: 1px solid black;} 
                tr:hover {background-color: #2b5335;}
                th {border-bottom: 1px solid #b3b3b3;} 
                thead {color: white; font-size: 16px; text-transform: capitalize;}
                thead:hover {background-color: #1bb551; color: white; font-size: 16px;} </style>"""
                
h1 = (len(newlist_df)*80+40)
components.html(style_newlist + html_content_newlist, height=h1)