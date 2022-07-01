import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import streamlit as st

st.set_page_config(page_title='House price predictor', layout='wide')

def encode(x):
    if x=='Yes':
        return 1
    else: 
        return 0
# Function to get Tier of a respective city
def find_tiers(x):
    tier1= 'Ahmedabad, Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune'
    tier2= '''Agra, Ajmer, Aligarh, Amravati, Amritsar, Anand, Asansol, Aurangabad, Bareilly, Belagavi, Bhavnagar, Bhiwandi, Bhopal, Bhubaneswar, 
            Bikaner, Bilaspur, Bokaro Steel City, Chandigarh, Coimbatore, Cuttack, Dehradun, Dhanbad, Bhilai, Durgapur, Erode, Faridabad, Firozabad, 
            Ghaziabad, Gorakhpur, Guntur, Gurugram, Guwahati, Gwalior, Hamirpur, Hubballi–Dharwad, Indore, Jabalpur, Jaipur, Jalandhar, Jalgaon, 
            Jammu, Jamnagar, Jamshedpur, Jhansi, Jodhpur, Kalaburagi, Kakinada, Kannur, Kanpur, Karnal, Kochi, Kolhapur, Kollam, Kozhikode, 
            Kurnool, Ludhiana, Lucknow, Madurai, Malappuram, Mathura, Mangaluru, Meerut, Moradabad, Mysuru, Nagpur, Nanded, Nashik, Nellore, 
            Noida, Patna, Puducherry, Purulia, Prayagraj, Raipur, Rajkot, Rajamahendravaram, Ranchi, Rourkela, Ratlam, Salem, Sangli, Shimla, 
            Siliguri, Solapur, Srinagar, Surat, Thanjavur, Thiruvananthapuram, Thrissur, Tiruchirappalli, Tirunelveli, Tiruvannamalai, Ujjain, 
            Vijayapura, Vadodara, Varanasi, Vasai-Virar, Vijayawada, Visakhapatnam, Vellore, Warangal'''

    tier1= [city.strip() for city in tier1.split(',')]
    tier2= [city.strip() for city in tier2.split(',')]

    if x in tier1: 
        return 'Tier 1'
    elif x in tier2: 
        return 'Tier 2'
    else: 
        return 'Tier 3'

# Check whether input city is a capital
def capital(x):
    if x in capitals:
        return 1
    else: 
        return 0

# @st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def load_data():
    path= 'X.pkl'
    with open(path, 'rb') as ref:
        df= pickle.load(ref)
    path= 'model.pkl'
    with open(path, 'rb') as ref:
        pipe= pickle.load(ref)
    
    return df, pipe

# To transform numbers to abbreviated format
def format_numbers(number, pos=None, fmt= '.0f'):
    fmt= '%'+fmt
    thousands, lacs, crores= 1_000, 1_00_000, 1_00_00_000
    if number/crores >=1:
        return (fmt+' Cr.') %(number/crores)
    elif number/lacs >=1:
        return (fmt+' Lacs.') %(number/lacs)
    elif number/thousands >=1:
        return (fmt+' K') %(number/thousands)
    else:
        return fmt %(number)

# Function for encoding multiple features
class CustomEncoder:

    def __init__(self, columns):
        self.columns= columns
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import LabelEncoder
        out= X.copy()
        if self.columns is not None:
            out[self.columns]= out[self.columns].apply(lambda x: LabelEncoder().fit_transform(x))
        else:
            out= out.apply(lambda x: LabelEncoder().fit_transform(x))
        return out
    
    def fit_transform(self, X, y=None):
        out= X.copy()
        return self.fit(out).transform(out)

def main():
    st.title('Houses Price predictor')

    cols= st.columns(4)
    state= cols[0].selectbox('State', states_city.keys(), key='state')
    city= cols[1].selectbox('City', states_city.get(st.session_state.state),key='city')
    locality= cols[2].selectbox('Locality', cities_local.get(st.session_state.city), key='locality')
    bhk= cols[3].slider('BHK', min_value=1, max_value= 10, key='bhk')
    

    cols= st.columns(5)
    square_ft= cols[0].number_input('Total area', key='square_ft', value= 200)
    ready_to_move= cols[1].radio('Ready to move', ['Yes','No'], key='ready')
    resale= cols[2].radio('Is property for resale',['Yes','No'], key='resale')
    posted= cols[3].radio('Posted by', df.posted_by.unique(), key='posted')
    rera= cols[4].radio('Approved by Rera', ['Yes','No'], key='rera')
    
    
    btn= st.button('Predict price', key='button')
    
    if btn:
        X= df.copy()
        longitude= lat_long_pop.get('longitude').get(st.session_state.city)
        latitude= lat_long_pop.get('latitude').get(st.session_state.city)
        population= lat_long_pop.get('population').get(st.session_state.city)
        is_capital= capital(st.session_state.city)
        tier= find_tiers(st.session_state.city)

        vals= [posted, encode(rera), bhk, square_ft, encode(ready_to_move), encode(resale), longitude, latitude, city, locality, population, state, is_capital, tier]

 
        X= X.append(dict(zip(X.columns, vals)), ignore_index=True)
        X['square_ft']= np.log(X.square_ft)

        X= pd.get_dummies(X, columns= ['posted_by','bhk_no.','tier'], drop_first=True)
        X= CustomEncoder(X.select_dtypes('O').columns).fit_transform(X)

        pred= pipe.predict(X.iloc[-2:,:])[-1]
        pred= format_numbers((pred**2)*100000, fmt='.1f')
        st.markdown('Price of the house would be approximately around  ₹%s'%pred)

path= 'capitals.pkl'
with open(path, 'rb') as ref:
    capitals= pickle.load(ref)

df, pipe= load_data()
states_city= df.groupby('state').city.agg(set).to_dict()
cities_local= df.groupby('city').locality.agg(set).to_dict()
lat_long_pop= df[['city','latitude','longitude','population']].set_index('city').to_dict()

if __name__=='__main__':
    main()

