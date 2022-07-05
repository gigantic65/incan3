import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import Prediction_app# import your pages here
from pages import Prediction2_app


# rest of the code

# Create an instance of the app 

app = MultiPage()

#st.set_page_config(layout="wide")
    
# Title of the main page
st.markdown("<h2 style='text-align: center; background-color:darkred; color: white;'>Incan 마감공정 조색 배합 예측 모델</h2>", unsafe_allow_html=True)



# Add all your applications (pages) here
app.add_page("Incan 조색 배합 예측 Model", Prediction_app.app)
app.add_page("점도 광택 예측 Model", Prediction2_app.app)
#app.add_page("Predict New Conditions", Prediction_app.app)
#app.add_page("Monitoring", Monitor_app.app)


# The main app
app.run()