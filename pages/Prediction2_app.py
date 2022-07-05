
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random
import itertools

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    

def feature_m(X,y):
    

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)
        
    rfe = MultiOutputRegressor(RandomForestRegressor())
    
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

      
    rfe.fit(rescaledX, y_train)
        
    f_importance = pd.DataFrame(rfe.estimators_[0].feature_importances_,columns=['importance'],index=X_train.columns)
    
    f_importance = f_importance.sort_values(by='importance', ascending = False)
    
    
    return f_importance

def app():
    
    st.write('')
    st.write('')
    
    st.markdown("<h6 style='text-align: right; color: black;'>적용 칼라: Blue, Gray </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: right; color: black;'>총 데이터 갯수: 26 조건</h6>", unsafe_allow_html=True)

    st.write("")

    
    with st.expander("Predict New Conditions Guide"):
        st.write(
                "1. 정확도 확인 : Model accuracy 버튼 클릭.\n"
                "2. 용제 투입과 소광제 투입에 따른 점도 및 광택 예측.\n"
        )



    st.sidebar.header('1. 예측 모델 업로드 ')
    
    
    df = pd.read_csv('train.csv')

    df = pd.DataFrame(df)
     
    
    def load_model(model):
        loaded_model = pickle.load(model)
        return loaded_model
            
    models = LinearRegression()
    
    with open('M_linear.pickle', 'rb') as f:
        models_a = pickle.load(f)
    
    with open('M_GBM.pickle', 'rb') as f:
        models_b = pickle.load(f)
        
    


    st.sidebar.write('')

    st.subheader('1. 데이터 및 머신러닝 모델')
    st.write('')

        
    x = list(df.columns[:-2])
    y = list(df.columns[df.shape[1]-2:])

        #Selected_X = st.sidebar.multiselect('X variables', x, x)
        #Selected_y = st.sidebar.multiselect('Y variables', y, y)
            
    Selected_X = np.array(x)
    Selected_y = np.array(y)
        
    st.write('**1.1 X인자 수 :**',Selected_X.shape[0])
    st.info(list(Selected_X))

    
    st.write('**1.2 Y인자 수:**',Selected_y.shape[0])
    st.info(list(Selected_y))

#    df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
        #df2 = df[df.columns == Selected_X]
    
             
    st.write('')   
            
    st.write('**1.3 선정 예측 모델 :**')
    model_list = ['Linear_Regression','Gradient_Boot_Machine']

    st.info(model_list)
            
    st.write("")
    st.write('**1.4 예측 모델 정확도 :**')
    

    results = []

    msg = []
    mean = []
    std = []
    names = []
    R2 = []

    columns = []    

 
            
    if st.button('Check Model Accuracy'):
        
        X = df[Selected_X]
        y = df[Selected_y]
                
                
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
                


        models_a.fit(X_train, y_train)
        models_b.fit(X_train, y_train)
                
        predictions = models_a.predict(X_test)
        predictions2 = models_b.predict(X_test)
        
        y_test = np.array(y_test)
        A_column = ['Actual_Gloss','Actual_Viscosity']
        y_test = pd.DataFrame(y_test,columns=A_column)
        
        
        P_column = ['Pred_Gloss_Linear','Pred_Viscosity_Linear']
        P_column2 = ['Pred_Gloss_GBM','Pred_Viscosity_GBM']
        predictions = pd.DataFrame(predictions,columns=P_column)
        predictions2 = pd.DataFrame(predictions2,columns=P_column2)
        
        
        new_result = pd.concat((y_test, predictions,predictions2),axis=1)
        
        #new_result['Delta_E'] = ((y_test['Actual_L'] - predictions['Pred_L'])**2+(y_test['Actual_a'] - predictions['Pred_a'])**2+(y_test['Actual_b'] - predictions['Pred_b'])**2)**0.5 

    
        
        #R2 =  3
        R2 = []
        R2.append(round(sm.r2_score(y_test,predictions),2))
        R2.append(round(sm.r2_score(y_test,predictions2),2))

        
        st.write('Model Accuracy for Test data ($R^2$):')
        
        #st.write(R2[0])
            
        st.write( 'Linear_Regression:',R2[0],'Gradient_Boost_Machine:',R2[1] )
        
        st.write(new_result)
                
        #st.write('Model Accuracy for Total data ($R^2$):')
                
        #R2 = list(R2)
        #st.info( R2[0] )
                        
                
        length = range(y_test.shape[0])
                
            #fig, axs = plt.subplots(ncols=3)
            
        fig, ax = plt.subplots(figsize=(10,4))
            
        g = sns.lineplot(x=length,y=predictions['Pred_Gloss_Linear'],color='blue',label='LM_prediction')
        g = sns.lineplot(x=length,y=predictions2['Pred_Gloss_GBM'],color='green',label='GBM_prediction')
        g = sns.lineplot(x=length,y=y_test['Actual_Gloss'],ax=g.axes, color='red',label='actual')
        g.set_ylabel("Gloss", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            
        g1 = sns.lineplot(x=length,y=predictions['Pred_Viscosity_Linear'],color='blue',label='LM_prediction')
        g = sns.lineplot(x=length,y=predictions2['Pred_Viscosity_GBM'],color='green',label='GBM_prediction')
        g1 = sns.lineplot(x=length,y=y_test['Actual_Viscosity'],ax=g1.axes, color='red',label='actual')
        g1.set_ylabel("Viscosity", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            

                
            
 
    #st.sidebar.write('3.1 Predict Single Condition')
            
    st.sidebar.write('')
    st.sidebar.write('')
    
    
    st.sidebar.header('2. 예측 모델 선정')
                            
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
                
    select = ['Select','점도/광택 예측']
    selected2 = st.sidebar.selectbox("점도/광택 예측", select)
    
    


                
    st.write('')
    st.write('')
    st.write('')        
    
                             
    st.subheader('**2. 점도/광택 및 용제/소광제 예측**')
                
    Target_n = []
    Target_v = []
        
    if selected2 == '점도/광택 예측':
                
        st.write('**2.1 점도/광택 예측**')
        

        color_list =[]        
        color_list2 = []    
    

        #color_list = ['Color','Solvent','Agent']

        #color_list = pd.DataFrame(color_list,columns=['color'])
        
        color_names = ['Gray','Blue']
        color = st.radio('Color', color_names)
        
        if color =='Gray':
            Target_v.append('1')
        if color =='Blue':
            Target_v.append('2')
        
        X = df[Selected_X]
        y = df[Selected_y]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

        models_a.fit(X_train, y_train)
        models_b.fit(X_train, y_train)
        
        
        value1 = st.number_input('Solvent',0.00, 1000.00, 0.0,format="%.2f")
        Target_v.append(value1)
        
        value2 = st.number_input('Agent',0.00, 1000.00, 0.0,format="%.2f")
        Target_v.append(value2)
        
        color_list = ['Color','Solvent','Agent']

        New_x = pd.DataFrame([Target_v],columns=color_list)
                    
        New_x2 = pd.DataFrame(X.iloc[0,:])
                
        New_x2 = New_x2.T

                    
        col1,col2 = st.columns([1,1])


        if st.button('Run Prediction'): 
                        

            New_x.index = ['New_case']        
                
            st.write(New_x)

                #model.fit(X_train, y_train)
                        
            predictions = models_a.predict(New_x)
            predictions2 = models_b.predict(New_x)
                        
            predictions = pd.DataFrame(predictions,columns = ['Pred_Gloss_Linear','Pred_Viscosity_Linear'])
            predictions2 = pd.DataFrame(predictions2,columns = ['Pred_Gloss_GBM','Pred_Viscosity_GBM'])
            
            new_result = pd.concat((predictions,predictions2),axis=1)
            
            st.write('')
            st.write('')
                    
            st.write('**2.2 광택/점도 예측 결과**')
                        
            new_result.index = ['Results'] 

            st.write(new_result)
                

   
            st.markdown('**Download Predicted Results for Multi Conditions**')
       
            st_pandas_to_csv_download_link(new_result, file_name = "Predicted_Results.csv")
            st.write('*Save directory setting : right mouse button -> save link as')  