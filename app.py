import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


def main():
    st.title('App para predição de transtornos do humor')
    st.subheader('Preencha as respostas das 8 perguntas abaixo e obtenha o resultado')

    QEDDEP04 = st.slider("Nos últimos 7 dias... eu me senti inútil", 1, 5)
    if (QEDDEP04 == 1):
        output_text = 'Nunca'
    elif (QEDDEP04 == 2):
        output_text = 'Raramente'
    elif (QEDDEP04 == 3):
        output_text = 'Às vezes'
    elif (QEDDEP04 == 4):
        output_text = 'Bastante'
    elif (QEDDEP04 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDDEP06 = st.slider("Nos últimos 7 dias... eu me senti desamparado", 1, 5)
    if (QEDDEP06 == 1):
        output_text = 'Nunca'
    elif (QEDDEP06 == 2):
        output_text = 'Raramente'
    elif (QEDDEP06 == 3):
        output_text = 'Às vezes'
    elif (QEDDEP06 == 4):
        output_text = 'Bastante'
    elif (QEDDEP06 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDDEP29 = st.slider("Nos últimos 7 dias... eu me senti deprimido", 1, 5)
    if (QEDDEP29 == 1):
        output_text = 'Nunca'
    elif (QEDDEP29 == 2):
        output_text = 'Raramente'
    elif (QEDDEP29 == 3):
        output_text = 'Às vezes'
    elif (QEDDEP29 == 4):
        output_text = 'Bastante'
    elif (QEDDEP29 == 5):
        output_text = 'Sempre'

    QEDDEP41 = st.slider("Nos últimos 7 dias... eu me senti sem esperanças", 1, 5)
    if (QEDDEP41 == 1):
        output_text = 'Nunca'
    elif (QEDDEP41 == 2):
        output_text = 'Raramente'
    elif (QEDDEP41 == 3):
        output_text = 'Às vezes'
    elif (QEDDEP41 == 4):
        output_text = 'Bastante'
    elif (QEDDEP41 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDANX01 = st.slider("Nos últimos 7 dias... eu me senti com medo", 1, 5)
    if (QEDANX01 == 1):
        output_text = 'Nunca'
    elif (QEDANX01 == 2):
        output_text = 'Raramente'
    elif (QEDANX01 == 3):
        output_text = 'Às vezes'
    elif (QEDANX01 == 4):
        output_text = 'Bastante'
    elif (QEDANX01 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDANX30 = st.slider("Nos últimos 7 dias... eu me senti muito preocupado", 1, 5)
    if (QEDANX30 == 1):
        output_text = 'Nunca'
    elif (QEDANX30 == 2):
        output_text = 'Raramente'
    elif (QEDANX30 == 3):
        output_text = 'Às vezes'
    elif (QEDANX30 == 4):
        output_text = 'Bastante'
    elif (QEDANX30 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDANX40 = st.slider("Nos últimos 7 dias... eu achei difícil focar em algo por conta da ansiedade", 1, 5)
    if (QEDANX40 == 1):
        output_text = 'Nunca'
    elif (QEDANX40 == 2):
        output_text = 'Raramente'
    elif (QEDANX40 == 3):
        output_text = 'Às vezes'
    elif (QEDANX40 == 4):
        output_text = 'Bastante'
    elif (QEDANX40 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)

    QEDANX53 = st.slider("Nos últimos 7 dias... eu não me senti bem comigo mesmo", 1, 5)
    if (QEDANX53 == 1):
        output_text = 'Nunca'
    elif (QEDANX53 == 2):
        output_text = 'Raramente'
    elif (QEDANX53 == 3):
        output_text = 'Às vezes'
    elif (QEDANX53 == 4):
        output_text = 'Bastante'
    elif (QEDANX53 == 5):
        output_text = 'Sempre'
    st.markdown(output_text)
# 0,67 a 7,12
    if st.button('Gerar predição'):  # when the submit button is pressed
        array = np.array([QEDDEP04, QEDDEP06, QEDDEP29, QEDDEP41, QEDANX01, QEDANX30, QEDANX40, QEDANX53]).reshape(1,-1)
        model_dep = pickle.load(open('modelo-lr.sav', 'rb'))
        model_ans = pickle.load(open('modelo-lr_ans.sav', 'rb'))
        proba_dep = model_dep.predict_proba(array)
        proba_ans = model_ans.predict_proba(array)
        p_dep = np.round(proba_dep[0], decimals=3)
        if p_dep[-1] < 0.4:
            st.success('Probabilidade de episódio depressivo é de {}%'.format((p_dep[-1]) * 100))
        else:
            st.warning('Probabilidade de episódio depressivo é de = {}%'.format((p_dep[-1]) * 100))
        p_ans = np.round(proba_ans[0], decimals=3)
        if p_ans[-1] < 0.4:
            st.success('Probabilidade de ansiedade generalizada é de = {}%'.format((p_ans[-1]) * 100))
        else:
            st.warning('Probabilidade de ansiedade generalizada é de = {}%'.format((p_ans[-1]) * 100))


if __name__ == '__main__':
    main()
