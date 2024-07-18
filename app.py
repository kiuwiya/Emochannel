import streamlit as st

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1581090464777-f3220bbe1b8b?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Emochannel X")

import pandas as pd
import numpy as np
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
from collections import Counter


import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Create button for navigation
    detect_emotions_button_clicked = st.sidebar.button("Home")
    contact_us_button_clicked = st.sidebar.button("Contact Us")


    if contact_us_button_clicked:
        contact_us()
    elif detect_emotions_button_clicked:
        detect_emotion()
    else:
        detect_emotion()

def detect_emotion():
    st.subheader("Detect Emotions In Tweets")
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

            # Generate a word cloud image
            wordcloud = WordCloud(width=4000, height=2500).generate(raw_text)

            # Create a column for the word cloud image and the hashtag frequency graph
            col1, col2 = st.columns(2)

            # Display the text and image side by side
            with col1:
                    st.write("WordCloud Result:")
                    plt.figure(figsize=(20, 15))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    fig = plt.gcf()
                    st.pyplot(fig)

                    # Create a donut chart
                    donut_chart_data = proba_df_clean
                    donut_chart = px.pie(values=donut_chart_data['probability'], names=donut_chart_data['emotions'],
                                         hole=0.5)
                    st.plotly_chart(donut_chart, use_container_width=True)

            with col2:
                    st.write("Hashtag Frequency:")
                    # Get the hashtags from the user input
                    hashtags = [word for word in raw_text.split() if word.startswith("#")]
                    hashtag_freq = Counter(hashtags)

                    # Create a pie chart for hashtag frequency
                    fig = px.pie(values=list(hashtag_freq.values()), names=list(hashtag_freq.keys()),
                                 title='Hashtag Frequency')
                    fig.update_layout(legend_title="Hashtags", legend_y=0.9)
                    fig.update_traces(textinfo='percent+label', textposition='inside')
                    st.plotly_chart(fig, use_container_width=True)


def contact_us():
    st.title("Contact Us")
    st.subheader("Contact Information")
    st.write("For any inquiries or support, please contact us at:")
    st.write("- Email: sofea@emochannelx.com")
    st.write("- Phone: 60+ 1156416813")

    st.subheader("Our Office Location")
    st.write("UITM JASIN ")
    st.write("MALACCA, MALAYSIA")

if __name__ == '__main__':
    main()
